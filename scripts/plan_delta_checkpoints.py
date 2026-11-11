"""Short, paired planning evaluation of a paused group's native checkpoints."""
import argparse
from contextlib import ExitStack
from dataclasses import replace, asdict
import fcntl
import inspect
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from audit_combined_experiment import check, read_json
from delta_sweep_config import VARIANTS
from evaluate_delta_experiment import verified_split
from evaluate_episode_experiment import file_hash, json_hash, now, write_json
from experiment_tracking import state_dict_sha256
from plan_delta_experiment import (PROTOCOL, TerminalCost, pin_json, run_trial,
                                   shared_trials, validate_record)
from run_episode_comparison import verify_source
from utils import get_img_preprocessor


def verify_paused_group(root):
    check(read_json(root / "control.json")["request"] == "pause", "Training must remain paused")
    check(read_json(root / "status.json")["status"] == "paused", "Pipeline has not completed its pause")
    record = read_json(root / "group-checkpoint.json")
    check(record["status"] == "paused" and set(record["members"]) == set(VARIANTS), "Incomplete paused group")
    check(record["record_sha256"] == json_hash({k:v for k,v in record.items() if k != "record_sha256"}),
          "Invalid common checkpoint checksum")
    verify_source(root)
    for name in ("jepa.py", "module.py", "utils.py"):
        check(file_hash(root / "source" / name) == file_hash(Path(__file__).resolve().parents[1] / name),
              f"Model source differs from the training snapshot: {name}")
    return record


def load_native(root, key, record):
    row = record["members"][key]
    run = root / "members" / key / "runs" / key
    path = Path(row["path"]).resolve(strict=True)
    check(path.parent == (run / "checkpoints").resolve()
          and path.name == row["generation_id"] + ".ckpt", "Checkpoint path is outside the member")
    check(path.stat().st_size == row["size_bytes"] and file_hash(path) == row["sha256"],
          "Native checkpoint checksum or size mismatch")
    cfg = OmegaConf.load(run / "config.yaml")
    identity = read_json(run / "config-identity.json")
    resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    fingerprint = json_hash({k:v for k,v in resolved.items() if k not in identity["excluded_runtime_keys"]})
    check(fingerprint == identity["config_sha256"] == row["config_sha256"], "Configuration checksum mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = payload["resumable"]
    check(payload["global_step"] == row["global_step"] == record["global_step"]
          and meta["generation_id"] == row["generation_id"]
          and meta["config_sha256"] == fingerprint
          and meta["manifest_sha256"] == record["manifest_sha256"]
          and meta["cursor"] == record["cursor"] == row["windows_seen"]
          and meta["order_sha256"] == record["order_sha256"]
          and meta["trace_sha256"] == record["trace_sha256"], "Inconsistent native checkpoint identity or cursor")
    variant = VARIANTS[key]
    check(cfg.loss.ldad.weight == variant.ldad_weight and cfg.loss.sigreg.weight == variant.sigreg_weight
          and (cfg.model.ldad is not None) == (variant.ldad_weight > 0), "Wrong checkpoint objective")
    model = hydra.utils.instantiate(cfg.model)
    state = {k.removeprefix("model."):v for k,v in payload["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    digest = state_dict_sha256(model.state_dict())
    del payload, state
    return model, digest, cfg


def verify_endpoints(inputs, env, trials):
    errors = []
    for trial in trials:
        ep = trial["episode"]
        for row, local_step in ((trial["start_row"], trial["start"]), (trial["goal_row"], trial["goal"])):
            check(int(inputs["ep_idx"][row]) == ep and int(inputs["step_idx"][row]) == local_step,
                  "Held-out endpoint row is inconsistent")
            env.reset(seed=ep, options={"state":inputs["proprio"][row], "target_state":inputs["proprio"][trial["goal_row"]]})
            actual, cached = env.render(), inputs["pixels"][row]
            check(actual.shape == cached.shape, "Simulator and recorded image shapes differ")
            error = int(np.abs(actual.astype(np.int16) - cached.astype(np.int16)).max())
            check(error <= 1, f"Simulator render differs from recorded state (episode {ep}, max error {error})")
            errors.append(error)
    return {"endpoints_checked":len(errors), "maximum_pixel_error":max(errors),
            "pixel_tolerance":1, "interpretation":"At most one uint8 level of renderer rounding."}


def evaluate(root, trajectories, seed, prepare_only):
    import gymnasium as gym
    import h5py
    import hdf5plugin
    import stable_worldmodel as swm
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv
    from stable_worldmodel.solver import CEMSolver

    record = verify_paused_group(root)
    bundle, _ = verified_split(root)
    check(bundle["manifest"]["manifest_sha256"] == record["manifest_sha256"], "Checkpoint split mismatch")
    protocol = replace(PROTOCOL, trajectories=trajectories, solver_seeds=(seed,))
    spec = shared_trials(bundle, protocol)
    folder = root / "planning" / f"checkpoint-{record['global_step']}"
    folder.mkdir(parents=True, exist_ok=True)
    pin_json(folder / "checkpoint.json", record)
    pin_json(folder / "trials.json", spec)
    env = TwoRoomEnv()
    inputs = h5py.File(bundle["manifest"]["dataset"]["path"], "r")
    try:
        endpoints = verify_endpoints(inputs, env, spec["trials"])
        pin_json(folder / "endpoint-check.json", endpoints)
        source = Path(__file__).resolve().parents[1]
        plan = {"checkpoint_record_sha256":record["record_sha256"], "trials_sha256":json_hash(spec),
                "protocol":json.loads(json.dumps(asdict(protocol))), "precision":"fp32", "allow_tf32":False,
                "script_sha256":file_hash(Path(__file__)),
                "planning_helper_sha256":file_hash(source / "scripts" / "plan_delta_experiment.py"),
                "solver_sha256":file_hash(Path(inspect.getfile(CEMSolver))),
                "environment_sha256":file_hash(Path(inspect.getfile(TwoRoomEnv))),
                "selection":"Uniformly select distinct held-out episodes, then a valid start within each episode.",
                "scope":"Short intermediate-checkpoint goal-reaching test using released Two-Room planning settings.",
                "endpoints":endpoints}
        fingerprint = json_hash(plan)
        pin_json(folder / "plan.json", {"plan":plan, "fingerprint":fingerprint})
        if prepare_only:
            for key in VARIANTS:
                model, digest, _ = load_native(root, key, record)
                print(json.dumps({"key":key,"checkpoint_step":record["global_step"],"state_sha256":digest}), flush=True)
                del model
            print(json.dumps({"prepared":True,"folder":str(folder),"endpoints":endpoints}), flush=True)
            return

        check(torch.cuda.is_available(), "Planning requires CUDA")
        device = torch.device("cuda")
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        mean = np.asarray(bundle["manifest"]["normalization"]["action"]["mean"], dtype=np.float32).reshape(1, 2)
        std = np.asarray(bundle["manifest"]["normalization"]["action"]["std"], dtype=np.float32).reshape(1, 2)
        preprocess = get_img_preprocessor("pixels", "pixels", 224)
        models = {key:{"status":"pending","completed_trials":0,"expected_trials":trajectories,
                       "successes":0,"success_rate_percent":0.0} for key in VARIANTS}
        started = time.perf_counter()
        fresh = 0
        active = None

        def publish(status, error=None):
            completed = sum(m["completed_trials"] for m in models.values())
            elapsed = time.perf_counter() - started
            progress = {"status":status,"pid":os.getpid(),"checkpoint_step":record["global_step"],
                        "completed_trials":completed,"expected_trials":trajectories*len(models),
                        "active_model":active,"models":models,"updated_utc":now(),"elapsed_seconds":elapsed,
                        "eta_seconds":elapsed*(trajectories*len(models)-completed)/fresh if fresh else None}
            if error:
                progress["error"] = str(error)
            write_json(folder / "progress.json", progress)
            return progress

        write_json(root / "planning" / "active.json", {"directory":folder.name, "pid":os.getpid(),
                                                      "checkpoint_step":record["global_step"]})
        publish("running")
        try:
            with torch.inference_mode():
                for key in VARIANTS:
                    active = key
                    models[key]["status"] = "running"
                    publish("running")
                    model, before, cfg = load_native(root, key, record)
                    model.to(device)
                    model_folder = folder / key
                    model_folder.mkdir(exist_ok=True)
                    trial_folder = model_folder / "trials"
                    trial_folder.mkdir(exist_ok=True)
                    member_plan = {"plan_sha256":fingerprint,"variant":key,"state_sha256":before,
                                   "checkpoint_sha256":record["members"][key]["sha256"]}
                    member_fingerprint = json_hash(member_plan)
                    pin_json(model_folder / "plan.json", member_plan)
                    solver = CEMSolver(TerminalCost(model),batch_size=1,num_samples=protocol.candidates,
                                       var_scale=protocol.initial_std,n_steps=protocol.iterations,
                                       topk=protocol.elites,device=device)
                    solver.configure(action_space=gym.spaces.Box(-1.0,1.0,shape=(1,2),dtype=np.float32),n_envs=1,
                                     config=swm.PlanConfig(horizon=protocol.horizon,action_block=protocol.action_block,
                                                           receding_horizon=protocol.horizon))
                    for trial in spec["trials"]:
                        check(read_json(root / "control.json")["request"] == "pause", "Training control changed during planning")
                        path = trial_folder / f"{trial['episode']}.json"
                        if not path.exists():
                            sub_seed = int(np.random.SeedSequence([seed,trial["episode"]]).generate_state(1)[0])
                            solver.torch_gen.manual_seed(sub_seed)
                            outcome = run_trial(model,env,solver,trial,inputs,mean,std,preprocess,device,protocol)
                            item = {"trial":trial,"evaluation_seed":seed,"solver_seed":sub_seed,
                                    "plan_sha256":member_fingerprint,"outcome":outcome}
                            write_json(path,{**item,"record_sha256":json_hash(item)})
                            fresh += 1
                        success = validate_record(read_json(path),trial,seed,member_fingerprint,protocol)
                        models[key]["completed_trials"] += 1
                        models[key]["successes"] += int(success)
                        models[key]["success_rate_percent"] = 100*models[key]["successes"]/models[key]["completed_trials"]
                        publish("running")
                        print(f"{now()} {key} {models[key]['completed_trials']}/{trajectories}, successes {models[key]['successes']}",flush=True)
                    after = state_dict_sha256(model.cpu().state_dict())
                    check(before == after, "Evaluation mutated model weights or buffers")
                    check(file_hash(Path(record["members"][key]["path"])) == record["members"][key]["sha256"],
                          "Checkpoint file changed during evaluation")
                    models[key].update(status="complete",model_state_unchanged=True,model_state_sha256=before,
                                       checkpoint_sha256=record["members"][key]["sha256"])
                    write_json(model_folder / "results.json", {"passed":True,**models[key],"checkpoint_step":record["global_step"]})
                    del solver, model
                    torch.cuda.empty_cache()
                    publish("running")
            check(verify_paused_group(root)["record_sha256"] == record["record_sha256"], "Common checkpoint changed")
            active = None
            result = {**publish("complete"),"passed":True,"plan_sha256":fingerprint,"trials_sha256":json_hash(spec),
                      "protocol":plan["protocol"],"test_episodes":trajectories,"finished_utc":now(),
                      "checkpoint_record_sha256":record["record_sha256"],"model_states_unchanged":True,
                      "interpretation":f"Intermediate checkpoints, one planner seed, {trajectories} paired held-out tasks; full paper-scale reproduction is not established."}
            write_json(folder / "results.json", result)
            print(json.dumps(result),flush=True)
        except BaseException as exc:
            publish("failed",exc)
            raise
    finally:
        inputs.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir",required=True,type=Path)
    parser.add_argument("--trajectories",type=int,default=20)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--prepare-only",action="store_true")
    args = parser.parse_args()
    check(args.trajectories > 0, "Trajectory count must be positive")
    root = args.experiment_dir.resolve(strict=True)
    with ExitStack() as stack:
        for name in ("pipeline.lock","group-training.lock"):
            lock = stack.enter_context((root/name).open("rb"))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        evaluate(root,args.trajectories,args.seed,args.prepare_only)


if __name__ == "__main__":
    main()
