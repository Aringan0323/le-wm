"""Evaluate continuous goal sequences from paused native Delta-JEPA checkpoints."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict, replace
import fcntl
import inspect
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from audit_combined_experiment import check, read_json
from delta_sweep_config import VARIANTS
from evaluate_delta_experiment import verified_split
from evaluate_episode_experiment import file_hash, json_hash, now, write_json
from experiment_tracking import state_dict_sha256
from plan_delta_checkpoints import load_native, verify_endpoints, verify_paused_group
from plan_delta_experiment import PROTOCOL, TerminalCost, image_embedding, pin_json, shared_trials
from utils import get_img_preprocessor


def goal_steps(length, spacing):
    check(length > 1 and spacing > 0, "Invalid episode length or goal spacing")
    return sorted(set(range(spacing, length, spacing)) | {length - 1})


def episode_specification(bundle, count, seed):
    protocol = replace(PROTOCOL, trajectories=count, solver_seeds=(seed,))
    selection = shared_trials(bundle, protocol)
    arrays = bundle["arrays"]
    episodes = []
    for trial in selection["trials"]:
        episode = trial["episode"]
        length = int(arrays["episode_lengths"][episode])
        offset = int(arrays["episode_offsets"][episode])
        steps = goal_steps(length, protocol.goal_offset)
        episodes.append({"episode":episode, "length":length, "start":0, "start_row":offset,
                         "goal_steps":steps, "goal_rows":[offset + step for step in steps]})
    spec = {"evaluation_mode":"continuous_episodes", "protocol":json.loads(json.dumps(asdict(protocol))),
            "split_arrays_sha256":selection["split_arrays_sha256"],
            "eligible_held_out_episodes":selection["eligible_held_out_episodes"], "episodes":episodes,
            "expected_tasks":sum(len(ep["goal_steps"]) for ep in episodes)}
    return protocol, spec


def position(env):
    return env.agent_position.detach().cpu().numpy().astype(float).tolist()


def run_episode(model, env, solver, episode, inputs, mean, std, preprocess, device,
                protocol, seed, on_goal=None):
    episode_id = episode["episode"]
    start = np.asarray(inputs["proprio"][episode["start_row"]], dtype=np.float32)
    first_target = np.asarray(inputs["proprio"][episode["goal_rows"][0]], dtype=np.float32)
    env.reset(seed=episode_id, options={"state":start, "target_state":first_target})
    check(np.allclose(position(env), start, rtol=0, atol=1e-6), "Episode reset did not restore frame zero")
    tasks = []
    cumulative_steps = 0
    for index, (goal_step, goal_row) in enumerate(zip(episode["goal_steps"], episode["goal_rows"])):
        target = np.asarray(inputs["proprio"][goal_row], dtype=np.float32)
        before_target_change = position(env)
        env._set_goal_state(target)
        check(position(env) == before_target_change, "Changing the goal moved the agent")
        solver_seed = int(np.random.SeedSequence([seed, episode_id, goal_step]).generate_state(1)[0])
        solver.torch_gen.manual_seed(solver_seed)
        actual_start = position(env)
        first_distance = float(torch.norm(env.agent_position - env.target_position))
        reached = first_distance < 16.0
        steps = replans = 0
        distance = first_distance
        if not reached:
            goal = image_embedding(model, inputs["pixels"][goal_row], preprocess, device)
        while not reached and steps < protocol.budget:
            state = image_embedding(model, env.render(), preprocess, device)
            outputs = solver({"start":state, "goal":goal})
            candidates = outputs["actions"][0].detach().cpu().reshape(-1, 2).numpy()
            actions = candidates * std + mean
            check(actions.shape == (protocol.horizon * protocol.action_block, 2)
                  and np.isfinite(actions).all(), "Invalid planned action sequence")
            replans += 1
            for action in actions:
                _, _, terminated, truncated, info = env.step(action)
                steps += 1
                distance = float(info["distance_to_target"])
                reached = distance < 16.0
                check(not truncated and np.isfinite(distance) and bool(terminated) == reached,
                      "Environment success threshold or transition changed")
                if reached or steps == protocol.budget:
                    break
        cumulative_start = cumulative_steps
        cumulative_steps += steps
        task = {"goal_index":index, "goal_step":goal_step, "goal_row":goal_row,
                "target_position":target.astype(float).tolist(), "actual_start_position":actual_start,
                "actual_end_position":position(env), "initial_distance":first_distance,
                "final_distance":distance, "success":bool(reached), "zero_action_success":bool(reached and steps == 0),
                "steps":steps, "replans":replans, "solver_seed":solver_seed,
                "cumulative_steps_before":cumulative_start, "cumulative_steps_after":cumulative_steps}
        tasks.append(task)
        if on_goal is not None:
            on_goal(tasks)
    return {"tasks":tasks, "success":all(task["success"] for task in tasks),
            "final_goal_success":tasks[-1]["success"], "physical_steps":cumulative_steps,
            "actual_initial_position":start.astype(float).tolist(), "actual_final_position":position(env),
            "environment_resets":1}


def validate_episode(record, episode, seed, fingerprint, protocol):
    content = {key:value for key,value in record.items() if key != "record_sha256"}
    check(json_hash(content) == record["record_sha256"] and record["plan_sha256"] == fingerprint
          and record["episode"] == episode and record["evaluation_seed"] == seed, "Invalid saved episode identity")
    outcome = record["outcome"]
    tasks = outcome["tasks"]
    check(len(tasks) == len(episode["goal_steps"]) and outcome["environment_resets"] == 1,
          "Incomplete continuous episode")
    previous_position = outcome["actual_initial_position"]
    cumulative = 0
    for index, task in enumerate(tasks):
        check(task["goal_index"] == index and task["goal_step"] == episode["goal_steps"][index]
              and task["goal_row"] == episode["goal_rows"][index], "Saved goal schedule changed")
        check(task["actual_start_position"] == previous_position and task["cumulative_steps_before"] == cumulative,
              "Saved episode violates state continuity")
        vectors = np.asarray([task[name] for name in
                              ("actual_start_position", "actual_end_position", "target_position")], dtype=float)
        check(vectors.shape == (3, 2) and np.isfinite(vectors).all(), "Invalid recorded positions")
        check(isinstance(task["success"], bool) and 0 <= task["steps"] <= protocol.budget
              and (task["success"] or task["steps"] == protocol.budget), "Incomplete saved goal")
        check(task["zero_action_success"] == (task["success"] and task["steps"] == 0)
              and task["success"] == (task["final_distance"] < 16.0), "Incorrect goal success classification")
        for field, a in (("initial_distance", vectors[0]), ("final_distance", vectors[1])):
            check(np.isclose(task[field], np.linalg.norm(a-vectors[2]), rtol=1e-5, atol=1e-5),
                  "Distance does not match recorded actual state")
        expected_replans = (task["steps"] + protocol.horizon*protocol.action_block - 1) // (protocol.horizon*protocol.action_block)
        check(task["replans"] == expected_replans, "Saved replan count differs")
        expected_seed = int(np.random.SeedSequence([seed, episode["episode"], task["goal_step"]]).generate_state(1)[0])
        check(task["solver_seed"] == expected_seed, "Saved planner random seed differs")
        cumulative += task["steps"]
        check(task["cumulative_steps_after"] == cumulative, "Saved physical step total differs")
        previous_position = task["actual_end_position"]
    check(outcome["success"] == all(task["success"] for task in tasks)
          and outcome["final_goal_success"] == tasks[-1]["success"]
          and outcome["physical_steps"] == cumulative and outcome["actual_final_position"] == previous_position,
          "Saved episode totals differ from its goals")
    return outcome


def summarize(outcomes, expected_episodes, expected_tasks, partial_tasks=()):
    tasks = [task for outcome in outcomes for task in outcome["tasks"]] + list(partial_tasks)
    completed_episodes, completed_tasks = len(outcomes), len(tasks)
    episode_successes = sum(int(outcome["success"]) for outcome in outcomes)
    task_successes = sum(int(task["success"]) for task in tasks)
    final_successes = sum(int(outcome["final_goal_success"]) for outcome in outcomes)
    task_rate = 100*task_successes/completed_tasks if completed_tasks else 0.0
    return {"completed_episodes":completed_episodes, "expected_episodes":expected_episodes,
            "episode_successes":episode_successes,
            "episode_success_rate_percent":100*episode_successes/completed_episodes if completed_episodes else 0.0,
            "completed_tasks":completed_tasks, "expected_tasks":expected_tasks,
            "task_successes":task_successes, "task_success_rate_percent":task_rate,
            "zero_action_successes":sum(int(task["zero_action_success"]) for task in tasks),
            "final_goal_successes":final_successes,
            "final_goal_success_rate_percent":100*final_successes/completed_episodes if completed_episodes else 0.0,
            "physical_steps":sum(task["steps"] for task in tasks),
            "completed_trials":completed_tasks, "expected_trials":expected_tasks,
            "successes":task_successes, "success_rate_percent":task_rate}


def group_totals(models):
    names = ("completed_episodes", "expected_episodes", "episode_successes", "completed_tasks", "expected_tasks",
             "task_successes", "zero_action_successes", "final_goal_successes", "physical_steps",
             "completed_trials", "expected_trials", "successes")
    totals = {name:sum(row[name] for row in models.values()) for name in names}
    for metric, successes, count in (
        ("episode_success_rate_percent", "episode_successes", "completed_episodes"),
        ("task_success_rate_percent", "task_successes", "completed_tasks"),
        ("final_goal_success_rate_percent", "final_goal_successes", "completed_episodes"),
        ("success_rate_percent", "successes", "completed_trials")):
        totals[metric] = 100*totals[successes]/totals[count] if totals[count] else 0.0
    return totals


def evaluate(root, count, seed, prepare_only):
    import gymnasium as gym
    import h5py
    import hdf5plugin
    import stable_worldmodel as swm
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv
    from stable_worldmodel.solver import CEMSolver

    record = verify_paused_group(root)
    bundle, _ = verified_split(root)
    check(bundle["manifest"]["manifest_sha256"] == record["manifest_sha256"], "Checkpoint split mismatch")
    protocol, spec = episode_specification(bundle, count, seed)
    folder = root / "planning" / f"episodes-checkpoint-{record['global_step']}"
    folder.mkdir(parents=True, exist_ok=True)
    pin_json(folder / "checkpoint.json", record)
    pin_json(folder / "episodes.json", spec)
    env = TwoRoomEnv()
    inputs = h5py.File(bundle["manifest"]["dataset"]["path"], "r")
    try:
        endpoints = [{"episode":episode["episode"], "start":0, "start_row":episode["start_row"],
                      "goal":step, "goal_row":row} for episode in spec["episodes"]
                     for step,row in zip(episode["goal_steps"],episode["goal_rows"])]
        rendered = verify_endpoints(inputs, env, endpoints)
        pin_json(folder / "endpoint-check.json", rendered)
        source = Path(__file__).resolve().parents[1]
        plan = {"evaluation_mode":"continuous_episodes", "checkpoint_record_sha256":record["record_sha256"],
                "episodes_sha256":json_hash(spec), "protocol":spec["protocol"], "precision":"fp32", "allow_tf32":False,
                "script_sha256":file_hash(Path(__file__)),
                "helper_sha256":{name:file_hash(source/"scripts"/name) for name in
                                  ("plan_delta_checkpoints.py","plan_delta_experiment.py")},
                "solver_sha256":file_hash(Path(inspect.getfile(CEMSolver))),
                "environment_sha256":file_hash(Path(inspect.getfile(TwoRoomEnv))),
                "goal_spacing":protocol.goal_offset, "final_recorded_goal_included":True,
                "budget_per_goal":protocol.budget, "success_distance_strictly_below":16,
                "episode_success":"Every scheduled goal succeeds during a single continuous rollout.",
                "failure_policy":"Continue the next goal from the actual final simulator state.",
                "resume_unit":"One complete episode.", "endpoints":rendered}
        fingerprint = json_hash(plan)
        pin_json(folder / "plan.json", {"plan":plan, "fingerprint":fingerprint})
        if prepare_only:
            for key in VARIANTS:
                model, digest, _ = load_native(root, key, record)
                print(json.dumps({"key":key,"checkpoint_step":record["global_step"],"state_sha256":digest}),flush=True)
                del model
            print(json.dumps({"prepared":True,"folder":str(folder),"episodes":count,
                              "tasks_per_model":spec["expected_tasks"],"endpoints":rendered}),flush=True)
            return

        check(torch.cuda.is_available(), "Continuous planning requires CUDA")
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device("cuda")
        normalization = bundle["manifest"]["normalization"]["action"]
        mean = np.asarray(normalization["mean"], dtype=np.float32).reshape(1, 2)
        std = np.asarray(normalization["std"], dtype=np.float32).reshape(1, 2)
        preprocess = get_img_preprocessor("pixels", "pixels", 224)
        models = {key:{"status":"pending",**summarize([],count,spec["expected_tasks"])} for key in VARIANTS}
        start_time, fresh_tasks, active = time.perf_counter(), 0, None

        def publish(status, error=None):
            totals = group_totals(models)
            elapsed = time.perf_counter()-start_time
            progress = {"evaluation_mode":"continuous_episodes", "status":status, "pid":os.getpid(),
                        "checkpoint_step":record["global_step"], "active_model":active, "models":models,
                        "updated_utc":now(), "elapsed_seconds":elapsed, **totals,
                        "eta_seconds":elapsed*(totals["expected_tasks"]-totals["completed_tasks"])/fresh_tasks
                                      if fresh_tasks else None}
            if error:
                progress["error"] = str(error)
            write_json(folder / "progress.json", progress)
            return progress

        write_json(root / "planning" / "active.json", {"directory":folder.name, "pid":os.getpid(),
                   "checkpoint_step":record["global_step"],"evaluation_mode":"continuous_episodes"})
        publish("running")
        try:
            with torch.inference_mode():
                for key in VARIANTS:
                    active = key
                    models[key]["status"] = "running"
                    publish("running")
                    model, before, _ = load_native(root, key, record)
                    model.to(device)
                    model_folder = folder / key
                    episode_folder = model_folder / "episodes"
                    episode_folder.mkdir(parents=True, exist_ok=True)
                    member_plan = {"plan_sha256":fingerprint, "variant":key, "state_sha256":before,
                                   "checkpoint_sha256":record["members"][key]["sha256"]}
                    member_fingerprint = json_hash(member_plan)
                    pin_json(model_folder / "plan.json", member_plan)
                    solver = CEMSolver(TerminalCost(model),batch_size=1,num_samples=protocol.candidates,
                                       var_scale=protocol.initial_std,n_steps=protocol.iterations,
                                       topk=protocol.elites,device=device)
                    solver.configure(action_space=gym.spaces.Box(-1.0,1.0,shape=(1,2),dtype=np.float32),n_envs=1,
                        config=swm.PlanConfig(horizon=protocol.horizon,action_block=protocol.action_block,
                                             receding_horizon=protocol.horizon))
                    outcomes = []
                    for episode in spec["episodes"]:
                        check(read_json(root/"control.json")["request"] == "pause", "Training control changed during planning")
                        path = episode_folder / f"{episode['episode']}.json"
                        if not path.exists():
                            def on_goal(tasks):
                                nonlocal fresh_tasks
                                check(read_json(root/"control.json")["request"] == "pause",
                                      "Training control changed during planning")
                                fresh_tasks += 1
                                models[key].update(summarize(outcomes,count,spec["expected_tasks"],tasks))
                                models[key].update(active_episode=episode["episode"],active_goal_step=tasks[-1]["goal_step"])
                                publish("running")
                                print(f"{now()} {key} episode {episode['episode']} goal {tasks[-1]['goal_step']} "
                                      f"success={tasks[-1]['success']} steps={tasks[-1]['steps']}",flush=True)
                            outcome = run_episode(model,env,solver,episode,inputs,mean,std,preprocess,device,
                                                  protocol,seed,on_goal)
                            item = {"episode":episode, "evaluation_seed":seed,
                                    "plan_sha256":member_fingerprint, "outcome":outcome}
                            write_json(path, {**item,"record_sha256":json_hash(item)})
                        outcome = validate_episode(read_json(path),episode,seed,member_fingerprint,protocol)
                        outcomes.append(outcome)
                        models[key].update(summarize(outcomes,count,spec["expected_tasks"]))
                        models[key].pop("active_episode",None)
                        models[key].pop("active_goal_step",None)
                        publish("running")
                    after = state_dict_sha256(model.cpu().state_dict())
                    check(before == after, "Continuous planning mutated model weights or buffers")
                    check(file_hash(Path(record["members"][key]["path"])) == record["members"][key]["sha256"],
                          "Native checkpoint changed during planning")
                    models[key].update(status="complete",model_state_unchanged=True,model_state_sha256=before,
                                       checkpoint_sha256=record["members"][key]["sha256"])
                    write_json(model_folder/"results.json", {"passed":True,"evaluation_mode":"continuous_episodes",
                               "checkpoint_step":record["global_step"],**models[key]})
                    del solver, model
                    torch.cuda.empty_cache()
                    publish("running")
            check(verify_paused_group(root)["record_sha256"] == record["record_sha256"], "Common checkpoint changed")
            active = None
            result = {**publish("complete"),"passed":True,"plan_sha256":fingerprint,"episodes_sha256":json_hash(spec),
                      "protocol":spec["protocol"],"test_episodes_per_model":count,"tasks_per_model":spec["expected_tasks"],
                      "checkpoint_record_sha256":record["record_sha256"],"model_states_unchanged":True,"finished_utc":now()}
            write_json(folder/"results.json",result)
            print(json.dumps(result),flush=True)
        except BaseException as exc:
            publish("failed",exc)
            raise
    finally:
        inputs.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir",type=Path,required=True)
    parser.add_argument("--episodes",type=int,default=20)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--prepare-only",action="store_true")
    args = parser.parse_args()
    check(args.episodes > 0, "Episode count must be positive")
    root = args.experiment_dir.resolve(strict=True)
    with ExitStack() as stack:
        for name in ("pipeline.lock","group-training.lock"):
            lock = stack.enter_context((root/name).open("rb"))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        evaluate(root,args.episodes,args.seed,args.prepare_only)


if __name__ == "__main__":
    main()
