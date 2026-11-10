"""Score a frozen Two-Room model by closed-loop goal reaching on matched test trajectories."""
import argparse
from dataclasses import asdict, dataclass
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from audit_combined_experiment import check, read_json
from delta_sweep_config import VARIANTS
from evaluate_delta_experiment import SPLIT_ARRAYS_SHA256, delta_checkpoint, verified_split
from evaluate_episode_experiment import file_hash, json_hash, now, write_json


@dataclass(frozen=True)
class PlanningProtocol:
    selection_seed: int = 20261110
    solver_seeds: tuple = (42, 43, 44)
    trajectories: int = 500
    # Released LeWM evaluation settings, corrected by its maintainer in issue 38.
    goal_offset: int = 25
    budget: int = 50
    horizon: int = 5
    action_block: int = 5
    candidates: int = 300
    iterations: int = 30
    elites: int = 30
    initial_std: float = 1.0


PROTOCOL = PlanningProtocol()


def shared_trials(bundle, protocol=PROTOCOL):
    arr = bundle["arrays"]
    held_out = arr["test_episode_ids"]
    check(not np.intersect1d(held_out, arr["train_episode_ids"]).size
          and not np.intersect1d(held_out, arr["unused_episode_ids"]).size, "Planning/test split overlaps training")
    eligible = held_out[arr["episode_lengths"][held_out] > protocol.goal_offset]
    check(len(eligible) >= protocol.trajectories, "Too few held-out trajectories with the requested goal offset")
    rng = np.random.default_rng(protocol.selection_seed)
    selected = np.sort(rng.choice(eligible, size=protocol.trajectories, replace=False))
    trials = []
    for ep in selected:
        start = int(rng.integers(int(arr["episode_lengths"][ep]) - protocol.goal_offset))
        start_row = int(arr["episode_offsets"][ep]) + start
        trials.append({"episode": int(ep), "start": start, "goal": start + protocol.goal_offset,
                       "start_row": start_row, "goal_row": start_row + protocol.goal_offset})
    return {"protocol": json.loads(json.dumps(asdict(protocol))), "split_arrays_sha256": SPLIT_ARRAYS_SHA256,
            "eligible_held_out_episodes": len(eligible), "trials": trials}


def pin_json(path, value):
    if path.exists():
        check(read_json(path) == value, f"Saved planning specification differs at {path}")
    else:
        write_json(path, value)


class TerminalCost(torch.nn.Module):
    """LeWM's five-block rollout and terminal goal cost with observation encodings cached per replan."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def get_cost(self, info, candidates):
        batch, samples, horizon, width = candidates.shape
        emb = info["start"].reshape(batch * samples, 1, -1)
        goal = info["goal"].reshape(batch * samples, 1, -1)
        actions = self.model.action_encoder(candidates.reshape(batch * samples, horizon, width))
        for t in range(horizon):
            first = max(0, t - 2)
            pred = self.model.predict(emb[:, -3:], actions[:, first:t + 1])[:, -1:]
            emb = torch.cat([emb, pred], dim=1)
        return (pred - goal).square().sum(dim=(1, 2)).reshape(batch, samples)


def image_embedding(model, image, preprocess, device):
    frame = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None]
    pixels = preprocess({"pixels": frame})["pixels"].to(device)
    return model.encode({"pixels": pixels.unsqueeze(0)})["emb"].detach()


def run_trial(model, env, solver, row, inputs, action_mean, action_std, preprocess, device, protocol=PROTOCOL):
    ep, start_row, goal_row = row["episode"], row["start_row"], row["goal_row"]
    check(int(inputs["ep_idx"][start_row]) == ep == int(inputs["ep_idx"][goal_row])
          and int(inputs["step_idx"][start_row]) == row["start"] and int(inputs["step_idx"][goal_row]) == row["goal"],
          "Trial endpoints do not match the held-out HDF5 rows")
    initial, target = inputs["proprio"][start_row], inputs["proprio"][goal_row]
    env.reset(seed=ep, options={"state": initial, "target_state": target})
    frame = inputs["pixels"][start_row]
    goal = image_embedding(model, inputs["pixels"][goal_row], preprocess, device)
    first_distance = float(np.linalg.norm(initial - target))
    steps = replans = 0
    distance = first_distance
    while steps < protocol.budget:
        state = image_embedding(model, frame, preprocess, device)
        outputs = solver({"start": state, "goal": goal})
        action_block = outputs["actions"][0].reshape(-1, 2).numpy()
        physical_actions = action_block * action_std + action_mean
        check(np.isfinite(physical_actions).all(), "Nonfinite planned action")
        replans += 1
        for action in physical_actions:
            _, _, reached, truncated, info = env.step(action)
            steps += 1
            distance = float(info["distance_to_target"])
            check(not truncated and np.isfinite(distance), "Invalid environment transition")
            if reached or steps == protocol.budget:
                return {"success": bool(reached), "steps": steps, "replans": replans,
                        "initial_distance": first_distance, "final_distance": distance}
        frame = env.render()


def validate_record(record, trial, seed, fingerprint, protocol):
    content = {k: v for k, v in record.items() if k != "record_sha256"}
    check(json_hash(content) == record["record_sha256"] and record["plan_sha256"] == fingerprint
          and record["trial"] == trial and record["evaluation_seed"] == seed, "Invalid saved trial")
    outcome = record["outcome"]
    check(isinstance(outcome["success"], bool) and 1 <= outcome["steps"] <= protocol.budget
          and outcome["replans"] == (outcome["steps"] - 1) // (protocol.horizon * protocol.action_block) + 1
          and (outcome["success"] or outcome["steps"] == protocol.budget), "Incomplete trial")
    return bool(outcome["success"])


def evaluate(experiment, variant, prepare_only=False):
    folder = experiment / "planning"
    folder.mkdir(parents=True, exist_ok=True)
    bundle, _ = verified_split(experiment)
    spec = shared_trials(bundle)
    pin_json(folder / "trials.json", spec)
    trial_hash = json_hash(spec)
    print(f"{now()} Fixed {len(spec['trials'])} held-out start/goal pairs with SHA256 {trial_hash}", flush=True)
    if prepare_only:
        return
    import gymnasium as gym
    import h5py
    import hdf5plugin
    import stable_worldmodel as swm
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv
    from stable_worldmodel.solver import CEMSolver
    from experiment_tracking import state_dict_sha256
    from utils import get_img_preprocessor

    device = torch.device("cuda")
    check(torch.cuda.is_available(), "Closed-loop full evaluation requires CUDA")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = bundle["manifest"]
    checkpoint = delta_checkpoint(variant, experiment, Path(os.environ["STABLEWM_HOME"]), manifest["manifest_sha256"])
    audit = read_json(experiment / "runs" / variant.label / "final-audit.json")["audit"]
    weights = Path(checkpoint["files"]["weights"]["path"])
    model = swm.wm.utils.load_pretrained(f"{variant.model}/{weights.name}").eval().requires_grad_(False)
    previous = state_dict_sha256(model.state_dict())
    check(previous == audit["export"]["state_sha256"], "Wrong model loaded for planning")
    model.to(device)
    env = TwoRoomEnv()
    protocol = PROTOCOL
    plan = {"trials_sha256": trial_hash, "variant": variant.label, "checkpoint": checkpoint,
            "model_state_sha256": previous, "precision": "fp32", "allow_tf32": False, "device": torch.cuda.get_device_name(0),
            "normalization": manifest["normalization"]["action"], "script_sha256": file_hash(Path(__file__)),
            "model_source_sha256": {name: file_hash(Path(__file__).resolve().parents[1] / name)
                                    for name in ("jepa.py", "module.py", "utils.py")},
            "solver_sha256": file_hash(Path(inspect.getfile(CEMSolver))), "env_sha256": file_hash(Path(inspect.getfile(TwoRoomEnv))),
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "stable-worldmodel", "gymnasium")},
            "cuda_runtime": torch.version.cuda}
    fingerprint = json_hash(plan)
    pin_json(folder / "plan.json", {"plan": plan, "fingerprint": fingerprint})
    preprocess = get_img_preprocessor("pixels", "pixels", 224)
    mean = np.asarray(plan["normalization"]["mean"], dtype=np.float32).reshape(1, 2)
    std = np.asarray(plan["normalization"]["std"], dtype=np.float32).reshape(1, 2)
    cost = TerminalCost(model)
    space = gym.spaces.Box(-1.0, 1.0, shape=(1, 2), dtype=np.float32)
    config = swm.PlanConfig(horizon=protocol.horizon, action_block=protocol.action_block,
                            receding_horizon=protocol.horizon)
    solver = CEMSolver(cost, batch_size=1, num_samples=protocol.candidates, var_scale=protocol.initial_std,
                       n_steps=protocol.iterations, topk=protocol.elites, device=device)
    solver.configure(action_space=space, n_envs=1, config=config)
    total = len(spec["trials"]) * len(protocol.solver_seeds)
    completed = fresh = successes = 0
    rates = {}
    tick = time.perf_counter()
    with h5py.File(manifest["dataset"]["path"], "r") as inputs, torch.inference_mode():
        for seed in protocol.solver_seeds:
            seed_successes = 0
            seed_folder = folder / "seeds" / str(seed)
            seed_folder.mkdir(parents=True, exist_ok=True)
            for trial in spec["trials"]:
                path = seed_folder / f"{trial['episode']}.json"
                if not path.exists():
                    # Per-(seed, episode) generators preserve paired draws after an interruption
                    # and when one model succeeds earlier than another.
                    sub_seed = int(np.random.SeedSequence([seed, trial["episode"]]).generate_state(1)[0])
                    solver.torch_gen.manual_seed(sub_seed)
                    outcome = run_trial(model, env, solver, trial, inputs, mean, std, preprocess, device, protocol)
                    content = {"trial": trial, "evaluation_seed": seed, "solver_seed": sub_seed,
                               "plan_sha256": fingerprint, "outcome": outcome}
                    write_json(path, {**content, "record_sha256": json_hash(content)})
                    fresh += 1
                success = validate_record(read_json(path), trial, seed, fingerprint, protocol)
                completed += 1
                successes += success
                seed_successes += success
                elapsed = time.perf_counter() - tick
                progress = {"phase": "planning", "completed_trials": completed, "expected_trials": total,
                            "seed": seed, "successes": successes, "success_rate_percent": 100 * successes / completed,
                            "elapsed_seconds": elapsed, "eta_seconds": (total - completed) * elapsed / fresh if fresh else None}
                write_json(folder / "progress.json", progress)
                if completed == 1 or completed % 25 == 0:
                    print(f"{now()} {completed}/{total} trials; {successes} reached the goal", flush=True)
            rates[str(seed)] = {"successes": seed_successes, "trials": protocol.trajectories,
                                "success_rate_percent": 100.0 * seed_successes / protocol.trajectories}
    env.close()
    final_hash = state_dict_sha256(model.cpu().state_dict())
    check(previous == final_hash and completed == total, "Incomplete planning evaluation or mutated model")
    percentages = [r["success_rate_percent"] for r in rates.values()]
    result = {"passed": True, "variant": variant.label, "plan_sha256": fingerprint, "trials_sha256": trial_hash,
              "test_episodes": protocol.trajectories, "evaluation_trials": total, "successes": successes,
              "mean_success_percent": float(np.mean(percentages)), "sample_sd_percent": float(np.std(percentages, ddof=1)),
              "per_seed": rates, "model_state_unchanged": True, "model_state_sha256": previous, "finished_utc": now(),
              "paper_two_room_success_percent": 100.0,
              "seed_interpretation": "Independent planner draws on the same episodes and frozen checkpoint."}
    write_json(folder / "results.json", result)
    write_json(folder / "progress.json", {**progress, "complete": True, "passed": True})
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true", help="Freeze shared held-out trials on CPU before training")
    args = parser.parse_args()
    variant = VARIANTS[args.variant]
    evaluate((args.experiment_dir or variant.experiment).resolve(strict=True), variant, args.prepare_only)
