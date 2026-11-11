"""Export verified continuous-episode traces as CPU-rendered, four-stream replays."""
import argparse
import inspect
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from audit_combined_experiment import check, read_json
from delta_sweep_config import VARIANTS
from evaluate_delta_experiment import verified_split
from evaluate_episode_experiment import file_hash, json_hash, now, write_json
from plan_delta_episodes import episode_specification, position, summarize, validate_episode, group_totals


LABELS = {"ground_truth": "Recorded dataset", "delta_l0_s09": "LDAD 0 · SIGReg 0.09",
          "delta_l50_s09": "LDAD 50 · SIGReg 0.09", "delta_l50_s0": "LDAD 50 · SIGReg 0"}


def same_position(actual, expected, message):
    actual, expected = np.asarray(actual), np.asarray(expected)
    check(actual.shape == expected.shape == (2,) and np.isfinite(actual).all()
          and np.isfinite(expected).all() and np.allclose(actual, expected, rtol=0, atol=1e-5), message)


def save_frame(stage, relative, pixels, assets):
    pixels = np.asarray(pixels)
    check(pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8, "Unexpected replay image")
    path = stage / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path, compress_level=1)
    assets[relative] = file_hash(path)
    return relative


def render_model(env, inputs, episode, outcome, write_image):
    """Re-execute every stored action, validating the live simulator against the trace."""
    tasks = outcome["tasks"]
    start = np.asarray(inputs["proprio"][episode["start_row"]], dtype=np.float32)
    env.reset(seed=episode["episode"], options={"state": start,
              "target_state": np.asarray(tasks[0]["target_position"], dtype=np.float32)})
    same_position(position(env), outcome["actual_initial_position"], "Replay initial state differs")
    same_position(outcome["initial_state"]["position"], start, "Trace initial state differs from dataset")
    check(outcome["initial_state"]["cumulative_step"] == 0, "Trace does not start at physical step zero")
    frames = [{"step": 0, "image": write_image(0, env.render()), "position": position(env),
               "events": [], "completed_tasks": 0, "goals_reached": 0, "action": None}]
    successes = 0
    for index, task in enumerate(tasks):
        target = np.asarray(inputs["proprio"][task["goal_row"]], dtype=np.float32)
        same_position(target, task["target_position"], "Replay goal differs from recorded dataset")
        same_position(position(env), task["actual_start_position"], "Replay goal boundary changed the agent state")
        env._set_goal_state(target)
        same_position(position(env), task["actual_start_position"], "Replay target update moved the agent")
        event = task["start_event"]
        check(event["cumulative_step"] == len(frames)-1 == task["cumulative_steps_before"]
              and event["goal_index"] == index and event["goal_step"] == task["goal_step"],
              "Replay goal event has the wrong timeline")
        same_position(event["position"], position(env), "Replay start event position differs")
        same_position(event["target_position"], target, "Replay start event goal differs")
        expected_event = "zero_action_success" if task["zero_action_success"] else "goal_start"
        check(event["event"] == expected_event, "Replay start event type differs")
        frames[-1].update(goal_index=index, goal_step=task["goal_step"], target_position=target.tolist(),
                          distance=float(torch.norm(env.agent_position-env.target_position)), status=expected_event)
        frames[-1]["events"].append({"event": expected_event, "goal": index+1})
        trajectory = task["trajectory"]
        check(len(trajectory) == task["steps"], "Replay does not contain every executed action")
        for action_index, transition in enumerate(trajectory):
            step = len(frames)-1
            check(transition["action_index"] == action_index and transition["goal_index"] == index
                  and transition["cumulative_step_before"] == step
                  and transition["cumulative_step_after"] == step+1, "Replay action timeline differs")
            same_position(transition["pre_position"], position(env), "Replay pre-action state differs")
            same_position(transition["target_position"], target, "Replay action goal differs")
            commanded = np.asarray(transition["commanded_action"], dtype=np.float32)
            same_position(np.clip(commanded, -1, 1), transition["applied_action"], "Replay action clipping differs")
            _, _, reached, truncated, info = env.step(commanded)
            same_position(position(env), transition["post_position"], "Replay dynamics differ from recorded state")
            distance = float(info["distance_to_target"])
            check(not truncated and np.isclose(distance, transition["distance_to_target"], rtol=0, atol=1e-5)
                  and bool(reached) == transition["success"] == (distance < 16), "Replay transition outcome differs")
            frames.append({"step": step+1, "image": write_image(step+1, env.render()), "position": position(env),
                           "action": transition["commanded_action"], "applied_action": transition["applied_action"],
                           "goal_index": index, "goal_step": task["goal_step"], "target_position": target.tolist(),
                           "distance": distance, "status": "moving", "completed_tasks": index,
                           "goals_reached": successes, "events": []})
        same_position(position(env), task["actual_end_position"], "Replay final goal state differs")
        distance = float(torch.norm(env.agent_position-env.target_position))
        check(np.isclose(distance, task["final_distance"], rtol=0, atol=1e-5)
              and task["success"] == (distance < 16), "Replay final goal distance differs")
        successes += int(task["success"])
        event_type = "goal_reached" if task["success"] else "goal_budget_exhausted"
        frames[-1].update(status=event_type, completed_tasks=index+1, goals_reached=successes)
        frames[-1]["events"].append({"event": event_type, "goal": index+1})
    check(len(frames)-1 == outcome["physical_steps"], "Replay physical-step coverage differs")
    same_position(position(env), outcome["actual_final_position"], "Replay episode endpoint differs")
    frames[-1].update(status="complete", episode_success=outcome["success"])
    return frames


def validate_inputs(root, folder):
    results = read_json(folder / "results.json")
    check(results.get("passed") and results.get("evaluation_mode") == "continuous_episodes"
          and results.get("model_states_unchanged"), "Replay requires a completed frozen-weight episode test")
    bundle, _ = verified_split(root)
    spec = read_json(folder / "episodes.json")
    record = read_json(folder / "checkpoint.json")
    check(record["record_sha256"] == json_hash({k:v for k,v in record.items() if k != "record_sha256"})
          == results["checkpoint_record_sha256"], "Pinned checkpoint record hash differs")
    check(record["manifest_sha256"] == bundle["manifest"]["manifest_sha256"]
          and record["global_step"] == results["checkpoint_step"], "Replay identifies another training snapshot")
    plan = read_json(folder / "plan.json")
    check(plan["fingerprint"] == json_hash(plan["plan"]) == results["plan_sha256"]
          and json_hash(spec) == results["episodes_sha256"] == plan["plan"]["episodes_sha256"]
          and plan["plan"]["checkpoint_record_sha256"] == record["record_sha256"],
          "Episode-test plan or selection hash differs")
    count = results["test_episodes_per_model"]
    seeds = spec["protocol"]["solver_seeds"]
    check(len(seeds) == 1 and set(results["models"]) == set(VARIANTS), "Unexpected replay comparison")
    seed = seeds[0]
    protocol, expected = episode_specification(bundle, count, seed)
    check(spec == expected and results["protocol"] == spec["protocol"], "Replay episode selection differs")
    check(plan["plan"].get("trajectory_schema_version") == 1, "Evaluation lacks replayable action traces")
    records = {}
    for key in VARIANTS:
        model_plan = read_json(folder / key / "plan.json")
        summary = results["models"][key]
        saved = read_json(folder / key / "results.json")
        check(saved.get("passed") and all(saved.get(k) == v for k,v in summary.items()),
              "Per-model result differs from final group result")
        check(summary.get("model_state_unchanged") and summary["status"] == "complete"
              and summary["model_state_sha256"] == model_plan["state_sha256"]
              and summary["checkpoint_sha256"] == record["members"][key]["sha256"] == model_plan["checkpoint_sha256"]
              and model_plan["plan_sha256"] == plan["fingerprint"], "Frozen model identity differs")
        outcomes = []
        for episode in spec["episodes"]:
            path = folder / key / "episodes" / f"{episode['episode']}.json"
            item = read_json(path)
            outcome = validate_episode(item, episode, seed, json_hash(model_plan), protocol)
            records[(key, episode["episode"])] = item
            outcomes.append(outcome)
        totals = summarize(outcomes, count, spec["expected_tasks"])
        check(all(summary.get(k) == v for k,v in totals.items()), "Replay outcome totals differ from test report")
    check(results["tasks_per_model"] == spec["expected_tasks"]
          and all(results.get(k) == v for k,v in group_totals(results["models"]).items()),
          "Final replay test totals differ")
    return bundle, spec, record, plan, results, records


def export(root, folder):
    import h5py
    import hdf5plugin
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    torch.set_num_threads(1)
    bundle, spec, record, plan, results, records = validate_inputs(root, folder)
    check(file_hash(Path(inspect.getfile(TwoRoomEnv))) == plan["plan"]["environment_sha256"],
          "Simulator implementation differs from the evaluated environment")
    source = {"evaluation_results_sha256":file_hash(folder / "results.json"),
              "plan_sha256":plan["fingerprint"], "manifest_sha256":bundle["manifest"]["manifest_sha256"],
              "episode_record_sha256":{f"{key}/{ep}":item["record_sha256"] for (key,ep),item in records.items()},
              "export_script_sha256":file_hash(Path(__file__)),
              "viewer_sha256":file_hash(Path(__file__).resolve().parents[1]/"tools/desktop/dashboard/replay.html")}
    destination = folder / "replay"
    if destination.exists():
        existing = read_json(destination / "manifest.json")
        check(existing.get("source") == source, "Existing replay belongs to another export")
        for name,digest in existing["asset_sha256"].items():
            asset=(destination/name).resolve(strict=True)
            check(asset.is_relative_to(destination.resolve()) and file_hash(asset)==digest,
                  "Existing replay asset checksum or path differs")
        print(f"Verified existing replay at {destination}", flush=True)
        return existing
    assets, episodes = {}, []
    env = TwoRoomEnv()
    try:
        with h5py.File(bundle["manifest"]["dataset"]["path"], "r") as inputs, TemporaryDirectory(
                prefix=".export-replay-", dir=folder) as temporary:
            stage = Path(temporary) / "replay"
            stage.mkdir()
            for episode in spec["episodes"]:
                episode_id, length, offset = episode["episode"], episode["length"], episode["start_row"]
                streams = []
                baseline = []
                for step in range(length):
                    row = offset + step
                    check(int(inputs["ep_idx"][row]) == episode_id and int(inputs["step_idx"][row]) == step,
                          "Dataset baseline row sequence differs")
                    baseline.append({"step":step, "image":save_frame(stage,
                        f"{episode_id}/ground_truth/{step:05d}.png", inputs["pixels"][row], assets),
                        "position":np.asarray(inputs["proprio"][row],dtype=float).tolist(),
                        "status":"complete" if step == length-1 else "recorded", "events":[]})
                streams.append({"key":"ground_truth", "label":LABELS["ground_truth"], "kind":"recorded",
                                "final_step":length-1, "frames":baseline})
                for key in VARIANTS:
                    outcome = records[(key, episode_id)]["outcome"]
                    def write_image(step, pixels):
                        return save_frame(stage, f"{episode_id}/{key}/{step:05d}.png", pixels, assets)
                    frames = render_model(env, inputs, episode, outcome, write_image)
                    streams.append({"key":key, "label":LABELS[key], "kind":"model", "final_step":len(frames)-1,
                                    "goal_count":len(outcome["tasks"]), "goals_reached":sum(t["success"] for t in outcome["tasks"]),
                                    "episode_success":outcome["success"], "frames":frames})
                episodes.append({"episode":episode_id, "source_length":length,
                                 "goal_steps":episode["goal_steps"], "streams":streams,
                                 "final_step":max(s["final_step"] for s in streams)})
                print(f"Exported episode {episode_id}: {sum(len(s['frames']) for s in streams)} verified frames", flush=True)
            manifest = {"schema_version":1, "created_utc":now(), "evaluation_mode":"continuous_episodes",
                        "checkpoint_step":record["global_step"], "source":source,
                        "timeline":"Executed physical action steps; each stream holds at its own end.",
                        "default_frames_per_second":10, "episode_count":len(episodes), "episodes":episodes,
                        "asset_sha256":assets, "all_actions_resimulated":True,
                        "baseline":"Original recorded dataset pixels for every frame of each selected episode."}
            viewer=Path(__file__).resolve().parents[1]/"tools/desktop/dashboard/replay.html"
            (stage/"index.html").write_bytes(viewer.read_bytes())
            assets["index.html"]=file_hash(stage/"index.html")
            write_json(stage / "manifest.json", manifest)
            stage.rename(destination)
            print(f"Replay exported to {destination}", flush=True)
            return manifest
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--evaluation-dir", required=True, type=Path)
    args = parser.parse_args()
    root, folder = args.experiment_dir.resolve(strict=True), args.evaluation_dir.resolve(strict=True)
    check(folder.parent == root / "planning", "Evaluation directory is outside this experiment's planning outputs")
    export(root, folder)


if __name__ == "__main__":
    main()
