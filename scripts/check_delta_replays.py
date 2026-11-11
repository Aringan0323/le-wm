"""Exercise action replay against real Two-Room dynamics without a model or GPU."""
import copy
from dataclasses import replace
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import torch
from stable_worldmodel.envs.two_room.env import TwoRoomEnv

import plan_delta_episodes as planner
from evaluate_episode_experiment import file_hash, json_hash
from export_delta_replays import render_model, save_frame


def main():
    torch.set_num_threads(1)
    episode = {"episode":42, "length":4, "start":0, "start_row":0,
               "goal_steps":[1,2,3], "goal_rows":[1,2,3]}
    inputs = {"proprio":np.asarray([[40,40],[160,160],[55,40],[80,40]],dtype=np.float32),
              "pixels":np.zeros((4,224,224,3),dtype=np.uint8)}
    protocol = replace(planner.PROTOCOL,budget=3,horizon=1,action_block=1)
    class Solver:
        torch_gen = torch.Generator()
        def __call__(self, _):
            return {"actions":torch.tensor([[[2.0,0.0]]])}
    old_embedding = planner.image_embedding
    planner.image_embedding = lambda *_: torch.zeros(1,1,1)
    env = TwoRoomEnv()
    try:
        outcome = planner.run_episode(None,env,Solver(),episode,inputs,np.zeros((1,2),dtype=np.float32),
                                      np.ones((1,2),dtype=np.float32),None,"cpu",protocol,42)
    finally:
        env.close()
        planner.image_embedding = old_embedding
    record = {"episode":episode,"evaluation_seed":42,"plan_sha256":"fixture","outcome":outcome}
    record["record_sha256"] = json_hash(record)
    planner.validate_episode(record,episode,42,"fixture",protocol)
    assert [task["success"] for task in outcome["tasks"]] == [False,True,True]
    assert [task["steps"] for task in outcome["tasks"]] == [3,0,2]
    replay_env = TwoRoomEnv()
    try:
        images = []
        def writer(step, pixels):
            images.append(pixels.copy())
            return f"42/model/{step:05d}.png"
        frames = render_model(replay_env,inputs,episode,outcome,writer)
        assert len(frames) == len(images) == 6
        assert frames[3]["goal_index"] == 2 and len(frames[3]["events"]) == 4
        assert frames[-1]["completed_tasks"] == 3 and frames[-1]["goals_reached"] == 2
        assert frames[-1]["position"] == [65.,40.] and frames[-1]["status"] == "complete"
        for mutate in ("command","missing","state","goal"):
            corrupt = copy.deepcopy(outcome)
            if mutate == "command":
                corrupt["tasks"][0]["trajectory"][0]["commanded_action"] = [0.,0.]
                corrupt["tasks"][0]["trajectory"][0]["applied_action"] = [0.,0.]
            elif mutate == "missing":
                corrupt["tasks"][0]["trajectory"].pop()
            elif mutate == "state":
                corrupt["tasks"][0]["trajectory"][0]["post_position"][0] += 1
            else:
                corrupt["tasks"][0]["target_position"][0] += 1
            try:
                render_model(replay_env,inputs,episode,corrupt,lambda step, pixels: str(step))
            except ValueError:
                pass
            else:
                raise AssertionError(f"Replay accepted {mutate} tampering")
        with TemporaryDirectory(prefix="delta-replay-check-") as temporary:
            root = Path(temporary)
            assets = {}
            relative = save_frame(root,"42/ground_truth/00000.png",images[0],assets)
            assert np.array_equal(np.asarray(Image.open(root/relative)),images[0])
            assert assets[relative] == file_hash(root/relative)
    finally:
        replay_env.close()
    print("Replay CPU checks passed: clipping, state continuity, failed-goal carryover, zero-action events, PNG fidelity, tamper rejection.")


if __name__ == "__main__":
    main()
