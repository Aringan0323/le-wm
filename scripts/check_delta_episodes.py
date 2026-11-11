"""CPU regression checks for continuous planning state and success accounting."""
from copy import deepcopy

import numpy as np
import torch

import plan_delta_episodes as planning
from evaluate_episode_experiment import json_hash


class Environment:
    def __init__(self):
        self.resets = 0
        self.goal_changes = 0

    def reset(self, *, seed, options):
        self.resets += 1
        self.agent_position = torch.tensor(options["state"], dtype=torch.float32)
        self.target_position = torch.tensor(options["target_state"], dtype=torch.float32)

    def _set_goal_state(self, target):
        self.goal_changes += 1
        self.target_position = torch.tensor(target, dtype=torch.float32)

    def render(self):
        return self.agent_position.numpy().copy()

    def step(self, action):
        self.agent_position += torch.as_tensor(action,dtype=torch.float32).clamp(-1,1)
        distance = float(torch.norm(self.agent_position-self.target_position))
        return None, 0, distance < 16, False, {"distance_to_target":distance}


class Solver:
    def __init__(self):
        self.torch_gen = torch.Generator()
        self.observations = []

    def __call__(self, info):
        self.observations.append(info["start"].reshape(2).tolist())
        actions = torch.zeros(1, 5, 10)
        actions[:, :, ::2] = 1
        return {"actions":actions}


def main():
    assert planning.goal_steps(101,25) == [25,50,75,100]
    assert planning.goal_steps(100,25) == [25,50,75,99]
    assert planning.goal_steps(10,25) == [9]
    protocol = planning.PROTOCOL
    episode = {"episode":123,"length":100,"start":0,"start_row":0,
               "goal_steps":[25,50,75,99],"goal_rows":[25,50,75,99]}
    points = np.zeros((100,2),dtype=np.float32)
    points[[25,50,75,99],0] = [5,25,100,65]
    inputs = {"proprio":points,"pixels":points.copy()}
    original_embedding = planning.image_embedding
    planning.image_embedding = lambda model,image,preprocess,device:torch.tensor(image).reshape(1,1,2)
    try:
        env, solver = Environment(), Solver()
        notifications = []
        outcome = planning.run_episode(None,env,solver,episode,inputs,np.zeros((1,2)),np.ones((1,2)),
                                       None,"cpu",protocol,42,lambda tasks:notifications.append(deepcopy(tasks)))
        tasks = outcome["tasks"]
        assert env.resets == 1 and env.goal_changes == 4
        assert [t["actual_start_position"][0] for t in tasks] == [0,0,10,60]
        assert [t["actual_end_position"][0] for t in tasks] == [0,10,60,60]
        assert [t["steps"] for t in tasks] == [0,10,50,0]
        assert [t["cumulative_steps_after"] for t in tasks] == [0,10,60,60]
        assert [t["zero_action_success"] for t in tasks] == [True,False,False,True]
        assert solver.observations == [[0,0],[10,0],[35,0]]
        assert outcome["initial_state"] == {"position":[0,0],"cumulative_step":0}
        assert [len(task["trajectory"]) for task in tasks] == [0,10,50,0]
        assert tasks[0]["start_event"]["event"] == "zero_action_success"
        assert tasks[3]["start_event"]["cumulative_step"] == 60
        assert tasks[2]["trajectory"][0]["pre_position"] == [10,0]
        assert tasks[2]["trajectory"][-1]["post_position"] == [60,0]
        assert tasks[2]["trajectory"][-1]["cumulative_step_after"] == 60
        assert len(notifications) == 4 and len(notifications[-1]) == 4
        assert not outcome["success"] and outcome["final_goal_success"] and outcome["physical_steps"] == 60

        record = {"episode":episode,"evaluation_seed":42,"plan_sha256":"fixture","outcome":outcome}
        record["record_sha256"] = json_hash(record)
        assert planning.validate_episode(record,episode,42,"fixture",protocol) == outcome
        summary = planning.summarize([outcome],20,76)
        assert summary["completed_episodes"] == 1 and summary["expected_episodes"] == 20
        assert summary["episode_successes"] == 0 and summary["episode_success_rate_percent"] == 0
        assert summary["completed_tasks"] == 4 and summary["expected_tasks"] == 76
        assert summary["task_successes"] == 3 and summary["task_success_rate_percent"] == 75
        assert summary["zero_action_successes"] == 2 and summary["final_goal_successes"] == 1
        assert summary["completed_trials"] == summary["completed_tasks"]
        partial = planning.summarize([],20,76,tasks[:2])
        assert partial["completed_episodes"] == 0 and partial["completed_tasks"] == 2
        total = planning.group_totals({"a":summary,"b":partial})
        assert total["expected_episodes"] == 40 and total["expected_tasks"] == 152
        assert total["completed_tasks"] == 6 and total["task_successes"] == 5

        for change in ("continuity","incomplete","action_trace","clipping","action_count","initial_state"):
            bad = deepcopy(record)
            bad.pop("record_sha256")
            if change == "continuity":
                bad["outcome"]["tasks"][2]["actual_start_position"][0] += 1
            elif change == "incomplete":
                bad["outcome"]["tasks"].pop()
            elif change == "action_trace":
                bad["outcome"]["tasks"][2]["trajectory"][3]["pre_position"][0] += 1
            elif change == "clipping":
                bad["outcome"]["tasks"][1]["trajectory"][0]["applied_action"][0] = 2
            elif change == "action_count":
                bad["outcome"]["tasks"][1]["trajectory"].pop()
            else:
                bad["outcome"]["initial_state"]["cumulative_step"] = 1
            bad["record_sha256"] = json_hash(bad)
            try:
                planning.validate_episode(bad,episode,42,"fixture",protocol)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Accepted {change} episode record")

        boundary = dict(episode,goal_steps=[25],goal_rows=[25])
        inputs["proprio"][25,0] = 16
        inputs["pixels"][25,0] = 16
        strict = planning.run_episode(None,Environment(),Solver(),boundary,inputs,np.zeros((1,2)),
                                      np.ones((1,2)),None,"cpu",protocol,42)
        assert strict["tasks"][0]["steps"] == 1 and not strict["tasks"][0]["zero_action_success"]
        clipped = planning.run_episode(None,Environment(),Solver(),boundary,inputs,np.zeros((1,2)),
                                       np.full((1,2),3.0),None,"cpu",protocol,42)
        action = clipped["tasks"][0]["trajectory"][0]
        assert action["commanded_action"] == [3,0] and action["applied_action"] == [1,0]
        assert action["post_position"] == [1,0]
        clipped_record = {"episode":boundary,"evaluation_seed":42,"plan_sha256":"fixture","outcome":clipped}
        clipped_record["record_sha256"] = json_hash(clipped_record)
        planning.validate_episode(clipped_record,boundary,42,"fixture",protocol)
        print("Passed one-reset continuity, failure carryover, current rendered observations, final remainder, "
              "zero-action/strict threshold, replay trace, clipping, per-goal publication, record rejection, "
              "and task/episode denominators.")
    finally:
        planning.image_embedding = original_embedding


if __name__ == "__main__":
    main()
