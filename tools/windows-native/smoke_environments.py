"""Reset, step and offscreen-render all four official LeWM environments."""
import json
import os
import time
import traceback

import gymnasium as gym
import numpy as np
from PIL import Image
import stable_worldmodel as swm

from runtime_paths import LOGS
CASES = {
    "tworoom": ("swm/TwoRoom-v1", {}),
    "pusht": ("swm/PushT-v1", {}),
    "cube": ("swm/OGBCube-v0", dict(env_type="single", ob_type="states", multiview=False,
            width=224, height=224, visualize_info=False, terminate_at_goal=True)),
    "reacher": ("swm/ReacherDMControl-v0", dict(task="qpos_match")),
}


def check_observation(obs):
    if isinstance(obs, dict):
        return {key: check_observation(value) for key, value in obs.items()}
    array = np.asarray(obs)
    assert np.isfinite(array).all(), "Nonfinite observation"
    return list(array.shape)


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    result = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "mujoco_gl": os.environ.get("MUJOCO_GL", "platform default"), "environments": {}}
    for name, (env_id, kwargs) in CASES.items():
        start = time.perf_counter()
        env = None
        item = {"env_id": env_id, "kwargs": kwargs, "render_mode": "rgb_array"}
        try:
            env = gym.make(env_id, max_episode_steps=5, render_mode="rgb_array", **kwargs)
            obs, info = env.reset(seed=42)
            env.action_space.seed(42)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            frame = np.asarray(env.render())
            assert frame.ndim == 3 and frame.shape[-1] == 3, frame.shape
            assert frame.dtype == np.uint8, frame.dtype
            assert np.isfinite(frame).all()
            assert float(frame.std()) > 0, "Rendered frame is blank"
            observation_shape = check_observation(obs)
            assert np.isfinite(reward), "Nonfinite reward"
            Image.fromarray(frame).save(LOGS / f"smoke-env-{name}.png")
            item.update(status="passed", observation_shape=observation_shape,
                        action_shape=list(action.shape), reward=float(reward),
                        terminated=bool(terminated), truncated=bool(truncated),
                        frame_shape=list(frame.shape), frame_std=float(frame.std()))
        except Exception as exc:
            item.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception as exc:
                    item["close_warning"] = str(exc)
            item["seconds"] = round(time.perf_counter() - start, 3)
        result["environments"][name] = item
        print(f"{name}: {item['status']} ({item['seconds']} s)", flush=True)
        if item["status"] == "failed":
            print(item["traceback"], flush=True)
    result["passed"] = all(item["status"] == "passed" for item in result["environments"].values())
    path = LOGS / "smoke-environments.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Report: {path}", flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
