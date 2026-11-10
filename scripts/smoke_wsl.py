"""WSL LeWM smoke runner; writes separate JSON, logs and rendered PNGs.

Run with the WSL virtualenv and activation variables already configured:
  python /home/adam/lewm/le-wm/scripts/smoke_wsl.py

Each case runs in a subprocess so native CUDA/GL failures cannot erase the
other results. Default checks require CUDA and the real Two-Room dataset.
These are execution/finite-output checks, not benchmark-quality estimates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

REPO = Path(os.environ.get("LEWM_REPO", "/home/adam/lewm/le-wm"))
STORAGE = Path(os.environ.get("STABLEWM_HOME", "/home/adam/lewm/storage"))
REPORT_ROOT = Path(os.environ.get("LEWM_REPORT_DIR", "/home/adam/lewm/logs"))
RUNTIME = Path(os.environ.get("LEWM_RUNTIME_ROOT", str(REPO.parent)))
MODELS = ("tworooms", "pusht", "cube", "reacher")
ENVS = {
    "tworoom": ("swm/TwoRoom-v1", {}),
    "pusht": ("swm/PushT-v1", {}),
    "cube": ("swm/OGBCube-v0", dict(env_type="single", ob_type="states", multiview=False,
             width=224, height=224, visualize_info=False, terminate_at_goal=True)),
    "reacher": ("swm/ReacherDMControl-v0", dict(task="qpos_match")),
}


def configure_runtime():
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPO))
    os.environ["STABLEWM_HOME"] = str(STORAGE)
    os.environ["LOCAL_DATASET_DIR"] = str(STORAGE)
    for key, path in {
        "TMPDIR": RUNTIME / "tmp", "HF_HOME": RUNTIME / "cache/huggingface",
        "TORCH_HOME": RUNTIME / "cache/torch", "XDG_CACHE_HOME": RUNTIME / "cache",
        "MPLCONFIGDIR": RUNTIME / "cache/matplotlib", "SPT_CACHE_DIR": RUNTIME / "training",
        "WANDB_DIR": RUNTIME / "logs/wandb", "WANDB_CACHE_DIR": RUNTIME / "cache/wandb",
        "WANDB_CONFIG_DIR": RUNTIME / "cache/wandb-config",
        "CUDA_CACHE_PATH": RUNTIME / "cache/cuda",
    }.items():
        os.environ.setdefault(key, str(path))
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    if os.environ["MUJOCO_GL"] == "osmesa":
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_summary(tensor, shape):
    import torch
    assert tuple(tensor.shape) == tuple(shape), (tensor.shape, shape)
    assert torch.isfinite(tensor).all().item(), "Nonfinite model output"
    return {"shape": list(tensor.shape), "finite": True,
            "min": tensor.min().item(), "max": tensor.max().item()}


def exercise(model, pixels, action, history):
    import torch
    batch, steps = pixels.shape[:2]
    assert batch == 1 and steps >= history + 1
    latent = int(model.predictor.pos_embedding.shape[-1])
    encoded = model.encode({"pixels": pixels.clone(), "action": action.clone()})
    predicted = model.predict(encoded["emb"][:, :history], encoded["act_emb"][:, :history])
    candidates, horizon = 4, history + 2
    actions = torch.randn(batch, candidates, horizon, action.shape[-1], device=pixels.device) * 0.1
    actions[:, :, :history] = action[:, None, :history]
    initial = pixels[:, None, :history].expand(-1, candidates, -1, -1, -1, -1)
    goal = pixels[:, None, history:history + 1].expand(-1, candidates, -1, -1, -1, -1)
    rollout = model.rollout({"pixels": initial.clone()}, actions.clone(), history_size=history)["predicted_emb"]
    cost = model.get_cost({"pixels": initial.clone(), "goal": goal.clone(),
                           "action": actions[:, :, :history].clone()}, actions.clone())
    mse = (predicted - encoded["emb"][:, 1:history + 1]).square().mean()
    assert torch.isfinite(mse).item()
    return {
        "encode": tensor_summary(encoded["emb"], (batch, steps, latent)),
        "predict": tensor_summary(predicted, (batch, history, latent)),
        "autoregressive_rollout": tensor_summary(rollout, (batch, candidates, horizon + 1, latent)),
        "candidate_cost": tensor_summary(cost, (batch, candidates)),
        "prediction_mse": mse.item(),
        "note": "Finite outputs and expected shapes; no task-quality threshold applied.",
    }


def real_tworoom_batch(history):
    import hdf5plugin  # Registers the compression filters before HDF5 reading.
    import torch
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from utils import get_column_normalizer, get_img_preprocessor

    path = STORAGE / "datasets/tworoom.h5"
    if not path.is_file():
        raise FileNotFoundError(f"Required real Two-Room dataset is missing: {path}")
    dataset = swm.data.load_dataset("tworoom.h5", cache_dir=str(STORAGE), num_steps=history + 1,
        frameskip=5, keys_to_load=["pixels", "action"], keys_to_cache=["action"], transform=None)
    dataset.transform = spt.data.transforms.Compose(
        get_img_preprocessor("pixels", "pixels", img_size=224),
        get_column_normalizer(dataset, "action", "action"))
    sample = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)))
    return sample["pixels"].cuda(), torch.nan_to_num(sample["action"], nan=0.0).cuda(), {
        "dataset_path": str(path), "dataset_length": len(dataset), "frameskip": 5,
        "preprocessing": "Repository ImageNet image preprocessing and full-dataset action z-score",
    }


def check_model(name, skip_dataset):
    import torch
    import stable_worldmodel as swm

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is false")
    torch.manual_seed(3072)
    torch.set_num_threads(8)
    torch.cuda.reset_peak_memory_stats()
    source = STORAGE / "checkpoints" / f"lewm-{name}"
    cfg = json.loads((source / "config.json").read_text(encoding="utf-8-sig"))
    manifest = json.loads((source / "source-manifest.json").read_text(encoding="utf-8-sig"))
    expected = next(item["lfs"]["oid"] for item in manifest if item["path"] == "weights.pt")
    digest = sha256(source / "weights.pt")
    assert digest == expected, "Checkpoint does not match its official SHA256 manifest"
    state = torch.load(source / "weights.pt", map_location="cpu", weights_only=True)
    model = swm.wm.utils.load_pretrained(f"lewm-{name}")
    loaded = model.state_dict()
    assert set(loaded) == set(state), "Loader returned different checkpoint keys"
    assert all(torch.equal(loaded[key], state[key]) for key in state), "Loader returned different weights"
    # Explicitly verify strict compatibility as well as the loader's default strict loading.
    model.load_state_dict(state, strict=True)
    model = model.cuda().eval().requires_grad_(False)
    history = int(cfg["predictor"]["num_frames"])
    size = int(cfg["encoder"]["image_size"])
    action_dim = int(cfg["action_encoder"]["input_dim"])
    pixels = torch.rand(1, history + 1, 3, size, size, device="cuda")
    mean = pixels.new_tensor([0.485, 0.456, 0.406])[None, None, :, None, None]
    std = pixels.new_tensor([0.229, 0.224, 0.225])[None, None, :, None, None]
    pixels = (pixels - mean) / std
    action = torch.randn(1, history + 1, action_dim, device="cuda") * 0.1
    result = {
        "loader": f"stable_worldmodel.wm.utils.load_pretrained('lewm-{name}')",
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "sha256": digest, "sha256_verified": True, "strict_load": True,
        "checkpoint_tensors_equal": True, "parameters": sum(p.numel() for p in model.parameters()),
        "gpu": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
    }
    with torch.inference_mode():
        result["synthetic"] = exercise(model, pixels, action, history)
        if name == "tworooms":
            if skip_dataset:
                result["real_dataset"] = {"status": "skipped", "reason": "Explicit --skip-dataset"}
            else:
                pixels, action, metadata = real_tworoom_batch(history)
                result["real_dataset"] = exercise(model, pixels, action, history)
                result["real_dataset"].update(metadata)
    torch.cuda.synchronize()
    result["peak_allocated_gpu_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 2)
    return result


def check_observation(obs):
    import numpy as np
    if isinstance(obs, dict):
        return {key: check_observation(value) for key, value in obs.items()}
    array = np.asarray(obs)
    assert np.isfinite(array).all(), "Nonfinite observation"
    return list(array.shape)


def check_environment(name, output_dir):
    import gymnasium as gym
    import numpy as np
    from PIL import Image
    import stable_worldmodel  # Registers environments.

    env_id, kwargs = ENVS[name]
    env = gym.make(env_id, max_episode_steps=5, render_mode="rgb_array", **kwargs)
    try:
        env.reset(seed=42)
        env.action_space.seed(42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        frame = np.asarray(env.render())
        assert frame.ndim == 3 and frame.shape[-1] == 3, frame.shape
        assert frame.dtype == np.uint8, frame.dtype
        assert np.isfinite(frame).all() and float(frame.std()) > 0, "Blank/nonfinite render"
        observation_shape = check_observation(obs)
        assert np.isfinite(reward), "Nonfinite reward"
        image_path = output_dir / f"environment-{name}.png"
        Image.fromarray(frame).save(image_path)
        return {"env_id": env_id, "kwargs": kwargs, "observation_shape": observation_shape,
                "action_shape": list(action.shape), "reward": float(reward),
                "terminated": bool(terminated), "truncated": bool(truncated),
                "frame_shape": list(frame.shape), "frame_std": float(frame.std()),
                "render_mode": "rgb_array", "mujoco_gl": os.environ.get("MUJOCO_GL"),
                "image": str(image_path)}
    finally:
        env.close()


def worker(args):
    started = time.perf_counter()
    result = {"case": args.worker_case, "started_at": timestamp()}
    try:
        configure_runtime()
        kind, name = args.worker_case.split(":", 1)
        result.update(check_model(name, args.skip_dataset) if kind == "model" else
                      check_environment(name, args.worker_dir))
        result["status"] = "passed"
    except Exception as exc:
        result.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        print(result["traceback"], flush=True)
    result["seconds"] = round(time.perf_counter() - started, 3)
    write_json(args.worker_dir / (args.worker_case.replace(":", "-") + ".json"), result)
    print(f"{args.worker_case}: {result['status']} ({result['seconds']} seconds)", flush=True)
    return 0 if result["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "inference", "environments"), default="all")
    parser.add_argument("--skip-dataset", action="store_true", help="Only for preliminary checks")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per case in seconds")
    parser.add_argument("--worker-case", help=argparse.SUPPRESS)
    parser.add_argument("--worker-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_case:
        return worker(args)
    if sys.platform != "linux":
        parser.error("Run this script inside WSL with its Linux virtualenv")
    configure_runtime()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = REPORT_ROOT / ("smoke-wsl-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}")
    run_dir.mkdir()
    versions = {}
    for name in ("torch", "torchvision", "stable-worldmodel", "stable-pretraining", "transformers", "mujoco", "dm-control"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = {"started_at": timestamp(), "platform": platform.platform(), "python": sys.version,
              "executable": sys.executable, "repo": str(REPO), "storage": str(STORAGE),
              "run_directory": str(run_dir), "versions": versions, "stage": args.stage,
              "dataset_required": not args.skip_dataset, "mujoco_gl": os.environ.get("MUJOCO_GL"),
              "cases": {}, "complete": False, "passed": False,
              "limitations": "Execution checks only; no training or benchmark performance claim."}
    cases = []
    if args.stage in ("all", "inference"):
        cases.extend(f"model:{name}" for name in MODELS)
    if args.stage in ("all", "environments"):
        cases.extend(f"environment:{name}" for name in ENVS)
    for case in cases:
        stem = case.replace(":", "-")
        log_path = run_dir / (stem + ".log")
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker-case", case,
                   "--worker-dir", str(run_dir)]
        if args.skip_dataset:
            command.append("--skip-dataset")
        print(f"Running {case}; log: {log_path}", flush=True)
        try:
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=args.timeout, check=False)
            path = run_dir / (stem + ".json")
            result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
                "status": "failed", "error": "Worker exited without a JSON result (see its log)"}
            result["return_code"] = proc.returncode
            if proc.returncode != 0:
                result["status"] = "failed"
        except Exception as exc:
            result = {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
        result["log"] = str(log_path)
        report["cases"][case] = result
        write_json(run_dir / "report.json", report)
        write_json(REPORT_ROOT / "smoke-wsl.json", report)
        print(f"{case}: {result['status']}", flush=True)
    report["complete"] = True
    report["finished_at"] = timestamp()
    report["passed"] = all(item["status"] == "passed" for item in report["cases"].values())
    write_json(run_dir / "report.json", report)
    write_json(REPORT_ROOT / "smoke-wsl.json", report)
    print(f"Report: {REPORT_ROOT / 'smoke-wsl.json'}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
