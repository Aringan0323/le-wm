"""GPU smoke checks for the four official LeWM checkpoints.

Run from an Activate-LeWM.ps1 shell. Synthetic checks exercise the network API,
not learned task quality. --require-dataset additionally verifies Two-Room on
real HDF5 samples using the repository's training preprocessing.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

from runtime_paths import DATASET_ROOT, RUNTIME as LOCAL, REPO, STORAGE, is_wsl_share
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO))
for key, value in {
    "STABLEWM_HOME": STORAGE,
    "LOCAL_DATASET_DIR": DATASET_ROOT,
    "TEMP": LOCAL / "tmp",
    "TMP": LOCAL / "tmp",
    "HF_HOME": LOCAL / "cache" / "huggingface",
    "TORCH_HOME": LOCAL / "cache" / "torch",
    "XDG_CACHE_HOME": LOCAL / "cache",
    "MPLCONFIGDIR": LOCAL / "cache" / "matplotlib",
    "WANDB_DIR": LOCAL / "logs" / "wandb",
    "WANDB_CACHE_DIR": LOCAL / "cache" / "wandb",
    "WANDB_CONFIG_DIR": LOCAL / "cache" / "wandb-config",
}.items():
    os.environ[key] = str(value)
os.environ["WANDB_MODE"] = "disabled"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
if is_wsl_share(DATASET_ROOT):
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import torch
import stable_pretraining as spt
from jepa import JEPA
from module import ARPredictor, Embedder, MLP


def plain_kwargs(config):
    return {key: value for key, value in config.items() if not key.startswith("_")}


def build_model(config):
    """Map the official HF config to the exact classes in the cloned repo."""
    def mlp(name):
        kwargs = plain_kwargs(config[name])
        norm = kwargs.pop("norm_fn", None)
        if norm and norm.get("_target_") != "torch.nn.BatchNorm1d":
            raise ValueError(f"Unexpected normalization config: {norm}")
        return MLP(**kwargs, norm_fn=torch.nn.BatchNorm1d if norm else None)

    return JEPA(
        encoder=spt.backbone.utils.vit_hf(**plain_kwargs(config["encoder"])),
        predictor=ARPredictor(**plain_kwargs(config["predictor"])),
        action_encoder=Embedder(**plain_kwargs(config["action_encoder"])),
        projector=mlp("projector"),
        pred_proj=mlp("pred_proj"),
    )


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_summary(tensor, expected_shape=None):
    if expected_shape is not None:
        assert tuple(tensor.shape) == tuple(expected_shape), (
            tensor.shape, expected_shape
        )
    assert torch.isfinite(tensor).all().item(), "Nonfinite model output"
    return {
        "shape": list(tensor.shape),
        "finite": True,
        "min": tensor.min().item(),
        "max": tensor.max().item(),
    }


def exercise(model, pixels, action, history, candidates=4):
    """Check encoding, next embedding prediction, rollout, and MPC cost."""
    batch, steps = pixels.shape[:2]
    assert batch == 1, "Smoke test uses one environment with multiple candidates"
    assert steps >= history + 1
    latent = model.predictor.pos_embedding.shape[-1]
    encoded = model.encode({"pixels": pixels.clone(), "action": action.clone()})
    prediction = model.predict(
        encoded["emb"][:, :history], encoded["act_emb"][:, :history]
    )
    horizon = history + 2
    action_candidates = torch.randn(
        batch, candidates, horizon, action.shape[-1], device=pixels.device
    ) * 0.1
    action_candidates[:, :, :history] = action[:, None, :history]
    initial_pixels = pixels[:, None, :history].expand(-1, candidates, -1, -1, -1, -1)
    rollout = model.rollout(
        {"pixels": initial_pixels.clone()}, action_candidates.clone(), history_size=history
    )["predicted_emb"]
    goal = pixels[:, None, history:history + 1].expand(-1, candidates, -1, -1, -1, -1)
    costs = model.get_cost(
        {
            "pixels": initial_pixels.clone(),
            "goal": goal.clone(),
            "action": action_candidates[:, :, :history].clone(),
        },
        action_candidates.clone(),
    )
    mse = (prediction - encoded["emb"][:, 1:history + 1]).square().mean()
    assert torch.isfinite(mse).item()
    return {
        "encode": tensor_summary(encoded["emb"], (batch, steps, latent)),
        "predict": tensor_summary(prediction, (batch, history, latent)),
        "autoregressive_rollout": tensor_summary(
            rollout, (batch, candidates, horizon + 1, latent)
        ),
        "candidate_cost": tensor_summary(costs, (batch, candidates)),
        "prediction_mse": mse.item(),
        "note": "Finite-output and shape checks; MSE has no task-quality pass threshold.",
    }


def check_library_api(repo_model, state, name, pixels, action, history, device):
    """Exercise the same stable-worldmodel loader used by eval.py."""
    import stable_worldmodel as swm

    library_model = swm.wm.utils.load_pretrained(f"lewm-{name}")
    # The library loader uses load_state_dict's strict=True default. Compare
    # every loaded tensor as an additional check that it loaded this checkpoint.
    loaded = library_model.state_dict()
    assert set(loaded) == set(state), "Library loader state keys differ"
    assert all(torch.equal(loaded[key], state[key]) for key in state), (
        "Library loader state values differ from the official checkpoint"
    )
    library_model = library_model.to(device).eval().requires_grad_(False)
    result = {
        "loader": f"stable_worldmodel.wm.utils.load_pretrained('lewm-{name}')",
        "model_class": f"{type(library_model).__module__}.{type(library_model).__name__}",
        "strict_load": True,
        "checkpoint_tensors_equal": True,
        "synthetic": exercise(library_model, pixels, action, history),
    }
    repo_encoded = repo_model.encode({"pixels": pixels.clone(), "action": action.clone()})
    library_encoded = library_model.encode({"pixels": pixels.clone(), "action": action.clone()})
    repo_pred = repo_model.predict(repo_encoded["emb"][:, :history], repo_encoded["act_emb"][:, :history])
    library_pred = library_model.predict(library_encoded["emb"][:, :history], library_encoded["act_emb"][:, :history])
    comparisons = {}
    for key, left, right in (
        ("encode", repo_encoded["emb"], library_encoded["emb"]),
        ("action_encode", repo_encoded["act_emb"], library_encoded["act_emb"]),
        ("predict", repo_pred, library_pred),
    ):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
        comparisons[key] = {"max_absolute_difference": (left - right).abs().max().item(),
                            "rtol": 1e-5, "atol": 1e-6, "passed": True}
    result["repository_equivalence"] = comparisons
    return result

def real_tworoom_batch(history, device):
    import hdf5plugin  # Registers filters required by the official HDF5 archive.
    import stable_worldmodel as swm
    from utils import get_column_normalizer, get_img_preprocessor

    dataset = swm.data.load_dataset(
        "tworoom.h5",
        cache_dir=str(DATASET_ROOT),
        num_steps=history + 1,
        frameskip=5,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
        transform=None,
    )
    dataset.transform = spt.data.transforms.Compose(
        get_img_preprocessor("pixels", "pixels", img_size=224),
        get_column_normalizer(dataset, "action", "action"),
    )
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)))
    pixels = batch["pixels"].to(device)
    action = torch.nan_to_num(batch["action"], nan=0.0).to(device)
    return pixels, action, {
        "path": str(DATASET_ROOT / "datasets" / "tworoom.h5"),
        "dataset_length": len(dataset),
        "frameskip": 5,
        "preprocessing": "Repository ImageNet image preprocessing and full-dataset action z-score",
        "pixels_shape": list(pixels.shape),
        "action_shape": list(action.shape),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["tworooms", "pusht", "cube", "reacher"],
                        choices=["tworooms", "pusht", "cube", "reacher"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--require-dataset", action="store_true")
    args = parser.parse_args()
    report = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": args.device,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["torch", "torchvision", "stable-worldmodel", "stable-pretraining", "transformers"]
        },
        "models": {},
        "limitations": "Smoke tests verify execution and finite outputs, not benchmark performance.",
    }
    torch.manual_seed(3072)
    torch.set_num_threads(8)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but unavailable")
        report["gpu"] = torch.cuda.get_device_name(0)
        report["cuda_runtime"] = torch.version.cuda
        report["compute_capability"] = list(torch.cuda.get_device_capability(0))
        torch.cuda.reset_peak_memory_stats()
    device = torch.device(args.device)
    for name in args.models:
        started = time.perf_counter()
        result = {}
        try:
            src = STORAGE / "checkpoints" / f"lewm-{name}"
            config = json.loads((src / "config.json").read_text(encoding="utf-8-sig"))
            weight_path = src / "weights.pt"
            digest = sha256(weight_path)
            manifest = json.loads((src / "source-manifest.json").read_text(encoding="utf-8-sig"))
            expected = next(item["lfs"]["oid"] for item in manifest if item["path"] == "weights.pt")
            assert digest == expected, "Checkpoint SHA256 does not match official HF manifest"
            result["sha256"] = digest
            result["sha256_verified"] = True
            model = build_model(config)
            state = torch.load(weight_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            result["strict_load"] = True
            result["parameters"] = sum(param.numel() for param in model.parameters())
            model = model.to(device).eval().requires_grad_(False)
            history = int(config["predictor"]["num_frames"])
            image_size = int(config["encoder"]["image_size"])
            action_dim = int(config["action_encoder"]["input_dim"])
            pixels = torch.rand(1, history + 1, 3, image_size, image_size, device=device)
            mean = pixels.new_tensor([0.485, 0.456, 0.406])[None, None, :, None, None]
            std = pixels.new_tensor([0.229, 0.224, 0.225])[None, None, :, None, None]
            pixels = (pixels - mean) / std
            action = torch.randn(1, history + 1, action_dim, device=device) * 0.1
            with torch.inference_mode():
                result["synthetic"] = exercise(model, pixels, action, history)
                result["library_api"] = check_library_api(
                    model, state, name, pixels, action, history, device
                )
                if name == "tworooms":
                    data_path = DATASET_ROOT / "datasets" / "tworoom.h5"
                    if data_path.exists():
                        pixels, action, metadata = real_tworoom_batch(history, device)
                        result["real_dataset"] = exercise(model, pixels, action, history)
                        result["real_dataset"].update(metadata)
                    elif args.require_dataset:
                        raise FileNotFoundError(data_path)
                    else:
                        result["real_dataset"] = {"status": "not_run", "reason": "Dataset not yet extracted"}
            if args.device == "cuda":
                torch.cuda.synchronize()
            result["status"] = "passed"
            del model, state
            if args.device == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = str(exc)
            result["traceback"] = traceback.format_exc()
        result["seconds"] = round(time.perf_counter() - started, 3)
        report["models"][name] = result
        print(f"{name}: {result['status']} ({result['seconds']} seconds)", flush=True)
        if result["status"] == "failed":
            print(result["traceback"], flush=True)
    if args.device == "cuda":
        report["peak_allocated_gpu_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 2)
    report["passed"] = all(item["status"] == "passed" for item in report["models"].values())
    report_path = LOCAL / "logs" / "smoke-inference.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {report_path}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
