"""Audit the checkpoint and final export of a native Train-LeWM.ps1 -Smoke run."""
import argparse
import csv
import json
import math
from pathlib import Path

import torch

from runtime_paths import LOGS, STORAGE


def audit(checkpoint_path, export_path, device):
    from smoke_inference import exercise, real_tworoom_batch
    import stable_worldmodel as swm

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    states = checkpoint["optimizer_states"][0]["state"]
    steps = [int(item["step"]) for item in states.values() if "step" in item]
    if checkpoint["global_step"] != 2 or not steps or min(steps) != 2 or max(steps) != 2:
        raise ValueError("Expected the completed two-update smoke checkpoint")
    if not any(torch.count_nonzero(item["exp_avg"]) for item in states.values() if "exp_avg" in item):
        raise ValueError("No nonzero optimizer moments")
    for item in states.values():
        for value in item.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise ValueError("Nonfinite optimizer state")
    with (checkpoint_path.parent.parent / "metrics.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    train = [float(row["fit/loss"]) for row in rows if row.get("fit/loss")]
    val = [float(row["validate/loss_epoch"]) for row in rows if row.get("validate/loss_epoch")]
    rates = [float(row["hparams/lr_model_opt"]) for row in rows if row.get("hparams/lr_model_opt")]
    if len(train) != 2 or len(val) != 1 or not all(map(math.isfinite, train + val)) or not rates or max(rates) <= 0:
        raise ValueError("Incomplete smoke-training metrics")
    weights = torch.load(export_path, map_location="cpu", weights_only=True)
    expected = {name.removeprefix("model."): tensor for name, tensor in checkpoint["state_dict"].items()
                if name.startswith("model.")}
    if set(weights) != set(expected):
        raise ValueError("Training checkpoint and export have different model keys")
    for name, value in weights.items():
        if not torch.isfinite(value).all() or not torch.equal(value, expected[name]):
            raise ValueError(f"Invalid exported model state at {name}")
    model = swm.wm.utils.load_pretrained(str(export_path.resolve())).to(device).eval().requires_grad_(False)
    with torch.inference_mode():
        pixels, actions, _ = real_tworoom_batch(3, torch.device(device))
        inference = exercise(model, pixels, actions, 3)
    return {"passed": True, "training_steps": 2, "train_losses": train, "validation_loss": val[0],
            "max_logged_learning_rate": max(rates), "optimizer_parameter_states": len(steps),
            "export_matches_lightning_checkpoint": True, "checkpoint": str(checkpoint_path),
            "export": str(export_path), "trained_checkpoint_real_data_inference": inference}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted last.ckpt from this smoke run")
    parser.add_argument("--export", type=Path, default=STORAGE / "checkpoints" / "tworoom-smoke" / "weights_epoch_1.pt")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    result = audit(args.checkpoint.resolve(), args.export.resolve(), args.device)
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "smoke-training.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
