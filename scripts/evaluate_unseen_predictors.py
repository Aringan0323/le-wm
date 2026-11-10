"""Paired next-embedding evaluation on explicitly unexposed episodes.

Run in the activated WSL environment. No model state is updated.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
MODEL_NAMES = {
    "sigreg": "tworoom-pilot-20261106-01",
    "action_predictor": "tworoom-action-pilot-20261106-01",
}
PRECISIONS = ("bf16_mixed", "fp32")


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arrays_hash(arrays):
    digest = hashlib.sha256()
    for name, array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def model_hash(model):
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_manifest(manifest, clip_indices):
    if "forbidden_episode_ids" not in manifest:
        raise ValueError("Manifest must explicitly supply forbidden_episode_ids")
    forbidden = set(map(int, manifest["forbidden_episode_ids"]))
    entries = manifest.get("test_windows", [])
    if not entries:
        raise ValueError("Manifest test_windows is empty")
    indices, seen, episodes = [], set(), set()
    for row in entries:
        index, ep, start = (int(row[key]) for key in ("dataset_index", "episode_id", "start"))
        if index in seen:
            raise ValueError(f"Repeated dataset index {index}")
        if ep in forbidden:
            raise ValueError(f"Forbidden episode {ep} appears in test set")
        if index < 0 or index >= len(clip_indices):
            raise ValueError(f"Invalid dataset index {index}")
        if tuple(map(int, clip_indices[index])) != (ep, start):
            raise ValueError(f"Dataset index {index} does not map to episode/start {(ep, start)}")
        seen.add(index)
        episodes.add(ep)
        indices.append(index)
    by_episode = defaultdict(list)
    for row in entries:
        by_episode[int(row["episode_id"])].append(int(row["start"]))
    for ep, starts in by_episode.items():
        starts.sort()
        if any(right < left + 20 for left, right in zip(starts, starts[1:])):
            raise ValueError(f"Overlapping test windows in episode {ep}")
    return indices, {
        "test_windows": len(indices), "test_episodes": len(episodes),
        "forbidden_episodes": len(forbidden), "test_episode_ids": sorted(episodes),
        "no_forbidden_episode_overlap": True, "no_duplicate_windows": True,
        "nonoverlapping_raw_windows_within_episode": True,
    }


def paired_comparison(rows, precision, field, bootstrap_draws, seed):
    paired = defaultdict(dict)
    for row in rows:
        if row["precision"] == precision:
            paired[int(row["dataset_index"])][row["model"]] = row
    if not paired or any(set(pair) != set(MODEL_NAMES) for pair in paired.values()):
        raise ValueError("Both models must have exactly one result for every test window")
    grouped = defaultdict(lambda: {"sigreg": [], "action_predictor": []})
    for pair in paired.values():
        a, b = pair["sigreg"], pair["action_predictor"]
        assert a["episode_id"] == b["episode_id"]
        for name in MODEL_NAMES:
            grouped[int(a["episode_id"])][name].append(float(pair[name][field]))
    episode_ids = sorted(grouped)
    count = np.array([len(grouped[ep]["sigreg"]) for ep in episode_ids], dtype=np.float64)
    baseline_sum = np.array([sum(grouped[ep]["sigreg"]) for ep in episode_ids])
    variant_sum = np.array([sum(grouped[ep]["action_predictor"]) for ep in episode_ids])
    baseline_mean = baseline_sum.sum() / count.sum()
    variant_mean = variant_sum.sum() / count.sum()
    rng = np.random.default_rng(seed)
    differences, relative_improvements = [], []
    for _ in range(bootstrap_draws):
        selected = rng.integers(0, len(episode_ids), size=len(episode_ids))
        denominator = count[selected].sum()
        base = baseline_sum[selected].sum() / denominator
        variant = variant_sum[selected].sum() / denominator
        differences.append(variant - base)
        relative_improvements.append(100 * (1 - variant / base) if base else 0.0)
    per_episode_difference = (variant_sum - baseline_sum) / count
    return {
        "precision": precision, "per_window_metric": field,
        "windows": int(count.sum()), "episodes": len(episode_ids),
        "sigreg_mse": float(baseline_mean), "action_predictor_mse": float(variant_mean),
        "action_minus_sigreg_mse": float(variant_mean - baseline_mean),
        "action_relative_reduction_percent": float(100 * (1 - variant_mean / baseline_mean)),
        "paired_episode_bootstrap_difference_95pct_ci": np.quantile(differences, [0.025, 0.975]).tolist(),
        "paired_episode_bootstrap_reduction_percent_95pct_ci": np.quantile(relative_improvements, [0.025, 0.975]).tolist(),
        "action_wins_episodes": int(np.sum(per_episode_difference < 0)),
        "sigreg_wins_episodes": int(np.sum(per_episode_difference > 0)),
        "tied_episodes": int(np.sum(per_episode_difference == 0)),
        "action_episode_win_fraction": float(np.mean(per_episode_difference < 0)),
        "bootstrap_draws": bootstrap_draws, "bootstrap_seed": seed,
        "weighting": "window-weighted MSE; whole paired episodes resampled for cluster bootstrap",
    }


def per_episode_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["precision"], int(row["episode_id"]))].append(row)
    result = []
    for (model, precision, ep), values in sorted(grouped.items()):
        result.append({
            "model": model, "precision": precision, "episode_id": ep, "windows": len(values),
            "native_mse": sum(float(row["native_mse"]) for row in values) / len(values),
            "fp32_accum_mse": sum(float(row["fp32_accum_mse"]) for row in values) / len(values),
        })
    return result


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def self_test():
    clips = [(0, 0), (0, 20), (1, 0), (2, 0)]
    good = {"forbidden_episode_ids": [2], "test_windows": [
        {"dataset_index": 0, "episode_id": 0, "start": 0},
        {"dataset_index": 1, "episode_id": 0, "start": 20},
        {"dataset_index": 2, "episode_id": 1, "start": 0}]}
    indices, report = validate_manifest(good, clips)
    assert indices == [0, 1, 2] and report["test_episodes"] == 2
    bad = json.loads(json.dumps(good))
    bad["forbidden_episode_ids"] = [0]
    try:
        validate_manifest(bad, clips)
    except ValueError:
        pass
    else:
        raise AssertionError("Exposure overlap was accepted")
    rows = []
    for index, (ep, base, variant) in enumerate([(0, 2.0, 1.0), (0, 4.0, 3.0), (1, 3.0, 1.0)]):
        for model, value in [("sigreg", base), ("action_predictor", variant)]:
            rows.append({"model": model, "precision": "bf16_mixed", "dataset_index": index,
                         "episode_id": ep, "native_mse": value, "fp32_accum_mse": value})
    comparison = paired_comparison(rows, "bf16_mixed", "native_mse", 2000, 123)
    assert comparison["sigreg_mse"] == 3
    assert math.isclose(comparison["action_predictor_mse"], 5 / 3)
    assert comparison["action_wins_episodes"] == 2
    assert comparison["paired_episode_bootstrap_difference_95pct_ci"][1] < 0
    assert len(per_episode_rows(rows)) == 4
    print(json.dumps({"self_test_passed": True, "manifest_exclusion_and_mapping": True,
                      "paired_cluster_bootstrap": True}), flush=True)


def evaluate(args):
    import torch
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from utils import get_img_preprocessor, get_column_normalizer

    if args.batch_size != 8:
        raise ValueError("This comparison fixes batch size 8 to match both pilot validations")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the BF16 comparison")
    torch.set_num_threads(8)
    torch.manual_seed(3072)
    np.random.seed(3072)
    output = args.output_dir.resolve()
    if (output / "results.json").exists() or (output / "per_window.csv").exists():
        raise FileExistsError(f"Evaluation outputs already exist: {output}")
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    storage = Path(os.environ["STABLEWM_HOME"])
    dataset = swm.data.load_dataset(
        "tworoom.h5", cache_dir=str(storage), num_steps=4, frameskip=5,
        keys_to_load=["pixels", "action"], keys_to_cache=["action"], transform=None)
    indices, checks = validate_manifest(manifest, dataset.clip_indices)
    dataset.transform = spt.data.transforms.Compose(
        get_img_preprocessor("pixels", "pixels", 224),
        get_column_normalizer(dataset, "action", "action"))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices), batch_size=8, shuffle=False,
        drop_last=False, num_workers=0, pin_memory=True)
    dataset_path = Path(dataset.h5_path)
    dataset_stat = dataset_path.stat()
    metadata = {
        "started_utc": now(), "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "dataset": {"path": str(dataset_path), "resolved_path": str(dataset_path.resolve()),
                    "file_bytes": dataset_stat.st_size, "mtime_ns": dataset_stat.st_mtime_ns,
                    "episodes": len(dataset.lengths), "valid_windows": len(dataset),
                    "index_metadata_sha256": arrays_hash([("lengths", dataset.lengths), ("offsets", dataset.offsets)]),
                    "full_file_sha256": sha256(dataset_path) if args.hash_dataset else None},
        "selection_checks": checks, "batch_size": 8, "history_size": 3,
        "prediction_offset": 1, "frameskip": 5, "embedding_width": 192,
        "normalization": "Original full-dataset action z-score and ImageNet image preprocessing; statistics were used by both training runs.",
        "normalization_caveat": "Test episodes are excluded from model fitting and prior selected evaluation, but their raw actions contributed to the original dataset-wide normalization statistics.",
        "metric_definitions": {
            "validation_arithmetic_mse": "Window-count-weighted mean of each batch's exact original (pred - target).pow(2).mean() inside the selected autocast context.",
            "native_mse": "Per-window (pred - target).pow(2).mean() with the same native intermediate dtype as original validation; used for primary paired episode confidence intervals.",
            "fp32_accum_mse": "Cast predictions and targets to float32 before subtraction, squaring, and averaging; auxiliary arithmetic check.",
            "transitions": "Three teacher-forced next-embedding predictions per window; averages all 192 coordinates. No auxiliary loss is evaluated.",
        },
        "software": {package: importlib.metadata.version(package) for package in
                     ("torch", "torchvision", "stable-worldmodel", "stable-pretraining", "transformers", "lightning")},
        "gpu": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "script_sha256": sha256(Path(__file__).resolve()),
        "checkpoints": {},
    }
    if manifest_path != (output / "test-manifest.json").resolve():
        shutil.copy2(manifest_path, output / "test-manifest.json")
    write_json(output / "metadata.json", metadata)
    all_rows, model_summaries = [], {}
    print(f"{now()} Evaluating {len(indices)} windows from {checks['test_episodes']} unexposed episodes", flush=True)
    for label, model_name in MODEL_NAMES.items():
        checkpoint = storage / "checkpoints" / model_name / "weights_epoch_1.pt"
        config = checkpoint.parent / "config.json"
        metadata["checkpoints"][label] = {"path": str(checkpoint),
                                         "sha256": sha256(checkpoint), "config_sha256": sha256(config)}
        model = swm.wm.utils.load_pretrained(f"{model_name}/weights_epoch_1.pt")
        model.eval().requires_grad_(False)
        assert all(not module.training for module in model.modules())
        initial_hash = model_hash(model)
        model = model.cuda()
        model_summaries[label] = {}
        for precision in PRECISIONS:
            exact_batch_total = 0.0
            offset, local_rows = 0, []
            dtypes = None
            mode_start = time.perf_counter()
            for batch_number, batch in enumerate(loader):
                batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
                batch["action"] = torch.nan_to_num(batch["action"], nan=0.0)
                amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if precision == "bf16_mixed" else nullcontext()
                with torch.inference_mode(), amp:
                    encoded = model.encode(batch)
                    predictions = model.predict(encoded["emb"][:, :3], encoded["act_emb"][:, :3])
                    targets = encoded["emb"][:, 1:4]
                    if predictions.shape != targets.shape or predictions.shape[1:] != (3, 192):
                        raise ValueError(f"Unexpected prediction/target shape: {predictions.shape}/{targets.shape}")
                    squared_native = (predictions - targets).pow(2)
                    exact_batch_loss = squared_native.mean()
                    native_windows = squared_native.mean(dim=(1, 2)).float()
                    native_transitions = squared_native.mean(dim=2).float()
                    squared_fp32 = (predictions.float() - targets.float()).square()
                    fp32_windows = squared_fp32.mean(dim=(1, 2))
                    fp32_transitions = squared_fp32.mean(dim=2)
                    if not all(torch.isfinite(value).all() for value in
                               (predictions, targets, exact_batch_loss, fp32_windows)):
                        raise FloatingPointError(f"Nonfinite outputs for {label}/{precision}/{batch_number}")
                dtypes = {"predictions": str(predictions.dtype), "targets": str(targets.dtype),
                          "native_squared_error": str(squared_native.dtype)}
                count = predictions.shape[0]
                exact_batch_total += float(exact_batch_loss) * count
                native_windows, fp32_windows = native_windows.cpu().tolist(), fp32_windows.cpu().tolist()
                native_transitions, fp32_transitions = native_transitions.cpu().tolist(), fp32_transitions.cpu().tolist()
                for local_index in range(count):
                    record = manifest["test_windows"][offset + local_index]
                    row = {"model": label, "precision": precision,
                           "dataset_index": int(record["dataset_index"]),
                           "episode_id": int(record["episode_id"]), "start": int(record["start"]),
                           "native_mse": native_windows[local_index],
                           "fp32_accum_mse": fp32_windows[local_index]}
                    for transition in range(3):
                        row[f"native_transition_{transition + 1}_mse"] = native_transitions[local_index][transition]
                        row[f"fp32_transition_{transition + 1}_mse"] = fp32_transitions[local_index][transition]
                    local_rows.append(row)
                offset += count
                if batch_number % 25 == 0 or offset == len(indices):
                    write_json(output / "progress.json", {"updated_utc": now(), "model": label,
                               "precision": precision, "completed_windows": offset, "total_windows": len(indices)})
                    print(f"{now()} {label}/{precision}: {offset}/{len(indices)} windows", flush=True)
            torch.cuda.synchronize()
            assert offset == len(indices)
            native_mean = sum(row["native_mse"] for row in local_rows) / len(local_rows)
            summary = {
                "windows": len(local_rows), "transitions": 3 * len(local_rows),
                "validation_arithmetic_mse": exact_batch_total / len(local_rows),
                "mean_native_window_mse": native_mean,
                "row_vs_batch_aggregate_abs_difference": abs(native_mean - exact_batch_total / len(local_rows)),
                "fp32_accum_mse": sum(row["fp32_accum_mse"] for row in local_rows) / len(local_rows),
                "per_transition_native_mse": [sum(row[f"native_transition_{i}_mse"] for row in local_rows) / len(local_rows) for i in (1, 2, 3)],
                "per_transition_fp32_accum_mse": [sum(row[f"fp32_transition_{i}_mse"] for row in local_rows) / len(local_rows) for i in (1, 2, 3)],
                "tensor_dtypes": dtypes, "elapsed_seconds": time.perf_counter() - mode_start,
            }
            model_summaries[label][precision] = summary
            all_rows.extend(local_rows)
            write_csv(output / "per_window.csv", all_rows)
            print(f"{now()} Finished {label}/{precision}: MSE={summary['validation_arithmetic_mse']:.10g}", flush=True)
        model.cpu()
        final_hash = model_hash(model)
        if initial_hash != final_hash:
            raise AssertionError(f"Model weights or buffers changed during evaluation: {label}")
        metadata["checkpoints"][label].update({
            "initial_state_sha256": initial_hash, "final_state_sha256": final_hash,
            "model_state_unchanged": True})
        del model
        torch.cuda.empty_cache()
    write_csv(output / "per_episode.csv", per_episode_rows(all_rows))
    comparisons = {}
    for precision in PRECISIONS:
        comparisons[precision] = {
            "native_window_metric": paired_comparison(all_rows, precision, "native_mse", args.bootstrap_draws, args.bootstrap_seed),
            "fp32_accum_metric": paired_comparison(all_rows, precision, "fp32_accum_mse", args.bootstrap_draws, args.bootstrap_seed),
        }
        base = model_summaries["sigreg"][precision]["validation_arithmetic_mse"]
        action = model_summaries["action_predictor"][precision]["validation_arithmetic_mse"]
        comparisons[precision]["original_validation_arithmetic"] = {
            "sigreg_mse": base, "action_predictor_mse": action,
            "action_minus_sigreg_mse": action - base,
            "action_relative_reduction_percent": 100 * (1 - action / base)}
    metadata["finished_utc"] = now()
    metadata["elapsed_seconds"] = time.perf_counter() - start
    write_json(output / "metadata.json", metadata)
    result = {"passed": True, "metadata": metadata, "models": model_summaries,
              "comparisons": comparisons, "primary_precision": "bf16_mixed",
              "primary_metric": "validation_arithmetic_mse",
              "ci_note": "Paired episode bootstrap uses native per-window MSE; the batch-vs-window reduction discrepancy is reported for each model.",
              "interpretation": "Comparison of next-embedding MSE in each model's own learned latent space. No auxiliary losses or planning scores are evaluated."}
    write_json(output / "results.json", result)
    write_json(output / "progress.json", {"finished_utc": now(), "complete": True, "passed": True})
    print(json.dumps({"passed": True, "elapsed_seconds": metadata["elapsed_seconds"],
                      "comparisons": {key: value["original_validation_arithmetic"] for key, value in comparisons.items()}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261106)
    parser.add_argument("--hash-dataset", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.manifest is None or args.output_dir is None:
        parser.error("--manifest and --output-dir are required")
    if args.bootstrap_draws < 100:
        parser.error("--bootstrap-draws must be at least 100")
    evaluate(args)


if __name__ == "__main__":
    main()
