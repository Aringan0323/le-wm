"""Restartable full-episode next-embedding comparison.

Each completed model/precision phase is protected by hashes and can be reused.
Incomplete phases restart from their beginning; model state is never updated.
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
import tempfile
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
PROTOCOL = "episode-next-embedding-v1"
LABELS = ("sigreg", "action_predictor")
CSV_FIELDS = ("dataset_index", "episode_id", "start", "native_mse", "fp32_accum_mse",
              "native_transition_1_mse", "native_transition_2_mse", "native_transition_3_mse",
              "fp32_transition_1_mse", "fp32_transition_2_mse", "fp32_transition_3_mse")


def now():
    return datetime.now(timezone.utc).isoformat()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


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


def checkpoint_info(storage, name, manifest_path, canonical_hash, expected_objective):
    import yaml
    folder = storage / "checkpoints" / name
    files = {key: folder / filename for key, filename in
             (("weights", "weights_epoch_1.pt"), ("model_config", "config.json"), ("training_config", "config.yaml"))}
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f"Completed local checkpoint artifact missing: {path}")
    config = yaml.safe_load(files["training_config"].read_text())
    if config["loss"].get("type", "sigreg") != expected_objective:
        raise ValueError(f"{name}: saved objective does not match the requested model label")
    configured_manifest = config.get("episode_manifest")
    if not configured_manifest or Path(configured_manifest).resolve() != manifest_path:
        raise ValueError(f"{name}: saved training config does not use this episode manifest")
    if config.get("episode_manifest_sha256") != canonical_hash:
        raise ValueError(f"{name}: saved training manifest hash does not match the experiment")
    if config["history_size"] != 3 or config["num_preds"] != 1 or config["img_size"] != 224:
        raise ValueError(f"{name}: unexpected next-embedding geometry")
    if config["data"]["dataset"]["frameskip"] != 5:
        raise ValueError(f"{name}: unexpected frame spacing")
    return {"name": name, "files": {key: {"path": str(path), "sha256": file_hash(path),
                "bytes": path.stat().st_size} for key, path in files.items()},
            "episode_manifest": configured_manifest, "episode_manifest_sha256": canonical_hash}


def load_completed_phase(folder, expected_fingerprint):
    summary_path = folder / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    if not summary.get("complete") or summary.get("phase_fingerprint") != expected_fingerprint:
        raise ValueError(f"Completed phase has incompatible configuration: {folder}")
    for filename, expected in summary["output_sha256"].items():
        path = folder / filename
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Completed phase output is missing or changed: {path}")
    if summary["initial_model_state_sha256"] != summary["final_model_state_sha256"]:
        raise ValueError(f"Completed phase recorded a mutated model: {folder}")
    return summary


def read_episode_rows(folder):
    with (folder / "per_episode.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        episode = int(row["episode_id"])
        if episode in result:
            raise ValueError(f"Duplicate episode in summary: {episode}")
        result[episode] = {"windows": int(row["windows"]), "native_sum": float(row["native_sum"]),
                           "fp32_sum": float(row["fp32_sum"])}
    return result


def compare_episodes(baseline, variant, draws, seed):
    if set(baseline) != set(variant):
        raise ValueError("Model phases contain different test episodes")
    episode_ids = sorted(baseline)
    count = np.array([baseline[ep]["windows"] for ep in episode_ids], dtype=np.float64)
    if not np.array_equal(count, [variant[ep]["windows"] for ep in episode_ids]):
        raise ValueError("Model phases contain different window counts by episode")
    if not len(count) or np.any(count <= 0):
        raise ValueError("Empty episode comparison")
    result = {}
    for field, name in (("native_sum", "native_window_metric"), ("fp32_sum", "fp32_accum_metric")):
        a = np.array([baseline[ep][field] for ep in episode_ids])
        b = np.array([variant[ep][field] for ep in episode_ids])
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise FloatingPointError("Nonfinite episode metric")
        base, action = a.sum() / count.sum(), b.sum() / count.sum()
        rng = np.random.default_rng(seed)
        differences, improvements = np.empty(draws), np.empty(draws)
        for draw in range(draws):
            sampled = rng.integers(0, len(count), size=len(count))
            denominator = count[sampled].sum()
            resampled_base = a[sampled].sum() / denominator
            resampled_action = b[sampled].sum() / denominator
            differences[draw] = resampled_action - resampled_base
            improvements[draw] = 100 * (1 - resampled_action / resampled_base) if resampled_base else 0
        episode_differences = (b - a) / count
        result[name] = {
            "sigreg_mse": float(base), "action_predictor_mse": float(action),
            "action_minus_sigreg_mse": float(action - base),
            "action_relative_reduction_percent": float(100 * (1 - action / base)) if base else None,
            "paired_episode_bootstrap_difference_95pct_ci": np.quantile(differences, [.025, .975]).tolist(),
            "paired_episode_bootstrap_reduction_percent_95pct_ci": np.quantile(improvements, [.025, .975]).tolist(),
            "action_wins_episodes": int(np.sum(episode_differences < 0)),
            "sigreg_wins_episodes": int(np.sum(episode_differences > 0)),
            "tied_episodes": int(np.sum(episode_differences == 0)),
            "action_episode_win_fraction": float(np.mean(episode_differences < 0)),
            "episodes": len(count), "windows": int(count.sum()),
            "bootstrap_draws": draws, "bootstrap_seed": seed,
            "weighting": "Window-weighted MSE; bootstrap resamples complete paired episodes and recomputes summed error / summed window count.",
        }
    return result


def write_episode_csv(path, episode_ids, counts, sums):
    fields = ("episode_id", "windows", "native_sum", "native_mse", "fp32_sum", "fp32_accum_mse")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for ep in episode_ids:
            count = int(counts[ep])
            if count <= 0:
                raise ValueError(f"Test episode {ep} has no evaluated windows")
            writer.writerow({"episode_id": int(ep), "windows": count,
                "native_sum": float(sums[ep, 0]), "native_mse": float(sums[ep, 0] / count),
                "fp32_sum": float(sums[ep, 1]), "fp32_accum_mse": float(sums[ep, 1] / count)})
        stream.flush()
        os.fsync(stream.fileno())


def evaluate_phase(model, loader, bundle, folder, label, precision, fingerprint,
                   initial_state_hash, progress_path, phase_number, phase_count):
    import torch
    folder.mkdir(parents=True, exist_ok=True)
    indices = bundle["arrays"]["test_clip_indices"]
    clip_episodes = bundle["arrays"]["clip_episode_ids"]
    clip_starts = bundle["arrays"]["clip_start_steps"]
    episode_ids = np.sort(bundle["arrays"]["test_episode_ids"])
    expected_counts = np.bincount(clip_episodes[indices], minlength=len(loader.dataset.dataset.lengths))
    counts = np.zeros_like(expected_counts, dtype=np.int64)
    sums = np.zeros((len(counts), 8), dtype=np.float64)
    partial_csv = folder / "per_window.csv.partial"
    started_utc, started = now(), time.perf_counter()
    offset, exact_batch_total = 0, 0.0
    native_dtypes = None
    with partial_csv.open("w", newline="", buffering=1024 * 1024) as stream:
        writer = csv.writer(stream)
        writer.writerow(CSV_FIELDS)
        for batch_index, batch in enumerate(loader):
            batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
            batch["action"] = torch.nan_to_num(batch["action"], nan=0.0)
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16_mixed" else nullcontext()
            with torch.inference_mode(), amp:
                encoded = model.encode(batch)
                prediction = model.predict(encoded["emb"][:, :3], encoded["act_emb"][:, :3])
                target = encoded["emb"][:, 1:4]
                if prediction.shape != target.shape or tuple(prediction.shape[1:]) != (3, 192):
                    raise ValueError("Unexpected prediction/target geometry")
                native = (prediction - target).pow(2)
                batch_loss = native.mean()
                accurate = (prediction.float() - target.float()).square()
                metrics = torch.cat((native.mean((1, 2)).float()[:, None],
                                     accurate.mean((1, 2))[:, None],
                                     native.mean(2).float(), accurate.mean(2)), dim=1)
                packed = torch.cat((metrics.reshape(-1), batch_loss.float().reshape(1))).cpu().numpy()
            if not np.isfinite(packed).all():
                raise FloatingPointError(f"Nonfinite MSE in {label}/{precision} batch {batch_index}")
            observed_dtypes = {"prediction": str(prediction.dtype), "target": str(target.dtype),
                               "native_squared_error": str(native.dtype)}
            if native_dtypes is not None and observed_dtypes != native_dtypes:
                raise ValueError("Tensor dtypes changed during evaluation")
            native_dtypes = observed_dtypes
            size = prediction.shape[0]
            values = packed[:-1].reshape(size, 8)
            exact_batch_total += float(packed[-1]) * size
            batch_indices = indices[offset:offset + size]
            batch_episodes = clip_episodes[batch_indices]
            np.add.at(counts, batch_episodes, 1)
            np.add.at(sums, batch_episodes, values)
            for i, index in enumerate(batch_indices):
                writer.writerow((int(index), int(batch_episodes[i]), int(clip_starts[index]),
                                 *map(float, values[i])))
            offset += size
            if batch_index % 100 == 0 or offset == len(indices):
                stream.flush()
                elapsed = time.perf_counter() - started
                remaining = elapsed * (len(indices) - offset) / offset
                write_json(progress_path, {"updated_utc": now(), "complete": False,
                    "phase": f"{label}--{precision}", "phase_number": phase_number, "phase_count": phase_count,
                    "completed_windows": offset, "total_windows": len(indices),
                    "elapsed_phase_seconds": elapsed, "estimated_remaining_phase_seconds": remaining,
                    "validation_arithmetic_mse_so_far": exact_batch_total / offset})
                print(f"{now()} {label}/{precision}: {offset}/{len(indices)} windows; "
                      f"MSE={exact_batch_total / offset:.9g}; remaining ~{remaining / 60:.1f} min", flush=True)
        stream.flush()
        os.fsync(stream.fileno())
    torch.cuda.synchronize()
    if offset != len(indices) or not np.array_equal(counts, expected_counts):
        raise AssertionError("Final evaluated windows do not exactly match the test split")
    final_state_hash = model_hash(model)
    if final_state_hash != initial_state_hash:
        raise AssertionError("Model weights or buffers changed during inference")
    episode_partial = folder / "per_episode.csv.partial"
    write_episode_csv(episode_partial, episode_ids, counts, sums)
    # Publish CSVs first, then the completion record; incomplete phases are rerun.
    partial_csv.replace(folder / "per_window.csv")
    episode_partial.replace(folder / "per_episode.csv")
    totals = sums.sum(axis=0)
    summary = {
        "complete": True, "phase_fingerprint": fingerprint, "model": label, "precision": precision,
        "started_utc": started_utc, "finished_utc": now(), "elapsed_seconds": time.perf_counter() - started,
        "windows": offset, "episodes": len(episode_ids), "transitions": 3 * offset,
        "validation_arithmetic_mse": exact_batch_total / offset,
        "mean_native_window_mse": float(totals[0] / offset),
        "row_vs_batch_aggregate_abs_difference": abs(float(totals[0] / offset) - exact_batch_total / offset),
        "fp32_accum_mse": float(totals[1] / offset),
        "per_transition_native_mse": (totals[2:5] / offset).tolist(),
        "per_transition_fp32_accum_mse": (totals[5:8] / offset).tolist(),
        "initial_model_state_sha256": initial_state_hash, "final_model_state_sha256": final_state_hash,
        "model_state_unchanged": True, "tensor_dtypes": native_dtypes,
        "output_sha256": {name: file_hash(folder / name) for name in ("per_window.csv", "per_episode.csv")},
    }
    write_json(folder / "summary.json", summary)
    print(f"{now()} COMPLETE {label}/{precision}: MSE={summary['validation_arithmetic_mse']:.10g}", flush=True)
    return summary


def run(args):
    import torch
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from episode_data import load_episode_experiment, episode_subset, get_episode_normalizer
    from utils import get_img_preprocessor

    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(3072)
    if args.batch_size != 8:
        raise ValueError("Batch size is fixed at 8 for this matched comparison")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation protocol")
    manifest_path = (args.experiment_dir / "manifest.json").resolve()
    storage = Path(os.environ["STABLEWM_HOME"])
    bundle = load_episode_experiment(manifest_path)
    dataset = swm.data.load_dataset(bundle["manifest"]["dataset"]["path"], cache_dir=str(storage), num_steps=4, frameskip=5,
                                    keys_to_load=["pixels", "action"], keys_to_cache=["action"])
    bundle = load_episode_experiment(manifest_path, dataset)
    arrays, manifest = bundle["arrays"], bundle["manifest"]
    canonical_hash = manifest["manifest_sha256"]
    dataset.transform = spt.data.transforms.Compose(get_img_preprocessor("pixels", "pixels", 224),
                                                     get_episode_normalizer(bundle, "action"))
    subset = episode_subset(dataset, bundle, "test")
    indices = np.asarray(arrays["test_clip_indices"])
    if not np.array_equal(np.asarray(subset.indices), indices):
        raise AssertionError("Test loader order disagrees with immutable split")
    if len(np.unique(indices)) != len(indices):
        raise AssertionError("Duplicate test clip index")
    test_episodes = set(map(int, arrays["test_episode_ids"]))
    if test_episodes.intersection(map(int, arrays["train_episode_ids"])):
        raise AssertionError("Train and test episodes overlap")
    if test_episodes.intersection(map(int, arrays["unused_episode_ids"])):
        raise AssertionError("Unused and test episodes overlap")
    if set(map(int, arrays["clip_episode_ids"][indices])) != test_episodes:
        raise AssertionError("Test window membership is incomplete")
    names = {"sigreg": args.sigreg_model, "action_predictor": args.action_model}
    checkpoints = {label: checkpoint_info(storage, name, manifest_path, canonical_hash,
                         "sigreg" if label == "sigreg" else "action_prediction")
                   for label, name in names.items()}
    plan = {
        "protocol": PROTOCOL, "manifest_path": str(manifest_path),
        "manifest_file_sha256": file_hash(manifest_path), "manifest_canonical_sha256": canonical_hash,
        "test_clip_order_sha256": array_hash(indices), "test_windows": len(indices),
        "test_episodes": len(test_episodes), "predicted_transitions_per_phase": 3 * len(indices),
        "dataset": manifest["dataset"], "normalization": manifest["normalization"]["action"],
        "checkpoints": checkpoints, "batch_size": 8, "loader_workers": args.num_workers, "cpu_threads": args.cpu_threads, "worker_start_method": "spawn" if args.num_workers else None, "history_size": 3, "frameskip": 5, "embedding_width": 192,
        "image_preprocessor_source_sha256": file_hash(REPO / "utils.py"),
        "episode_data_source_sha256": file_hash(REPO / "episode_data.py"),
        "evaluator_source_sha256": file_hash(Path(__file__).resolve()),
        "model_source_sha256": {name: file_hash(REPO / name) for name in ("jepa.py", "module.py")},
        "software": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchvision", "stable-worldmodel", "stable-pretraining", "transformers", "lightning")},
        "gpu": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "metric_protocol": {
            "primary": "BF16 autocast, exact original batch (prediction - target).pow(2).mean(), weighted by batch window count.",
            "native_mse": "Same arithmetic per window, used for paired whole-episode bootstrap; tiny reduction discrepancies are reported.",
            "fp32_accum_mse": "Cast prediction and target to float32 before subtraction, squaring and averaging.",
            "scope": "All three next-embedding predictions per window. No SIGReg, inverse/action-head loss, or planning metric is evaluated.",
            "window_policy": "All valid windows in every held-out test episode, including overlapping windows; confidence intervals resample episodes.",
            "normalization": "Saved statistics fitted only on the 5000 training episodes. No test statistics are fitted or updated.",
        },
    }
    fingerprint = json_hash(plan)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "evaluation-plan.json"
    if plan_path.exists():
        saved = json.loads(plan_path.read_text())
        if saved["fingerprint"] != fingerprint or saved["plan"] != plan:
            raise ValueError("Existing evaluation plan differs from current inputs/code/environment; choose a new output directory.")
    else:
        write_json(plan_path, {"fingerprint": fingerprint, "created_utc": now(), "plan": plan})
    loader = torch.utils.data.DataLoader(subset, batch_size=8, shuffle=False, drop_last=False,
                                         num_workers=args.num_workers, pin_memory=True,
                                         persistent_workers=args.num_workers > 0,
                                         prefetch_factor=2 if args.num_workers else None,
                                         multiprocessing_context="spawn" if args.num_workers else None)
    start, invocation_started = time.perf_counter(), now()
    summaries, resumed = {}, []
    phase_count = len(LABELS) * len(args.precisions)
    phase_number = 0
    print(f"{now()} Full test: {len(test_episodes)} episodes, {len(indices)} windows; "
          f"phases={phase_count}; output={output}", flush=True)
    for label in LABELS:
        summaries[label], model, initial_hash = {}, None, None
        for precision in args.precisions:
            phase_number += 1
            folder = output / "phases" / f"{label}--{precision}"
            phase_fingerprint = json_hash({"plan": fingerprint, "model": label, "precision": precision})
            completed = load_completed_phase(folder, phase_fingerprint)
            if completed is not None:
                if completed["windows"] != len(indices) or completed["episodes"] != len(test_episodes):
                    raise ValueError("Completed phase has incorrect test counts")
                summaries[label][precision] = completed
                resumed.append(f"{label}--{precision}")
                print(f"{now()} REUSE {label}/{precision}: verified completed CSV hashes", flush=True)
                continue
            if model is None:
                model = swm.wm.utils.load_pretrained(f"{names[label]}/weights_epoch_1.pt")
                model.eval().requires_grad_(False)
                assert all(not module.training for module in model.modules())
                initial_hash = model_hash(model)
                model.cuda()
            summaries[label][precision] = evaluate_phase(
                model, loader, bundle, folder, label, precision, phase_fingerprint,
                initial_hash, output / "progress.json", phase_number, phase_count)
        if model is not None:
            model.cpu()
            del model
            torch.cuda.empty_cache()
    comparisons = {}
    for precision in args.precisions:
        baseline = read_episode_rows(output / "phases" / f"sigreg--{precision}")
        action = read_episode_rows(output / "phases" / f"action_predictor--{precision}")
        comparison = compare_episodes(baseline, action, args.bootstrap_draws, args.bootstrap_seed)
        a = summaries["sigreg"][precision]["validation_arithmetic_mse"]
        b = summaries["action_predictor"][precision]["validation_arithmetic_mse"]
        comparison["original_validation_arithmetic"] = {
            "sigreg_mse": a, "action_predictor_mse": b, "action_minus_sigreg_mse": b - a,
            "action_relative_reduction_percent": 100 * (1 - b / a) if a else None}
        comparisons[precision] = comparison
    result = {
        "passed": True, "protocol": PROTOCOL, "plan_fingerprint": fingerprint,
        "started_utc": invocation_started, "finished_utc": now(),
        "invocation_elapsed_seconds": time.perf_counter() - start,
        "summed_phase_elapsed_seconds": sum(s["elapsed_seconds"] for model in summaries.values() for s in model.values()),
        "resumed_completed_phases": resumed, "requested_precisions": args.precisions,
        "primary_precision": "bf16_mixed", "test_episodes": len(test_episodes), "test_windows": len(indices),
        "transitions_per_model_precision": 3 * len(indices),
        "train_test_episode_disjoint": True, "train_only_normalization_verified": True,
        "models": summaries, "comparisons": comparisons,
        "ci_note": "Window-weighted estimates with paired episode-cluster bootstrap. Within-episode overlapping windows are never treated as independent bootstrap units.",
        "interpretation": "Next-embedding MSE in each model's own learned latent space, with no auxiliary objective included.",
    }
    write_json(output / "results.json", result)
    write_json(output / "progress.json", {"complete": True, "passed": True, "finished_utc": now(),
                                        "requested_precisions": args.precisions})
    print(json.dumps({"passed": True, "resumed_completed_phases": resumed,
                      "comparisons": {p: c["original_validation_arithmetic"] for p, c in comparisons.items()}}), flush=True)


def self_test():
    baseline = {0: {"windows": 2, "native_sum": 6., "fp32_sum": 6.},
                1: {"windows": 1, "native_sum": 3., "fp32_sum": 3.}}
    action = {0: {"windows": 2, "native_sum": 4., "fp32_sum": 4.},
              1: {"windows": 1, "native_sum": 1., "fp32_sum": 1.}}
    result = compare_episodes(baseline, action, 2000, 123)
    assert result["native_window_metric"]["sigreg_mse"] == 3
    assert math.isclose(result["native_window_metric"]["action_predictor_mse"], 5 / 3)
    assert result["native_window_metric"]["paired_episode_bootstrap_difference_95pct_ci"][1] < 0
    assert result["native_window_metric"]["action_wins_episodes"] == 2
    with tempfile.TemporaryDirectory(prefix="episode-eval-check-") as name:
        folder = Path(name)
        (folder / "per_window.csv").write_text("test\n1\n")
        (folder / "per_episode.csv").write_text("test\n1\n")
        summary = {"complete": True, "phase_fingerprint": "expected",
                   "initial_model_state_sha256": "same", "final_model_state_sha256": "same",
                   "output_sha256": {f: file_hash(folder / f) for f in ("per_window.csv", "per_episode.csv")}}
        write_json(folder / "summary.json", summary)
        assert load_completed_phase(folder, "expected") == summary
        (folder / "per_window.csv").write_text("changed\n")
        try:
            load_completed_phase(folder, "expected")
        except ValueError:
            pass
        else:
            raise AssertionError("Corrupted completed phase was accepted")
    print(json.dumps({"self_test_passed": True, "window_weighted_episode_bootstrap": True,
                      "completed_phase_reuse_and_corruption_detection": True}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--sigreg-model", default="tworoom-sigreg-episode50-20261106-01")
    parser.add_argument("--action-model", default="tworoom-action-episode50-20261106-01")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--precisions", nargs="+", choices=("bf16_mixed", "fp32"), default=["bf16_mixed", "fp32"])
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261106)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.experiment_dir is None or args.output_dir is None:
        parser.error("--experiment-dir and --output-dir are required")
    if len(set(args.precisions)) != len(args.precisions) or "bf16_mixed" not in args.precisions:
        parser.error("Precisions must be unique and include primary bf16_mixed")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    if args.bootstrap_draws < 100:
        parser.error("--bootstrap-draws must be at least 100")
    run(args)


if __name__ == "__main__":
    main()
