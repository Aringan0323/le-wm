"""Measure held-out next-embedding MSE and encoder variation for a trained Delta-JEPA model."""
import argparse
from pathlib import Path

import yaml
import numpy as np

from audit_combined_experiment import check, read_json
from delta_sweep_config import VARIANTS
from delta_group_layout import model_name, sibling
from evaluate_combined_experiment import (
    PRECISIONS, add_candidate_reference, publish_results, score_candidate,
)
from evaluate_episode_experiment import PROTOCOL, array_hash, file_hash, json_hash
from evaluate_mse_experiment import read_variation, score_with_variation
from episode_data import load_episode_experiment


# Pin the original full-test row order independently of workstation paths or earlier model exports.
TEST_ORDER_SHA256 = "ddc698c484186274d3f6afe9546f93e5b6527e70980efcc5d7abcef4ae71babd"
SPLIT_ARRAYS_SHA256 = "7ca5f87c3ae40a6023cf8d24c933c27d9ebd9a4a819be43c8f88e3da852dcd5f"


def verified_split(experiment):
    bundle = load_episode_experiment(experiment / "manifest.json")
    arrays = bundle["arrays"]
    manifest = bundle["manifest"]
    check(json_hash(manifest["array_hashes"]) == SPLIT_ARRAYS_SHA256,
          "Episode memberships, row mapping or clip indices differ from the matched split")
    norm = manifest["normalization"]["action"]
    check(norm["source_split"] == "train" and norm["train_episode_ids_sha256"] == manifest["array_hashes"]["train_episode_ids"]
          and norm["valid_rows"] == 455657 and norm["excluded_nan_rows"] == 5000,
          "The held-out evaluation requires action normalization fitted only on the recorded training episodes")
    indices = arrays["test_clip_indices"]
    episode_ids, counts = np.unique(arrays["clip_episode_ids"][indices], return_counts=True)
    check(len(indices) == 218790 and array_hash(indices) == TEST_ORDER_SHA256 and len(episode_ids) == 3000
          and np.array_equal(episode_ids, np.sort(arrays["test_episode_ids"])), "Wrong full held-out episode/window set")
    check(not np.intersect1d(episode_ids, arrays["train_episode_ids"]).size
          and not np.intersect1d(episode_ids, arrays["unused_episode_ids"]).size, "Split overlap")
    return bundle, dict(zip(map(int, episode_ids), map(int, counts)))


def delta_checkpoint(variant, experiment, storage, canonical_hash, group=None):
    audit = read_json(experiment / "runs" / variant.label / "final-audit.json")
    check(audit["passed"] and audit["audit"]["objective"] == variant.objective, "Missing passing Delta-JEPA audit")
    record = audit["audit"]
    name = model_name(experiment, variant, group)
    folder = storage / "checkpoints" / name
    config_path = folder / "config.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    check(cfg["output_model_name"] == name, "Saved configuration labels another model")
    manifest = (experiment / "manifest.json").resolve()
    check(record["manifest_sha256"] == cfg["episode_manifest_sha256"] == canonical_hash
          and Path(cfg["episode_manifest"]).resolve() == manifest, "Wrong split for final model")
    loss = cfg["loss"]
    check(loss["type"] == variant.objective and loss["ldad"]["weight"] == variant.ldad_weight
          and loss["sigreg"]["weight"] == variant.sigreg_weight,
          "Final training coefficients differ from the requested Delta-JEPA variant")
    check(cfg["history_size"] == 3 and cfg["num_preds"] == 1 and cfg["img_size"] == 224
          and cfg["data"]["dataset"]["frameskip"] == 5, "Unexpected evaluation geometry")
    final_epoch = int(cfg["trainer"]["max_epochs"])
    weights = folder / f"weights_epoch_{final_epoch}.pt"
    check(final_epoch >= 1 and weights.resolve() == Path(record["export"]["path"]).resolve(),
          "Training audit did not identify the final-epoch export")
    files = {"weights": weights, "training_config": config_path, "model_config": folder / "config.json"}
    details = {key: {"path": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size}
               for key, path in files.items()}
    check(details["weights"]["sha256"] == record["export"]["weights_file_sha256"],
          "Final Delta-JEPA weights have changed since training audit")
    if group is not None:
        check(record["group_commit"]["group_record_sha256"] == read_json(group / "group-checkpoint.json")["record_sha256"]
              and final_epoch == int(read_json(experiment / "plan.json")["epochs"]),
              "Evaluation does not identify the audited final group generation")
        has_ldad = variant.ldad_weight > 0
        check((cfg["model"].get("ldad") is not None) == has_ldad
              and record["delta_objective"]["ldad_decoder_enabled"] is has_ldad,
              "Group control/decoder architecture differs from its audited model")
    return {"name": name, "files": details, "epoch": final_epoch,
            "episode_manifest": str(manifest), "episode_manifest_sha256": canonical_hash}


def references(experiment, variant, group=None):
    bundle, counts = verified_split(experiment)
    plan = {
        "protocol": PROTOCOL, "test_clip_order_sha256": TEST_ORDER_SHA256,
        "normalization": bundle["manifest"]["normalization"]["action"], "history_size": 3, "frameskip": 5,
        "embedding_width": 192, "batch_size": 8, "cpu_threads": 1, "loader_workers": 4,
        "worker_start_method": "spawn", "float32_matmul_precision": "highest", "cuda_matmul_allow_tf32": False,
        "evaluator_source_sha256": "33d7366da9a482761ba9d983c7fafe06548fc07a5044f3e66a558bf6e5b58fef",
        "software": ["torch", "torchvision", "stable-worldmodel", "stable-pretraining", "transformers", "lightning"],
        "metric_protocol": {
            "primary": "BF16 autocast, window-weighted native batch mean over three predictions and 192 features.",
            "paired": "Resample entire episodes, preserving window counts, using native per-window MSE.",
            "control": "Repeat held-out scoring with full float32 model inference.",
        },
    }
    evidence = {"protocol_spec_sha256": json_hash(plan), "scores": {p: {} for p in PRECISIONS}}
    data = bundle["manifest"], plan, evidence, {p: {} for p in PRECISIONS}, counts
    saved = experiment / "evaluation" / "evaluation-plan.json"
    if saved.is_file():
        stored = read_json(saved)["plan"]["reference"].get("additional_references", {})
        paths = {label: Path(r["experiment_dir"]) for label, r in stored.items()}
    else:
        paths = {}
        for earlier in VARIANTS.values():
            if earlier.label == variant.label:
                break
            other_dir = sibling(experiment, earlier, group)
            if group is not None or (other_dir / "evaluation" / "results.json").is_file():
                paths[earlier.label] = other_dir
    if group is not None:
        expected = {}
        for earlier in VARIANTS.values():
            if earlier.label == variant.label:
                break
            expected[earlier.label] = (group / "members" / earlier.label).resolve(strict=True)
        check({label: path.resolve(strict=True) for label, path in paths.items()} == expected,
              "Group evaluation must compare exactly the completed earlier members of its own group")
    for label, path in paths.items():
        if group is not None:
            other = VARIANTS[label]
            identity = read_json(path / "evaluation" / "evaluation-plan.json")["plan"]["checkpoints"][label]
            other_audit = read_json(path / "runs" / label / "final-audit.json")
            check(other_audit["passed"] and identity["name"] == model_name(path, other, group)
                  and identity["files"]["weights"]["sha256"] == other_audit["audit"]["export"]["weights_file_sha256"]
                  and other_audit["audit"]["group_commit"]["group_record_sha256"] == read_json(group / "group-checkpoint.json")["record_sha256"],
                  "Sibling evaluation does not match its audited group export")
        add_candidate_reference(data, path, label)
    return data


def report(experiment, result):
    variant = VARIANTS[result["variant"]]
    lines = [f"Two-Room held-out next-embedding prediction, {variant.display_label}",
             f"{result['test_episodes']:,} test episodes, {result['test_windows']:,} windows, "
             f"{result['transitions_per_model_precision']:,} predicted transitions per precision.",
             "Each model predicts its own encoder's next image embedding.", ""]
    for precision in PRECISIONS:
        comparison = result["comparisons"][precision]
        values = comparison["original_validation_arithmetic"]
        score = values[variant.label + "_mse"]
        lines.append(f"{precision}    Next-embedding MSE {score:.12g}")
        for label, stats in comparison["paired_by_reference"].items():
            change = values[f"{variant.label}_minus_{label}_mse"]
            ci = stats["native_window_metric"]["paired_episode_bootstrap_difference_95pct_ci"]
            lines.append(f"  {label} {values[label + '_mse']:.12g}. Candidate minus reference "
                         f"{change:+.12g}, paired episode-bootstrap interval [{ci[0]:.8g}, {ci[1]:.8g}].")
        latent = result["test_latent_variation"][precision]
        lines.extend([f"  Encoder variance {latent['latent_cross_example_var']:.12g}, "
                      f"RMS standard deviation {latent['latent_cross_example_rms_std']:.12g}, "
                      f"maximum observed channel range {latent['latent_global_max_range']:.12g}.", ""])
    path = experiment / "report.txt"
    temporary = path.with_suffix(".txt.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def evaluate(experiment, variant, group=None):
    planned_name = model_name(experiment, variant, group)
    def checkpoint_reader(storage, name, manifest, manifest_hash, objective):
        check(name == planned_name and manifest == experiment / "manifest.json" and objective == variant.objective,
              "Unexpected model or split requested")
        return delta_checkpoint(variant, experiment, storage, manifest_hash, group)

    result = score_candidate(experiment, planned_name, variant.label, variant.objective,
                             references(experiment, variant, group), phase_runner=score_with_variation,
                             entrypoint=Path(__file__), checkpoint_reader=checkpoint_reader)
    result["variant"] = variant.label
    if group is not None:
        result["group_root"] = str(group)
        result["group_record_sha256"] = read_json(group / "group-checkpoint.json")["record_sha256"]
        result["model_name"] = planned_name
        result["ldad_decoder_enabled"] = variant.ldad_weight > 0
    result["loss_weights"] = {"next_embedding_mse": 1.0, "ldad": variant.ldad_weight, "sigreg": variant.sigreg_weight}
    result["test_latent_variation"] = {
        p: read_variation(experiment / "evaluation" / "phases" / f"{variant.label}--{p}", result["models"][variant.label][p])
        for p in PRECISIONS
    }
    publish_results(experiment, result, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--group-root", type=Path)
    args = parser.parse_args()
    variant = VARIANTS[args.variant]
    evaluate((args.experiment_dir or variant.experiment).resolve(strict=True), variant,
             args.group_root.resolve(strict=True) if args.group_root else None)
