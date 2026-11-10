"""Score only the combined model against experiment 03's verified held-out results."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from audit_combined_experiment import (
    MODEL_NAME, OBJECTIVE, REFERENCE_DIR, check, check_matching_manifests, read_json,
)
from evaluate_episode_experiment import (
    PROTOCOL, REPO, array_hash, checkpoint_info, compare_episodes, evaluate_phase, file_hash,
    json_hash, load_completed_phase, model_hash, now, read_episode_rows, write_json,
)

PRECISIONS = ("bf16_mixed", "fp32")
REFERENCES = ("sigreg", "action_predictor")
LABEL = "combined"


def phase_details(folder, label, precision, fingerprint, episode_counts):
    phase_key = json_hash({"plan": fingerprint, "model": label, "precision": precision})
    summary = load_completed_phase(folder, phase_key)
    check(summary is not None, f"Missing completed phase in {folder}")
    windows = sum(episode_counts.values())
    check(summary["model"] == label and summary["precision"] == precision, f"Mislabeled phase in {folder}")
    check(summary["windows"] == windows and summary["episodes"] == len(episode_counts)
          and summary["transitions"] == 3 * windows, f"Incomplete evaluation in {folder}")
    check(summary["model_state_unchanged"] is True, f"Model state changed in {folder}")
    check(set(summary["output_sha256"]) == {"per_episode.csv", "per_window.csv"}, "Incomplete metric artifacts")
    episodes = read_episode_rows(folder)
    check({ep: values["windows"] for ep, values in episodes.items()} == episode_counts,
          f"Per-episode window coverage differs in {folder}")
    for field, key in (("native_sum", "mean_native_window_mse"), ("fp32_sum", "fp32_accum_mse")):
        score = math.fsum(row[field] for row in episodes.values()) / windows
        check(math.isclose(score, summary[key], rel_tol=1e-10, abs_tol=1e-15),
              f"Per-episode scores disagree with {key} in {folder}")
    return summary, episodes


def verified_reference(reference):
    from episode_data import load_episode_experiment

    previous = load_episode_experiment(reference / "manifest.json")
    arrays, manifest = previous["arrays"], previous["manifest"]
    indices = arrays["test_clip_indices"]
    episodes, counts = np.unique(arrays["clip_episode_ids"][indices], return_counts=True)
    episode_counts = dict(zip(map(int, episodes), map(int, counts)))
    check(len(indices) == 218790 and len(episode_counts) == 3000, "Expected the full experiment 03 test set")
    check(set(episode_counts) == set(map(int, arrays["test_episode_ids"])), "Missing held-out episodes")
    check(set(episode_counts).isdisjoint(arrays["train_episode_ids"])
          and set(episode_counts).isdisjoint(arrays["unused_episode_ids"]), "Test episodes overlap another split")
    folder = reference / "evaluation"
    saved = read_json(folder / "evaluation-plan.json")
    result = read_json(folder / "results.json")
    plan = saved["plan"]
    check(saved["fingerprint"] == result["plan_fingerprint"] == json_hash(plan), "Reference plan fingerprint differs")
    check(result["passed"] and result["protocol"] == plan["protocol"] == PROTOCOL, "Reference protocol differs")
    check(plan["manifest_canonical_sha256"] == manifest["manifest_sha256"]
          and plan["test_clip_order_sha256"] == array_hash(indices), "Reference split or test order differs")
    check(result["requested_precisions"] == list(PRECISIONS) and result["train_test_episode_disjoint"]
          and result["train_only_normalization_verified"], "Reference evaluation incomplete")
    states, scores, rows = {}, {}, {}
    for precision in PRECISIONS:
        scores[precision], rows[precision] = {}, {}
        for label in REFERENCES:
            phase_folder = folder / "phases" / f"{label}--{precision}"
            summary, episodes = phase_details(phase_folder, label, precision, saved["fingerprint"], episode_counts)
            check(summary == result["models"][label][precision], f"Reference summary changed for {label}/{precision}")
            mse = result["comparisons"][precision]["original_validation_arithmetic"][f"{label}_mse"]
            check(summary["validation_arithmetic_mse"] == mse and math.isfinite(mse) and mse > 0,
                  f"Reference headline MSE differs for {label}/{precision}")
            scores[precision][f"{label}_mse"] = mse
            rows[precision][label] = episodes
            states[f"{label}--{precision}"] = file_hash(phase_folder / "summary.json")
    evidence = {
        "experiment_dir": str(reference), "results_file": str(folder / "results.json"),
        "results_sha256": file_hash(folder / "results.json"), "plan_fingerprint": saved["fingerprint"],
        "phase_summary_sha256": states, "scores": scores,
    }
    return manifest, plan, evidence, rows, episode_counts


def add_candidate_reference(data, experiment, label):
    """Extend the two verified original reference scores with one completed, matching candidate."""
    from episode_data import load_episode_experiment

    previous_manifest, previous_plan, evidence, rows, episode_counts = data
    folder = experiment / "evaluation"
    result = read_json(folder / "results.json")
    check(result.get("passed") is True, f"{label} reference evaluation did not pass")
    manifest = load_episode_experiment(experiment / "manifest.json")["manifest"]
    check_matching_manifests(manifest, previous_manifest)
    saved = read_json(folder / "evaluation-plan.json")
    plan = saved["plan"]
    check(plan["model"] == label and saved["fingerprint"] == json_hash(plan) == result["plan_fingerprint"],
          f"{label} reference identity differs")
    protocol_fields = ("test_clip_order_sha256", "normalization", "batch_size", "metric_protocol")
    check(all(plan[key] == previous_plan[key] for key in protocol_fields)
          and plan["manifest_canonical_sha256"] == manifest["manifest_sha256"]
          and result["reference"]["results_sha256"] == evidence["results_sha256"]
          and result["protocol"] == previous_plan["protocol"], f"{label} reference scoring protocol differs")
    summaries = {}
    for precision in PRECISIONS:
        phase = folder / "phases" / f"{label}--{precision}"
        summary, episode_rows = phase_details(phase, label, precision, saved["fingerprint"], episode_counts)
        check(summary == result["models"][label][precision], "Reference phase summary differs from final result")
        mse = result["comparisons"][precision]["original_validation_arithmetic"][f"{label}_mse"]
        check(mse == summary["validation_arithmetic_mse"] and math.isfinite(mse) and mse >= 0,
              f"Invalid {label} reference score")
        rows[precision][label] = episode_rows
        evidence["scores"][precision][f"{label}_mse"] = mse
        summaries[precision] = file_hash(phase / "summary.json")
    evidence.setdefault("additional_references", {})[label] = {
        "experiment_dir": str(experiment), "results_sha256": file_hash(folder / "results.json"),
        "plan_fingerprint": saved["fingerprint"], "phase_summary_sha256": summaries,
    }
    return data


def relative_reduction(reference_mse, combined_mse):
    return 100 * (1 - combined_mse / reference_mse) if reference_mse else None


def paired_statistics(previous, combined, candidate_label=LABEL):
    # Reuse exactly the original paired whole-episode bootstrap, naming the models explicitly.
    result = compare_episodes(previous, combined, 2000, 20261106)
    names = {
        "sigreg_mse": "reference_mse", "action_predictor_mse": f"{candidate_label}_mse",
        "action_minus_sigreg_mse": f"{candidate_label}_minus_reference_mse",
        "action_relative_reduction_percent": f"{candidate_label}_relative_reduction_percent",
        "action_wins_episodes": f"{candidate_label}_wins_episodes", "sigreg_wins_episodes": "reference_wins_episodes",
        "action_episode_win_fraction": f"{candidate_label}_episode_win_fraction",
    }
    return {metric: {names.get(key, key): value for key, value in values.items()}
            for metric, values in result.items()}


def compare_phases(summaries, rows, reference_scores, reference_rows, candidate_label=LABEL):
    comparisons = {}
    for precision in PRECISIONS:
        mse = summaries[precision]["validation_arithmetic_mse"]
        headline = {**reference_scores[precision], f"{candidate_label}_mse": mse}
        paired = {}
        for label in reference_rows[precision]:
            prior = headline[f"{label}_mse"]
            headline[f"{candidate_label}_minus_{label}_mse"] = mse - prior
            headline[f"{candidate_label}_relative_reduction_vs_{label}_percent"] = relative_reduction(prior, mse)
            paired[label] = paired_statistics(reference_rows[precision][label], rows[precision], candidate_label)
        comparisons[precision] = {"original_validation_arithmetic": headline, "paired_by_reference": paired}
    return comparisons


def publish_report(experiment, result):
    lines = [
        "Two-Room combined-objective held-out next-embedding MSE",
        f"Test: {result['test_episodes']:,} held-out episodes, {result['test_windows']:,} windows, "
        f"{result['transitions_per_model_precision']:,} predicted transitions per model/precision.",
        f"Reference results: {result['reference']['results_file']}",
        "",
    ]
    for precision in PRECISIONS:
        scores = result["comparisons"][precision]["original_validation_arithmetic"]
        lines.extend([precision, f"  SIGReg:          {scores['sigreg_mse']:.12g}",
                      f"  Action predictor: {scores['action_predictor_mse']:.12g}",
                      f"  Combined:         {scores['combined_mse']:.12g}"])
        for label in REFERENCES:
            reduction = scores[f"combined_relative_reduction_vs_{label}_percent"]
            stats = result["comparisons"][precision]["paired_by_reference"][label]["native_window_metric"]
            lower, upper = stats["paired_episode_bootstrap_reduction_percent_95pct_ci"]
            lines.append(f"  Combined relative MSE reduction vs {label}: {reduction:.6f}% "
                         f"(paired episode-bootstrap 95% interval {lower:.6f}% to {upper:.6f}%).")
        lines.append("")
    final_path = experiment / "report.txt"
    temporary = final_path.with_suffix(".txt.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(final_path)


def score_candidate(experiment, model_name, label, objective, reference_data, *,
                    phase_runner=evaluate_phase, entrypoint=None):
    """Run the existing two precision phases for one audited candidate model."""
    import torch
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from episode_data import episode_subset, get_episode_normalizer, load_episode_experiment
    from experiment_tracking import state_dict_sha256
    from utils import get_img_preprocessor

    output = experiment / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    audit_path = experiment / "runs" / label / "final-audit.json"
    final_audit = read_json(audit_path)
    check(final_audit.get("passed") is True and final_audit["audit"]["objective"] == objective,
          "Training audit must pass before evaluation")
    started, tick = now(), time.perf_counter()
    prior_manifest, prior_plan, evidence, reference_rows, episode_counts = reference_data
    manifest_path = experiment / "manifest.json"
    bundle = load_episode_experiment(manifest_path)
    manifest, arrays = bundle["manifest"], bundle["arrays"]
    check_matching_manifests(manifest, prior_manifest)
    check(final_audit["audit"]["manifest_sha256"] == manifest["manifest_sha256"], "Audit belongs to another manifest")
    check(prior_plan["batch_size"] == 8 and prior_plan["history_size"] == 3 and prior_plan["frameskip"] == 5
          and prior_plan["embedding_width"] == 192, "Reference evaluation geometry differs")
    check(prior_plan["normalization"] == manifest["normalization"]["action"], "Reference normalization differs")
    check(file_hash(REPO / "scripts" / "evaluate_episode_experiment.py") == prior_plan["evaluator_source_sha256"],
          "Reference and candidate scoring implementations differ")
    storage = Path(os.environ["STABLEWM_HOME"])
    checkpoint = checkpoint_info(storage, model_name, manifest_path, manifest["manifest_sha256"], objective)
    check(checkpoint["files"]["weights"]["sha256"] == final_audit["audit"]["export"]["weights_file_sha256"],
          "Export no longer matches the audited model")
    torch.set_num_threads(prior_plan["cpu_threads"])
    torch.manual_seed(3072)
    torch.set_float32_matmul_precision(prior_plan["float32_matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = prior_plan["cuda_matmul_allow_tf32"]
    check(torch.cuda.is_available(), "CUDA is required for the original mixed-precision protocol")
    dataset = swm.data.load_dataset(manifest["dataset"]["path"], cache_dir=str(storage), num_steps=4, frameskip=5,
                                    keys_to_load=["pixels", "action"], keys_to_cache=["action"])
    bundle = load_episode_experiment(manifest_path, dataset)
    dataset.transform = spt.data.transforms.Compose(get_img_preprocessor("pixels", "pixels", 224),
                                                     get_episode_normalizer(bundle, "action"))
    subset = episode_subset(dataset, bundle, "test")
    indices = arrays["test_clip_indices"]
    check(np.array_equal(np.asarray(subset.indices), indices)
          and array_hash(indices) == prior_plan["test_clip_order_sha256"], "Candidate test window order differs")
    workers = prior_plan["loader_workers"]
    loader = torch.utils.data.DataLoader(
        subset, batch_size=prior_plan["batch_size"], shuffle=False, drop_last=False, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0, prefetch_factor=2 if workers else None,
        multiprocessing_context=prior_plan["worker_start_method"] if workers else None,
    )
    plan = {
        "protocol": PROTOCOL, "model": label, "precisions": list(PRECISIONS),
        "reference": evidence, "manifest_path": str(manifest_path),
        "manifest_file_sha256": file_hash(manifest_path), "manifest_canonical_sha256": manifest["manifest_sha256"],
        "test_clip_order_sha256": array_hash(indices), "test_windows": len(indices), "test_episodes": len(episode_counts),
        "predicted_transitions_per_phase": 3 * len(indices), "dataset": manifest["dataset"],
        "normalization": manifest["normalization"]["action"], "checkpoints": {label: checkpoint},
        "batch_size": prior_plan["batch_size"], "loader_workers": workers, "cpu_threads": prior_plan["cpu_threads"],
        "worker_start_method": prior_plan["worker_start_method"], "history_size": 3, "frameskip": 5,
        "embedding_width": 192, "image_preprocessor_source_sha256": file_hash(REPO / "utils.py"),
        "episode_data_source_sha256": file_hash(REPO / "episode_data.py"),
        "inference_implementation_sha256": prior_plan["evaluator_source_sha256"],
        "evaluator_source_sha256": file_hash(Path(__file__).resolve()),
        "model_source_sha256": {name: file_hash(REPO / name) for name in ("jepa.py", "module.py")},
        "software": {name: importlib.metadata.version(name) for name in prior_plan["software"]},
        "gpu": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "metric_protocol": prior_plan["metric_protocol"], "audit_export_sha256": final_audit["audit"]["export"]["weights_file_sha256"],
    }
    if entrypoint is not None:
        plan["candidate_entrypoint_sha256"] = file_hash(entrypoint)
    fingerprint = json_hash(plan)
    plan_path = output / "evaluation-plan.json"
    if plan_path.exists():
        saved = read_json(plan_path)
        check(saved["fingerprint"] == fingerprint and saved["plan"] == plan,
              "Existing evaluation phases belong to different evidence or a different protocol")
    else:
        write_json(plan_path, {"created_utc": now(), "fingerprint": fingerprint, "plan": plan})
    summaries, rows, reused = {}, {}, []
    model = None
    for index, precision in enumerate(PRECISIONS, 1):
        folder = output / "phases" / f"{label}--{precision}"
        phase_key = json_hash({"plan": fingerprint, "model": label, "precision": precision})
        if (folder / "summary.json").exists():
            reused.append(f"{label}--{precision}")
            print(f"{now()} Reusing completed phase after checksum verification: {precision}", flush=True)
        else:
            if model is None:
                model = swm.wm.utils.load_pretrained(f"{model_name}/weights_epoch_1.pt")
                model.eval().requires_grad_(False)
                check(all(not module.training for module in model.modules()), "Model is in training mode")
                check(state_dict_sha256(model.state_dict()) == final_audit["audit"]["export"]["state_sha256"],
                      "The loaded evaluation model differs from the audited export")
                initial_hash = model_hash(model)
                model.cuda()
            phase_runner(model, loader, bundle, folder, label, precision, phase_key, initial_hash,
                           output / "progress.json", index, len(PRECISIONS))
        summaries[precision], rows[precision] = phase_details(folder, label, precision, fingerprint, episode_counts)
    if model is not None:
        model.cpu()
        del model
        torch.cuda.empty_cache()
    result = {
        "passed": True, "protocol": PROTOCOL, "plan_fingerprint": fingerprint,
        "started_utc": started, "finished_utc": now(), "invocation_elapsed_seconds": time.perf_counter() - tick,
        "summed_phase_elapsed_seconds": sum(item["elapsed_seconds"] for item in summaries.values()),
        "resumed_completed_phases": reused, "requested_precisions": list(PRECISIONS),
        "primary_precision": "bf16_mixed", "test_episodes": len(episode_counts), "test_windows": len(indices),
        "transitions_per_model_precision": 3 * len(indices),
        "train_test_episode_disjoint": True, "train_only_normalization_verified": True,
        "models": {label: summaries}, "reference": evidence,
        "comparisons": compare_phases(summaries, rows, evidence["scores"], reference_rows, label),
        "interpretation": "Next-embedding MSE against each model's own encoded next image, averaged over three predictions per window.",
    }
    return result


def publish_results(experiment, result, report_writer=publish_report):
    output = experiment / "evaluation"
    report_writer(experiment, result)
    write_json(output / "results.json", result)
    write_json(output / "progress.json", {"complete": True, "passed": True, "finished_utc": now(),
                                         "requested_precisions": list(PRECISIONS)})
    print(json.dumps({"passed": True, "results": str(output / "results.json"), "report": str(experiment / "report.txt"),
                      "scores": {p: c["original_validation_arithmetic"] for p, c in result["comparisons"].items()}}), flush=True)



def evaluate(experiment, reference, model_name):
    result = score_candidate(experiment, model_name, LABEL, OBJECTIVE, verified_reference(reference))
    publish_results(experiment, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE_DIR)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--verify-reference-only", action="store_true", help="Verify prior files and scores entirely on CPU")
    args = parser.parse_args()
    reference = args.reference_dir.resolve(strict=True)
    if args.verify_reference_only:
        _, _, evidence, _, _ = verified_reference(reference)
        print(json.dumps(evidence, indent=2), flush=True)
        return
    if args.experiment_dir is None:
        parser.error("--experiment-dir is required for evaluation")
    evaluate(args.experiment_dir.resolve(strict=True), reference, args.model)


if __name__ == "__main__":
    main()
