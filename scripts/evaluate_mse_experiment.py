"""Evaluate the MSE-only model on the matched test windows and measure encoder variation."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from audit_combined_experiment import REFERENCE_DIR, check, read_json
from audit_mse_experiment import LABEL, MODEL_NAME
from evaluate_combined_experiment import (
    PRECISIONS, add_candidate_reference, publish_results, score_candidate, verified_reference,
)
from evaluate_episode_experiment import evaluate_phase, json_hash, write_json

COMBINED_DIR = Path("/home/adam/lewm/experiments/episode50-combined-20261107-01")


class EncoderVariation:
    """Stream per-time-position, per-channel moments and ranges over held-out windows."""

    def __init__(self):
        self.count = 0
        self.mean = np.zeros((4, 192), dtype=np.float64)
        self.m2 = np.zeros_like(self.mean)
        self.low = np.full_like(self.mean, np.inf)
        self.high = np.full_like(self.mean, -np.inf)

    def add(self, frames):
        frames = np.asarray(frames, dtype=np.float64)
        check(frames.ndim == 3 and frames.shape[1:] == self.mean.shape and len(frames) > 0,
              "Unexpected encoder geometry for held-out latent variation")
        check(np.all(np.isfinite(frames)), "Nonfinite held-out encoder vector")
        batch_count = len(frames)
        batch_mean = frames.mean(axis=0)
        batch_m2 = np.square(frames - batch_mean).sum(axis=0)
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.m2 += batch_m2 + delta * delta * (self.count * batch_count / total)
        self.mean += delta * (batch_count / total)
        self.low = np.minimum(self.low, frames.min(axis=0))
        self.high = np.maximum(self.high, frames.max(axis=0))
        self.count = total

    def metrics(self):
        check(self.count > 1, "Cross-example variance needs at least two windows")
        variance = float((self.m2 / (self.count - 1)).mean())
        rms_std = math.sqrt(variance)
        latent_rms = math.sqrt(float((self.m2 / self.count + self.mean ** 2).mean()))
        cross_range = float((self.high - self.low).max())
        global_range = float((self.high.max(axis=0) - self.low.min(axis=0)).max())
        return {
            "windows": self.count, "encoded_frame_occurrences": self.count * 4,
            "latent_cross_example_var": variance, "latent_cross_example_rms_std": rms_std,
            "latent_cross_example_max_range": cross_range, "latent_global_max_range": global_range,
            "latent_rms": latent_rms, "rms_std_over_latent_rms": rms_std / latent_rms if latent_rms else None,
            "exactly_same_vector_for_all_frames": global_range == 0,
            "exactly_same_vector_across_examples_at_each_time_position": cross_range == 0,
        }


def score_with_variation(model, loader, bundle, folder, label, precision, fingerprint,
                         initial_state_hash, progress_path, phase_number, phase_count):
    expected_windows = len(bundle["arrays"]["test_clip_indices"])
    moments = EncoderVariation()

    def observe(_projector, _inputs, latent):
        # JEPA's image projector returns its actual encoder output as (batch * 4, 192).
        frames = latent.detach().float().cpu().numpy().reshape(-1, 4, 192)
        moments.add(frames)
        check(moments.count <= expected_windows, "Duplicate encoder observations")
        if moments.count == expected_windows:
            record = {
                "phase_fingerprint": fingerprint, "precision": precision,
                "initial_model_state_sha256": initial_state_hash, **moments.metrics(),
            }
            # The original evaluator commits summary.json after this last forward call.
            # This ordering makes completing and then resuming a phase safe.
            write_json(folder / "latent-variation.json", {**record, "record_sha256": json_hash(record)})

    with model.projector.register_forward_hook(observe):
        summary = evaluate_phase(model, loader, bundle, folder, label, precision, fingerprint,
                                 initial_state_hash, progress_path, phase_number, phase_count)
    check(moments.count == expected_windows, "Incomplete held-out encoder diagnostic")
    return summary


def read_variation(folder, summary):
    record = read_json(folder / "latent-variation.json")
    content = {key: value for key, value in record.items() if key != "record_sha256"}
    check(json_hash(content) == record["record_sha256"], "Encoder diagnostic changed after evaluation")
    check(record["phase_fingerprint"] == summary["phase_fingerprint"]
          and record["initial_model_state_sha256"] == summary["initial_model_state_sha256"]
          and record["precision"] == summary["precision"] and record["windows"] == summary["windows"],
          "Encoder diagnostic is not paired with this model and test phase")
    return record


def load_references(reference, combined, experiment):
    data = verified_reference(reference)
    saved_plan = experiment / "evaluation" / "evaluation-plan.json"
    linked = None
    if saved_plan.exists():
        linked = read_json(saved_plan)["plan"]["reference"].get("additional_references", {}).get("combined")
        combined = Path(linked["experiment_dir"]) if linked else None
    if combined is None or not (combined / "evaluation" / "results.json").exists():
        if linked:
            raise FileNotFoundError("The previously included combined reference is missing")
        return data
    return add_candidate_reference(data, combined, "combined")


def publish_mse_report(experiment, result):
    lines = [
        "Two-Room MSE-only held-out evaluation",
        f"Scored {result['test_windows']:,} windows from {result['test_episodes']:,} held-out episodes.",
        f"Each model and precision made {result['transitions_per_model_precision']:,} next-embedding predictions.",
        "",
    ]
    names = {"sigreg": "SIGReg", "action_predictor": "Action predictor", "combined": "Combined", LABEL: "MSE only"}
    for precision in PRECISIONS:
        comparison = result["comparisons"][precision]
        scores = comparison["original_validation_arithmetic"]
        lines.extend([precision, "Model                 Next-embedding MSE"])
        for name, title in names.items():
            if f"{name}_mse" in scores:
                lines.append(f"{title:<22}{scores[f'{name}_mse']:.12g}")
        for name in comparison["paired_by_reference"]:
            reduction = scores[f"{LABEL}_relative_reduction_vs_{name}_percent"]
            if reduction is not None:
                direction = "lower" if reduction >= 0 else "higher"
                ci = comparison["paired_by_reference"][name]["native_window_metric"]["paired_episode_bootstrap_reduction_percent_95pct_ci"]
                lines.append(f"MSE only was {abs(reduction):.6g}% {direction} than {names[name]}. "
                             f"Paired 95% interval for the percentage reduction [{ci[0]:.6g}, {ci[1]:.6g}].")
        variation = result["test_latent_variation"][precision]
        lines.append(f"Encoder cross-example variance {variation['latent_cross_example_var']:.12g}, "
                     f"RMS standard deviation {variation['latent_cross_example_rms_std']:.12g}, "
                     f"maximum channel range across all frames {variation['latent_global_max_range']:.12g}.")
        lines.append(f"Every held-out frame yielded exactly the same vector: "
                     f"{variation['exactly_same_vector_for_all_frames']}.")
        lines.append("")
    lines.append("Encoder variance uses all test windows at matching time positions. "
                 "Overlapping windows count the same frames more than once.")
    lines.append(f"Experiment 03 reference results: {result['reference']['results_file']}")
    output = experiment / "report.txt"
    temp = output.with_suffix(".txt.tmp")
    temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temp.replace(output)


def evaluate_mse(experiment, reference, combined, model_name):
    references = load_references(reference, combined, experiment)
    result = score_candidate(experiment, model_name, LABEL, LABEL, references,
                             phase_runner=score_with_variation, entrypoint=Path(__file__))
    result["test_latent_variation"] = {
        precision: read_variation(experiment / "evaluation" / "phases" / f"{LABEL}--{precision}",
                                  result["models"][LABEL][precision])
        for precision in PRECISIONS
    }
    audit = read_json(experiment / "runs" / LABEL / "final-audit.json")
    result["training_latent_variation"] = audit["audit"]["mse_only_objective"]
    result["variation_protocol"] = (
        "Cross-example sample variance uses correction 1 over all test windows at each time position and "
        "latent coordinate, then averages across time/coordinates. Global maximum range considers all "
        "four encoder frame occurrences per window across all windows. The measurements use eval mode "
        "and the same encoder outputs consumed by the next-embedding scoring pass."
    )
    publish_results(experiment, result, publish_mse_report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE_DIR)
    parser.add_argument("--combined-reference-dir", type=Path, default=COMBINED_DIR)
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()
    evaluate_mse(args.experiment_dir.resolve(strict=True), args.reference_dir.resolve(strict=True),
                 args.combined_reference_dir.resolve(), args.model)


if __name__ == "__main__":
    main()
