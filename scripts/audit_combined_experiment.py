"""Audit a completed combined-objective training run against experiment 03."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
import traceback

from evaluate_episode_experiment import now, write_json

REFERENCE_DIR = Path("/home/adam/lewm/experiments/episode50-30-20261106-03")
MODEL_NAME = "tworoom-combined-episode50-20261107-01"
OBJECTIVE = "combined"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def check_matching_manifests(new, previous):
    # Each bundle's loader has independently verified its archive and hashes.
    fields = ("dataset", "counts", "normalization", "training_order", "array_hashes")
    for field in fields:
        check(new[field] == previous[field], f"Experiment 03 differs in {field}")
    check(new["training_order"]["batch_size"] == 32, "Expected training batch size 32")
    check(new["training_order"]["steps_per_epoch"] == 11427, "Expected 11,427 optimizer updates")


VARIABILITY_FIELDS = (
    "latent_cross_example_var", "latent_cross_example_rms_std", "latent_cross_example_max_range",
    "latent_global_max_range",
)


def verify_latent_variation_rows(rows, batch_size, final_batch_size):
    """Check how much encoded observations vary across examples at each optimizer step."""
    check(bool(rows), "Missing training metrics")
    values = {name: [] for name in VARIABILITY_FIELDS}
    for index, row in enumerate(rows):
        count = float(row["fit/latent_cross_example_count"])
        expected = final_batch_size if index + 1 == len(rows) else batch_size
        check(count == expected, f"Wrong variability sample count at step {index + 1}")
        for name in VARIABILITY_FIELDS:
            raw = row["fit/" + name]
            if count == 1 and name != "latent_global_max_range":
                check(raw in ("", None), f"Singleton cross-example statistic {name} should be undefined")
                continue
            value = float(raw)
            check(math.isfinite(value) and value >= 0, f"Invalid {name} at step {index + 1}")
            values[name].append(value)
        if count == 1:
            continue
        variance = values["latent_cross_example_var"][-1]
        rms = values["latent_cross_example_rms_std"][-1]
        check(math.isclose(variance, rms * rms, rel_tol=2e-6, abs_tol=1e-40), "Variance/RMS mismatch")
        check(values["latent_global_max_range"][-1] >= values["latent_cross_example_max_range"][-1],
              "Cross-example range exceeds the global channel range")
    summary = {}
    for name, observations in values.items():
        if not observations:
            summary[name] = {"defined_steps": 0}
            continue
        summary[name] = {
            "defined_steps": len(observations), "first": observations[0], "last": observations[-1],
            "minimum": min(observations), "maximum": max(observations),
            "last_100_mean": float(sum(observations[-100:]) / len(observations[-100:])),
        }
    return {
        "steps_checked": len(rows),
        "training_batch_variability": summary,
        "exactly_constant_all_frames_steps": sum(v == 0 for v in values["latent_global_max_range"]),
        "exactly_constant_at_each_time_position_steps": sum(v == 0 for v in values["latent_cross_example_max_range"]),
    }


def verify_weighted_rows(rows, sigreg_weight=0.09, action_weight=0.09):
    """Recreate the training arithmetic from each row's logged, unweighted losses."""
    import torch

    check(bool(rows), "Missing weighted-objective rows")
    fields = ["fit/pred_loss", "fit/action_loss", "fit/loss"]
    if sigreg_weight is None:
        check(all("fit/sigreg_loss" not in row for row in rows), "Action-only training unexpectedly logged SIGReg")
    else:
        fields.append("fit/sigreg_loss")
    values = {}
    for field in fields:
        values[field] = torch.tensor([float(row[field]) for row in rows], dtype=torch.float32)
        check(torch.isfinite(values[field]).all().item(), f"Nonfinite {field}")
    expected = values["fit/pred_loss"]
    if sigreg_weight is not None:
        # SIGReg returns BF16. Weighting it rounds once before the FP32 addition.
        expected = expected + (sigreg_weight * values["fit/sigreg_loss"].bfloat16()).float()
    expected = expected + action_weight * values["fit/action_loss"]
    torch.testing.assert_close(values["fit/loss"], expected, rtol=2e-6, atol=2e-8)
    return {
        "steps_checked": len(rows),
        "max_absolute_total_loss_error": (values["fit/loss"] - expected).abs().max().item(),
        "next_embedding_weight": 1.0,
        "sigreg_enabled": sigreg_weight is not None,
        "sigreg_weight": 0.0 if sigreg_weight is None else sigreg_weight,
        "action_prediction_weight": action_weight,
        "logged_components": {
            field: {"logged_count": len(rows), "first": series[0].item(), "last": series[-1].item(),
                    "minimum": series.min().item(), "maximum": series.max().item()}
            for field, series in values.items()
        },
    }


def audit_combined(experiment, reference, model_name, *, label="combined", sigreg_weight=0.09, action_weight=0.09,
                   track_cross_example_latent_spread=False, objective=OBJECTIVE):
    from audit_episode_experiment import audit as audit_training, file_sha256
    from episode_data import load_episode_experiment
    from omegaconf import OmegaConf

    manifest_path = experiment / "manifest.json"
    run = experiment / "runs" / label
    manifest = load_episode_experiment(manifest_path)["manifest"]
    previous_manifest = load_episode_experiment(reference / "manifest.json")["manifest"]
    check_matching_manifests(manifest, previous_manifest)
    audit = audit_training(SimpleNamespace(run_dir=run, manifest=manifest_path,
                                           expected_objective=objective, paired_run_dir=None,
                                           expected_sigreg_weight=sigreg_weight, expected_action_weight=action_weight))
    check(audit["checkpoint"]["recovery_protocol"] is not None, "Expected native resumable checkpoint")
    check(audit["data_provenance"]["completed_steps"] == 11427, "Combined training is incomplete")
    check(audit["data_provenance"]["windows"] == 365657, "Incomplete training window coverage")
    check(audit["data_provenance"]["last_batch_size"] == 25, "Wrong final batch size")
    check(audit["process_exit_code"] in (0, None), "Training process failed")

    config_path = Path(audit["export"]["path"]).parent / "config.yaml"
    cfg = OmegaConf.load(config_path)
    check(cfg.output_model_name == model_name, "Unexpected combined model export")
    check(objective in ("combined", "action_prediction") and cfg.loss.type == objective
          and cfg.loss.action_prediction.weight == action_weight, "Unexpected action loss objective or coefficient")
    with_sigreg = objective == "combined"
    if with_sigreg:
        check(cfg.loss.sigreg.weight == sigreg_weight, "Unexpected SIGReg coefficient")
    else:
        check(sigreg_weight == 0.0 and cfg.loss.sigreg is None, "Action-only training must omit SIGReg")

    check(cfg.get("track_cross_example_latent_spread", False) == track_cross_example_latent_spread,
          "Unexpected cross-example latent diagnostic setting")
    paired = {}
    for prior_label in ("sigreg", "action_predictor"):
        prior_run = reference / "runs" / prior_label
        previous_report = read_json(prior_run / "final-audit.json")
        check(previous_report.get("passed") is True, f"Reference {prior_label} training audit did not pass")
        previous = previous_report["audit"]
        check(previous["manifest_sha256"] == previous_manifest["manifest_sha256"], "Reference audit manifest differs")
        check(previous["data_provenance"]["order_sha256"] == audit["data_provenance"]["order_sha256"],
              f"Actual consumed training order differs from {prior_label}")
        check(previous["initialization"]["shared_state_sha256"] == audit["initialization"]["shared_state_sha256"],
              f"Shared initialization differs from {prior_label}")
        check(previous["data_provenance"]["completed_steps"] == audit["data_provenance"]["completed_steps"],
              f"Training step count differs from {prior_label}")
        paired[prior_label] = {
            "reference_run": str(prior_run),
            "audit_sha256": file_sha256(prior_run / "final-audit.json"),
            "actual_consumed_order_identical": True,
            "shared_initial_weights_identical": True,
        }
        if prior_label == "action_predictor":
            check(previous["initialization"]["state_sha256"] == audit["initialization"]["state_sha256"],
                  "Full initialization differs from the action-predictor reference")
            old_cfg = OmegaConf.load(Path(previous["export"]["path"]).parent / "config.yaml")
            for key in ("model", "optimizer", "seed", "embed_dim", "img_size", "history_size", "num_preds"):
                check(cfg[key] == old_cfg[key], f"Combined/action model config differs in {key}")
            check(cfg.trainer.gradient_clip_val == old_cfg.trainer.gradient_clip_val, "Gradient clipping changed")
            paired[prior_label]["full_initial_weights_identical"] = True
            paired[prior_label]["same_model_and_optimizer_hyperparameters"] = True
        elif with_sigreg:
            old_cfg = OmegaConf.load(Path(previous["export"]["path"]).parent / "config.yaml")
            candidate_options = {key: value for key, value in cfg.loss.sigreg.items() if key != "weight"}
            old_options = {key: value for key, value in old_cfg.loss.sigreg.items() if key != "weight"}
            check(candidate_options == old_options, "SIGReg options differ from the reference")

    with (run / "metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    check(len(rows) == 11427, "Expected one logged objective per optimizer step")
    terms = verify_weighted_rows(rows, sigreg_weight if with_sigreg else None, action_weight)
    audit["training_metrics"] = terms.pop("logged_components")
    audit["combined_objective" if with_sigreg else "action_only_objective"] = terms
    if track_cross_example_latent_spread:
        data = audit["data_provenance"]
        audit["training_latent_variation"] = verify_latent_variation_rows(rows, data["batch_size"], data["last_batch_size"])
    audit["paired_run_checks"] = paired
    audit["export"]["weights_file_sha256"] = file_sha256(audit["export"]["path"])
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", default=REFERENCE_DIR, type=Path)
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()
    experiment = args.experiment_dir.resolve(strict=True)
    output = experiment / "runs" / "combined" / "final-audit.json"
    result = {"passed": False, "started_utc": now()}
    try:
        import torch
        torch.set_num_threads(4)
        result["audit"] = audit_combined(experiment, args.reference_dir.resolve(strict=True), args.model)
        result["passed"] = True
    except Exception as error:
        result.update(error=str(error), traceback=traceback.format_exc())
    result["finished_utc"] = now()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(json.dumps({"passed": result["passed"], "report": str(output), "error": result.get("error")}), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

