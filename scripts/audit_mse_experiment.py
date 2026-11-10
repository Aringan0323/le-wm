"""Audit an MSE-only one-pass run, including its encoder variability trace."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
import traceback

from audit_combined_experiment import REFERENCE_DIR, check, check_matching_manifests, read_json, verify_latent_variation_rows
from evaluate_episode_experiment import now, write_json

LABEL = "mse_only"
MODEL_NAME = "tworoom-mse-episode50-20261107-01"
def check_training_metrics(rows, batch_size, final_batch_size):
    for index, row in enumerate(rows):
        pred = float(row["fit/pred_loss"])
        check(math.isfinite(pred) and pred >= 0 and pred == float(row["fit/loss"]),
              f"Total does not equal next-embedding MSE at step {index + 1}")
        check(not any(row.get(field) for field in ("fit/sigreg_loss", "fit/action_loss")),
              "An auxiliary loss was logged for the MSE-only run")
    result = verify_latent_variation_rows(rows, batch_size, final_batch_size)
    result["total_equals_next_embedding_loss_every_step"] = True
    return result


def audit_mse(experiment, reference, model_name):
    from audit_episode_experiment import audit as audit_training, file_sha256
    from episode_data import load_episode_experiment
    from omegaconf import OmegaConf

    run = experiment / "runs" / LABEL
    manifest_path = experiment / "manifest.json"
    manifest = load_episode_experiment(manifest_path)["manifest"]
    reference_manifest = load_episode_experiment(reference / "manifest.json")["manifest"]
    check_matching_manifests(manifest, reference_manifest)
    audit = audit_training(SimpleNamespace(run_dir=run, manifest=manifest_path,
                                           expected_objective=LABEL, paired_run_dir=None))
    data = audit["data_provenance"]
    check(audit["checkpoint"]["recovery_protocol"] is not None, "Missing native checkpoint provenance")
    check((data["completed_steps"], data["windows"], data["last_batch_size"]) == (11427, 365657, 25),
          "Incomplete MSE-only training pass")
    cfg = OmegaConf.load(Path(audit["export"]["path"]).parent / "config.yaml")
    check(cfg.output_model_name == model_name and cfg.loss.type == LABEL, "Unexpected model/objective")
    check(cfg.loss.get("sigreg") is None and cfg.loss.get("action_prediction") is None
          and cfg.model.get("action_predictor") is None, "MSE-only config enables an auxiliary objective or head")

    paired = {}
    for prior_label in ("sigreg", "action_predictor"):
        other_run = reference / "runs" / prior_label
        report = read_json(other_run / "final-audit.json")
        check(report.get("passed") is True, f"Missing passing reference audit for {prior_label}")
        previous = report["audit"]
        check(previous["manifest_sha256"] == reference_manifest["manifest_sha256"], "Reference manifest mismatch")
        check(previous["data_provenance"]["order_sha256"] == data["order_sha256"], "Training order differs")
        check(previous["initialization"]["shared_state_sha256"] == audit["initialization"]["shared_state_sha256"],
              f"Shared fresh weights differ from {prior_label}")
        paired[prior_label] = {
            "audit_sha256": file_sha256(other_run / "final-audit.json"),
            "shared_initial_weights_identical": True, "actual_consumed_order_identical": True,
        }
        if prior_label == "sigreg":
            check(previous["initialization"]["state_sha256"] == audit["initialization"]["state_sha256"],
                  "Full fresh initialization differs from SIGReg")
            old_cfg = OmegaConf.load(Path(previous["export"]["path"]).parent / "config.yaml")
            for key in ("model", "optimizer", "seed", "embed_dim", "img_size", "history_size", "num_preds"):
                check(cfg[key] == old_cfg[key], f"Model/optimization setting differs from SIGReg: {key}")
            check(cfg.trainer.gradient_clip_val == old_cfg.trainer.gradient_clip_val, "Gradient clipping differs")
            paired[prior_label]["full_initial_weights_identical"] = True
            paired[prior_label]["same_model_and_optimizer_hyperparameters"] = True
    with (run / "metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    check(len(rows) == data["completed_steps"], "Incomplete MSE-only metric trace")
    audit["mse_only_objective"] = check_training_metrics(rows, data["batch_size"], data["last_batch_size"])
    audit["paired_run_checks"] = paired
    audit["export"]["weights_file_sha256"] = file_sha256(audit["export"]["path"])
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE_DIR)
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()
    experiment = args.experiment_dir.resolve(strict=True)
    output = experiment / "runs" / LABEL / "final-audit.json"
    result = {"passed": False, "started_utc": now()}
    try:
        import torch
        torch.set_num_threads(4)
        result["audit"] = audit_mse(experiment, args.reference_dir.resolve(strict=True), args.model)
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

