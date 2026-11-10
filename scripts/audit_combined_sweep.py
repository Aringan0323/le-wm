"""Audit one combined loss variant against experiment 03 and the original combined run."""
import argparse
import json
from pathlib import Path
import traceback

from audit_combined_experiment import audit_combined, check, check_matching_manifests, read_json
from combined_sweep_config import ACTION_WEIGHT, COMBINED, REFERENCE, VARIANTS
from evaluate_episode_experiment import file_hash, json_hash, now, write_json


def configuration_evidence(config):
    """Preserve model, optimizer, dataset, and loader settings for a direct comparison."""
    # Importing the same training package registers the saved config's "eval" resolver.
    import stable_pretraining
    from omegaconf import OmegaConf

    resolved = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    identity = {key: value for key, value in resolved.items() if key not in ("resume_from", "control_file")}
    full_hash = json_hash(identity)
    excluded = ("subdir", "output_model_name", "run_metadata_dir", "episode_manifest", "episode_manifest_sha256",
                "track_cross_example_latent_spread")
    matched = {key: value for key, value in identity.items() if key not in excluded}
    matched["trainer"] = dict(matched["trainer"])
    matched["trainer"].pop("default_root_dir", None)
    matched["wandb"] = dict(matched["wandb"])
    matched["wandb"]["config"] = {key: val for key, val in matched["wandb"]["config"].items()
                                  if key not in ("name", "id")}
    matched["loss"] = dict(matched["loss"])
    for term in ("sigreg", "action_prediction"):
        if matched["loss"].get(term) is not None:
            matched["loss"][term] = {key: val for key, val in matched["loss"][term].items() if key != "weight"}
    return full_hash, matched


def audit_sweep(variant, experiment, reference, parent_combined):
    from episode_data import load_episode_experiment
    from omegaconf import OmegaConf

    audit = audit_combined(experiment, reference, variant.model, label=variant.label,
                           sigreg_weight=variant.sigreg_weight, action_weight=variant.action_weight,
                           track_cross_example_latent_spread=variant.track_cross_example_latent_spread, objective=variant.objective)
    current_run = experiment / "runs" / variant.label
    with_sigreg = variant.objective == "combined"
    parent = parent_combined if with_sigreg else reference
    previous_label = "combined" if with_sigreg else "action_predictor"
    previous_run = parent / "runs" / previous_label
    previous_report = read_json(previous_run / "final-audit.json")
    check(previous_report.get("passed") is True, "The comparison training audit must pass")
    previous = previous_report["audit"]
    check(previous["objective"] == variant.objective, "Wrong reference objective")
    manifest = load_episode_experiment(experiment / "manifest.json")["manifest"]
    previous_manifest = load_episode_experiment(parent / "manifest.json")["manifest"]
    check_matching_manifests(manifest, previous_manifest)
    check(previous["manifest_sha256"] == previous_manifest["manifest_sha256"], "Combined reference manifest differs")

    current_cfg = OmegaConf.load(Path(audit["export"]["path"]).parent / "config.yaml")
    previous_cfg = OmegaConf.load(Path(previous["export"]["path"]).parent / "config.yaml")
    check(previous_cfg.loss.type == variant.objective and previous_cfg.loss.action_prediction.weight == ACTION_WEIGHT,
          "Expected the original action coefficient of 0.09")
    if with_sigreg:
        check(previous_cfg.loss.sigreg.weight == 0.09, "Expected original reference SIGReg weight 0.09")
    else:
        check(previous_cfg.loss.sigreg is None, "Original action-only reference must omit SIGReg")
    current_hash, current_matched = configuration_evidence(current_cfg)
    previous_hash, previous_matched = configuration_evidence(previous_cfg)
    for config_hash, run, report in ((current_hash, current_run, audit), (previous_hash, previous_run, previous)):
        check(config_hash == read_json(run / "config-identity.json")["config_sha256"]
              == report["checkpoint"]["recovery_protocol"]["config_sha256"], "Config/checkpoint fingerprint differs")
    check(current_matched == previous_matched, "Model, optimizer, data, or trainer settings differ from the original matching run")

    previous_order = previous["data_provenance"]["order_sha256"]
    check(file_hash(previous_run / "training-indices.i64") == previous_order
          == audit["data_provenance"]["order_sha256"], "Consumed order differs from the 0.09 run")
    for key in ("state_sha256", "shared_state_sha256"):
        check(read_json(previous_run / "initialization.json")[key] == previous["initialization"][key]
              == audit["initialization"][key], f"Initialization differs from the 0.09 run: {key}")
    check(previous["data_provenance"]["completed_steps"] == audit["data_provenance"]["completed_steps"],
          "Training pass differs from the 0.09 run")
    audit["paired_run_checks"][previous_label] = {
        "reference_run": str(previous_run), "audit_sha256": file_hash(previous_run / "final-audit.json"),
        "full_initial_weights_identical": True, "shared_initial_weights_identical": True,
        "actual_consumed_order_identical": True, "candidate_config_sha256": current_hash,
        "reference_config_sha256": previous_hash, "matched_effective_settings_sha256": json_hash(current_matched),
        "matched_settings_exclusions": ["experiment-specific output/control/resume paths and run IDs",
                                       "manifest path/hash checked through verified split arrays",
                                       "independently checked loss weights and latent metric logging"],
        "sigreg_weight": variant.sigreg_weight, "reference_sigreg_weight": 0.09 if with_sigreg else 0.0,
        "action_weight": variant.action_weight, "reference_action_weight": ACTION_WEIGHT,
        "track_cross_example_latent_spread": variant.track_cross_example_latent_spread,
    }
    audit["variant"] = variant.label
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE)
    parser.add_argument("--combined-reference-dir", type=Path, default=COMBINED)
    args = parser.parse_args()
    variant = VARIANTS[args.variant]
    experiment = (args.experiment_dir or variant.experiment).resolve(strict=True)
    output = experiment / "runs" / variant.label / "final-audit.json"
    result = {"passed": False, "started_utc": now()}
    try:
        import torch
        torch.set_num_threads(4)
        result["audit"] = audit_sweep(variant, experiment, args.reference_dir.resolve(strict=True),
                                      args.combined_reference_dir.resolve(strict=True))
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

