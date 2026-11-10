"""Evaluate a matched objective-weight variant and its held-out encoder variation."""
import argparse
from pathlib import Path

import yaml

from audit_combined_experiment import check, read_json
from combined_sweep_config import COMBINED, MSE_ONLY, REFERENCE, VARIANTS
from evaluate_combined_experiment import (
    PRECISIONS, add_candidate_reference, publish_results, score_candidate, verified_reference,
)
from evaluate_mse_experiment import read_variation, score_with_variation
from evaluate_episode_experiment import file_hash


def action_reference_evidence(reference, plan):
    label = "action_predictor"
    audit_path = reference / "runs" / label / "final-audit.json"
    audited = read_json(audit_path)
    audit = audited["audit"]
    check(audited.get("passed") is True and audit["objective"] == "action_prediction",
          "Missing the audited action-only reference")
    export = audit["export"]
    checkpoint = plan["checkpoints"][label]["files"]
    weights = Path(checkpoint["weights"]["path"])
    config_path = Path(checkpoint["training_config"]["path"])
    check(export["matches_final_checkpoint_exactly"] and export["strict_load_passed"]
          and weights.resolve() == Path(export["path"]).resolve()
          and file_hash(weights) == checkpoint["weights"]["sha256"],
          "The action-only reference scores do not match its audited export")
    check(file_hash(config_path) == checkpoint["training_config"]["sha256"],
          "The scored action-only reference training configuration has changed")
    loss = yaml.safe_load(config_path.read_text())["loss"]
    check(loss["type"] == "action_prediction" and loss["sigreg"] is None
          and loss["action_prediction"]["weight"] == 0.09, "Expected the action-only 0.09 reference")
    return {"label": label, "sigreg_weight": 0.0, "action_embedding_mse_weight": 0.09,
            "audited_weights_sha256": checkpoint["weights"]["sha256"], "audit_sha256": file_hash(audit_path)}


def mse_reference_variation(experiment):
    folder = experiment / "evaluation"
    result = read_json(folder / "results.json")
    records = {}
    for precision in PRECISIONS:
        record = read_variation(folder / "phases" / f"mse_only--{precision}", result["models"]["mse_only"][precision])
        check(record == result["test_latent_variation"][precision]
              and record["encoded_frame_occurrences"] == 4 * record["windows"],
              "The MSE-only encoder spread does not match the verified held-out evaluation")
        records[precision] = record
    return records


def sweep_references(experiment, variant, reference, combined, mse):
    data = verified_reference(reference)
    paths = {"combined": combined, "mse_only": mse}
    if variant.label == "action_w50":
        paths = {"mse_only": mse}
    if variant.tag in ("s03", "a100"):
        paths["combined_s06"] = VARIANTS["combined_s06"].experiment
    if variant.tag == "a100":
        paths["combined_s03"] = VARIANTS["combined_s03"].experiment
    plan_path = experiment / "evaluation" / "evaluation-plan.json"
    if plan_path.exists():
        saved = read_json(plan_path)["plan"]["reference"].get("additional_references", {})
        paths = {label: Path(evidence["experiment_dir"]) for label, evidence in saved.items()}
        check(all((path / "evaluation" / "results.json").is_file() for path in paths.values()),
              "A reference used by the saved evaluation plan is missing")
    if variant.tag == "a100":
        check("combined" in paths and (paths["combined"] / "evaluation" / "results.json").is_file(),
              "The action-weight comparison requires the completed original combined reference")
        original = paths["combined"]
        original_audit = read_json(original / "runs" / "combined" / "final-audit.json")
        check(original_audit.get("passed") is True, "Original combined training audit did not pass")
        terms = original_audit["audit"]["combined_objective"]
        check(terms["sigreg_weight"] == 0.09 and terms["action_prediction_weight"] == 0.09,
              "Expected original combined objective weights 0.09 SIGReg and 0.09 action")
        weights_hash = original_audit["audit"]["export"]["weights_file_sha256"]
        scored = read_json(original / "evaluation" / "evaluation-plan.json")["plan"]["checkpoints"]["combined"]
        check(scored["files"]["weights"]["sha256"] == weights_hash,
              "The reference scores do not match the audited original combined weights")
        data[2]["primary_reference"] = {
            "label": "combined", "sigreg_weight": 0.09, "action_embedding_mse_weight": 0.09,
            "audited_weights_sha256": weights_hash,
        }
    for label, path in paths.items():
        if (path / "evaluation" / "results.json").is_file():
            add_candidate_reference(data, path, label)
    if variant.label == "action_w50":
        check("mse_only" in data[2].get("additional_references", {}),
              "The action-only weight-50 evaluation requires the completed MSE-only control")
        data[2]["primary_reference"] = action_reference_evidence(reference, data[1])
        data[2]["mse_only_test_latent_variation"] = mse_reference_variation(paths["mse_only"])
    return data


def publish_sweep_report(experiment, result):
    label = result["variant"]
    weights = result["loss_weights"]
    names = {"sigreg": "SIGReg", "action_predictor": "Action predictor", "combined": "Combined 0.09",
             "mse_only": "MSE only", "combined_s06": "Combined 0.06", "combined_s03": "Combined 0.03"}
    if label == "combined_a100":
        # Put the matched comparison first and distinguish both loss weights.
        names = {
            "combined": "Combined 0.09 / 0.09", label: "Combined 0.09 / 1.00",
            "combined_s06": "Combined 0.06 / 0.09", "combined_s03": "Combined 0.03 / 0.09",
            "mse_only": names["mse_only"], "sigreg": names["sigreg"],
            "action_predictor": names["action_predictor"],
        }
    title = f"Two-Room combined objective with SIGReg weight {weights['sigreg']} and action weight {weights['action_embedding_mse']}"
    if label == "action_w50":
        title = "Two-Room action-only objective with next-embedding MSE weight 1 and action MSE weight 50"
        names = {label: "Action weight 50", "action_predictor": "Action weight 0.09",
                 "mse_only": "MSE only", "sigreg": "SIGReg"}
    lines = [
        title,
        f"Scored {result['test_windows']:,} windows from {result['test_episodes']:,} held-out episodes.",
        "Each score averages three next-embedding predictions per window.", "",
    ]
    if label == "combined_a100":
        lines.extend(["Primary comparison uses SIGReg weight 0.09 and changes action MSE weight from 0.09 to 1.00.",
                      "Combined model labels list SIGReg weight / action weight.", ""])
    elif label == "action_w50":
        lines.extend(["The matched action-only reference used action MSE weight 0.09. "
                      "Both models omitted SIGReg and used next-embedding MSE weight 1.",
                      "Next-embedding MSE measures predictions against each model’s own image embeddings.", ""])
    for precision in PRECISIONS:
        comparison = result["comparisons"][precision]
        scores = comparison["original_validation_arithmetic"]
        lines.extend([precision, "Model                 Next-embedding MSE"])
        for key, title in names.items():
            if f"{key}_mse" in scores:
                lines.append(f"{title:<22}{scores[f'{key}_mse']:.12g}")
        other_labels = list(comparison["paired_by_reference"])
        primary = result.get("primary_reference")
        if primary:
            other_labels = [primary] + [other for other in other_labels if other != primary]
        for other in other_labels:
            stats = comparison["paired_by_reference"][other]
            reduction = scores[f"{label}_relative_reduction_vs_{other}_percent"]
            metrics = stats["native_window_metric"]
            if reduction is None:
                lines.append(f"The change from {names[other]} was {scores[f'{label}_minus_{other}_mse']:+.12g} MSE. "
                             "A zero reference MSE has no percentage reduction.")
            else:
                interval = metrics["paired_episode_bootstrap_reduction_percent_95pct_ci"]
                direction = "lower" if reduction >= 0 else "higher"
                lines.append(f"{names[label]} scored {abs(reduction):.6g}% {direction} than {names[other]}.")
                if interval is not None:
                    lines.append(f"Paired 95% interval for the percentage reduction [{interval[0]:.6g}, {interval[1]:.6g}].")
        latent = result["test_latent_variation"][precision]
        lines.append(f"Held-out encoder cross-example variance {latent['latent_cross_example_var']:.12g}, "
                     f"RMS standard deviation {latent['latent_cross_example_rms_std']:.12g}, "
                     f"maximum channel range across all frames {latent['latent_global_max_range']:.12g}.")
        lines.append(f"All observed frame vectors exactly equal: {latent['exactly_same_vector_for_all_frames']}.")
        if label == "action_w50":
            control = result["reference_test_latent_variation"]["mse_only"][precision]
            lines.append(f"MSE-only held-out encoder cross-example variance {control['latent_cross_example_var']:.12g}, "
                         f"RMS standard deviation {control['latent_cross_example_rms_std']:.12g}, "
                         f"maximum channel range across all frames {control['latent_global_max_range']:.12g}.")
        lines.append("")
    output = experiment / "report.txt"
    temporary = output.with_suffix(".txt.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(output)


def evaluate_sweep(experiment, variant, reference, combined, mse):
    references = sweep_references(experiment, variant, reference, combined, mse)
    audit = read_json(experiment / "runs" / variant.label / "final-audit.json")
    check(audit.get("passed") is True and audit["audit"].get("variant") == variant.label,
          "Missing passing audit for the requested strength")
    action_weight = variant.action_weight
    if variant.label == "action_w50":
        terms = audit["audit"]["action_only_objective"]
        check(terms["next_embedding_weight"] == 1.0 and terms["sigreg_enabled"] is False
              and variant.sigreg_weight == 0.0 and action_weight == 50.0,
              "Expected next-embedding MSE weight 1, action MSE weight 50 and SIGReg disabled")
    else:
        terms = audit["audit"]["combined_objective"]
    check(terms["sigreg_weight"] == variant.sigreg_weight and terms["action_prediction_weight"] == action_weight,
          "Audited training coefficients differ from the requested variant")
    if variant.tag == "a100":
        check(variant.sigreg_weight == 0.09 and action_weight == 1.0, "Incorrect action-weight-1.0 variant")
    result = score_candidate(experiment, variant.model, variant.label, variant.objective, references,
                             phase_runner=score_with_variation, entrypoint=Path(__file__))
    # A collapsed MSE-only reference can have exactly zero test error. Absolute
    # differences remain defined, while a percentage reduction/interval does not.
    for comparison in result["comparisons"].values():
        for pair in comparison["paired_by_reference"].values():
            for metric in pair.values():
                if metric["reference_mse"] == 0:
                    metric["paired_episode_bootstrap_reduction_percent_95pct_ci"] = None
    result["variant"] = variant.label
    result["loss_weights"] = {"sigreg": variant.sigreg_weight, "action_embedding_mse": action_weight}
    if variant.tag == "a100":
        result["primary_reference"] = "combined"
    elif variant.label == "action_w50":
        result["primary_reference"] = "action_predictor"
        result["loss_weights"]["next_embedding_mse"] = terms["next_embedding_weight"]
        result["reference_test_latent_variation"] = {"mse_only": references[2]["mse_only_test_latent_variation"]}
        result["training_latent_variation"] = audit["audit"]["training_latent_variation"]
    result["test_latent_variation"] = {
        precision: read_variation(experiment / "evaluation" / "phases" / f"{variant.label}--{precision}",
                                  result["models"][variant.label][precision])
        for precision in PRECISIONS
    }
    result["variation_protocol"] = (
        "Per-time-position and per-channel sample variance over all held-out windows uses correction 1. "
        "Global channel ranges include all four image encodings per window. Overlapping windows count "
        "each observed frame occurrence, using the model's ordinary held-out prediction passes."
    )
    publish_results(experiment, result, publish_sweep_report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE)
    parser.add_argument("--combined-reference-dir", type=Path, default=COMBINED)
    parser.add_argument("--mse-reference-dir", type=Path, default=MSE_ONLY)
    args = parser.parse_args()
    variant = VARIANTS[args.variant]
    evaluate_sweep((args.experiment_dir or variant.experiment).resolve(strict=True), variant,
                   args.reference_dir.resolve(strict=True), args.combined_reference_dir.resolve(),
                   args.mse_reference_dir.resolve())


if __name__ == "__main__":
    main()

