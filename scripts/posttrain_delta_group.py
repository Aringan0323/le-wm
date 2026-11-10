"""Audit and score all three independent models after their shared training pass."""
import argparse
from pathlib import Path
import sys

from audit_combined_experiment import check, read_json
from delta_group_layout import model_name
from delta_sweep_config import VARIANTS
from evaluate_combined_experiment import PRECISIONS, paired_statistics, phase_details
from evaluate_delta_experiment import verified_split
from evaluate_episode_experiment import PROTOCOL, file_hash, json_hash, now, write_json
from evaluate_mse_experiment import read_variation


def run_group_posttraining(pipeline, env):
    group = pipeline.experiment.resolve(strict=True)
    scripts = group / "source" / "scripts"
    members = {label: group / "members" / label for label in VARIANTS}
    for label, member in members.items():
        completion = read_json(member / "runs" / label / "runcompletion.json")
        plan = read_json(member / "plan.json")
        check(completion["passed"] and completion["global_step"] == plan["steps_per_model"]
              and completion["epochs"] == plan["epochs"] and completion["objective"] == VARIANTS[label].objective,
              f"Shared training did not finish for {label}")
    def run(script, phase, member, label, log, progress=None):
        command = [sys.executable, "-u", str(scripts / script), "--group-root", str(group),
                   "--experiment-dir", str(member), "--variant", label]
        pipeline.child(phase, command, log, env, progress)

    for label, member in members.items():
        run_dir = member / "runs" / label
        run("audit_delta_experiment.py", f"audit_{label}", member, label, run_dir / "audit-console.log")
        check(read_json(run_dir / "final-audit.json")["passed"], f"{label} training audit failed")
    for label, member in members.items():
        output = member / "evaluation"
        run("evaluate_delta_experiment.py", f"evaluate_{label}", member, label, output / "console.log",
            output / "progress.json")
        check(read_json(output / "results.json")["passed"], f"{label} evaluation failed")
    output = group / "evaluation"
    pipeline.child("compare_next_embedding", [sys.executable, "-u", str(scripts / "posttrain_delta_group.py"),
                   "--group-root", str(group)], output / "comparison.log", env)
    result = read_json(output / "results.json")
    check(result["passed"] and set(result["models"]) == set(members), "Missing group test results")
    return result


def member_evidence(group, variant, counts, manifest_hash, group_digest):
    label = variant.label
    member = group / "members" / label
    folder = member / "evaluation"
    result = read_json(folder / "results.json")
    audit_file = member / "runs" / label / "final-audit.json"
    audit = read_json(audit_file)
    check(audit["passed"] and audit["audit"]["group_commit"]["group_record_sha256"] == group_digest,
          f"Unverified group audit for {label}")
    plan_file = read_json(folder / "evaluation-plan.json")
    plan = plan_file["plan"]
    fingerprint = plan_file["fingerprint"]
    checkpoint = plan["checkpoints"][label]
    export = audit["audit"]["export"]
    check(result["passed"] and result["variant"] == label and result["protocol"] == PROTOCOL
          and result["group_root"] == str(group) and result["group_record_sha256"] == group_digest
          and result["train_test_episode_disjoint"] and result["train_only_normalization_verified"]
          and plan["manifest_canonical_sha256"] == manifest_hash
          and fingerprint == json_hash(plan) == result["plan_fingerprint"]
          and result["requested_precisions"] == list(PRECISIONS)
          and result["model_name"] == checkpoint["name"] == model_name(member, variant, group)
          and Path(checkpoint["files"]["weights"]["path"]).resolve() == Path(export["path"]).resolve()
          and checkpoint["files"]["weights"]["sha256"] == export["weights_file_sha256"] == file_hash(export["path"]),
          f"Test results do not match {label}'s audited export and data split")
    check(result["loss_weights"] == {"next_embedding_mse": 1.0, "ldad": variant.ldad_weight, "sigreg": variant.sigreg_weight},
          f"Wrong reported loss weights for {label}")
    has_ldad = variant.ldad_weight > 0
    check(result["ldad_decoder_enabled"] is has_ldad and audit["audit"]["delta_objective"]["ldad_decoder_enabled"] is has_ldad,
          f"Wrong architecture for {label}")
    summaries, episodes, spreads = {}, {}, {}
    for p in PRECISIONS:
        phase = folder / "phases" / f"{label}--{p}"
        summaries[p], episodes[p] = phase_details(phase, label, p, fingerprint, counts)
        spreads[p] = read_variation(phase, summaries[p])
        check(summaries[p] == result["models"][label][p] and spreads[p] == result["test_latent_variation"][p]
              and spreads[p]["encoded_frame_occurrences"] == 4 * sum(counts.values()),
              f"Published {label}/{p} differs from verified test artifacts")
    evidence = {"model_name": checkpoint["name"], "epochs": checkpoint["epoch"],
                "weights_sha256": export["weights_file_sha256"], "audit_sha256": file_hash(audit_file),
                "result_sha256": file_hash(folder / "results.json"), "evaluation_plan_sha256": fingerprint,
                "loss_weights": result["loss_weights"], "initial_state_sha256": audit["audit"]["initialization"]["state_sha256"],
                "shared_initial_state_sha256": audit["audit"]["initialization"]["shared_state_sha256"],
                "ldad_decoder_enabled": has_ldad,
                "training_order_sha256": audit["audit"]["data_provenance"]["order_sha256"]}
    return evidence, summaries, episodes, spreads, result


def compare(group):
    group = group.resolve(strict=True)
    bundle, counts = verified_split(group)
    commit = read_json(group / "group-checkpoint.json")
    group_digest = commit["record_sha256"]
    check(commit["status"] == "complete" and set(commit["members"]) == set(VARIANTS)
          and json_hash({k: v for k, v in commit.items() if k != "record_sha256"}) == group_digest,
          "Missing or invalid completed shared-training commit")
    members, models, rows, spreads, raw = {}, {}, {}, {}, {}
    for label, variant in VARIANTS.items():
        members[label], models[label], rows[label], spreads[label], raw[label] = member_evidence(
            group, variant, counts, bundle["manifest"]["manifest_sha256"], group_digest)
    check(len({v["initial_state_sha256"] for v in members.values() if v["ldad_decoder_enabled"]}) == 1
          and len({v["shared_initial_state_sha256"] for v in members.values()}) == 1
          and all(v["initial_state_sha256"] == v["shared_initial_state_sha256"] for v in members.values() if not v["ldad_decoder_enabled"])
          and {v["training_order_sha256"] for v in members.values()} == {commit["order_sha256"]}
          and len({v["epochs"] for v in members.values()}) == 1, "Initialization, epochs or training order differ")
    comparisons = {}
    labels = list(VARIANTS)
    for p in PRECISIONS:
        scores = {k + "_mse": models[k][p]["validation_arithmetic_mse"] for k in labels}
        pairs = {}
        for i, candidate in enumerate(labels):
            published = raw[candidate]["comparisons"][p]
            check(set(published["paired_by_reference"]) == set(labels[:i]), "Missing or unpaired sibling comparison")
            for other in labels[:i]:
                pair = paired_statistics(rows[other][p], rows[candidate][p], candidate)
                check(pair == published["paired_by_reference"][other], "Bootstrap differs from verified paired episodes")
                reference_mse, candidate_mse = scores[other + "_mse"], scores[candidate + "_mse"]
                difference = candidate_mse - reference_mse
                reduction = 100 * (1 - candidate_mse / reference_mse) if reference_mse else None
                check(difference == published["original_validation_arithmetic"][f"{candidate}_minus_{other}_mse"],
                      "Published paired difference disagrees with phase scores")
                pairs[f"{candidate}--{other}"] = {"candidate": candidate, "reference": other,
                    "difference_mse": difference, "relative_reduction_percent": reduction,
                    "paired_episode_statistics": pair}
        comparisons[p] = {"original_validation_arithmetic": scores, "pairs": pairs,
                          "ranking_by_next_embedding_mse": sorted(labels, key=lambda k: scores[k + "_mse"])}
    result = {"passed": True, "protocol": PROTOCOL, "group_root": str(group), "group_record_sha256": group_digest,
              "manifest_sha256": bundle["manifest"]["manifest_sha256"], "primary_precision": "bf16_mixed",
              "requested_precisions": list(PRECISIONS), "test_episodes": len(counts), "test_windows": sum(counts.values()),
              "transitions_per_model_precision": 3 * sum(counts.values()), "train_test_episode_disjoint": True,
              "train_only_normalization_verified": True, "members": members, "models": models,
              "test_latent_variation": spreads, "comparisons": comparisons, "finished_utc": now()}
    lines = ["Two-Room Delta-JEPA held-out next-embedding MSE",
             f"{result['test_episodes']:,} episodes, {result['test_windows']:,} windows and "
             f"{result['transitions_per_model_precision']:,} predicted transitions per model and precision.",
             "Each model predicts its own image encoder's next embedding.", ""]
    for p in PRECISIONS:
        lines.extend([p, "LDAD       SIGReg      Next-embedding MSE       Encoder RMS standard deviation"])
        for label in labels:
            v = VARIANTS[label]
            lines.append(f"{v.ldad_weight:<10g} {v.sigreg_weight:<10g} {models[label][p]['validation_arithmetic_mse']:<24.12g} "
                         f"{spreads[label][p]['latent_cross_example_rms_std']:.12g}")
        for pair in comparisons[p]["pairs"].values():
            ci = pair["paired_episode_statistics"]["native_window_metric"]["paired_episode_bootstrap_difference_95pct_ci"]
            lines.append(f"{VARIANTS[pair['candidate']].display_label} minus {VARIANTS[pair['reference']].display_label}: "
                         f"{pair['difference_mse']:+.8g} MSE, paired episode-bootstrap 95% interval [{ci[0]:.8g}, {ci[1]:.8g}].")
        lines.append("")
    report = group / "report.txt"
    temp = report.with_suffix(".txt.tmp")
    temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temp.replace(report)
    folder = group / "evaluation"
    folder.mkdir(exist_ok=True)
    write_json(folder / "results.json", result)
    write_json(folder / "progress.json", {"complete": True, "passed": True, "finished_utc": result["finished_utc"]})
    print(f"Verified all six phases and three paired comparisons per precision. Results in {folder}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-root", type=Path, required=True)
    args = parser.parse_args()
    compare(args.group_root)
