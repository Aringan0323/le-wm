"""Verify an observed-displacement Delta-JEPA run, its actual data and weighted losses."""
import argparse
import csv
import copy
import hashlib
import json
import math
import sys
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import hydra
import torch
from omegaconf import OmegaConf, open_dict
from audit_episode_experiment import audit as audit_training, read_json
from combined_sweep_config import REFERENCE
from delta_sweep_config import VARIANTS
from delta_group_layout import committed_generation, model_name, sibling
from episode_data import load_episode_experiment
from experiment_tracking import state_dict_sha256
from resumable_training import atomic_json, file_sha256


def verify_delta_metrics(path, variant, epoch_steps, epochs, batch_size, tail, has_ldad=True):
    """Reconstruct exact training arithmetic; memory use is bounded by 4096 rows."""
    with_sigreg = variant.sigreg_weight > 0
    fields = ["fit/pred_loss", "fit/loss"] + (["fit/ldad_loss"] if has_ldad else []) + (["fit/sigreg_loss"] if with_sigreg else [])
    dtype_names = read_json(path.parent / "first-step-gradients.json")["term_dtypes"]
    dtype = {"torch.float32": torch.float32, "torch.bfloat16": torch.bfloat16}
    assert (not has_ldad or dtype_names["ldad_loss"] == "torch.float32") and set(dtype_names) == {k.split("/")[1] for k in fields}
    terms = {key: [] for key in fields}
    max_error = 0.0
    spread_last = deque(maxlen=100)
    spread_first = spread_min = spread_max = None
    exact_constant = 0
    count = 0
    def verify_block():
        series = {k: torch.tensor(values, dtype=torch.float32) for k, values in terms.items()}
        for values in series.values():
            assert torch.isfinite(values).all() and (values >= 0).all()
        expected = series["fit/pred_loss"].to(dtype[dtype_names["pred_loss"]])
        if with_sigreg:
            expected = expected + variant.sigreg_weight * series["fit/sigreg_loss"].to(dtype[dtype_names["sigreg_loss"]])
        if variant.ldad_weight:
            expected = expected + variant.ldad_weight * series["fit/ldad_loss"]
        actual = series["fit/loss"]
        assert str(expected.dtype) == dtype_names["loss"]
        expected = expected.float()
        torch.testing.assert_close(expected, actual, rtol=2e-6, atol=2e-8)
        for values in terms.values():
            values.clear()
        return float((actual - expected).abs().max())

    with path.open(newline="") as stream:
        rows = csv.DictReader(stream)
        assert set(fields) <= set(rows.fieldnames)
        assert ("fit/sigreg_loss" in rows.fieldnames) == with_sigreg
        assert ("fit/ldad_loss" in rows.fieldnames) == has_ldad
        assert "fit/action_loss" not in rows.fieldnames
        for count, row in enumerate(rows, 1):
            assert int(row["step"]) == count and int(row["epoch"]) == (count - 1) // epoch_steps
            for key in fields:
                terms[key].append(float(row[key]))
            expected_size = tail if count % epoch_steps == 0 else batch_size
            assert float(row["fit/latent_cross_example_count"]) == expected_size
            if expected_size > 1:
                rms = float(row["fit/latent_cross_example_rms_std"])
                var = float(row["fit/latent_cross_example_var"])
                max_range = float(row["fit/latent_cross_example_max_range"])
                global_range = float(row["fit/latent_global_max_range"])
                assert all(math.isfinite(x) and x >= 0 for x in (rms, var, max_range, global_range))
                assert math.isclose(rms * rms, var, rel_tol=2e-6, abs_tol=1e-40) and global_range >= max_range
                exact_constant += int(max_range == 0)
                spread_last.append(rms)
                if spread_first is None:
                    spread_first = spread_min = spread_max = rms
                spread_min, spread_max = min(spread_min, rms), max(spread_max, rms)
            if len(terms["fit/loss"]) == 4096:
                max_error = max(max_error, verify_block())
    if terms["fit/loss"]:
        max_error = max(max_error, verify_block())
    assert count == epoch_steps * epochs
    return {"steps_checked": count, "next_embedding_weight": 1.0, "ldad_weight": variant.ldad_weight, "ldad_decoder_enabled": has_ldad,
            "sigreg_weight": variant.sigreg_weight, "sigreg_enabled": with_sigreg,
            "max_absolute_total_loss_error": max_error,
            "training_latent_rms_std": {"first": spread_first, "last": spread_last[-1], "min": spread_min,
                "max": spread_max, "last_100_mean": sum(spread_last) / len(spread_last),
                "exactly_constant_batches": exact_constant}}


def assert_matching_protocol(manifest, previous):
    # File paths, archive timestamps and creation times can change on a fresh checkout.
    for field in ("schema_version", "split", "window", "counts", "training_order", "normalization", "array_hashes"):
        assert manifest[field] == previous[field], f"Historical data protocol differs: {field}"
    for field in ("columns", "episode_count", "row_count", "episode_lengths_sha256", "episode_offsets_sha256"):
        assert manifest["dataset"][field] == previous["dataset"][field], f"Historical dataset differs: {field}"


def compare_history(reference, manifest, data, init):
    if reference is None or not reference.exists():
        return {"available": False}
    previous = read_json(reference / "manifest.json")
    from resumable_training import json_hash
    assert previous["manifest_sha256"] == json_hash({k:v for k,v in previous.items() if k != "manifest_sha256"})
    assert file_sha256(reference / previous["arrays_file"]) == previous["arrays_sha256"]
    assert_matching_protocol(manifest, previous)
    old = reference / "runs/sigreg"
    old_init, old_order = read_json(old / "initialization.json"), read_json(old / "data-order.json")
    assert old_init["manifest_sha256"] == old_order["manifest_sha256"] == previous["manifest_sha256"]
    assert old_order["status"] == "complete" and old_order["counts_match"] is True
    assert data["epoch_order_sha256"][0] == old_order["order_sha256"] == file_sha256(old / "training-indices.i64")
    assert init["shared_state_sha256"] == old_init["shared_state_sha256"]
    return {"available": True, "reference": str(reference), "matched_data_protocol": True,
            "identical_epoch_one_order": True, "identical_shared_initialization": True}


def audit_delta(experiment, variant, historical_reference=REFERENCE, group=None):
    manifest_path = experiment / "manifest.json"
    manifest = load_episode_experiment(manifest_path)["manifest"]
    plan = read_json(experiment / "plan.json")
    epochs = int(plan["epochs"])
    has_ldad = (variant.ldad_weight > 0) if group is not None else True
    if group is not None:
        assert plan["ldad_decoder_enabled"] is has_ldad
    run = experiment / "runs" / variant.label
    result = audit_training(SimpleNamespace(run_dir=run, manifest=manifest_path, paired_run_dir=None,
        expected_objective="delta_jepa", expected_epochs=epochs, expected_ldad_weight=variant.ldad_weight,
        expected_sigreg_weight=variant.sigreg_weight, expected_ldad_decoder_enabled=has_ldad))
    data, init = result["data_provenance"], result["initialization"]
    export = Path(result["export"]["path"])
    cfg = OmegaConf.load(export.parent / "config.yaml")
    expected_model = model_name(experiment, variant, group)
    assert cfg.output_model_name == expected_model and export.parent.name == expected_model and export.name == f"weights_epoch_{epochs}.pt"
    assert cfg.data.dataset.frameskip == 5 and cfg.history_size == 3 and cfg.num_preds == 1
    if has_ldad:
        assert cfg.model.ldad.horizon == 5 and cfg.model.ldad.action_dim == 2
    else:
        assert cfg.model.get("ldad") is None and init["state_sha256"] == init["shared_state_sha256"]
    assert cfg.optimizer.type == "AdamW" and cfg.optimizer.lr == 5e-5 and cfg.trainer.gradient_clip_val == 1.0
    assert cfg.track_cross_example_latent_spread is True
    assert data["all_train_windows_once_per_epoch"] and data["test_and_unused_episodes_not_consumed"]
    assert data["windows_per_epoch"] == 365657 and data["steps_per_epoch"] == 11427 and data["last_batch_size"] == 25
    protocol = read_json(run / "order-protocol.json")
    assert protocol["epochs"] == epochs and protocol["order_sha256"] == data["order_sha256"]
    assert protocol["epoch_sha256"] == data["epoch_order_sha256"] and protocol["manifest_sha256"] == manifest["manifest_sha256"]
    base_cfg = copy.deepcopy(cfg.model)
    with open_dict(base_cfg):
        if "ldad" in base_cfg:
            del base_cfg.ldad
    torch.manual_seed(cfg.seed)
    base_model = hydra.utils.instantiate(base_cfg)
    local_shared_hash = state_dict_sha256(base_model.state_dict())
    assert local_shared_hash == init["shared_state_sha256"]
    del base_model
    history = compare_history(historical_reference, manifest, data, init)

    initial_generation, = (run / "checkpoints").glob("step-00000000-*.ckpt")
    sidecar = read_json(initial_generation.with_suffix(".json"))
    assert sidecar["global_step"] == 0 and file_sha256(initial_generation) == sidecar["sha256"]
    initial_payload = torch.load(initial_generation, map_location="cpu", weights_only=False)
    initial_state = {k[6:]: t for k,t in initial_payload["state_dict"].items() if k.startswith("model.")}
    assert any(k.startswith("ldad.") for k in initial_state) == has_ldad
    assert initial_payload["global_step"] == 0 and state_dict_sha256(initial_state) == init["state_sha256"]
    assert initial_payload["resumable"]["config_sha256"] == result["checkpoint"]["recovery_protocol"]["config_sha256"]
    assert (variant.sigreg_weight > 0) == any(k.startswith("sigreg.") for k in initial_payload["state_dict"])
    del initial_payload, initial_state
    gradients = read_json(run / "first-step-gradients.json")
    assert gradients["step"] == 1 and gradients["config_sha256"] == result["checkpoint"]["recovery_protocol"]["config_sha256"]
    for key, enabled in (("shared", True), ("ldad", variant.ldad_weight > 0)):
        if key == "ldad" and not has_ldad:
            if key in gradients:
                assert gradients[key]["parameter_tensors"] == gradients[key]["gradient_tensors"] == gradients[key]["l2_norm"] == 0
            continue
        row = gradients[key]
        assert row["parameter_tensors"] > 0 and row["gradient_tensors"] == (row["parameter_tensors"] if enabled else 0)
        assert math.isfinite(row["l2_norm"]) and (row["l2_norm"] > 0 if enabled else row["l2_norm"] == 0)
    comparisons = {}
    for other in VARIANTS.values():
        if other.label == variant.label:
            continue
        other_root = sibling(experiment, other, group)
        other_run = other_root / "runs" / other.label
        if not (other_run / "initialization.json").is_file():
            assert group is None, f"Missing group initialization for {other.label}"
            continue
        other_init = read_json(other_run / "initialization.json")
        other_manifest = load_episode_experiment(other_root / "manifest.json")["manifest"]
        assert_matching_protocol(manifest, other_manifest)
        assert other_init["manifest_sha256"] == other_manifest["manifest_sha256"]
        assert other_init["shared_state_sha256"] == init["shared_state_sha256"]
        match_full = group is None or (has_ldad and other.ldad_weight > 0)
        if match_full:
            assert other_init["state_sha256"] == init["state_sha256"]
        evidence = {"identical_full_initialization": match_full, "identical_shared_initialization": True}
        other_report = other_run / "final-audit.json"
        if other_report.is_file() and (group is None or read_json(other_report).get("passed")):
            other_audit = read_json(other_report)
            assert other_audit["passed"] and other_audit["audit"]["data_provenance"]["order_sha256"] == data["order_sha256"]
            assert file_sha256(other_run / "training-indices.i64") == data["order_sha256"]
            evidence["identical_all_epoch_order"] = True
        comparisons[other.label] = evidence
    result.update(variant=variant.label, delta_objective=verify_delta_metrics(run / "metrics.csv", variant,
        data["steps_per_epoch"], epochs, data["batch_size"], data["last_batch_size"], has_ldad),
        first_step_gradients=gradients, paired_run_checks=comparisons, historical_reference=history)
    result["initialization"].update(independent_seeded_base_model_sha256=local_shared_hash,
                                    step_zero_checkpoint_sha256=sidecar["sha256"])
    result["export"]["weights_file_sha256"] = file_sha256(export)
    if group is not None:
        result["group_commit"] = committed_generation(group, variant, run, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--group-root", type=Path, help="Use committed, namespaced checkpoints and siblings from this group")
    parser.add_argument("--historical-reference-dir", type=Path, help="Optional prior experiment03 bundle for strict comparison")
    args = parser.parse_args()
    variant = VARIANTS[args.variant]
    experiment = (args.experiment_dir or variant.experiment).resolve(strict=True)
    output = experiment / "runs" / variant.label / "final-audit.json"
    torch.set_num_threads(2)
    record = {"passed":False, "started_utc":datetime.now(timezone.utc).isoformat()}
    try:
        group = args.group_root.resolve(strict=True) if args.group_root else None
        reference = args.historical_reference_dir.resolve(strict=True) if args.historical_reference_dir else (None if group else REFERENCE)
        record["audit"] = audit_delta(experiment, variant, reference, group)
        record["passed"] = True
    except Exception as exc:
        record.update(error=str(exc), traceback=traceback.format_exc())
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(output, record)
    print(json.dumps({"passed":record["passed"], "report":str(output), "error":record.get("error")}), flush=True)
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
