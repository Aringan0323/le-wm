"""Names and final checkpoint evidence for standalone or grouped Delta-JEPA runs."""
import json
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def model_name(experiment, variant, group=None):
    plan = read_json(experiment / "plan.json")
    name = plan["model_names"][variant.label]
    expected = variant.model
    if group is not None:
        group = group.resolve(strict=True)
        group_plan = read_json(group / "plan.json")
        if experiment.resolve(strict=True) != (group / "members" / variant.label).resolve(strict=True):
            raise ValueError(f"Unexpected group member directory for {variant.label}")
        for item in ("manifest.json", "arrays.npz", "source"):
            if not (experiment / item).samefile(group / item):
                raise ValueError(f"{variant.label} does not use shared {item}")
        names = group_plan["model_names"]
        expected = names[variant.label]
        if (not group_plan["shared_batch"] or plan.get("shared_batch") is not True
                or Path(plan["group_root"]).resolve() != group or plan["epochs"] != group_plan["epochs"]
                or variant.label not in group_plan["members"] or len(set(names.values())) != len(names)
                or expected == variant.model or expected in (".", "..") or not expected or Path(expected).name != expected):
            raise ValueError(f"Invalid shared group namespace for {variant.label}")
    if name != expected or plan["variant"] != variant.label or plan["objective"] != variant.objective:
        raise ValueError(f"Wrong planned checkpoint or objective for {variant.label}")
    return name


def sibling(experiment, other, group=None):
    return group / "members" / other.label if group is not None else experiment.parent / other.experiment.name


def committed_generation(group, variant, run, audit):
    from resumable_training import json_hash
    record = read_json(group / "group-checkpoint.json")
    content = {key: value for key, value in record.items() if key != "record_sha256"}
    data = audit["data_provenance"]
    checkpoint = audit["checkpoint"]
    if (record["record_sha256"] != json_hash(content) or record["version"] != 1 or record["status"] != "complete"
            or record["global_step"] != data["completed_steps"] or record["cursor"] != data["windows"]
            or record["manifest_sha256"] != audit["manifest_sha256"] or record["order_sha256"] != data["order_sha256"]
            or record["trace_sha256"] != data["order_sha256"]):
        raise ValueError("Completed group ledger does not match the audited training data")
    member = record["members"][variant.label]
    generation = Path(member["path"]).resolve(strict=True)
    sidecar = read_json(generation.with_suffix(".json"))
    recovery = checkpoint["recovery_protocol"]
    if (member["status"] != "complete" or generation.parent != (run / "checkpoints").resolve(strict=True)
            or generation.name != member["generation_id"] + ".ckpt"
            or member["global_step"] != record["global_step"] or member["windows_seen"] != record["cursor"]
            or member["config_sha256"] != checkpoint["recovery_protocol"]["config_sha256"]
            or generation.stat().st_size != member["size_bytes"]
            or member["sha256"] != sidecar["sha256"] or sidecar["generation_id"] != member["generation_id"]
            or not generation.samefile(checkpoint["path"])
            or recovery["generation_id"] != member["generation_id"] or recovery["checksum_verified"] is not True
            or recovery["sha256"] != member["sha256"]):
        raise ValueError("Member audit does not match its committed group checkpoint")
    return {"group_record_sha256": record["record_sha256"], "group_generation_id": record["generation_id"],
            "member_generation_id": member["generation_id"], "checkpoint_sha256": member["sha256"]}
