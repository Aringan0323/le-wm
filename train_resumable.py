"""One-pass paired training with batch-boundary pause/resume and 5-minute saves."""

import csv
import hashlib
import json
import math
import os
import random
import shutil
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf, open_dict

from episode_data import get_episode_normalizer, load_episode_experiment
from experiment_tracking import initialization_record
from resumable_training import (
    CheckpointManager, DurableTrainingTrace, EpochOrder, FixedOrderBatchSampler,
    IndexedWindowDataset, atomic_json, file_sha256, json_hash, make_fixed_order,
    capture_rng_state, pause_requested, restore_rng_state, utc_now,
)
from train import CROSS_EXAMPLE_FIELDS, configure_training_objective, tracks_cross_example_latents
from utils import get_img_preprocessor


class MetricsWriter:
    """CSV append stream reconciled to the exact byte prefix in each checkpoint."""
    def __init__(self, path, fields, resume=None):
        self.path = Path(path)
        self.fields = fields
        self.digest = hashlib.sha256()
        if resume is None:
            self.stream = self.path.open("xb")
            self.writer = csv.DictWriter(self, fields, lineterminator="\n")
            self.writer.writeheader()
        else:
            length = int(resume["metrics_bytes"])
            with self.path.open("rb") as source:
                remaining = length
                while remaining:
                    block = source.read(min(remaining, 1024 * 1024))
                    if not block:
                        raise ValueError("Metrics file is shorter than its checkpoint")
                    self.digest.update(block)
                    remaining -= len(block)
                if self.digest.hexdigest() != resume["metrics_sha256"]:
                    raise ValueError("CSV metrics prefix differs from the resume checkpoint")
                tail = self.path.stat().st_size - length
                if tail:
                    backup = self.path.with_name(f"metrics.abandoned-{uuid.uuid4().hex}.csv")
                    with backup.open("xb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                        output.flush()
                        os.fsync(output.fileno())
            if tail:
                with self.path.open("r+b") as output:
                    output.truncate(length)
                    output.flush()
                    os.fsync(output.fileno())
            self.stream = self.path.open("ab")
            self.writer = csv.DictWriter(self, fields, lineterminator="\n")

    def write(self, text):
        raw = text.encode("utf-8")
        self.stream.write(raw)
        self.digest.update(raw)
        return len(text)

    def append(self, values):
        self.writer.writerow(values)

    def checkpoint_metadata(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())
        return {"metrics_bytes": self.path.stat().st_size,
                "metrics_sha256": self.digest.hexdigest()}

    def flush(self):
        self.stream.flush()

    def close(self):
        if not self.stream.closed:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()


def delta_gradient_evidence(model, decoder_enabled):
    """Check the first actual backward pass, before clipping or the optimizer update."""
    record = {}
    for label, decoder in (("shared", False), ("ldad", True)):
        parameters = [p for name, p in model.named_parameters() if p.requires_grad and name.startswith("ldad.") == decoder]
        gradients = [p.grad for p in parameters if p.grad is not None]
        required = not decoder or decoder_enabled
        if not parameters or len(gradients) != (len(parameters) if required else 0):
            raise ValueError(f"Unexpected first-step gradient coverage for {label}")
        norm = torch.stack([g.detach().norm().float() for g in gradients]).norm().item() if gradients else 0.0
        if not math.isfinite(norm) or (required and norm == 0):
            raise FloatingPointError(f"Unexpected first-step gradient norm for {label}: {norm}")
        record[label] = {"parameter_tensors": len(parameters), "gradient_tensors": len(gradients), "l2_norm": norm}
    return record


def _train(cfg):
    if not cfg.get("episode_manifest") or not cfg.get("run_metadata_dir"):
        raise ValueError("Resumable training requires episode_manifest and run_metadata_dir")
    objective = cfg.loss.get("type", "sigreg")
    epochs = cfg.trainer.max_epochs
    if type(epochs) is not int or epochs < 1 or cfg.trainer.get("max_steps", -1) != -1:
        raise ValueError("Specify a positive number of complete epochs without a step cap")
    if epochs != 1 and objective != "delta_jepa":
        raise ValueError("Existing non-Delta episode experiments are single-pass")
    if cfg.get("compile_training", False):
        raise ValueError("The resumable experiment uses eager execution for RNG-continuous recovery")
    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cfg.get("cpu_threads") is not None:
        torch.set_num_threads(int(cfg.cpu_threads))
    run_dir = Path(cfg.run_metadata_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    resume_path = cfg.get("resume_from")

    dataset_config = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_config.pop("name")
    dataset = swm.data.load_dataset(dataset_name, transform=None,
                                   cache_dir=os.environ.get("LOCAL_DATASET_DIR"), **dataset_config)
    bundle = load_episode_experiment(cfg.episode_manifest, dataset)
    manifest = bundle["manifest"]
    batch_size = int(cfg.loader.batch_size)
    if batch_size != manifest["training_order"]["batch_size"]:
        raise ValueError("Batch size differs from the immutable experiment manifest")
    transforms = [get_img_preprocessor("pixels", "pixels", cfg.img_size)]
    if objective == "delta_jepa":
        from utils import PreserveExecutedActions
        transforms.insert(0, PreserveExecutedActions(cfg.data.dataset.frameskip))
    for column in cfg.data.dataset.keys_to_load:
        if not column.startswith("pixels"):
            transforms.append(get_episode_normalizer(bundle, column))
    dataset.transform = spt.data.transforms.Compose(*transforms)
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")
        cfg.episode_manifest = bundle["manifest_path"]
        cfg.episode_manifest_sha256 = manifest["manifest_sha256"]
    resolved_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    fingerprint_config = {key: value for key, value in resolved_config.items()
                          if key not in ("resume_from", "control_file")}
    config_sha256 = json_hash(fingerprint_config)
    train_indices = bundle["arrays"]["train_clip_indices"]
    order = EpochOrder(train_indices, seed, epochs) if objective == "delta_jepa" else make_fixed_order(train_indices, seed)
    sampler = FixedOrderBatchSampler(order, batch_size)
    total_steps = sampler.total_steps
    if sampler.steps_per_epoch != manifest["training_order"]["steps_per_epoch"]:
        raise ValueError("Expected step count differs from the experiment manifest")

    if isinstance(order, EpochOrder):
        schedule_record = {**order.protocol, "manifest_sha256": manifest["manifest_sha256"],
                           "order_sha256": order.order_hash, "batch_size": batch_size,
                           "steps_per_epoch": sampler.steps_per_epoch, "expected_steps": sampler.total_steps}
        schedule_path = run_dir / "order-protocol.json"
        if resume_path:
            if json.loads(schedule_path.read_text()) != schedule_record:
                raise ValueError("Saved epoch order protocol differs from this configuration")
        else:
            atomic_json(schedule_path, schedule_record)

    model = hydra.utils.instantiate(cfg.model)
    if not resume_path:
        initialization_record(model, cfg, bundle, run_dir)
    elif not (run_dir / "initialization.json").exists():
        raise ValueError("Resume requires the original fresh initialization evidence")
    accelerator = cfg.trainer.accelerator
    if accelerator in ("gpu", "cuda"):
        device = torch.device("cuda:0")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
    elif accelerator == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError("Choose an explicit gpu or cpu accelerator")
    model.to(device).train()
    forward, sigreg = configure_training_objective(cfg, model)
    if sigreg is not None:
        sigreg.to(device)
    optimizer_config = OmegaConf.to_container(cfg.optimizer, resolve=True)
    if optimizer_config.pop("type") != "AdamW":
        raise ValueError("This paired experiment requires AdamW")
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)
    from stable_pretraining.optim.lr_scheduler import LinearWarmupCosineAnnealingLR
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, warmup_steps=max(1, int(0.01 * total_steps)), max_steps=total_steps,
        warmup_start_lr=0.0, eta_min=0.0)
    checkpoints = CheckpointManager(
        run_dir, manifest["manifest_sha256"], config_sha256, batch_size, order,
        interval_seconds=float(cfg.get("checkpoint_every_seconds", 300)))
    restored = checkpoints.load(resume_path, model, optimizer, scheduler, sigreg,
                                 sampler=sampler, restore_rng=False) if resume_path else None
    metadata = restored["resumable"] if restored else None
    trace = DurableTrainingTrace(run_dir, order, resume=metadata)
    metric_names = ["fit/pred_loss", "fit/loss", "hparams/lr_model_opt"]
    if sigreg is not None:
        metric_names += ["fit/sigreg_loss"]
    if objective in ("action_prediction", "combined"):
        metric_names += ["fit/action_loss", "fit/latent_std", "fit/action_embedding_std", "fit/predicted_action_std"]
    if objective == "delta_jepa":
        metric_names.append("fit/ldad_loss")
    if tracks_cross_example_latents(cfg):
        metric_names.extend("fit/" + name for name in CROSS_EXAMPLE_FIELDS)
    metrics = MetricsWriter(run_dir / "metrics.csv", ["step", "epoch", *metric_names],
                            resume=metadata["extra"] if metadata else None)

    export_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), cfg.output_model_name)
    export_dir.mkdir(parents=True, exist_ok=True)
    if not resume_path:
        OmegaConf.save(cfg, export_dir / "config.yaml")
        OmegaConf.save(cfg, run_dir / "config.yaml")
        atomic_json(run_dir / "config-identity.json", {
            "config_sha256": config_sha256, "manifest_sha256": manifest["manifest_sha256"],
            "excluded_runtime_keys": ["resume_from", "control_file"]})
        events = []
        atomic_json(run_dir / "resume_events.json", events)
    else:
        events_path = run_dir / "resume_events.json"
        events = json.loads(events_path.read_text()) if events_path.exists() else []
        events.append({"resumed_utc": utc_now(), "checkpoint": str(Path(resume_path).resolve()),
                       "global_step": sampler.global_step, "windows_seen": sampler.cursor,
                       "config_sha256": config_sha256})
        atomic_json(events_path, events)

    loader_config = OmegaConf.to_container(cfg.loader, resolve=True)
    loader_config.pop("batch_size")
    for key in ("shuffle", "drop_last", "sampler", "batch_sampler", "generator"):
        loader_config.pop(key, None)
    loader = torch.utils.data.DataLoader(
        IndexedWindowDataset(dataset), batch_sampler=sampler,
        generator=torch.Generator().manual_seed(seed), **loader_config)
    # Start workers only after a durable checkpoint and visible progress exist.
    iterator = None

    resumed_elapsed = float(metadata["extra"].get("elapsed_seconds", 0)) if metadata else 0.0
    session_start = time.monotonic()
    started_utc = metadata["extra"].get("started_utc", utc_now()) if metadata else utc_now()
    last_values = dict(metadata["extra"].get("last_values", {})) if metadata else {}
    holder = SimpleNamespace(model=model, sigreg=sigreg, logged={})
    holder.log_dict = lambda values, **kwargs: setattr(holder, "logged", values)
    precision = str(cfg.trainer.precision)
    if precision not in ("bf16", "bf16-mixed", "32", "32-true"):
        raise ValueError("Only BF16 mixed or full FP32 precision is supported")
    use_bf16 = precision in ("bf16", "bf16-mixed")
    log_interval = max(1, int(cfg.trainer.get("log_every_n_steps", 10)))

    def progress(status, error=None):
        elapsed = resumed_elapsed + time.monotonic() - session_start
        record = {
            "schema_version": 1, "backend": "native_pytorch", "status": status,
            "started_utc": started_utc, "updated_utc": utc_now(),
            "manifest_sha256": manifest["manifest_sha256"],
            "resume_count": len(events),
            "expected_steps": total_steps, "expected_windows": len(order),
            "optimizer_steps": sampler.global_step, "batches_seen": sampler.global_step,
            "last_batch_size": sampler.last_batch_size,
            "epochs": epochs, "steps_per_epoch": sampler.steps_per_epoch, "windows_per_epoch": sampler.windows_per_epoch,
            "completed_epochs": sampler.cursor // sampler.windows_per_epoch,
            "current_epoch": min(epochs, sampler.cursor // sampler.windows_per_epoch + 1),
            "windows_in_epoch": sampler.cursor % sampler.windows_per_epoch if sampler.cursor < len(order) else sampler.windows_per_epoch,
            "elapsed_seconds": elapsed,
            "eta_seconds": elapsed * (total_steps - sampler.global_step) / sampler.global_step if sampler.global_step else None,
            "last_train_loss": last_values.get("fit/loss"),
            "last_losses": {key.split("/", 1)[-1]: value for key, value in last_values.items() if "loss" in key},
            "nonfinite_loss_name": next((key for key, value in last_values.items() if "loss" in key and value is None), None),
            "counts_match": sampler.cursor == len(order) and sampler.global_step == total_steps,
            **trace.summary(),
        }
        if checkpoints.latest_record:
            record["latest_checkpoint"] = checkpoints.latest_record["path"]
            record["checkpoint_saved_utc"] = checkpoints.latest_record["saved_utc"]
            record["checkpoint_step"] = checkpoints.latest_record["global_step"]
        if error is not None:
            record["exception_type"] = type(error).__name__
            record["exception_message"] = str(error)
        atomic_json(run_dir / "progress.json", record)
        return record

    first_gradient_hash = metadata["extra"].get("first_step_gradient_sha256") if metadata else None
    if objective == "delta_jepa" and sampler.cursor:
        if not first_gradient_hash or file_sha256(run_dir / "first-step-gradients.json") != first_gradient_hash:
            raise ValueError("Initial gradient evidence differs from the resume checkpoint")

    def save(status="running", final=False):
        extra = {"elapsed_seconds": resumed_elapsed + time.monotonic() - session_start,
                 "started_utc": started_utc, "objective": objective,
                 "last_values": last_values, **metrics.checkpoint_metadata()}
        if objective == "delta_jepa":
            extra["first_step_gradient_sha256"] = first_gradient_hash
        return checkpoints.save(model, optimizer, scheduler, sigreg, sampler, trace,
                                status=status, extra=extra, final=final)

    try:
        # A pause before worker startup must still preserve the checkpoint's RNG.
        intended_rng = metadata["rng"] if restored else capture_rng_state()
        restore_rng_state(intended_rng)
        if not restored:
            save()  # Step zero is always recoverable, even before the first batch.
        progress("running")
        if sampler.cursor < len(order) and pause_requested(cfg.get("control_file")):
            save("paused")
            progress("paused")
            return 75
        iterator = iter(loader)
        # Restore LAST after worker startup, on both fresh start and resume.
        restore_rng_state(intended_rng)
        for batch in iterator:
            window_ids = batch["_window_index"]
            device_batch = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) and key != "_window_index" else value
                            for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                output = forward(holder, device_batch, "fit", cfg)
            if not torch.isfinite(output["loss"]).item():
                raise FloatingPointError("Nonfinite training loss; retaining the last valid durable checkpoint")
            output["loss"].backward()
            if objective == "delta_jepa" and sampler.global_step == 0:
                gradients = delta_gradient_evidence(model, cfg.loss.ldad.weight > 0)
                term_dtypes = {name: str(value.dtype) for name, value in output.items() if name in ("pred_loss", "ldad_loss", "sigreg_loss", "loss")}
                atomic_json(run_dir / "first-step-gradients.json", {"step": 1, "config_sha256": config_sha256,
                                                                    "term_dtypes": term_dtypes, **gradients})
                first_gradient_hash = file_sha256(run_dir / "first-step-gradients.json")
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.trainer.gradient_clip_val),
                                          error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            # No prefetch cursor enters checkpoints: commit only the completed step.
            sampler.commit(window_ids)
            trace.append(window_ids)
            last_values = {key: float(value.detach().float().cpu()) for key, value in holder.logged.items()}
            last_values["hparams/lr_model_opt"] = float(optimizer.param_groups[0]["lr"])
            if not all(math.isfinite(value) for value in last_values.values()):
                last_values = {key: value if math.isfinite(value) else None for key, value in last_values.items()}
                raise FloatingPointError("Nonfinite training diagnostic; retaining the last valid checkpoint")
            metrics.append({"step": sampler.global_step, "epoch": (sampler.cursor - 1) // sampler.windows_per_epoch, **last_values})
            if pause_requested(cfg.get("control_file")):
                save("paused")
                progress("paused")
                return 75
            if checkpoints.due():
                save()
                progress("running")
            elif sampler.global_step % log_interval == 0:
                metrics.flush()
                if sampler.global_step % 100 == 0:
                    trace.flush()
                progress("running")
        if sampler.cursor != len(order) or sampler.global_step != total_steps:
            raise RuntimeError("Training ended before all fixed-order windows were committed")
        final_checkpoint = save("complete", final=True)
        swm.wm.utils.save_pretrained(model, run_name=cfg.output_model_name,
                                     config=cfg.model, filename=f"weights_epoch_{epochs}.pt")
        summary = progress("complete")
        atomic_json(run_dir / "data-order.json", summary)
        atomic_json(run_dir / "runcompletion.json", {
            "passed": True, "backend": "native_pytorch", "global_step": sampler.global_step,
            "expected_steps": total_steps, "expected_windows": len(order), "epochs": epochs,
            "steps_per_epoch": sampler.steps_per_epoch, "windows_per_epoch": sampler.windows_per_epoch,
            "final_ckpt_path": str(final_checkpoint),
            "export_path": str(export_dir / f"weights_epoch_{epochs}.pt"),
            "registry_run": str(run_dir), "manifest_sha256": manifest["manifest_sha256"],
            "objective": objective, "resume_count": len(events), "completed_utc": utc_now()})
        return 0
    except BaseException as error:
        # A partially executed step must not be checkpointed under the old cursor.
        progress("failed", error)
        raise
    finally:
        metrics.close()
        trace.close()
        shutdown = getattr(iterator if iterator is not None else getattr(loader, "_iterator", None), "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    result = _train(cfg)
    if result:
        raise SystemExit(result)


if __name__ == "__main__":
    run()
