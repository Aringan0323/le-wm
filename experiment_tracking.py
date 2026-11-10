"""Initialization and actual data-order evidence for fresh episode experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, value, *, overwrite=False):
    """Publish complete JSON; exclusive creation never replaces existing evidence."""
    path = Path(path)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if overwrite:
            os.replace(temporary, path)
        else:
            # A hard link is an atomic create-if-absent operation on this filesystem.
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def state_dict_sha256(state_dict, *, exclude_prefixes=()):
    """Hash sorted names, dtypes, shapes, and exact contiguous tensor bytes.

    Header and raw-byte lengths are encoded as little-endian unsigned 64-bit
    integers. The raw tensor is viewed as bytes, supporting BF16 and scalars.
    """
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        if any(name.startswith(prefix) for prefix in exclude_prefixes):
            continue
        tensor = state_dict[name].detach().cpu().contiguous()
        header = json.dumps({"name": name, "dtype": str(tensor.dtype),
                             "shape": list(tensor.shape)}, sort_keys=True,
                            separators=(",", ":")).encode("utf-8")
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def _config_value(cfg, key, default=None):
    if OmegaConf.is_config(cfg):
        return OmegaConf.select(cfg, key, default=default)
    value = cfg
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def initialization_record(model, cfg, bundle, run_dir):
    """Record a fresh JEPA model before training, including all state buffers."""
    resume_from = _config_value(cfg, "resume_from")
    if resume_from:
        raise ValueError("Episode experiments require fresh training; loader-state resume is unsupported")
    run_dir = Path(run_dir).expanduser().resolve()
    path = run_dir / "initialization.json"
    if path.exists():
        raise FileExistsError(f"Initialization evidence already exists: {path}")
    state = model.state_dict()
    excluded = ("action_predictor.", "ldad.") if getattr(model, "ldad", None) is not None else ("action_predictor.",)
    record = {
        "schema_version": 1, "created_utc": _utc_now(),
        "seed": int(_config_value(cfg, "seed", 3072)),
        "resume_from": None,
        "encoder_pretrained": bool(_config_value(cfg, "model.encoder.pretrained", False)),
        "model_target": _config_value(cfg, "model._target_", f"{type(model).__module__}.{type(model).__qualname__}"),
        "manifest_sha256": bundle["manifest"]["manifest_sha256"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "state_sha256": state_dict_sha256(state),
        "shared_state_sha256": state_dict_sha256(state, exclude_prefixes=excluded),
        "shared_state_excluded_prefixes": list(excluded),
        "state_entry_count": len(state),
        "shared_state_entry_count": sum(not name.startswith(excluded) for name in state),
        "hash_format": "sorted state names; length-prefixed canonical name/dtype/shape JSON and contiguous raw tensor bytes; lengths uint64 little-endian",
    }
    _write_json(path, record)
    return record


class IndexedSubsetWrapper(torch.utils.data.Dataset):
    """Attach global dataset window IDs without changing observations or losses."""

    def __init__(self, subset):
        if not hasattr(subset, "indices"):
            raise TypeError("Expected a subset with global dataset indices")
        self.subset = subset
        self.indices = subset.indices

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        item = self.subset[index]
        if not isinstance(item, dict) or "_window_index" in item:
            raise ValueError("Expected a sample dictionary without a reserved _window_index field")
        return {**item, "_window_index": torch.tensor(int(self.indices[index]), dtype=torch.int64)}

    def __getitems__(self, indices):
        return [self[index] for index in indices]


class TrainingTraceCallback(Callback):
    """Record batches that actually completed training; no resume or append mode."""

    def __init__(self, run_dir, expected_steps, expected_windows, manifest_sha):
        super().__init__()
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.expected_steps = int(expected_steps)
        self.expected_windows = int(expected_windows)
        if self.expected_steps <= 0 or self.expected_windows <= 0:
            raise ValueError("Expected training steps/windows must be positive")
        self.manifest_sha = str(manifest_sha)
        self.trace_path = self.run_dir / "training-indices.i64"
        self.progress_path = self.run_dir / "progress.json"
        self.summary_path = self.run_dir / "data-order.json"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if any(path.exists() for path in (self.trace_path, self.progress_path, self.summary_path)):
            raise FileExistsError("Training trace already exists; episode loader-state resume is unsupported")
        self._stream = self.trace_path.open("xb")
        self._digest = hashlib.sha256()
        self._started = None
        self._started_utc = None
        self._finished = False
        self._batches = 0
        self._windows = 0
        self._step = 0
        self._first_ids = []
        self._last_ids = []
        self._last_batch_size = 0
        self._last_losses = {}
        self._nonfinite_loss_name = None
        try:
            _write_json(self.progress_path, self._snapshot("pending"))
        except BaseException:
            self._stream.close()
            raise

    def _snapshot(self, status, *, exception=None):
        elapsed = max(0.0, time.monotonic() - self._started) if self._started is not None else 0.0
        eta = elapsed * max(0, self.expected_steps - self._step) / self._step if self._step else None
        result = {
            "schema_version": 1, "status": status,
            "updated_utc": _utc_now(), "started_utc": self._started_utc,
            "manifest_sha256": self.manifest_sha,
            "trace_file": self.trace_path.name, "trace_dtype": "<i8",
            "expected_steps": self.expected_steps, "expected_windows": self.expected_windows,
            "optimizer_steps": self._step, "batches_seen": self._batches,
            "windows_seen": self._windows, "trace_bytes": self._windows * 8,
            "order_sha256": self._digest.hexdigest(),
            "first_ids": self._first_ids, "last_ids": self._last_ids,
            "last_batch_size": self._last_batch_size,
            "elapsed_seconds": elapsed, "eta_seconds": eta,
            "last_train_loss": self._last_losses.get("loss"),
            "last_losses": self._last_losses,
            "nonfinite_loss_name": self._nonfinite_loss_name,
            "counts_match": self._step == self.expected_steps and self._batches == self.expected_steps and self._windows == self.expected_windows,
        }
        if exception is not None:
            result["exception_type"] = type(exception).__name__
            result["exception_message"] = str(exception)
        return result

    def _flush(self):
        if not self._stream.closed:
            self._stream.flush()
            os.fsync(self._stream.fileno())

    def on_fit_start(self, trainer, pl_module):
        if int(trainer.global_step) != 0:
            raise ValueError("Training trace requires global_step=0; resume is unsupported")
        self._started = time.monotonic()
        self._started_utc = _utc_now()
        _write_json(self.progress_path, self._snapshot("running"), overwrite=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._finished:
            raise RuntimeError("Training trace is already closed")
        if self._started is None:
            raise RuntimeError("Training trace did not receive on_fit_start")
        values = batch["_window_index"]
        if torch.is_tensor(values):
            values = values.detach().cpu().numpy()
        indices = np.asarray(values)
        if indices.dtype.kind not in "iu" or indices.ndim != 1 or len(indices) == 0 or np.any(indices < 0):
            raise ValueError("Expected a nonempty vector of nonnegative global window IDs")
        encoded = indices.astype("<i8", copy=False).tobytes(order="C")
        self._stream.write(encoded)
        self._digest.update(encoded)
        self._windows += len(indices)
        self._batches += 1
        self._step = int(trainer.global_step)
        ids = indices.tolist()
        self._first_ids = (self._first_ids + ids)[:16]
        self._last_ids = (self._last_ids + ids)[-16:]
        self._last_batch_size = len(ids)

        if torch.is_tensor(outputs):
            outputs = {"loss": outputs}
        if not isinstance(outputs, dict) or "loss" not in outputs:
            raise ValueError("Training output must include a scalar loss")
        losses = {}
        for name, value in outputs.items():
            if name != "loss" and not name.endswith("_loss"):
                continue
            if torch.is_tensor(value):
                if value.numel() != 1:
                    raise ValueError(f"Expected scalar training metric: {name}")
                value = value.detach().float().cpu().item()
            value = float(value)
            if not math.isfinite(value):
                self._nonfinite_loss_name = name
                self._last_losses = {**losses, name: None}
                raise FloatingPointError(f"Nonfinite training loss: {name}")
            losses[name] = value
        self._last_losses = losses
        if self._step != self._batches:
            raise ValueError("Expected exactly one optimizer step per recorded training batch")
        if self._step > self.expected_steps or self._windows > self.expected_windows:
            raise ValueError("Training exceeded the declared one-pass experiment budget")
        if self._step % 100 == 0 or self._step == self.expected_steps:
            self._flush()
            _write_json(self.progress_path, self._snapshot("running"), overwrite=True)

    def on_fit_end(self, trainer, pl_module):
        self._step = int(trainer.global_step)
        if not (self._step == self.expected_steps and self._batches == self.expected_steps and self._windows == self.expected_windows):
            error = RuntimeError(f"Incomplete training: steps {self._step}/{self.expected_steps}, windows {self._windows}/{self.expected_windows}")
            self.on_exception(trainer, pl_module, error)
            raise error
        self._flush()
        self._stream.close()
        summary = self._snapshot("complete")
        _write_json(self.summary_path, summary)
        _write_json(self.progress_path, summary, overwrite=True)
        self._finished = True

    def on_exception(self, trainer, pl_module, exception):
        if self._finished:
            return
        self._step = int(getattr(trainer, "global_step", self._step))
        try:
            self._flush()
        finally:
            self._stream.close()
        summary = self._snapshot("failed", exception=exception)
        if not self.summary_path.exists():
            _write_json(self.summary_path, summary)
        _write_json(self.progress_path, summary, overwrite=True)
        self._finished = True
