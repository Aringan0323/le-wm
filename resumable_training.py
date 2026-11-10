"""Durable batch-boundary checkpoints with explicit sample and RNG cursors.

DataLoader prefetch never commits samples. The caller commits only after a full
optimizer/scheduler step, then may checkpoint or honor a pause request. Resume
requires deterministic preprocessing and a dedicated DataLoader generator.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _indices(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iu" or np.any(array < 0):
        raise ValueError("Expected one-dimensional nonnegative integer window IDs")
    return np.ascontiguousarray(array, dtype="<i8")


def order_sha256(order):
    return hashlib.sha256(_indices(order).tobytes()).hexdigest()


def make_fixed_order(train_indices, seed=3072):
    """Match the original seeded DataLoader base-seed draw, then its randperm."""
    indices = _indices(train_indices)
    generator = torch.Generator().manual_seed(seed)
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    permutation = torch.randperm(len(indices), generator=generator).numpy()
    return indices[permutation].copy()


class EpochOrder:
    """A deterministic schedule requiring only two cached epoch permutations (for prefetch)."""
    def __init__(self, indices, seed, epochs):
        from collections import OrderedDict
        self.indices = _indices(indices).copy()
        self.indices.flags.writeable = False
        self.seed, self.epochs = int(seed), int(epochs)
        self.windows_per_epoch = len(self.indices)
        if not self.windows_per_epoch or self.epochs < 1 or len(np.unique(self.indices)) != self.windows_per_epoch:
            raise ValueError("Each epoch must contain a nonempty set of distinct windows")
        self._cache = OrderedDict()
        digest = hashlib.sha256()
        epoch_hashes = []
        for epoch in range(self.epochs):
            order = make_fixed_order(self.indices, self.seed + epoch)
            digest.update(order)
            epoch_hashes.append(hashlib.sha256(order).hexdigest())
        self.order_hash = digest.hexdigest()
        self.protocol = {"algorithm": "torch_base_seed_draw_randperm_seed_plus_epoch_v1", "seed": self.seed,
                         "epochs": self.epochs, "windows_per_epoch": self.windows_per_epoch,
                         "train_indices_sha256": order_sha256(self.indices), "epoch_sha256": epoch_hashes}

    def __len__(self):
        return self.windows_per_epoch * self.epochs

    def epoch(self, epoch):
        if epoch not in self._cache:
            indices = make_fixed_order(self.indices, self.seed + epoch)
            indices.flags.writeable = False
            self._cache[epoch] = indices
            if len(self._cache) > 2:
                self._cache.popitem(last=False)
        self._cache.move_to_end(epoch)
        return self._cache[epoch]

    def __getitem__(self, key):
        start, stop, stride = key.indices(len(self))
        if stride != 1:
            raise ValueError("A training schedule supports only contiguous slices")
        parts = []
        while start < stop:
            epoch, offset = divmod(start, self.windows_per_epoch)
            amount = min(stop - start, self.windows_per_epoch - offset)
            parts.append(self.epoch(epoch)[offset:offset + amount])
            start += amount
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts) if parts else np.empty(0, dtype="<i8")


class FixedOrderBatchSampler(torch.utils.data.Sampler):
    def __init__(self, order, batch_size, cursor=0):
        if isinstance(order, EpochOrder):
            self.order = order
            self.windows_per_epoch = order.windows_per_epoch
            self.epochs = order.epochs
        else:
            self.order = _indices(order).copy()
            if len(np.unique(self.order)) != len(self.order):
                raise ValueError("A one-pass training order cannot repeat windows")
            self.order.flags.writeable = False
            self.windows_per_epoch, self.epochs = len(self.order), 1
        self.batch_size = int(batch_size)
        if self.batch_size < 1 or not self.windows_per_epoch:
            raise ValueError("Batch size and the number of windows must be positive")
        self.steps_per_epoch = math.ceil(self.windows_per_epoch / self.batch_size)
        self.total_steps = self.steps_per_epoch * self.epochs
        self.restore_cursor(cursor)

    def restore_cursor(self, cursor):
        cursor = int(cursor)
        if cursor < 0 or cursor > len(self.order) or (cursor % self.windows_per_epoch) % self.batch_size:
            raise ValueError("Checkpoint cursor is not a valid epoch-local batch boundary")
        self.cursor = cursor

    @property
    def global_step(self):
        epoch, offset = divmod(self.cursor, self.windows_per_epoch)
        return epoch * self.steps_per_epoch + math.ceil(offset / self.batch_size)

    @property
    def last_batch_size(self):
        if not self.cursor:
            return 0
        within = self.cursor % self.windows_per_epoch or self.windows_per_epoch
        return within % self.batch_size or self.batch_size

    def __len__(self):
        return self.total_steps - self.global_step

    def batch_end(self, start):
        next_epoch = (start // self.windows_per_epoch + 1) * self.windows_per_epoch
        return min(start + self.batch_size, next_epoch)

    def __iter__(self):
        # Each epoch retains its partial last batch; prefetch never commits the cursor.
        start = self.cursor
        while start < len(self.order):
            end = self.batch_end(start)
            yield self.order[start:end].tolist()
            start = end

    def commit(self, indices):
        indices = _indices(indices)
        expected = self.order[self.cursor:self.batch_end(self.cursor)]
        if not len(indices) or not np.array_equal(indices, expected):
            raise ValueError("Completed batch does not match the next committed window IDs")
        self.cursor += len(indices)


class IndexedWindowDataset(torch.utils.data.Dataset):
    """Wrap the full dataset; sampler indices and reported IDs are both global."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if not isinstance(sample, dict) or "_window_index" in sample:
            raise ValueError("Expected a dictionary without the reserved _window_index key")
        return {**sample, "_window_index": torch.tensor(int(index), dtype=torch.int64)}

    def __getitems__(self, indices):
        return [self[index] for index in indices]


def capture_rng_state(synchronize=True):
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch_cpu": torch.get_rng_state().clone(), "torch_cuda": None}
    if torch.cuda.is_initialized():
        if synchronize:
            torch.cuda.synchronize()
        state["torch_cuda"] = [item.clone().cpu() for item in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"] is not None:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from the saved RNG state")
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


def pause_requested(control_file):
    if not control_file or not Path(control_file).exists():
        return False
    control = json.loads(Path(control_file).read_text())
    request = control.get("request", "run")
    if request not in ("run", "pause"):
        raise ValueError(f"Unknown training control request: {request}")
    return request == "pause"


class DurableTrainingTrace:
    def __init__(self, run_dir, order, resume=None):
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "training-indices.i64"
        self.order = order if isinstance(order, EpochOrder) else _indices(order)
        self.cursor = int(resume["cursor"]) if resume else 0
        self.digest = hashlib.sha256()
        if resume is None:
            self.stream = self.path.open("xb")
        else:
            length = self.cursor * 8
            if not self.path.exists() or self.path.stat().st_size < length:
                raise ValueError("Training trace is missing committed checkpoint samples")
            with self.path.open("rb") as source:
                for start in range(0, self.cursor, 65536):
                    expected = self.order[start:min(self.cursor, start + 65536)]
                    raw = source.read(expected.nbytes)
                    if raw != expected.tobytes():
                        raise ValueError("Training trace prefix does not match the checkpoint")
                    self.digest.update(raw)
                if self.digest.hexdigest() != resume["trace_sha256"]:
                    raise ValueError("Training trace checksum differs from the checkpoint")
                tail = self.path.stat().st_size - length
                if tail:
                    backup = self.run_dir / f"training-indices.abandoned-{uuid.uuid4().hex}.i64"
                    with backup.open("xb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                        output.flush()
                        os.fsync(output.fileno())
            if tail:
                with self.path.open("r+b") as output:
                    output.truncate(length)
                    output.flush()
                    os.fsync(output.fileno())
                _fsync_directory(self.run_dir)
            self.stream = self.path.open("ab")

    def append(self, indices):
        indices = _indices(indices)
        if len(indices) == 0 or not np.array_equal(indices, self.order[self.cursor:self.cursor + len(indices)]):
            raise ValueError("Trace batch is not the next committed sample sequence")
        raw = indices.tobytes()
        self.stream.write(raw)
        self.digest.update(raw)
        self.cursor += len(indices)

    def flush(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())
        _fsync_directory(self.run_dir)

    def summary(self):
        return {"trace_file": self.path.name, "trace_dtype": "<i8",
                "windows_seen": self.cursor, "trace_bytes": self.cursor * 8,
                "order_sha256": self.digest.hexdigest(),
                "first_ids": self.order[:min(self.cursor, 16)].tolist(),
                "last_ids": self.order[max(0, self.cursor - 16):self.cursor].tolist()}

    def close(self):
        if not self.stream.closed:
            self.flush()
            self.stream.close()


class CheckpointManager:
    """Atomic checkpoint generations; last.ckpt is promoted only after validation.

    Immutable generations and checksum sidecars are retained. last.ckpt/final.ckpt
    are atomic hard-link aliases, so a crash cannot separate their bytes from the
    generation's checksum. CPU tests never initialize CUDA merely to save RNG.
    """
    def __init__(self, run_dir, manifest_sha256, config_sha256, batch_size, order,
                 interval_seconds=300, clock=time.monotonic):
        self.run_dir = Path(run_dir).resolve()
        self.directory = self.run_dir / "checkpoints"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_sha256 = str(manifest_sha256)
        self.config_sha256 = str(config_sha256)
        self.batch_size = int(batch_size)
        self.multi_epoch = isinstance(order, EpochOrder)
        self.order = order if self.multi_epoch else _indices(order).copy()
        self.order_hash = order.order_hash if self.multi_epoch else order_sha256(self.order)
        self.interval_seconds = float(interval_seconds)
        if self.interval_seconds <= 0:
            raise ValueError("Checkpoint interval must be positive")
        self.clock = clock
        self.last_saved_at = None
        self.latest_record = None

    def due(self):
        return self.last_saved_at is None or self.clock() - self.last_saved_at >= self.interval_seconds

    def _promote(self, generation, name):
        target = self.directory / name
        temporary = self.directory / f".{name}.{uuid.uuid4().hex}.tmp"
        try:
            os.link(generation, temporary)
            os.replace(temporary, target)
            _fsync_directory(self.directory)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def stage(self, model, optimizer, scheduler, sigreg, sampler, trace,
              status="running", extra=None, rng=None):
        """Durably write a generation and checksum without advancing any live alias."""
        if sampler.cursor != trace.cursor:
            raise ValueError("Cannot checkpoint inconsistent model/sample trace cursors")
        same_order = sampler.order is self.order if self.multi_epoch else np.array_equal(sampler.order, self.order)
        if sampler.batch_size != self.batch_size or not same_order:
            raise ValueError("Sampler differs from immutable checkpoint order/batch size")
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        trace.flush()
        generation_id = f"step-{sampler.global_step:08d}-{uuid.uuid4().hex[:12]}"
        saved_utc = utc_now()
        state = {f"model.{key}": value for key, value in model.state_dict().items()}
        if sigreg is not None:
            state.update({f"sigreg.{key}": value for key, value in sigreg.state_dict().items()})
        metadata = {
            "version": 2 if self.multi_epoch else 1, "backend": "native_pytorch", "generation_id": generation_id,
            "status": status, "saved_utc": saved_utc,
            "manifest_sha256": self.manifest_sha256, "config_sha256": self.config_sha256,
            "batch_size": self.batch_size, "cursor": sampler.cursor,
            "total_windows": len(self.order), "expected_steps": sampler.total_steps,
            "order_sha256": self.order_hash,
            "trace_sha256": trace.digest.hexdigest(), "trace_bytes": trace.cursor * 8,
            "rng": capture_rng_state() if rng is None else rng, "extra": extra or {},
        }
        if self.multi_epoch:
            metadata["order_protocol"] = self.order.protocol
        else:
            metadata["order"] = torch.from_numpy(self.order.copy())
        payload = {"state_dict": state, "optimizer_states": [optimizer.state_dict()],
                   "lr_schedulers": [scheduler.state_dict()],
                   "global_step": sampler.global_step, "epoch": max(0, sampler.cursor - 1) // sampler.windows_per_epoch,
                   "resumable": metadata}
        generation = self.directory / f"{generation_id}.ckpt"
        temporary = self.directory / f".{generation_id}.tmp"
        try:
            with temporary.open("xb") as output:
                torch.save(payload, output)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, generation)
            checksum = file_sha256(generation)
            atomic_json(generation.with_suffix(".json"), {
                "sha256": checksum, "size_bytes": generation.stat().st_size,
                "generation_id": generation_id, "global_step": sampler.global_step,
                "windows_seen": sampler.cursor, "saved_utc": saved_utc})
        finally:
            temporary.unlink(missing_ok=True)
        return {"path": str(generation), "global_step": sampler.global_step, "windows_seen": sampler.cursor,
                "saved_utc": saved_utc, "size_bytes": generation.stat().st_size, "sha256": checksum,
                "status": status, "generation_id": generation_id, "config_sha256": self.config_sha256}

    def publish(self, record, final=False, previous_generation=None, initial_generation=None):
        """Promote a staged or recovered generation after its enclosing commit."""
        generation_id = record["generation_id"]
        generation = Path(record["path"])
        if generation.parent != self.directory or generation.name != generation_id + ".ckpt":
            raise ValueError("Staged checkpoint path differs from the run directory")
        if previous_generation is None:
            previous_generation = (self.latest_record or {}).get("generation_id")
        selected = self._promote(generation, "last.ckpt")
        if final:
            selected = self._promote(generation, "final.ckpt")
        elif (self.directory / "final.ckpt").exists():
            # Reconciliation to an earlier valid group generation invalidates a stale final alias.
            (self.directory / "final.ckpt").unlink()
            _fsync_directory(self.directory)
        self.latest_record = dict(record, path=str(selected))
        atomic_json(self.run_dir / "checkpoint.json", self.latest_record)
        if self.multi_epoch:
            # Publish and fsync the new aliases/record before pruning exact known
            # generations. Preserve the initial state and the previous durable save.
            initial = sorted(self.directory.glob("step-00000000-*.ckpt"), key=lambda p: p.stat().st_mtime_ns)
            keep = {generation_id, previous_generation, initial_generation or (initial[0].stem if initial else None)}
            for candidate in self.directory.glob("step-*.ckpt"):
                if candidate.stem in keep:
                    continue
                sidecar = candidate.with_suffix(".json")
                if sidecar.is_file() and json.loads(sidecar.read_text()).get("generation_id") == candidate.stem:
                    candidate.unlink()
            _fsync_directory(self.directory)
        self.last_saved_at = self.clock()
        return selected

    def save(self, model, optimizer, scheduler, sigreg, sampler, trace,
             status="running", extra=None, final=False):
        record = self.stage(model, optimizer, scheduler, sigreg, sampler, trace, status, extra)
        return self.publish(record, final=final)

    def load(self, path, model, optimizer, scheduler, sigreg=None, sampler=None, restore_rng=False):
        # Checkpoints are locally authored trusted artifacts; RNG includes NumPy state.
        path = Path(path).resolve(strict=True)
        if path.parent != self.directory:
            raise ValueError("Resume checkpoint must belong to this run's checkpoint directory")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = payload.get("resumable", {})
        if metadata.get("version") != (2 if self.multi_epoch else 1) or metadata.get("backend") != "native_pytorch":
            raise ValueError("Checkpoint does not contain the native resumable cursor protocol")
        generation_id = metadata["generation_id"]
        if Path(generation_id).name != generation_id:
            raise ValueError("Invalid checkpoint generation ID")
        sidecar = self.directory / f"{generation_id}.json"
        expected_checksum = json.loads(sidecar.read_text())
        if path.stat().st_size != expected_checksum["size_bytes"] or file_sha256(path) != expected_checksum["sha256"]:
            raise ValueError("Checkpoint checksum mismatch")
        for key, expected in (("manifest_sha256", self.manifest_sha256),
                              ("config_sha256", self.config_sha256),
                              ("batch_size", self.batch_size), ("order_sha256", self.order_hash),
                              ("total_windows", len(self.order))):
            if metadata.get(key) != expected:
                raise ValueError(f"Resume experiment mismatch: {key}")
        if self.multi_epoch:
            if metadata.get("order_protocol") != self.order.protocol:
                raise ValueError("Checkpoint epoch schedule differs from the experiment")
        elif not np.array_equal(_indices(metadata["order"]), self.order):
            raise ValueError("Checkpoint order differs from the experiment")
        if sampler is None:
            sampler = FixedOrderBatchSampler(self.order, self.batch_size)
        sampler.restore_cursor(metadata["cursor"])
        if (payload["global_step"] != sampler.global_step or metadata["trace_bytes"] != sampler.cursor * 8
                or metadata["expected_steps"] != sampler.total_steps):
            raise ValueError("Checkpoint step/cursor/trace sizes disagree")
        model.load_state_dict({key[6:]: value for key, value in payload["state_dict"].items()
                               if key.startswith("model.")}, strict=True)
        regularizer_state = {key[7:]: value for key, value in payload["state_dict"].items()
                             if key.startswith("sigreg.")}
        if sigreg is not None:
            sigreg.load_state_dict(regularizer_state, strict=True)
        elif regularizer_state:
            raise ValueError("SIGReg checkpoint cannot resume an objective without SIGReg")
        optimizer.load_state_dict(payload["optimizer_states"][0])
        scheduler.load_state_dict(payload["lr_schedulers"][0])
        if restore_rng:
            restore_rng_state(metadata["rng"])
        self.last_saved_at = self.clock()
        self.latest_record = {"path": str(path), "global_step": sampler.global_step,
                              "windows_seen": sampler.cursor, "saved_utc": metadata["saved_utc"],
                              "size_bytes": path.stat().st_size, "sha256": expected_checksum["sha256"],
                              "status": metadata["status"], "generation_id": generation_id}
        atomic_json(self.run_dir / "checkpoint.json", self.latest_record)
        return payload
