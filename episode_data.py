"""Immutable episode splits and train-only normalization for paired experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 - registers the dataset's compression filters
import numpy as np
import torch


SCHEMA_VERSION = 1
SPLITS = ("train", "test", "unused")


def _json_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _array_hash(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)},
                             sort_keys=True, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_metadata(path):
    path = Path(path).expanduser().resolve(strict=True)
    stat = path.stat()
    with h5py.File(path, "r") as h5:
        lengths = np.asarray(h5["ep_len"], dtype=np.int64)
        offsets = np.asarray(h5["ep_offset"], dtype=np.int64)
        columns = {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                   for name, value in h5.items()}
    if lengths.ndim != 1 or offsets.shape != lengths.shape or np.any(lengths <= 0):
        raise ValueError("Invalid episode lengths/offsets")
    if not np.array_equal(offsets, np.r_[0, np.cumsum(lengths)[:-1]]):
        raise ValueError("Expected contiguous episode offsets matching episode lengths")
    for name in ("pixels", "action", "proprio"):
        if columns[name]["shape"][0] != int(lengths.sum()):
            raise ValueError(f"Dataset column {name} does not match episode row counts")
    metadata = {
        "path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "columns": columns, "episode_count": len(lengths),
        "row_count": int(lengths.sum()),
        "episode_lengths_sha256": _array_hash(lengths),
        "episode_offsets_sha256": _array_hash(offsets),
        "identity_scope": "file size/mtime, column shapes/dtypes, complete episode lengths/offsets; not a full pixel-content hash",
    }
    metadata["metadata_sha256"] = _json_hash(metadata)
    return metadata, lengths, offsets


def _clip_mapping(lengths, span):
    counts = np.maximum(lengths - span + 1, 0)
    episodes = np.repeat(np.arange(len(lengths), dtype=np.int64), counts)
    starts = np.concatenate([np.arange(int(count), dtype=np.int64)
                             for count in counts])
    return episodes, starts


def prepare_episode_experiment(dataset_path, output_dir, *, split_seed=20261106,
                               train_fraction=0.5, test_fraction=0.3,
                               frameskip=5, num_steps=4, loader_seed=3072,
                               batch_size=8):
    """Write a sealed manifest and every global clip index; never reuse outputs."""
    output_dir = Path(output_dir).expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    arrays_path = output_dir / "arrays.npz"
    if manifest_path.exists() or arrays_path.exists():
        raise FileExistsError(f"Experiment manifest already exists in {output_dir}")
    if not (0 < train_fraction < 1 and 0 < test_fraction < 1
            and train_fraction + test_fraction <= 1):
        raise ValueError("Invalid episode split fractions")
    if min(frameskip, num_steps, batch_size) < 1:
        raise ValueError("Frame spacing, sequence length, and batch size must be positive")

    metadata, lengths, offsets = _dataset_metadata(dataset_path)
    n_episodes = len(lengths)
    train_count = int(math.floor(n_episodes * train_fraction))
    test_count = int(math.floor(n_episodes * test_fraction))
    if min(train_count, test_count) == 0:
        raise ValueError("Training and test partitions must contain episodes")
    permutation = np.random.default_rng(split_seed).permutation(n_episodes).astype(np.int64)
    span = int(frameskip * num_steps)
    clip_episodes, clip_starts = _clip_mapping(lengths, span)
    arrays = {
        "episode_permutation": permutation,
        "episode_lengths": lengths,
        "episode_offsets": offsets,
        "clip_episode_ids": clip_episodes,
        "clip_start_steps": clip_starts,
        "train_episode_ids": permutation[:train_count],
        "test_episode_ids": permutation[train_count:train_count + test_count],
        "unused_episode_ids": permutation[train_count + test_count:],
    }
    counts = {}
    for split in SPLITS:
        ids = arrays[f"{split}_episode_ids"]
        arrays[f"{split}_clip_indices"] = np.flatnonzero(np.isin(clip_episodes, ids))
        counts[split] = {"episodes": len(ids), "rows": int(lengths[ids].sum()),
                         "windows": len(arrays[f"{split}_clip_indices"])}

    # Select raw rows once: overlapping clips must not reweight normalization.
    train_episode_mask = np.isin(np.arange(n_episodes), arrays["train_episode_ids"])
    train_row_mask = np.repeat(train_episode_mask, lengths)
    normalization = {}
    with h5py.File(metadata["path"], "r") as h5:
        for column in ("action", "proprio"):
            selected = np.asarray(h5[column])[train_row_mask]
            if selected.ndim != 2:
                raise ValueError(f"Expected a matrix for normalization column {column}")
            valid = ~np.isnan(selected).any(axis=1)
            tensor = torch.from_numpy(selected[valid].copy())
            if len(tensor) < 2 or not torch.isfinite(tensor).all():
                raise ValueError(f"Insufficient/nonfinite training values in {column}")
            mean = tensor.mean(0, keepdim=True)
            std = tensor.std(0, keepdim=True, correction=1)
            if not torch.isfinite(std).all() or not (std > 0).all():
                raise ValueError(f"Invalid training standard deviation for {column}")
            normalization[column] = {
                "mean": mean.tolist(), "std": std.tolist(),
                "dtype": str(selected.dtype), "correction": 1,
                "source_split": "train", "source_unit": "raw rows, each counted once",
                "valid_rows": int(valid.sum()), "excluded_nan_rows": int((~valid).sum()),
                "train_episode_ids_sha256": _array_hash(arrays["train_episode_ids"]),
                "selected_train_values_sha256": _array_hash(selected[valid]),
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_arrays = output_dir / "arrays.npz.tmp"
    with open(temporary_arrays, "wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary_arrays, arrays_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": metadata,
        "split": {"seed": int(split_seed), "rng": "numpy.random.default_rng/PCG64",
                  "train_fraction": train_fraction, "test_fraction": test_fraction,
                  "unused_fraction": 1 - train_fraction - test_fraction,
                  "membership": "ordered episode permutation slices",
                  "clip_order": "global dataset index ascending within each split"},
        "window": {"frameskip": int(frameskip), "num_steps": int(num_steps),
                   "span": span, "total_windows": len(clip_episodes)},
        "counts": counts,
        "training_order": {"seed": int(loader_seed), "shuffle": True,
                           "drop_last": False, "batch_size": int(batch_size),
                           "steps_per_epoch": math.ceil(counts["train"]["windows"] / batch_size),
                           "instruction": "Use a fresh independent torch.Generator with this seed for each model; the episode split does not consume it."},
        "normalization": normalization,
        "arrays_file": arrays_path.name,
        "arrays_sha256": _file_hash(arrays_path),
        "array_hashes": {name: _array_hash(value) for name, value in arrays.items()},
    }
    manifest["manifest_sha256"] = _json_hash(manifest)
    temporary_manifest = output_dir / "manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    os.replace(temporary_manifest, manifest_path)
    load_episode_experiment(manifest_path)
    return manifest_path


def load_episode_experiment(manifest_path, dataset=None):
    """Verify manifest, array hashes, episode membership, and dataset clip mapping."""
    manifest_path = Path(manifest_path).expanduser().resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported episode manifest schema")
    digest = manifest.get("manifest_sha256")
    if digest != _json_hash({k: v for k, v in manifest.items() if k != "manifest_sha256"}):
        raise ValueError("Manifest hash mismatch")
    arrays_path = (manifest_path.parent / manifest["arrays_file"]).resolve(strict=True)
    if arrays_path.parent != manifest_path.parent:
        raise ValueError("Array archive must be beside the manifest")
    if _file_hash(arrays_path) != manifest["arrays_sha256"]:
        raise ValueError("Array archive hash mismatch")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if set(arrays) != set(manifest["array_hashes"]):
        raise ValueError("Array archive keys do not match manifest")
    for name, value in arrays.items():
        if value.dtype != np.int64 or value.ndim != 1:
            raise ValueError(f"Expected a one-dimensional int64 index array: {name}")
        if _array_hash(value) != manifest["array_hashes"][name]:
            raise ValueError(f"Array hash mismatch: {name}")

    dataset_path = dataset.h5_path if dataset is not None else manifest["dataset"]["path"]
    metadata, lengths, offsets = _dataset_metadata(dataset_path)
    if metadata != manifest["dataset"]:
        raise ValueError("Dataset metadata changed since split preparation")
    if not np.array_equal(lengths, arrays["episode_lengths"]) or not np.array_equal(offsets, arrays["episode_offsets"]):
        raise ValueError("Saved episode mapping differs from the dataset")
    permutation = np.random.default_rng(manifest["split"]["seed"]).permutation(len(lengths))
    if not np.array_equal(permutation, arrays["episode_permutation"]):
        raise ValueError("Episode permutation does not match the declared seed")
    window = manifest["window"]
    if window["span"] != window["frameskip"] * window["num_steps"]:
        raise ValueError("Invalid window span")
    clip_episodes, clip_starts = _clip_mapping(lengths, window["span"])
    if len(clip_episodes) != window["total_windows"]:
        raise ValueError("Incorrect total window count")
    if not np.array_equal(clip_episodes, arrays["clip_episode_ids"]) or not np.array_equal(clip_starts, arrays["clip_start_steps"]):
        raise ValueError("Saved clip mapping does not match episode boundaries")

    start = 0
    for split in SPLITS:
        counts = manifest["counts"][split]
        ids = arrays[f"{split}_episode_ids"]
        if not np.array_equal(ids, permutation[start:start + counts["episodes"]]):
            raise ValueError(f"Invalid or overlapping {split} episode membership")
        start += counts["episodes"]
        indices = np.flatnonzero(np.isin(clip_episodes, ids))
        if not np.array_equal(indices, arrays[f"{split}_clip_indices"]):
            raise ValueError(f"Missing, out-of-split, or unsorted {split} windows")
        if len(indices) != counts["windows"] or int(lengths[ids].sum()) != counts["rows"]:
            raise ValueError(f"Incorrect {split} counts")
    if start != len(lengths):
        raise ValueError("Episode partitions do not cover the dataset")
    for column, stats in manifest["normalization"].items():
        if stats["source_split"] != "train" or stats["correction"] != 1:
            raise ValueError(f"Invalid normalization provenance: {column}")
        if stats["train_episode_ids_sha256"] != _array_hash(arrays["train_episode_ids"]):
            raise ValueError(f"Normalization uses different training episodes: {column}")
        if stats["valid_rows"] + stats["excluded_nan_rows"] != manifest["counts"]["train"]["rows"]:
            raise ValueError(f"Incorrect normalization row count: {column}")
        if not np.isfinite(stats["mean"]).all() or not (np.asarray(stats["std"]) > 0).all() or not np.isfinite(stats["std"]).all():
            raise ValueError(f"Invalid saved normalization statistics: {column}")
    if dataset is not None:
        if dataset.frameskip != window["frameskip"] or dataset.num_steps != window["num_steps"]:
            raise ValueError("Dataset window settings differ from the experiment")
        mapping = np.asarray(dataset.clip_indices, dtype=np.int64)
        expected = np.column_stack([clip_episodes, clip_starts])
        if len(dataset) != len(clip_episodes) or not np.array_equal(mapping, expected):
            raise ValueError("Dataset implementation uses a different clip index mapping")
    for value in arrays.values():
        value.flags.writeable = False
    return {"manifest_path": str(manifest_path), "manifest": manifest, "arrays": arrays}


def episode_subset(dataset, bundle, split):
    """Return all windows from one verified partition in canonical index order."""
    if split not in SPLITS:
        raise ValueError(f"Unknown episode split: {split}")
    return torch.utils.data.Subset(dataset, bundle["arrays"][f"{split}_clip_indices"].tolist())


def get_episode_normalizer(bundle, column, source=None, target=None):
    """Use the saved train-only statistics in the existing preprocessing path."""
    from stable_pretraining import data as dt
    from utils import ZScoreNormalizer

    stats = bundle["manifest"]["normalization"][column]
    dtype = getattr(torch, stats["dtype"])
    mean = torch.tensor(stats["mean"], dtype=dtype)
    std = torch.tensor(stats["std"], dtype=dtype)
    return dt.transforms.WrapTorchTransform(
        ZScoreNormalizer(mean, std), source=source or column, target=target or column)
