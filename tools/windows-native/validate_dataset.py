"""Read Two-Room episode metadata and either sampled image rows or every compressed pixel chunk."""
import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin  # Registers the dataset's compression filters.
import numpy as np

from runtime_paths import DATASET_ROOT, LOGS, is_wsl_share


def validate(path, full=False):
    with h5py.File(path, "r", locking=False if is_wsl_share(path) else None) as dataset:
        lengths, offsets = dataset["ep_len"][:], dataset["ep_offset"][:]
        frames = dataset["pixels"]
        if (not len(lengths) or np.any(lengths <= 0) or int(lengths.sum()) != len(frames)
                or not np.array_equal(offsets, np.r_[0, np.cumsum(lengths[:-1])])):
            raise ValueError("Inconsistent episode lengths or frame offsets")
        if frames.dtype != np.uint8 or frames.shape[1:] != (224, 224, 3):
            raise ValueError(f"Unexpected pixels shape or type: {frames.shape}, {frames.dtype}")
        sample = np.unique(np.linspace(0, len(frames) - 1, 8, dtype=np.int64))
        if full:
            block = frames.chunks[0] if frames.chunks else 64
            for start in range(0, len(frames), block):
                _ = frames[start:start + block]
        else:
            _ = frames[sample]
        return {"passed": True, "path": str(path), "file_bytes": path.stat().st_size,
                "frames": len(frames), "episodes": len(lengths), "episode_length_min": int(lengths.min()),
                "episode_length_max": int(lengths.max()), "episode_length_mean": float(lengths.mean()),
                "metadata_consistent": True, "checked_all_pixel_chunks": full,
                "pixel_rows_read": len(frames) if full else len(sample),
                "columns": {key: {"shape": list(col.shape), "dtype": str(col.dtype)} for key, col in dataset.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_ROOT / "datasets" / "tworoom.h5")
    parser.add_argument("--full", action="store_true", help="Decode every frame, including every compressed pixel chunk")
    parser.add_argument("--no-report", action="store_true", help="Print the verification result without writing a file")
    args = parser.parse_args()
    result = validate(args.dataset, args.full)
    if not args.no_report:
        LOGS.mkdir(parents=True, exist_ok=True)
        (LOGS / "dataset-validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "columns"}, indent=2))
