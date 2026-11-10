"""Verify the official Two-Room archive and extract its HDF5 dataset."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile

import zstandard

from runtime_paths import DATASET_ROOT, LOGS, RUNTIME


ARCHIVE_SHA256 = "494b1a02f0765cd9a0d9daf1786c419ced1009977fc45d01e3158932f8d080ca"
BLOCK = 16 * 1024 * 1024


def digest(stream):
    value = hashlib.sha256()
    while block := stream.read(BLOCK):
        value.update(block)
    return value.hexdigest()


def extract(archive, destination):
    with archive.open("rb") as source:
        if digest(source) != ARCHIVE_SHA256:
            raise ValueError(f"Archive SHA256 mismatch for {archive}")
        source.seek(0)
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / "tworoom.h5"
        report = None
        with zstandard.ZstdDecompressor().stream_reader(source) as reader, tarfile.open(fileobj=reader, mode="r|") as tar:
            for member in tar:
                if member.isdir():
                    continue
                if not member.isfile() or member.name not in ("tworoom.h5", "tworoom.hdf5") or report is not None:
                    raise ValueError(f"Unexpected archive member {member.name!r}")
                with tar.extractfile(member) as stream:
                    if output.exists():
                        with output.open("rb") as existing:
                            if output.stat().st_size != member.size or digest(existing) != digest(stream):
                                raise FileExistsError(f"Existing {output} differs from the archive. Choose a separate destination.")
                    else:
                        if shutil.disk_usage(destination).free <= member.size + 5 * 1024**3:
                            raise OSError("Insufficient free space for the dataset and a 5 GiB reserve")
                        with tempfile.NamedTemporaryFile(dir=destination, suffix=".partial", delete=False) as tmp:
                            partial = Path(tmp.name)
                        try:
                            with partial.open("wb") as data:
                                shutil.copyfileobj(stream, data, length=BLOCK)
                            if partial.stat().st_size != member.size:
                                raise OSError("Extracted HDF5 file has an incorrect size")
                            # A hard link publishes the complete file and fails if another process
                            # has already created that destination. The temporary name is then removed.
                            os.link(partial, output)
                        finally:
                            partial.unlink(missing_ok=True)
                report = {"archive": str(archive), "archive_sha256": ARCHIVE_SHA256,
                          "dataset": str(output), "bytes": member.size}
        if report is None:
            raise ValueError("No Two-Room HDF5 member in the archive")
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=RUNTIME / "downloads" / "tworoom.tar.zst")
    parser.add_argument("--destination", type=Path, default=DATASET_ROOT / "datasets")
    args = parser.parse_args()
    record = extract(args.archive, args.destination)
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "dataset-extraction.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
