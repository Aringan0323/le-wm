"""Measure one candidate batch on all three resident models without writing to a training run."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import torch
from omegaconf import open_dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_delta_group import (Member, compose_members, load_shared_data, seed_all, transfer_batch)
from resumable_training import EpochOrder, FixedOrderBatchSampler, IndexedWindowDataset


def device_memory():
    smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    command = [smi, "-i", "0", "--query-gpu=memory.used,memory.free,memory.total",
               "--format=csv,noheader,nounits"]
    used, free, total = map(int, subprocess.check_output(command, text=True, timeout=5).strip().split(","))
    return {"used_mib": used, "free_mib": free, "total_mib": total}


def run(root, batch_size, steps, temp):
    configs = compose_members(root)
    first = next(iter(configs.values()))
    seed_all(first.seed)
    torch.set_num_threads(int(first.get("cpu_threads", 1)))
    dataset, bundle = load_shared_data(first, root)
    order = EpochOrder(bundle["arrays"]["train_clip_indices"], first.seed, first.trainer.max_epochs)
    sampler = FixedOrderBatchSampler(order, batch_size)
    device = torch.device("cuda:0")
    baseline = device_memory()
    members = []
    for key, cfg in configs.items():
        with open_dict(cfg):
            cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")
            cfg.loader.batch_size = batch_size
            cfg.run_metadata_dir = str(temp / key / "run")
            cfg.output_model_name = str(temp / key / "export")
        members.append(Member(key, cfg, "calibration", bundle, order, sampler, device, resuming=False))
    dataset = IndexedWindowDataset(dataset)
    # The real, pinned batch exercises the same shapes, dtypes and data fields as production.
    indices = next(iter(sampler))
    host_batch = next(iter(torch.utils.data.DataLoader(dataset, batch_sampler=[indices], pin_memory=True)))
    readings, durations = [], []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        start = time.perf_counter()
        batch = transfer_batch(host_batch, device)
        for member in members:
            member.step(batch, step)
        del batch
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
        readings.append(device_memory())
    stats = torch.cuda.memory_stats()
    return {
        "status": "ok", "batch_size": batch_size, "steps": steps, "device": torch.cuda.get_device_name(),
        "members": [m.key for m in members], "device_before_models": baseline, "device_after_steps": readings,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "device_used_peak_sample_mib": max(r["used_mib"] for r in readings),
        "num_alloc_retries": stats["num_alloc_retries"], "group_compute_seconds": durations,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--steps", default=3, type=int)
    args = parser.parse_args()
    if args.batch_size < 2 or args.steps < 2:
        parser.error("Choose a batch size and number of steps of at least two.")
    root = args.experiment_dir.resolve(strict=True)
    # A calibration must not overlap a live pipeline in the reference experiment.
    with (root / "pipeline.lock").open("rb") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        with TemporaryDirectory(prefix="delta-batch-probe-") as temp:
            try:
                result = run(root, args.batch_size, args.steps, Path(temp))
            except torch.OutOfMemoryError as exc:
                result = {"status": "out_of_memory", "batch_size": args.batch_size, "error": str(exc)}
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
