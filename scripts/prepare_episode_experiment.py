"""Prepare a shared 50% train / 30% test / 20% unused episode manifest."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from episode_data import load_episode_experiment, prepare_episode_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="/home/adam/lewm/storage/datasets/tworoom.h5")
    parser.add_argument("--output-dir", default="/home/adam/lewm/experiments/episode50-30-20261106-01")
    parser.add_argument("--split-seed", type=int, default=20261106)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    path = prepare_episode_experiment(args.dataset, args.output_dir, split_seed=args.split_seed, batch_size=args.batch_size)
    manifest = load_episode_experiment(path)["manifest"]
    print(json.dumps({"manifest": str(path), "manifest_sha256": manifest["manifest_sha256"],
                      "counts": manifest["counts"], "training_order": manifest["training_order"],
                      "normalization": manifest["normalization"]}, indent=2))


if __name__ == "__main__":
    main()
