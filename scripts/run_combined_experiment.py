"""Train a fresh combined model using the split and optimizer settings of the paired run."""
import os
import shutil
import sys
from pathlib import Path

from run_episode_comparison import (
    ROOT, Pipeline, read_json, run_locked, train_command, verify_source, sha,
)

EXPERIMENT = ROOT / "experiments/episode50-combined-20261107-01"
REFERENCE = ROOT / "experiments/episode50-30-20261106-03"
MODEL = "tworoom-combined-episode50-20261107-01"


class CombinedPipeline(Pipeline):
    experiment = EXPERIMENT
    host_output = Path("/mnt/c/Users/Adam/Desktop/AI/lewm-wsl/experiments") / EXPERIMENT.name

    def run(self):
        verify_source(self.experiment)
        source = self.experiment / "source"
        run_dir = self.experiment / "runs/combined"
        env = dict(os.environ, LEWM_REPO=str(source), PYTHONPATH=str(source),
                   PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        manifest = read_json(self.experiment / "manifest.json")
        expected_steps = manifest["training_order"]["steps_per_epoch"]
        completion_path = run_dir / "runcompletion.json"

        if not completion_path.exists():
            command = train_command("combined", "lewm_combined", MODEL, self.experiment)
            last = run_dir / "checkpoints/last.ckpt"
            if last.exists():
                command.append(f"+resume_from={last}")
            elif (run_dir / "initialization.json").exists():
                raise RuntimeError("A previous initialization exists without a native checkpoint")
            train_env = dict(env, SPT_CACHE_DIR=str(ROOT / "training" / MODEL))
            self.child("train_combined", command, run_dir / "console.log", train_env,
                       run_dir / "progress.json")

        completion = read_json(completion_path)
        if not completion["passed"] or completion["global_step"] != expected_steps:
            raise ValueError("Combined model did not complete the full training pass")
        verify_source(self.experiment)
        post_files = read_json(self.experiment / "evaluation-source-snapshot.json")["files_sha256"]
        for name in ("audit_combined_experiment.py", "evaluate_combined_experiment.py"):
            if sha(source / "scripts" / name) != post_files["scripts/" + name]:
                raise ValueError(f"Evaluation source hash mismatch: {name}")
        args = ["--experiment-dir", str(self.experiment), "--reference-dir", str(REFERENCE)]
        scripts = source / "scripts"
        self.child("audit_combined", [sys.executable, "-u", str(scripts / "audit_combined_experiment.py"), *args],
                   run_dir / "audit-console.log", env)
        if not read_json(run_dir / "final-audit.json")["passed"]:
            raise ValueError("Combined training audit failed")

        output = self.experiment / "evaluation"
        self.child("evaluate_next_embedding",
                   [sys.executable, "-u", str(scripts / "evaluate_combined_experiment.py"), *args,
                    "--model", MODEL],
                   output / "console.log", env, output / "progress.json")
        results = read_json(output / "results.json")
        counts = manifest["counts"]["test"]
        if not results["passed"] or results["test_episodes"] != counts["episodes"] or results["test_windows"] != counts["windows"]:
            raise ValueError("Evaluation did not cover the expected held-out episodes and windows")

        shutil.copytree(run_dir, self.host_output / "runs/combined", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("checkpoints"))
        shutil.copytree(output, self.host_output / "evaluation", dirs_exist_ok=True)
        shutil.copy2(self.experiment / "report.txt", self.host_output / "report.txt")
        self.state.update(status="complete", phase="complete", finished_utc=results["finished_utc"],
                          results=str(output / "results.json"), report=str(self.host_output / "report.txt"))
        self.state.pop("child_progress", None)
        self.publish()


if __name__ == "__main__":
    run_locked(CombinedPipeline)
