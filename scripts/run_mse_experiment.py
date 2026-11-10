"""Train, audit, and score the MSE-only Two-Room baseline using a frozen source snapshot."""
import os
from pathlib import Path
import shutil
import sys

from audit_mse_experiment import LABEL, MODEL_NAME
from run_episode_comparison import ROOT, Pipeline, read_json, run_locked, train_command, verify_source

EXPERIMENT = ROOT / "experiments/episode50-mse-20261107-01"
REFERENCE = ROOT / "experiments/episode50-30-20261106-03"
COMBINED = ROOT / "experiments/episode50-combined-20261107-01"


class MSEPipeline(Pipeline):
    experiment = EXPERIMENT
    host_output = Path("/mnt/c/Users/Adam/Desktop/AI/lewm-wsl/experiments") / EXPERIMENT.name

    def run(self):
        verify_source(self.experiment)
        source = self.experiment / "source"
        scripts = source / "scripts"
        run_dir = self.experiment / "runs" / LABEL
        env = dict(os.environ, LEWM_REPO=str(source), PYTHONPATH=str(source), PYTHONUNBUFFERED="1",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        manifest = read_json(self.experiment / "manifest.json")
        expected = manifest["training_order"]["steps_per_epoch"]
        done = run_dir / "runcompletion.json"
        if not done.exists():
            command = train_command(LABEL, "lewm_mse_only", MODEL_NAME, self.experiment)
            checkpoint = run_dir / "checkpoints/last.ckpt"
            if checkpoint.exists():
                command.append(f"+resume_from={checkpoint}")
            elif (run_dir / "initialization.json").exists():
                raise RuntimeError("MSE-only initialization exists without a native checkpoint")
            train_env = dict(env, SPT_CACHE_DIR=str(ROOT / "training" / MODEL_NAME))
            self.child(f"train_{LABEL}", command, run_dir / "console.log", train_env,
                       run_dir / "progress.json")

        completion = read_json(done)
        if not completion["passed"] or completion["global_step"] != expected or completion["objective"] != LABEL:
            raise ValueError("MSE-only training did not finish the expected optimizer updates")
        verify_source(self.experiment)
        shared_args = ["--experiment-dir", str(self.experiment), "--reference-dir", str(REFERENCE),
                       "--model", MODEL_NAME]
        self.child(f"audit_{LABEL}", [sys.executable, "-u", str(scripts / "audit_mse_experiment.py"), *shared_args],
                   run_dir / "audit-console.log", env)
        if not read_json(run_dir / "final-audit.json")["passed"]:
            raise ValueError("MSE-only training audit failed")

        output = self.experiment / "evaluation"
        self.child("evaluate_next_embedding",
                   [sys.executable, "-u", str(scripts / "evaluate_mse_experiment.py"), *shared_args,
                    "--combined-reference-dir", str(COMBINED)],
                   output / "console.log", env, output / "progress.json")
        results = read_json(output / "results.json")
        counts = manifest["counts"]["test"]
        if not results["passed"] or results["test_episodes"] != counts["episodes"] or results["test_windows"] != counts["windows"]:
            raise ValueError("MSE-only evaluation did not cover the sealed test split")
        if set(results["test_latent_variation"]) != {"bf16_mixed", "fp32"}:
            raise ValueError("Missing held-out encoder variation diagnostic")

        shutil.copytree(run_dir, self.host_output / "runs" / LABEL, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("checkpoints"))
        shutil.copytree(output, self.host_output / "evaluation", dirs_exist_ok=True)
        shutil.copy2(self.experiment / "report.txt", self.host_output / "report.txt")
        self.state.update(status="complete", phase="complete", finished_utc=results["finished_utc"],
                          results=str(output / "results.json"), report=str(self.host_output / "report.txt"))
        self.state.pop("child_progress", None)
        self.publish()


if __name__ == "__main__":
    run_locked(MSEPipeline)

