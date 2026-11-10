"""Train, audit, and test one variant of the matched Two-Room loss comparison."""
import argparse
import os
from pathlib import Path
import shutil
import sys

from combined_sweep_config import COMBINED, MSE_ONLY, REFERENCE, VARIANTS
from run_episode_comparison import ROOT, Pipeline, read_json, run_locked, train_command, verify_source


class SweepPipeline(Pipeline):
    def run(self):
        variant = self.variant
        verify_source(self.experiment)
        source = self.experiment / "source"
        scripts = source / "scripts"
        run_dir = self.experiment / "runs" / variant.label
        env = dict(os.environ, LEWM_REPO=str(source), PYTHONPATH=str(source), PYTHONUNBUFFERED="1",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        manifest = read_json(self.experiment / "manifest.json")
        expected = manifest["training_order"]["steps_per_epoch"]
        done = run_dir / "runcompletion.json"
        if not done.exists():
            command = train_command(variant.label, variant.config, variant.model, self.experiment)
            checkpoint = run_dir / "checkpoints/last.ckpt"
            if checkpoint.exists():
                command.append(f"+resume_from={checkpoint}")
            elif (run_dir / "initialization.json").exists():
                raise RuntimeError(f"{variant.label} initialization exists without a native checkpoint")
            train_env = dict(env, SPT_CACHE_DIR=str(ROOT / "training" / variant.model))
            self.child(f"train_{variant.label}", command, run_dir / "console.log", train_env,
                       run_dir / "progress.json")

        completion = read_json(done)
        if not completion["passed"] or completion["global_step"] != expected or completion["objective"] != variant.objective:
            raise ValueError(f"{variant.label} did not finish the expected optimizer updates")
        verify_source(self.experiment)
        shared_args = ["--variant", variant.label, "--experiment-dir", str(self.experiment),
                       "--reference-dir", str(REFERENCE), "--combined-reference-dir", str(COMBINED)]
        self.child(f"audit_{variant.label}", [sys.executable, "-u", str(scripts / "audit_combined_sweep.py"), *shared_args],
                   run_dir / "audit-console.log", env)
        if not read_json(run_dir / "final-audit.json")["passed"]:
            raise ValueError(f"{variant.label} training audit failed")

        output = self.experiment / "evaluation"
        self.child("evaluate_next_embedding",
                   [sys.executable, "-u", str(scripts / "evaluate_combined_sweep.py"), *shared_args,
                    "--mse-reference-dir", str(MSE_ONLY)],
                   output / "console.log", env, output / "progress.json")
        results = read_json(output / "results.json")
        counts = manifest["counts"]["test"]
        if not results["passed"] or results["variant"] != variant.label or results["test_episodes"] != counts["episodes"] or results["test_windows"] != counts["windows"]:
            raise ValueError(f"{variant.label} evaluation did not cover the sealed test split")
        if set(results["test_latent_variation"]) != {"bf16_mixed", "fp32"}:
            raise ValueError(f"{variant.label} held-out encoder diagnostics are incomplete")
        shutil.copytree(run_dir, self.host_output / "runs" / variant.label, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("checkpoints"))
        shutil.copytree(output, self.host_output / "evaluation", dirs_exist_ok=True)
        shutil.copy2(self.experiment / "report.txt", self.host_output / "report.txt")
        self.state.update(status="complete", phase="complete", finished_utc=results["finished_utc"],
                          results=str(output / "results.json"), report=str(self.host_output / "report.txt"))
        self.state.pop("child_progress", None)
        self.publish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    args = parser.parse_args()
    SweepPipeline.variant = VARIANTS[args.variant]
    SweepPipeline.experiment = SweepPipeline.variant.experiment
    SweepPipeline.host_output = Path("/mnt/c/Users/Adam/Desktop/AI/lewm-wsl/experiments") / SweepPipeline.experiment.name
    run_locked(SweepPipeline)


if __name__ == "__main__":
    main()

