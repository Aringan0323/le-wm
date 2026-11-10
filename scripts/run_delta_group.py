"""Own the shared training process, its GPU telemetry, final audits and held-out tests."""
import argparse
from pathlib import Path
import os
import shutil

from delta_sweep_config import VARIANTS
from group_telemetry import TrainingTelemetry
from run_episode_comparison import Pipeline, now, read_json, run_locked, verify_source


class GroupPipeline(Pipeline):
    def run(self):
        root = self.experiment
        verify_source(root)
        source = root/"source"
        plan = read_json(root/"plan.json")
        env = dict(os.environ, LEWM_REPO=str(source), PYTHONPATH=str(source),
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
        run_dirs = {k:root/"members"/k/"runs"/k for k in VARIANTS}
        if not all((rd/"runcompletion.json").exists() for rd in run_dirs.values()):
            with TrainingTelemetry(root):
                self.child("train_shared", plan["training_command"], root/"training.log", env,
                           root/"group-progress.json")
        for k,rd in run_dirs.items():
            done = read_json(rd/"runcompletion.json")
            if not done.get("passed") or done.get("global_step") != plan["steps_per_model"]:
                raise ValueError(f"{k} did not finish the planned shared steps")
            (rd/"training-exit.code").write_text("0\n")
        verify_source(root)
        from posttrain_delta_group import run_group_posttraining
        run_group_posttraining(self, env)
        results = read_json(root/"evaluation/results.json")
        if not results.get("passed") or set(results["models"]) != set(VARIANTS):
            raise ValueError("Shared group comparison is incomplete")
        for key,rd in run_dirs.items():
            member = root/"members"/key
            target = self.host_output/"members"/key
            shutil.copytree(rd,target/"runs"/key,dirs_exist_ok=True,ignore=shutil.ignore_patterns("checkpoints"))
            shutil.copytree(member/"evaluation",target/"evaluation",dirs_exist_ok=True)
            shutil.copy2(member/"report.txt",target/"report.txt")
        shutil.copytree(root/"evaluation",self.host_output/"evaluation",dirs_exist_ok=True)
        for name in ("report.txt","gpu-telemetry.csv","gpu-telemetry.json","group-progress.json","group-checkpoint.json"):
            if (root/name).is_file():
                shutil.copy2(root/name,self.host_output/name)
        self.state.update(status="complete",phase="complete",finished_utc=now(),report=str(self.host_output/"report.txt"))
        self.state.pop("child_progress",None)
        self.publish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    exp = parser.parse_args().experiment_dir.resolve(strict=True)
    GroupPipeline.experiment = exp
    GroupPipeline.host_output = Path(read_json(exp/"plan.json")["report_directory"])
    run_locked(GroupPipeline)
