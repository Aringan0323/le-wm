"""Freeze three matched Delta models for a single shared-batch training process."""
import argparse
import json
import shlex
import shutil
import sys
from importlib.metadata import distributions
from pathlib import Path
from tempfile import TemporaryDirectory

from delta_sweep_config import VARIANTS
from experiment_paths import REPORTS, ROOT
from prepare_variant import REPO, describe
from run_episode_comparison import now, read_json, sha, train_command, verify_source, write_json


def prepare(group, split, epochs=50, reports=REPORTS):
    group, split = group.resolve(), split.resolve()
    mirror = reports.resolve() / group.name
    if epochs < 1:
        raise ValueError("epochs must be positive")
    # Use the group name to generate model namespaces and to permit later repetitions.
    if group.name == "episode50-delta-shared-20261110-01":
        names = {k: v.model + "-shared" for k, v in VARIANTS.items()}
    else:
        names = {k: f"{group.name}-{k}" for k in VARIANTS}
    for path in (group, mirror, *(ROOT / "storage/checkpoints" / name for name in names.values()),
                 *(ROOT / "training" / name for name in names.values())):
        if path.exists():
            raise FileExistsError(f"Fresh namespace is already in use: {path}")
    manifest = read_json(split / "manifest.json")
    if sha(split / "arrays.npz") != manifest["arrays_sha256"] or manifest["training_order"]["batch_size"] != 32:
        raise ValueError("Expected the verified batch-32 Two-Room split")
    stamp = now()
    steps = manifest["training_order"]["steps_per_epoch"] * epochs
    train, test, unused = (manifest["counts"][s] for s in ("train", "test", "unused"))
    common = dict(created_utc=stamp, manifest_sha256=manifest["manifest_sha256"], epochs=epochs,
                  steps_per_model=steps, training_batch_size=32, checkpoint_interval_seconds=300,
                  train_episodes=train["episodes"], test_episodes=test["episodes"], unused_episodes=unused["episodes"],
                  train_windows=train["windows"], train_window_visits=train["windows"]*epochs, test_windows=test["windows"],
                  seed=3072, fresh_initialization=True, same_training_order=True, pause_resume=True,
                  evaluate_planning=False, precisions=["bf16_mixed","fp32"], primary_metric="Next-embedding MSE",
                  training_backend="native_pytorch_shared_batch", track_cross_example_latent_spread=True)
    group.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".prepare-shared-", dir=group.parent) as tmp:
        stage = Path(tmp) / group.name
        source = stage / "source"
        source.mkdir(parents=True)
        for f in (*REPO.glob("*.py"), *(REPO/"scripts").glob("*.py"), *(REPO/"config").rglob("*.yaml")):
            dest = source / f.relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
        for f in ("manifest.json", "arrays.npz"):
            shutil.copy2(split/f, stage/f)
        write_json(stage/"source-snapshot.json", dict(created_utc=stamp, source=str(group/"source"),
                   files_sha256={str(f.relative_to(source)):sha(f) for f in sorted(source.rglob("*")) if f.is_file()}))
        verify_source(stage)
        write_json(stage/"control.json", {"request":"run", "requested_utc":stamp})
        write_json(stage/"status.json", {"status":"prepared", "phase":"prepared", "updated_utc":stamp})
        plan = dict(common, shared_batch=True, model_names=names, members=list(names),
                    report_directory=str(mirror), loss_weights={k:describe(v)[1] for k,v in VARIANTS.items()},
                    training_command=[sys.executable,"-u",str(group/"source/train_delta_group.py"),
                                      "--experiment-dir",str(group)])
        write_json(stage/"plan.json", plan)
        for k,v in VARIANTS.items():
            rel = Path("members") / k
            member = group / rel
            here = stage / rel
            here.mkdir(parents=True)
            for f in ("source", "source-snapshot.json", "manifest.json", "arrays.npz", "control.json"):
                (here/f).symlink_to(Path("../..")/f, target_is_directory=(f=="source"))
            command = train_command(k,v.config,names[k],member,epochs,manifest["dataset"]["path"])
            command = [f"+control_file={group/'control.json'}" if arg.startswith("+control_file=") else arg
                       for arg in command]
            if not v.ldad_weight:
                command.append("model.ldad=null")
            write_json(here/"plan.json", dict(common, shared_batch=True, group_root=str(group), variant=k,
                       objective=v.objective, loss_weights=describe(v)[1], ldad_decoder_enabled=bool(v.ldad_weight),
                       training_commands={k:command},
                       model_names={k:names[k]}, model_exports={k:str(ROOT/"storage/checkpoints"/names[k])},
                       report_directory=str(mirror/rel)))
        (stage/"environment.txt").write_text("\n".join(sorted(
            f"{pkg.metadata['Name']}=={pkg.version}" for pkg in distributions() if pkg.metadata["Name"]))+"\n")
        runner = stage/"run.sh"
        runner.write_text("#!/usr/bin/env bash\nset -euo pipefail\n"
                          f"export LEWM_ROOT={shlex.quote(str(ROOT))}\nsource {shlex.quote(str(REPO/'activate.sh'))}\n"
                          "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1\n"
                          f"exec python -u {shlex.quote(str(group/'source/scripts/run_delta_group.py'))} "
                          f"--experiment-dir {shlex.quote(str(group))}\n")
        runner.chmod(0o755)
        stage.rename(group)
    mirror.mkdir(parents=True)
    for f in ("plan.json","source-snapshot.json","manifest.json","arrays.npz","environment.txt","status.json"):
        shutil.copy2(group/f,mirror/f)
    for k in names:
        dest = mirror/"members"/k
        dest.mkdir(parents=True)
        shutil.copy2(group/"members"/k/"plan.json",dest/"plan.json")
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--reports", type=Path, default=REPORTS)
    args = parser.parse_args()
    plan = prepare(args.experiment_dir, args.manifest_dir, args.epochs, args.reports)
    print(json.dumps({"experiment":str(args.experiment_dir),"models":plan["model_names"],"shared_steps":plan["steps_per_model"]}))
