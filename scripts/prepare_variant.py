"""Freeze a new experiment for native Linux/WSL, optionally handing it to the desktop queue."""
import argparse
import json
import os
import shlex
import shutil
from importlib.metadata import distributions
from pathlib import Path, PureWindowsPath
from tempfile import TemporaryDirectory

from combined_sweep_config import REFERENCE, VARIANTS as LEGACY
from delta_sweep_config import VARIANTS as DELTA
from experiment_paths import HOST_ROOT, REPORTS, ROOT
from run_episode_comparison import now, read_json, sha, train_command, verify_source, write_json

REPO = Path(__file__).resolve().parents[1]
VARIANTS = {**LEGACY, **DELTA}


def describe(v):
    if v.objective == "delta_jepa":
        return v.display_label, {"next_embedding_mse": 1.0, "ldad": v.ldad_weight, "sigreg": v.sigreg_weight}
    text = (f"Action {v.action_weight:g} (no SIGReg)" if v.objective == "action_prediction"
            else f"SIGReg {v.sigreg_weight:g} + action {v.action_weight:.1f}")
    return text, {"next_embedding_mse": 1.0, "action_embedding_mse": v.action_weight, "sigreg": v.sigreg_weight}


def make_queue(prior, v, display, stamp, steps, manifest):
    old_dir = PureWindowsPath(prior["windows_path"])
    old_key, = prior["model_names"]
    predecessor = dict(key=old_key, label=prior["label"].removeprefix("Two-Room · "),
                       windows_path=str(old_dir), linux_experiment=prior["linux_experiment"])
    target = dict(key=v.label, label=display, windows_path=str(old_dir.parent / v.experiment.name),
                  linux_experiment=str(v.experiment))
    pointer = dict(prior, windows_path=target["windows_path"], linux_experiment=str(v.experiment),
                   host_output=str(PureWindowsPath(prior["host_output"]).parent / v.experiment.name),
                   linux_run_script=str(v.experiment / "run.sh"), model_names={v.label: v.model},
                   steps_per_model=steps, label="Two-Room · " + display,
                   training_batch_size=manifest["training_order"]["batch_size"],
                   train_episodes=manifest["counts"]["train"]["episodes"],
                   test_episodes=manifest["counts"]["test"]["episodes"],
                   comparison_experiments=[*prior.get("comparison_experiments", []), predecessor])
    pointer.pop("queued_experiments", None)
    return dict(created_utc=stamp, predecessor=predecessor, target=target, pointer=pointer)


def prepare(args):
    v = VARIANTS[args.key]
    display, weights = describe(v)
    exp = v.experiment
    desktop = args.manifest_dir is None
    reports = REPORTS if desktop else Path(os.environ.get("LEWM_REPORTS", ROOT / "reports"))
    mirror = reports / exp.name
    for namespace in (exp, mirror, ROOT / "storage/checkpoints" / v.model, ROOT / "training" / v.model):
        if namespace.exists():
            raise FileExistsError(f"Existing experiment or model namespace: {namespace}")
    if desktop:
        active = read_json(HOST_ROOT / "lewm-wsl/active-experiment.json")
        prior = read_json(args.after / "queue.json")["pointer"] if args.after else active
        split = Path(prior["linux_experiment"])
        verify_source(split)
    else:
        if args.after:
            raise ValueError("--after applies to desktop queues; --manifest-dir is for standalone runs")
        split = args.manifest_dir.resolve()
    manifest = read_json(split / "manifest.json")
    if sha(split / "arrays.npz") != manifest["arrays_sha256"] or manifest["training_order"]["batch_size"] != 32:
        raise ValueError("Expected a matching Two-Room split sealed at batch size 32")
    epochs = args.epochs
    if epochs is None:
        epochs = 50 if v.label in DELTA else 1
    if epochs < 1 or (v.label not in DELTA and (epochs != 1 or args.planning)):
        raise ValueError("Historical variants support one epoch and MSE evaluation")
    steps = epochs * manifest["training_order"]["steps_per_epoch"]
    train, test, unused = (manifest["counts"][s] for s in ("train", "test", "unused"))
    stamp = now()
    command = train_command(v.label, v.config, v.model, exp, epochs=epochs, dataset=manifest["dataset"]["path"])
    plan = dict(created_utc=stamp, manifest_sha256=manifest["manifest_sha256"], variant=v.label,
                objective=v.objective, loss_weights=weights, model_names={v.label: v.model},
                train_episodes=train["episodes"], test_episodes=test["episodes"], unused_episodes=unused["episodes"],
                train_windows=train["windows"], train_window_visits=epochs*train["windows"], test_windows=test["windows"],
                training_batch_size=32, steps_per_model=steps, epochs=epochs, checkpoint_interval_seconds=300,
                fresh_initialization=True, same_training_order=True, pause_resume=True, normalization="Training episodes only",
                seed=3072, training_backend="native_pytorch", loader_workers=4, prefetch_batches_per_worker=2,
                primary_metric="Next-embedding MSE", precisions=["bf16_mixed", "fp32"],
                evaluate_planning=args.planning, track_cross_example_latent_spread=v.track_cross_example_latent_spread,
                training_commands={v.label: command}, model_exports={v.label: str(ROOT/"storage/checkpoints"/v.model)},
                report_directory=str(mirror), queued_after=str(split) if desktop else None)
    exp.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".prepare-", dir=exp.parent) as temp:
        stage = Path(temp) / exp.name
        source = stage / "source"
        source.mkdir(parents=True)
        paths = [*REPO.glob("*.py"), *(REPO / "scripts").glob("*.py"), *(REPO / "config").rglob("*.yaml")]
        for original in paths:
            dest = source / original.relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, dest)
        for name in ("manifest.json", "arrays.npz"):
            shutil.copy2(split / name, stage / name)
        write_json(stage / "plan.json", plan)
        write_json(stage / "control.json", {"request": "run", "requested_utc": stamp})
        write_json(stage / "status.json", dict(schema_version=1, status="queued", phase="queued",
                   experiment=str(exp), host_output=str(mirror), updated_utc=stamp))
        if desktop:
            write_json(stage / "queue.json", make_queue(prior, v, display, stamp, steps, manifest))
        write_json(stage / "source-snapshot.json", dict(created_utc=stamp, source=str(exp/"source"),
                   files_sha256={str(p.relative_to(source)): sha(p) for p in sorted(source.rglob("*")) if p.is_file()}))
        verify_source(stage)
        packages = sorted(f"{pkg.metadata['Name']}=={pkg.version}" for pkg in distributions() if pkg.metadata["Name"])
        (stage / "environment.txt").write_text("\n".join(packages) + "\n")
        script = exp / "source/scripts/run_combined_sweep.py"
        runner = stage / "run.sh"
        runner.write_text("#!/usr/bin/env bash\nset -euo pipefail\n"
                          f"export LEWM_ROOT={shlex.quote(str(ROOT))}\nsource {shlex.quote(str(REPO/'activate.sh'))}\n"
                          "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1\n"
                          f"exec python -u {shlex.quote(str(script))} --variant {shlex.quote(v.label)}\n")
        runner.chmod(0o755)
        stage.rename(exp)
    mirror.mkdir(parents=True)
    for f in exp.iterdir():
        if f.is_file() and f.name not in ("run.sh", "control.json"):
            shutil.copy2(f, mirror / f.name)
    print(json.dumps(dict(experiment=str(exp), weights=weights, epochs=epochs, steps=steps, runner=str(exp/"run.sh"))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("key", choices=VARIANTS)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--planning", action="store_true")
    parser.add_argument("--after", type=Path, help="Prepared desktop queue predecessor directory")
    parser.add_argument("--manifest-dir", type=Path, help="Native WSL/Linux mode using an existing sealed split")
    prepare(parser.parse_args())
