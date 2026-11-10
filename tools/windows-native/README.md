# Native Windows commands

Run these PowerShell entry points from a Windows checkout. They use the Python source and Hydra configuration in that checkout. Checkpoints, datasets, cache files, downloads, logs, Python installations and generated images live in a separate runtime directory.

```powershell
$native = Join-Path (Get-Location) 'tools\windows-native'
$env:LEWM_WINDOWS_RUNTIME = 'D:\LeWM\lewm-local'
# For an existing native virtual environment, supply its directory.
$env:LEWM_WINDOWS_VENV = 'D:\LeWM\le-wm\.venv'
. "$native\Activate-LeWM.ps1"
& "$native\Test-LeWM.ps1" -RequireDataset
```

If the variables are unset, the runtime defaults to `lewm-local` beside the checkout. The default venv is `venv` inside that runtime, with detection of an existing sibling `le-wm\.venv` containing Windows Python. Each entry point also accepts `-RuntimeRoot` and `-VenvPath`. For a fresh environment, install `uv` and run `Setup-LeWM.ps1`. Setup syncs the chosen venv to the Windows CUDA 12.8 package lock and applies the Windows signal fix for stable-pretraining 0.1.8.

The inference checks use local checkpoint directories `storage\checkpoints\lewm-{tworooms,pusht,cube,reacher}` containing `config.json`, `weights.pt` and the model repository's Hugging Face file listing in `source-manifest.json`. Select individual models or a CPU check with `-ExtraArguments @('--models','tworooms','--device','cpu')`. The real-data smoke pass uses `storage\datasets\tworoom.h5`. To reuse a dataset in another location, set `LEWM_WINDOWS_DATASET_ROOT` to the directory containing the `datasets` folder before activation.

```powershell
& "$native\Train-LeWM.ps1" -Smoke
& "$native\Plan-LeWM.ps1" -Episodes 2
& $LeWMPython "$native\smoke_environments.py"
& $LeWMPython "$native\validate_dataset.py" --full
& $LeWMPython "$native\audit_training.py" --checkpoint 'D:\LeWM\lewm-local\training\runs\...\checkpoints\last.ckpt'
```

`Train-LeWM.ps1` normally trains Two-Room for 100 epochs at batch size 8. Its smoke mode runs two updates and one validation batch. `Plan-LeWM.ps1` defaults to a short planning check. `-Full` uses the checkout's Two-Room planner configuration. Additional Hydra overrides can be supplied with `-ExtraArguments`. To inspect either configuration without starting a job, pass `-ExtraArguments @('--cfg','job')`.

For an existing archive at `downloads\tworoom.tar.zst`, `extract_dataset.py` verifies its SHA256, streams the HDF5 member to the runtime's dataset directory and checks any existing file against the archive. A different destination can be selected with `--destination`. Extraction uses an NTFS hard link to publish a complete file without replacing an existing dataset. The default `validate_dataset.py` checks eight image rows. Use `--full` to decode all pixel chunks.
