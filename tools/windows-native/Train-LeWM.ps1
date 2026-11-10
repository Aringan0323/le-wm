param(
    [ValidateRange(1, 2147483647)][int]$Epochs = 100,
    [ValidateRange(1, 2147483647)][int]$BatchSize = 8,
    [switch]$Smoke,
    [string[]]$ExtraArguments,
    [string]$RuntimeRoot = $env:LEWM_WINDOWS_RUNTIME,
    [string]$VenvPath = $env:LEWM_WINDOWS_VENV
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\Activate-LeWM.ps1" -RuntimeRoot $RuntimeRoot -VenvPath $VenvPath
$run = if ($Smoke) { 'tworoom-smoke' } else { 'tworoom-local' }
if ($Smoke) { $Epochs = 1 }
$out = $LeWMLocal.Replace('\', '/')
$arguments = @(
    (Join-Path $LeWMRepo 'train.py'), 'data=tworoom', "subdir=$run", "output_model_name=$run",
    "trainer.max_epochs=$Epochs", "loader.batch_size=$BatchSize", 'num_workers=0', 'loader.persistent_workers=false',
    'loader.prefetch_factor=null', 'wandb.enabled=false', 'trainer.precision=bf16-mixed', 'trainer.devices=1',
    "+trainer.default_root_dir=$out/training/$run", "hydra.run.dir=$out/logs/$run"
)
if ($Smoke) {
    $arguments += @('+trainer.limit_train_batches=2', '+trainer.limit_val_batches=1', '+trainer.log_every_n_steps=1')
}
$arguments += $ExtraArguments
Invoke-LeWMReadOnlyDatasetPython -PythonArguments $arguments
if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE" }
