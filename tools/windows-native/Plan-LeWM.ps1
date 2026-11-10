param(
    [ValidateRange(1, 10000)][int]$Episodes = 2,
    [switch]$Full,
    [string[]]$ExtraArguments,
    [string]$RuntimeRoot = $env:LEWM_WINDOWS_RUNTIME,
    [string]$VenvPath = $env:LEWM_WINDOWS_VENV
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\Activate-LeWM.ps1" -RuntimeRoot $RuntimeRoot -VenvPath $VenvPath
$arguments = @(
    (Join-Path $LeWMRepo 'eval.py'), '--config-name=tworoom', 'policy=lewm-tworooms',
    "eval.num_eval=$Episodes", "cache_dir=$($env:LOCAL_DATASET_DIR.Replace('\','/'))",
    "hydra.run.dir=$($LeWMLocal.Replace('\','/'))/logs/planning", 'output.filename=tworoom-local-results.txt'
)
if (-not $Full) {
    $arguments += @('solver.num_samples=32', 'solver.topk=8', 'solver.n_steps=3', 'eval.eval_budget=25')
}
$arguments += $ExtraArguments
Invoke-LeWMReadOnlyDatasetPython -PythonArguments $arguments
if ($LASTEXITCODE -ne 0) { throw "Planning failed with exit code $LASTEXITCODE" }
