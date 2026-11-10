param(
    [switch]$RequireDataset,
    [string[]]$ExtraArguments,
    [string]$RuntimeRoot = $env:LEWM_WINDOWS_RUNTIME,
    [string]$VenvPath = $env:LEWM_WINDOWS_VENV
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\Activate-LeWM.ps1" -RuntimeRoot $RuntimeRoot -VenvPath $VenvPath
$arguments = @("$LeWMNativeTools\smoke_inference.py")
if ($RequireDataset) { $arguments += '--require-dataset' }
$arguments += $ExtraArguments
Invoke-LeWMReadOnlyDatasetPython -PythonArguments $arguments
if ($LASTEXITCODE -ne 0) { throw "Inference checks failed with exit code $LASTEXITCODE" }
