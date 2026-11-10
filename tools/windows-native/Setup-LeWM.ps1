param(
    [string]$RuntimeRoot = $env:LEWM_WINDOWS_RUNTIME,
    [string]$VenvPath = $env:LEWM_WINDOWS_VENV
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\Activate-LeWM.ps1" -RuntimeRoot $RuntimeRoot -VenvPath $VenvPath
foreach ($directory in $env:TEMP, $env:SPT_CACHE_DIR, $env:STABLEWM_HOME, (Join-Path $LeWMLocal 'logs')) {
    [System.IO.Directory]::CreateDirectory($directory) | Out-Null
}
uv python install 3.10.20
if ($LASTEXITCODE -ne 0) { throw 'Python installation failed' }
if (-not (Test-Path -LiteralPath $LeWMPython)) {
    uv venv --python 3.10.20 $env:VIRTUAL_ENV
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
}
uv pip sync --python $LeWMPython "$LeWMNativeTools\requirements-lock.txt" --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match --no-cache
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& $LeWMPython "$LeWMNativeTools\apply_windows_patches.py"
if ($LASTEXITCODE -ne 0) { throw 'Compatibility patch failed' }
uv pip check --python $LeWMPython
if ($LASTEXITCODE -ne 0) { throw 'Dependency compatibility check failed' }
