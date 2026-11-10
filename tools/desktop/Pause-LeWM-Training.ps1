$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'runtime.ps1')
$active = Get-Content $LeWMPointer -Raw | ConvertFrom-Json
$controlPath = Join-Path $active.windows_path 'control.json'
$temporaryPath = "$controlPath.tmp"
$control = @{ request = 'pause'; requested_utc = [DateTime]::UtcNow.ToString('o') }
[System.IO.File]::WriteAllText($temporaryPath, ($control | ConvertTo-Json), [System.Text.UTF8Encoding]::new($false))
Move-Item -LiteralPath $temporaryPath -Destination $controlPath -Force
@{ status = 'pause_requested'; message = 'Training will save a checkpoint after the current batch and release the GPU.' } | ConvertTo-Json -Compress