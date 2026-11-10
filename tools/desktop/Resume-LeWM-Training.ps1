$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'runtime.ps1')
$active = Get-Content $LeWMPointer -Raw | ConvertFrom-Json
$statusPath = Join-Path $active.windows_path 'status.json'
if (Test-Path -LiteralPath $statusPath) {
    $status = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
    if ($status.status -eq 'complete') { @{ status = 'complete'; message = 'The experiment is already complete.' } | ConvertTo-Json -Compress; exit 0 }
    if ($status.status -eq 'running') {
        $linuxPid = [int]$status.pid
        $savedPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        & wsl.exe -d $LeWMDistro --exec kill -0 $linuxPid 2>$null
        $processIsAlive = $LASTEXITCODE -eq 0
        $ErrorActionPreference = $savedPreference
        if ($processIsAlive) { @{ status = 'already_running'; message = 'The training pipeline is already running.' } | ConvertTo-Json -Compress; exit 0 }
    }
}
$controlPath = Join-Path $active.windows_path 'control.json'
$temporaryPath = "$controlPath.tmp"
[System.IO.File]::WriteAllText($temporaryPath, (@{ request = 'run'; requested_utc = [DateTime]::UtcNow.ToString('o') } | ConvertTo-Json), [System.Text.UTF8Encoding]::new($false))
Move-Item -LiteralPath $temporaryPath -Destination $controlPath -Force
[void][System.IO.Directory]::CreateDirectory($active.host_output)
$stamp = [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss-fff')
$process = Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $PSScriptRoot 'Run-LeWM-Experiment.ps1')) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $active.host_output "pipeline-$stamp.stdout.log") -RedirectStandardError (Join-Path $active.host_output "pipeline-$stamp.stderr.log")
$launch = @{ status = 'launched'; windows_pid = $process.Id; launched_utc = [DateTime]::UtcNow.ToString('o'); launcher = (Join-Path $PSScriptRoot 'Run-LeWM-Experiment.ps1'); experiment = $active.linux_experiment }
[System.IO.File]::WriteAllText((Join-Path $active.host_output 'launch.json'), ($launch | ConvertTo-Json), [System.Text.UTF8Encoding]::new($false))
$launch | ConvertTo-Json -Compress