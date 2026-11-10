param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'runtime.ps1')
$taskRoot = $PSScriptRoot
$taskPython = Join-Path $LeWMDesktop 'lewm-local\python\cpython-3.10.20-windows-x86_64-none\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { $taskPython = (Get-Command python.exe -ErrorAction Stop).Source }
$taskScript = Join-Path $taskRoot 'Training-Dashboard.py'
$taskLogs = Join-Path $LeWMState 'dashboard'
$taskUri = 'http://127.0.0.1:8765'
$taskMutex = [System.Threading.Mutex]::new($false, 'Local\LeWMTrainingDashboardLaunch')
$taskLocked = $false
try {
    $taskLocked = $taskMutex.WaitOne(15000)
    if (-not $taskLocked) { throw 'Another dashboard launch is still in progress.' }
    $taskReady = $false
    try {
        $taskStatus = Invoke-RestMethod -Uri "$taskUri/api/status" -TimeoutSec 2
        if ($taskStatus.control_token) { $taskReady = $true }
        else { throw 'Port 8765 is being used by another service.' }
    } catch {
        if ($_.Exception.Message -like '*another service*') { throw }
    }
    if (-not $taskReady) {
        [IO.Directory]::CreateDirectory($taskLogs) | Out-Null
        $taskProcess = Start-Process -FilePath $taskPython -ArgumentList @('"' + $taskScript + '"') -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskLogs 'server.log') -RedirectStandardError (Join-Path $taskLogs 'server-error.log')
        [IO.File]::WriteAllText((Join-Path $taskLogs 'server.pid'), [string]$taskProcess.Id)
        for ($taskAttempt = 0; $taskAttempt -lt 30; $taskAttempt++) {
            Start-Sleep -Milliseconds 200
            try {
                $taskStatus = Invoke-RestMethod -Uri "$taskUri/api/status" -TimeoutSec 1
                if ($taskStatus.control_token) { $taskReady = $true; break }
            } catch {}
        }
        if (-not $taskReady) { throw "Dashboard did not start. See $taskLogs\server-error.log" }
    }
    if (-not $NoBrowser) { Start-Process $taskUri }
    @{status='ready'; url=$taskUri} | ConvertTo-Json -Compress
} finally {
    if ($taskLocked) { $taskMutex.ReleaseMutex() }
    $taskMutex.Dispose()
}
