param([string]$RunScript)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'runtime.ps1')
if (-not $RunScript) {
    $active = Get-Content $LeWMPointer -Raw | ConvertFrom-Json
    $RunScript = $active.linux_run_script
}
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class TrainingPowerRequest {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@
$previous = [TrainingPowerRequest]::SetThreadExecutionState([uint32]2147483649)
if ($previous -eq 0) { throw 'Could not request temporary system-awake state' }
$runExit = 1
try {
    & wsl.exe -d $LeWMDistro --exec bash $RunScript
    $runExit = $LASTEXITCODE
} finally {
    [void][TrainingPowerRequest]::SetThreadExecutionState([uint32]2147483648)
}
exit $runExit
