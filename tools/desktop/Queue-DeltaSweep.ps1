param([ValidateRange(1,1000)][int]$Epochs = 50, [switch]$Planning, [string]$WslRepo = $env:LEWM_WSL_REPO)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'runtime.ps1')
$active = Get-Content -LiteralPath $LeWMPointer -Raw | ConvertFrom-Json
if ($active.queued_experiments) { throw 'The queue already has entries. Check the dashboard before adding another sweep.' }
if (-not $WslRepo) {
    $linuxUserHome = ((& wsl.exe -d $LeWMDistro --exec printenv HOME) | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $linuxUserHome) { throw 'Could not resolve WSL home directory' }
    $WslRepo = "$linuxUserHome/lewm/le-wm"
}
$python = (Get-Command python.exe -ErrorAction Stop).Source
$worker = Join-Path $PSScriptRoot 'run-queued-experiment.py'
$previous = $null
$queues = @()
foreach ($variant in 'delta_l0_s09','delta_l50_s09','delta_l50_s0') {
    $argv = @('-d',$LeWMDistro,'--exec','bash',"$WslRepo/lewm.sh",'python',"$WslRepo/scripts/prepare_variant.py",$variant,'--epochs',[string]$Epochs)
    if ($previous) { $argv += @('--after',$previous) }
    if ($Planning) { $argv += '--planning' }
    $json = & wsl.exe @argv
    if ($LASTEXITCODE -ne 0) { throw "Could not prepare $variant. No queue workers have been launched." }
    $result = ($json | Out-String) | ConvertFrom-Json
    $previous = $result.experiment
    $name = ($previous -split '/')[-1]
    $queues += ,(Join-Path $LeWMState "experiments\$name\queue.json")
}
foreach ($queue in $queues) {
    $q = Get-Content -Raw -LiteralPath $queue | ConvertFrom-Json
    $folder = Split-Path -Parent $queue
    $proc = Start-Process -FilePath $python -ArgumentList @('-u', ('"{0}"' -f $worker), ('"{0}"' -f $queue), '--register') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $folder 'queue-worker.log') -RedirectStandardError (Join-Path $folder 'queue-worker-error.log')
    $deadline = [DateTime]::UtcNow.AddSeconds(25)
    do {
        $now = Get-Content -Raw -LiteralPath $LeWMPointer | ConvertFrom-Json
        $registered = $now.linux_experiment -eq $q.target.linux_experiment -or $now.queued_experiments.linux_experiment -contains $q.target.linux_experiment
        if (-not $registered) {
            if ($proc.HasExited -or [DateTime]::UtcNow -gt $deadline) { throw "Queue registration failed. See $folder\queue-worker.log." }
            Start-Sleep -Milliseconds 250
        }
    } until ($registered)
    [pscustomobject]@{model=$q.target.label; predecessor=$q.predecessor.key; worker=$proc.Id}
}
