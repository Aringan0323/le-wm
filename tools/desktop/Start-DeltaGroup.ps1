param(
    [string]$GroupName = 'episode50-delta-shared-20261110-01',
    [ValidateRange(1,1000)][int]$Epochs = 50,
    [ValidateRange(1,2147483647)][int]$BatchSize = 32,
    [string]$WslRepo = $env:LEWM_WSL_REPO,
    [switch]$Prepared
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'runtime.ps1')
if ($GroupName -notmatch '^[a-zA-Z0-9][a-zA-Z0-9_-]+$') { throw 'Invalid group directory name.' }
$prior = Get-Content -Raw -LiteralPath $LeWMPointer | ConvertFrom-Json
$oldStatus = Get-Content -Raw -LiteralPath (Join-Path $prior.windows_path 'status.json') | ConvertFrom-Json
if ($prior.queued_experiments -or $oldStatus.status -notin @('cancelled','complete')) {
    throw 'An experiment or its queue is still active.'
}
if (-not $WslRepo) {
    $linuxUserHome = ((& wsl.exe -d $LeWMDistro --exec printenv HOME) | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $linuxUserHome) {throw 'Cannot locate WSL home'}
    $WslRepo = "$linuxUserHome/lewm/le-wm"
}
$linuxGroup = ($prior.linux_experiment -replace '/[^/]+$', '') + '/' + $GroupName
$hostGroup = Join-Path (Split-Path $prior.host_output -Parent) $GroupName
$uncGroup = Join-Path (Split-Path $prior.windows_path -Parent) $GroupName
if (-not $Prepared) {
    & wsl.exe -d $LeWMDistro --exec bash "$WslRepo/lewm.sh" python "$WslRepo/scripts/prepare_delta_group.py" --manifest-dir $prior.linux_experiment --experiment-dir $linuxGroup --epochs $Epochs --batch-size $BatchSize
    if ($LASTEXITCODE -ne 0) { throw 'Could not prepare shared-batch experiment.' }
}
$plan = Get-Content -Raw -LiteralPath (Join-Path $uncGroup 'plan.json') | ConvertFrom-Json
$ready = Get-Content -Raw -LiteralPath (Join-Path $uncGroup 'status.json') | ConvertFrom-Json
$control = Get-Content -Raw -LiteralPath (Join-Path $uncGroup 'control.json') | ConvertFrom-Json
if (-not $plan.shared_batch -or $plan.epochs -ne $Epochs -or $plan.training_batch_size -ne $BatchSize -or $ready.status -ne 'prepared' -or $control.request -ne 'run') {
    throw 'Group is not ready for fresh launch. For a paused active group use Resume.'
}
$history = @($prior.comparison_experiments | Where-Object {$_})
if ($oldStatus.status -eq 'complete' -and $prior.model_names.PSObject.Properties.Name.Count -eq 1) {
    $oldKey = @($prior.model_names.PSObject.Properties.Name)[0]
    if ($oldKey -notin $plan.members) {
        $history += [pscustomobject]@{key=$oldKey;label=$prior.label;windows_path=$prior.windows_path;linux_experiment=$prior.linux_experiment}
    }
}
$next = [ordered]@{linux_experiment=$linuxGroup;host_output=$hostGroup;windows_path=$uncGroup;linux_run_script="$linuxGroup/run.sh";
    label='Two-Room · Shared-batch Delta-JEPA';shared_batch=$true;model_names=$plan.model_names;training_batch_size=$plan.training_batch_size;
    steps_per_model=$plan.steps_per_model;train_episodes=$plan.train_episodes;test_episodes=$plan.test_episodes;
    reference_experiment_windows_path=$prior.reference_experiment_windows_path;reference_experiment_linux_path=$prior.reference_experiment_linux_path;
    comparison_experiments=$history}
$check = Get-Content -Raw -LiteralPath $LeWMPointer | ConvertFrom-Json
if ($check.linux_experiment -ne $prior.linux_experiment -or $check.queued_experiments) {throw 'Active pointer changed before launch'}
Copy-Item -LiteralPath $LeWMPointer -Destination (Join-Path $hostGroup 'predecessor-pointer.json')
$temp = "$LeWMPointer.$PID.tmp"
[IO.File]::WriteAllText($temp,($next|ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
Move-Item -LiteralPath $temp -Destination $LeWMPointer -Force
$launcher=Join-Path $PSScriptRoot 'Run-LeWM-Experiment.ps1'
$proc=Start-Process -FilePath powershell.exe -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',('"{0}"' -f $launcher),'-RunScript',('"{0}"' -f $next.linux_run_script)) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $hostGroup 'pipeline.stdout.log') -RedirectStandardError (Join-Path $hostGroup 'pipeline.stderr.log')
@{status='launched';windows_pid=$proc.Id;experiment=$linuxGroup}|ConvertTo-Json -Compress
