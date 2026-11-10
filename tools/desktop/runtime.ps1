$LeWMDesktop = if ($env:LEWM_DESKTOP_ROOT) { $env:LEWM_DESKTOP_ROOT } else {
    Join-Path ([Environment]::GetFolderPath('Desktop')) 'AI'
}
$LeWMState = Join-Path $LeWMDesktop 'lewm-wsl'
$LeWMPointer = Join-Path $LeWMState 'active-experiment.json'
$LeWMDistro = if ($env:LEWM_DISTRO) { $env:LEWM_DISTRO } else { 'Ubuntu-24.04' }
