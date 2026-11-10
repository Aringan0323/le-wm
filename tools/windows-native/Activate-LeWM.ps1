param(
    [string]$RuntimeRoot = $env:LEWM_WINDOWS_RUNTIME,
    [string]$VenvPath = $env:LEWM_WINDOWS_VENV
)

$LeWMNativeTools = $PSScriptRoot
$LeWMRepo = (Get-Item -LiteralPath "$PSScriptRoot\..\..").FullName
if (-not $RuntimeRoot) {
    $RuntimeRoot = Join-Path (Split-Path -Parent $LeWMRepo) 'lewm-local'
}
$LeWMLocal = [System.IO.Path]::GetFullPath($RuntimeRoot)
if (-not $VenvPath) {
    $VenvPath = Join-Path $LeWMLocal 'venv'
    $legacy = Join-Path (Split-Path -Parent $LeWMLocal) 'le-wm\.venv'
    if (-not (Test-Path -LiteralPath "$VenvPath\Scripts\python.exe") -and
        (Test-Path -LiteralPath "$legacy\Scripts\python.exe")) {
        $VenvPath = $legacy
    }
}
$env:LEWM_WINDOWS_RUNTIME = $LeWMLocal
$env:LEWM_WINDOWS_VENV = [System.IO.Path]::GetFullPath($VenvPath)
$env:VIRTUAL_ENV = $env:LEWM_WINDOWS_VENV
$LeWMPython = Join-Path $env:VIRTUAL_ENV 'Scripts\python.exe'
$scriptsPath = Join-Path $env:VIRTUAL_ENV 'Scripts'
$env:PATH = (@($scriptsPath) + @(($env:PATH -split ';') | Where-Object { $_ -and $_ -ne $scriptsPath })) -join ';'
$env:STABLEWM_HOME = Join-Path $LeWMLocal 'storage'
$env:LOCAL_DATASET_DIR = if ($env:LEWM_WINDOWS_DATASET_ROOT) {
    [System.IO.Path]::GetFullPath($env:LEWM_WINDOWS_DATASET_ROOT)
} else { $env:STABLEWM_HOME }
$env:TEMP = Join-Path $LeWMLocal 'tmp'
$env:TMP = $env:TEMP
$env:UV_CACHE_DIR = Join-Path $LeWMLocal 'cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $LeWMLocal 'python'
$env:UV_PYTHON_INSTALL_BIN = '0'
$env:UV_PYTHON_INSTALL_REGISTRY = '0'
$env:PIP_CACHE_DIR = Join-Path $LeWMLocal 'cache\pip'
$env:HF_HOME = Join-Path $LeWMLocal 'cache\huggingface'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:HF_HUB_DISABLE_XET = '1'
$env:TORCH_HOME = Join-Path $LeWMLocal 'cache\torch'
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $LeWMLocal 'cache\torchinductor'
$env:TRITON_CACHE_DIR = Join-Path $LeWMLocal 'cache\triton'
$env:XDG_CACHE_HOME = Join-Path $LeWMLocal 'cache'
$env:MPLCONFIGDIR = Join-Path $LeWMLocal 'cache\matplotlib'
$env:WANDB_DIR = Join-Path $LeWMLocal 'logs\wandb'
$env:WANDB_CACHE_DIR = Join-Path $LeWMLocal 'cache\wandb'
$env:WANDB_CONFIG_DIR = Join-Path $LeWMLocal 'cache\wandb-config'
$env:WANDB_MODE = 'disabled'
$env:SDL_AUDIODRIVER = 'dummy'
$env:SDL_VIDEODRIVER = 'dummy'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:HYDRA_FULL_ERROR = '1'
$env:SPT_CACHE_DIR = Join-Path $LeWMLocal 'training'
$env:PYTORCH_ALLOC_CONF = 'expandable_segments:False'
$env:CUDA_CACHE_PATH = Join-Path $LeWMLocal 'cache\cuda'
$env:HF_DATASETS_CACHE = Join-Path $LeWMLocal 'cache\huggingface\datasets'
$env:MINARI_DATASETS_PATH = Join-Path $LeWMLocal 'cache\minari'

function Invoke-LeWMReadOnlyDatasetPython {
    param([string[]]$PythonArguments)
    # Windows HDF5 cannot lock immutable datasets on the WSL share. Limit this
    # setting to the child command that reads that dataset.
    $previousLock = $env:HDF5_USE_FILE_LOCKING
    try {
        if ($env:LOCAL_DATASET_DIR -match '^\\\\(?:wsl\.localhost|wsl\$)\\') {
            $env:HDF5_USE_FILE_LOCKING = 'FALSE'
        }
        & $LeWMPython @PythonArguments
    } finally {
        $env:HDF5_USE_FILE_LOCKING = $previousLock
    }
}
