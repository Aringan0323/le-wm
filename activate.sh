#!/usr/bin/env bash
# Source this file in Bash: source ./activate.sh
export LEWM_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LEWM_ROOT="${LEWM_ROOT:-$(dirname -- "$LEWM_REPO")}"
export LEWM_HOST_ROOT="${LEWM_HOST_ROOT:-/mnt/c/Users/$(id -un)/Desktop/AI}"
export LEWM_WSL_LOGS="$LEWM_ROOT/logs"
export LEWM_REPORT_DIR="$LEWM_WSL_LOGS"
export VIRTUAL_ENV="$LEWM_ROOT/.venv"
export PATH="$VIRTUAL_ENV/bin:$LEWM_ROOT/bin:$PATH"
export STABLEWM_HOME="$LEWM_ROOT/storage"
export LOCAL_DATASET_DIR="$STABLEWM_HOME"
export TMPDIR="$LEWM_ROOT/tmp"
export UV_CACHE_DIR="$LEWM_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$LEWM_ROOT/python"
export UV_PYTHON_INSTALL_BIN=0
export UV_PYTHON_INSTALL_REGISTRY=0
export PIP_CACHE_DIR="$LEWM_ROOT/cache/pip"
export HF_HOME="$LEWM_ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET=1
export TORCH_HOME="$LEWM_ROOT/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$LEWM_ROOT/cache/torchinductor"
export TRITON_CACHE_DIR="$LEWM_ROOT/cache/triton"
export CUDA_CACHE_PATH="$LEWM_ROOT/cache/cuda"
export XDG_CACHE_HOME="$LEWM_ROOT/cache"
export MPLCONFIGDIR="$LEWM_ROOT/cache/matplotlib"
export WANDB_DIR="$LEWM_ROOT/logs/wandb"
export WANDB_CACHE_DIR="$LEWM_ROOT/cache/wandb"
export WANDB_CONFIG_DIR="$LEWM_ROOT/cache/wandb-config"
export WANDB_MODE=disabled
export SPT_CACHE_DIR="$LEWM_ROOT/training"
export MINARI_DATASETS_PATH="$LEWM_ROOT/cache/minari"
export SDL_AUDIODRIVER=dummy
export SDL_VIDEODRIVER=dummy
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYTHONUTF8=1
export PYTHONDONTWRITEBYTECODE=1
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p "$LEWM_WSL_LOGS" "$TMPDIR"
