"""Source and data paths for native Windows commands, usable from any working directory."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNTIME = Path(os.environ.get("LEWM_WINDOWS_RUNTIME", REPO.parent / "lewm-local")).resolve()
STORAGE = Path(os.environ.get("STABLEWM_HOME", RUNTIME / "storage")).resolve()
DATASET_ROOT = Path(os.environ.get("LEWM_WINDOWS_DATASET_ROOT", os.environ.get("LOCAL_DATASET_DIR", STORAGE))).resolve()
LOGS = RUNTIME / "logs"


def is_wsl_share(path):
    return os.name == "nt" and str(path).lower().startswith(("\\\\wsl.localhost\\", "\\\\wsl$\\"))
