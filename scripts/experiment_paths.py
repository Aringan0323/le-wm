"""Runtime assets live outside the source checkout and outside frozen run snapshots."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("LEWM_ROOT", Path.home() / "lewm")).expanduser().resolve()
HOST_ROOT = Path(os.environ.get("LEWM_HOST_ROOT", Path("/mnt/c/Users") / Path.home().name / "Desktop/AI"))
REPORTS = Path(os.environ.get("LEWM_REPORTS", HOST_ROOT / "lewm-wsl/experiments"))
