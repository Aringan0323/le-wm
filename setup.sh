#!/usr/bin/env bash
# Restore the locked Python environment in this already-provisioned Ubuntu distro.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/activate.sh"
uv python install 3.10.20
if [ ! -x "$VIRTUAL_ENV/bin/python" ]; then
    uv venv --python 3.10.20 "$VIRTUAL_ENV"
fi
uv pip sync --python "$VIRTUAL_ENV/bin/python" "$LEWM_REPO/scripts/requirements-linux-lock.txt" \
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match --no-cache
uv pip check --python "$VIRTUAL_ENV/bin/python"
