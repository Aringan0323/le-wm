#!/usr/bin/env bash
# Run the three matched ablations sequentially from a Linux or WSL checkout.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 1 || "$1" == "--help" ]]; then
    echo "Usage: ./delta-sweep.sh SEALED_MANIFEST_DIR [--epochs N] [--planning]"
    exit 0
fi
split="$(realpath "$1")"
shift
source "$repo/activate.sh"
for model in delta_l0_s09 delta_l50_s09 delta_l50_s0; do
    path="$(python -c 'import sys; sys.path.insert(0,sys.argv[1]); from delta_sweep_config import VARIANTS; print(VARIANTS[sys.argv[2]].experiment)' "$repo/scripts" "$model")"
    if [[ ! -e "$path" ]]; then
        python "$repo/scripts/prepare_variant.py" "$model" --manifest-dir "$split" "$@"
    fi
    if ! python - "$path/control.json" <<'PY'
import json, sys
sys.exit(0 if json.load(open(sys.argv[1]))["request"] == "run" else 1)
PY
    then
        echo "$model is paused. Set its control request to run before continuing."
        exit 0
    fi
    bash "$path/run.sh"
    python - "$path/status.json" <<'PY'
import json, sys
status = json.load(open(sys.argv[1]))["status"]
if status != "complete":
    sys.exit(f"Sequence stopped with experiment status {status}")
PY
done
