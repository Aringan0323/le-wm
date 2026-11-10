#!/usr/bin/env bash
# Action-prediction variant of the 1,000-step Two-Room pilot; fresh initialization.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/activate.sh"
cd "$LEWM_REPO"
run_name="${1:-tworoom-action-pilot-$(TZ=America/New_York date +%Y%m%d-%H%M%S)}"
if [[ ! "$run_name" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "Run name must contain only letters, digits, underscores, and hyphens." >&2
    exit 2
fi
log_dir="$LEWM_WSL_LOGS/$run_name"
# Refuse reuse so the run cannot overwrite previous logs or model exports.
if [[ -e "$log_dir" || -e "$STABLEWM_HOME/checkpoints/$run_name" || -e "$LEWM_ROOT/training/$run_name" ]]; then
    echo "Run name already exists: $run_name" >&2
    exit 2
fi
mkdir -p "$log_dir" "$LEWM_ROOT/training/$run_name"
export SPT_CACHE_DIR="$LEWM_ROOT/training/$run_name"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=3072
printf 'Training logs: %s/console.log\n' "$log_dir"
exec >"$log_dir/console.log" 2>&1
trap 'result=$?; printf "%s\n" "$result" > "$log_dir/exit.code"' EXIT
mkdir "$log_dir/source"
cp train.py jepa.py module.py utils.py train-action-pilot.sh "$log_dir/source/"
cp -r config "$log_dir/source/"
git rev-parse HEAD > "$log_dir/source-commit.txt"
git diff -- train.py eval.py jepa.py module.py > "$log_dir/source-diff.patch"
command=(python -u train.py --config-name lewm_action
    data=tworoom "subdir=$run_name" "output_model_name=$run_name"
    trainer.max_epochs=1 +trainer.max_steps=1000
    +trainer.limit_train_batches=1000 +trainer.limit_val_batches=100
    +trainer.val_check_interval=250 +trainer.log_every_n_steps=10
    +trainer.enable_progress_bar=false
    loader.batch_size=8 num_workers=0 loader.persistent_workers=false loader.prefetch_factor=null
    wandb.enabled=false trainer.precision=bf16-mixed trainer.devices=1 seed=3072
    +checkpoint_every_n_steps=250
    "+trainer.default_root_dir=$LEWM_ROOT/training/$run_name"
    "hydra.run.dir=$log_dir")
printf '%q ' "${command[@]}" > "$log_dir/command.txt"
printf '\n' >> "$log_dir/command.txt"
printf 'Started (New York): %s\n' "$(TZ=America/New_York date --iso-8601=seconds)"
"${command[@]}" &
training_pid=$!
printf '%s\n' "$training_pid" > "$log_dir/train.pid"
wait "$training_pid"
