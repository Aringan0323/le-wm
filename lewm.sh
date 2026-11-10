#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/activate.sh"
cd "$LEWM_REPO"
mode="${1:-shell}"
if [ "$#" -gt 0 ]; then shift; fi
mkdir -p "$LEWM_WSL_LOGS" "$TMPDIR"
# Hydra's argparse parser needs options before its positional overrides.
hydra_options=()
hydra_overrides=()
expects_value=0
for argument in "$@"; do
    if (( expects_value )); then
        hydra_options+=("$argument")
        expects_value=0
        continue
    fi
    case "$argument" in
        --cfg|-c|--package|-p|--config-path|-cp|--config-name|-cn|--config-dir|-cd|--experimental-rerun)
            hydra_options+=("$argument"); expects_value=1 ;;
        -*) hydra_options+=("$argument") ;;
        *=*|'~'*|+*) hydra_overrides+=("$argument") ;;
        *) hydra_options+=("$argument") ;;
    esac
done
case "$mode" in
    shell)
        exec bash --noprofile --rcfile "$LEWM_REPO/scripts/shellrc.sh" -i
        ;;
    inference)
        exec python "$LEWM_REPO/scripts/smoke_wsl.py" --stage inference "$@"
        ;;
    test)
        exec python "$LEWM_REPO/scripts/smoke_wsl.py" "$@"
        ;;
    train|train-smoke)
        run=tworoom-wsl
        epochs=100
        extra=()
        if [ "$mode" = train-smoke ]; then
            run=tworoom-wsl-smoke
            epochs=1
            extra+=(+trainer.limit_train_batches=2 +trainer.limit_val_batches=1 +trainer.log_every_n_steps=1)
        fi
        exec python train.py "${hydra_options[@]}" data=tworoom "subdir=$run" "output_model_name=$run" \
            "trainer.max_epochs=$epochs" loader.batch_size=8 num_workers=0 \
            loader.persistent_workers=false loader.prefetch_factor=null \
            wandb.enabled=false trainer.precision=bf16-mixed trainer.devices=1 \
            "+trainer.default_root_dir=$LEWM_ROOT/training/$run" \
            "hydra.run.dir=$LEWM_WSL_LOGS/$run" "${extra[@]}" "${hydra_overrides[@]}"
        ;;
    plan)
        exec python eval.py "${hydra_options[@]}" --config-name=tworoom policy=lewm-tworooms \
            eval.num_eval=2 solver.num_samples=32 solver.topk=8 solver.n_steps=3 \
            eval.eval_budget=25 "cache_dir=$STABLEWM_HOME" \
            "hydra.run.dir=$LEWM_WSL_LOGS/planning" output.filename=tworoom-wsl-results.txt "${hydra_overrides[@]}"
        ;;
    python)
        exec python "$@"
        ;;
    help|--help|-h)
        cat <<'HELP'
Usage: ./lewm.sh [shell|inference|test|train|train-smoke|plan|python] [arguments...]

./inference.sh                  Run all four GPU inference checks, including real Two-Room data
./test.sh                       Run inference and simulator smoke checks
./train-smoke.sh                Run two training batches and one validation batch
./train.sh trainer.max_epochs=10 loader.batch_size=8
./plan.sh                       Run the small pretrained Two-Room planning test
./lewm.sh python your_script.py Run Python with the project environment
source ./activate.sh            Activate the environment in your current Bash shell

Training/planning arguments are Hydra overrides. Pass --help to those entry points
for their configuration help. Logs: /home/adam/lewm/logs
HELP
        ;;
    *)
        printf 'Unknown mode: %s\nUse --help to list commands.\n' "$mode" >&2
        exit 2
        ;;
esac
