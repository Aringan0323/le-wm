# Two-Room experiments in WSL

The three models share a randomly initialized encoder, an action-conditioned forward predictor and a latent difference action decoder (LDAD). Each clip contains four observations separated by five simulator steps. For each of the three adjacent observation pairs, LDAD receives the observed difference `z(t+5) - z(t)` and reconstructs the five recorded 2D actions spanning that interval. The forward predictor receives training-set-standardized action inputs. LDAD targets are the recorded actions in their original units.

We use the paper's three noncausal Transformer layers, five learned action queries, AdaLN conditioning, eight heads of width 64 and feed-forward width 512. The decoder residual width is 192 and decoder dropout is zero. The authors do not specify these latter settings. The main paper uses action weight 10 and 50 epochs. Weight 50 was a Push-T ablation. The two configurations below that include SIGReg are additional experiments. [Delta-JEPA paper](https://arxiv.org/html/2606.31232v1)

| Configuration | Next-embedding MSE | Raw-action LDAD MSE | SIGReg |
|---|---:|---:|---:|
| `delta_l0_s09` | 1 | 0 | 0.09 |
| `delta_l50_s09` | 1 | 50 | 0.09 |
| `delta_l50_s0` | 1 | 50 | 0 |

## Run in Ubuntu or WSL

Use an Ubuntu 24.04 environment with CUDA, `uv`, and a Two-Room HDF5 dataset. The commands below install the locked Python environment in the runtime directory, outside the checkout. Set `LEWM_ROOT` before activation to select a different runtime directory.

```bash
mkdir -p ~/lewm && cd ~/lewm
git clone --branch delta-jepa https://github.com/Aringan0323/le-wm.git
cd le-wm
./setup.sh
source ./activate.sh

# Set this to your copy of the LeWM Two-Room dataset.
DATASET="$LEWM_ROOT/storage-local/datasets/tworoom.h5"
SPLIT="$LEWM_ROOT/splits/tworoom-50-30"
python scripts/prepare_episode_experiment.py --dataset "$DATASET" --output-dir "$SPLIT" --batch-size 32
python scripts/check_delta_jepa.py --manifest "$SPLIT/manifest.json"
./delta-sweep.sh "$SPLIT" --epochs 50
```

All three runs use one fixed split of 5,000 train and 3,000 test episodes. The remaining 2,000 episodes are unused. One epoch covers 365,657 windows in 11,427 optimizer steps including the final batch of 25. At 50 epochs the training budget is 571,350 updates per model. Each run evaluates frozen weights in BF16 and FP32 on 218,790 held-out windows and records the encoder's cross-example spread. Add `--planning` on first preparation to evaluate goal-reaching success on 500 held-out trajectories with three planner seeds per checkpoint. This can take significant extra GPU time.

The native runner writes into `$LEWM_ROOT/experiments`. Each run contains a plan, frozen source and split, training progress, native recovery checkpoints, an audit, and held-out results. The checkpoint cadence is five minutes and the trainer writes a checkpoint when paused. In a standalone run, set the experiment's `control.json` request to `pause` using an atomic file replacement. To resume, set it back to `run` and run the same sweep command. Already-complete experiments are skipped.

## Windows dashboard and queue

On the configured desktop, `tools/desktop/Open-Training-Dashboard.cmd` opens the dashboard at <http://127.0.0.1:8765/>. It supports pause/resume, live test metrics, loss curve toggles and encoder spread. `Queue-DeltaSweep.ps1 -Epochs 50` stages the same three runs behind the experiment in the active pointer. Add `-Planning` to include the planning benchmark. The desktop queue and standalone command are alternative launchers for the same experiment directories.

Desktop helpers read the pointer at `%LEWM_DESKTOP_ROOT%\lewm-wsl\active-experiment.json`. The default desktop root is the current user's Desktop/AI directory. Set `LEWM_DISTRO` and `LEWM_WSL_REPO` to override Ubuntu-24.04 and the WSL checkout location. In WSL, set `LEWM_HOST_ROOT` to the corresponding mounted desktop root when using a nondefault installation. Datasets, private configuration, reference outputs and model weights are local runtime assets.

## Review changes

The fork's `main` is the upstream baseline. Open the [cross-fork comparison](https://github.com/Aringan0323/le-wm/compare/main...delta-jepa) or compare in a fresh WSL clone:

```bash
git remote add upstream https://github.com/lucas-maes/le-wm.git
git fetch upstream main
git diff --stat upstream/main...HEAD
git diff upstream/main...HEAD -- train.py module.py
```
