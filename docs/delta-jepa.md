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

### Shared-batch training

The shared trainer keeps all three models on one GPU. It transfers one batch and performs three independent optimizer updates in sequence, releasing each model's gradients and intermediate activations before the next update. The zero-LDAD control omits the decoder completely. The two positive-weight members have separate decoders. All members have their own action-conditioned forward predictor, optimizer, scheduler, stochastic state and loss records.

From an activated WSL checkout, create a fresh group and start it:

```bash
GROUP="$LEWM_ROOT/experiments/episode50-delta-shared-20261110-01"
python scripts/prepare_delta_group.py --experiment-dir "$GROUP" --manifest-dir "$SPLIT" --epochs 50
bash "$GROUP/run.sh"
```

The training batch size defaults to 32. Pass `--batch-size 80` in WSL or `-BatchSize 80` on the desktop launcher to start a fresh group at batch size 80. On the fixed Two-Room split, this gives 4,571 updates per epoch and a final batch of 57. The same windows are visited in the same order. All held-out evaluations use batch size 8. To measure a candidate on a free GPU, run `python scripts/probe_delta_batch.py --experiment-dir "$GROUP" --batch-size 80 --steps 4` against a stopped group.

The dashboard launcher for a fresh group is `tools/desktop/Start-DeltaGroup.ps1`. It uses the active experiment's sealed split after the old run has completed or been cancelled. One Pause request checkpoints all three models at the same committed batch. Resume restores this common generation. The live group counter advances once after updating all three models. `gpu-telemetry.csv` records device-wide samples and `gpu-telemetry.json` gives a rolling utilization average and a training ETA. Held-out evaluation for each model follows training.

### Planning from a paused checkpoint

The short checkpoint evaluation uses the released LeWM Two-Room planner that Delta-JEPA follows. It chooses 20 distinct held-out episodes with one planner seed, uses goals 25 physical steps ahead and allows 50 steps to reach them. CEM evaluates 300 sequences for 30 iterations, retaining 30 elites. Each sequence contains five chunks of five actions, and the chosen 25 actions execute before replanning. Success uses the simulator's 16-pixel distance threshold.

The 25-step goal offset and 50-step budget follow the [maintainer's correction](https://github.com/lucas-maes/le-wm/issues/38). The original Delta-JEPA table uses 500 test trajectories and three evaluation seeds. Our short test selects distinct held-out episodes and then a start within each, while released LeWM evaluation samples valid starting rows. All three models receive the same tasks and sampling seeds. Models load from one verified native checkpoint generation and use the training split's action normalization.

### Continuous complete-episode planning

To evaluate a paused shared checkpoint through complete held-out episodes:

```bash
./lewm.sh python scripts/plan_delta_episodes.py --experiment-dir "$GROUP" --episodes 20 --seed 42
```

Each episode resets once to its first recorded state. Its ordered goals are the recorded states at steps 25, 50, 75, and so on, including the final state even when the final interval is shorter than 25 steps. The simulator retains its actual agent position between goals, and the planner encodes the current rendered observation. It never teleports to the recorded intermediate states. Every model uses the same selected episodes, goal sequence, and per-goal sampling seeds.

Each goal has a budget of 50 physical actions. Reaching a distance strictly below 16 pixels succeeds; a goal already within that tolerance succeeds with zero actions and is counted separately. After a failed goal, evaluation continues to the next goal from the actual final position. Task success is reached goals divided by attempted goals. Episode success requires reaching every goal in that episode; final-goal success is also recorded separately. These episode metrics extend the isolated start/goal test and should not be treated as the paper's original task-success benchmark.

Results and continuity records are written under `planning/episodes-checkpoint-STEP`, preserving earlier single-task tests. The dashboard displays episode and task success together. The evaluator validates one common native checkpoint generation, keeps weights and buffers frozen, and holds the training locks until testing completes. It does not automatically resume training; resume through the normal controls after verifying results. Interrupted tests reuse only complete, checksum-validated episode records, and rerun an unfinished episode from its original start.

### Sequential queue

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
