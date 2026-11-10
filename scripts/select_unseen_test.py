"""Select an outcome-independent, episode-disjoint test manifest from exposure replay."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--exposure-json', type=Path, required=True)
parser.add_argument('--exposure-npz', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--episodes', type=int, default=512)
parser.add_argument('--seed', type=int, default=20261106)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(f'Refusing to replace the predefined test set: {args.output}')
exposure = json.loads(args.exposure_json.read_text())
arrays = np.load(args.exposure_npz, allow_pickle=False)
lengths = arrays['episode_lengths'].astype(np.int64)
pool_key = 'conservative_unused_episode_ids' if 'conservative_unused_episode_ids' in arrays else 'unused_episode_ids'
forbidden = set(map(int, arrays['all_seen_episode_ids']))
if pool_key == 'conservative_unused_episode_ids':
    forbidden.update(set(range(len(lengths))) - set(map(int, arrays[pool_key])))
candidates = np.array(sorted(set(map(int, arrays[pool_key])) - forbidden), dtype=np.int64)
assert len(candidates) >= args.episodes, (len(candidates), args.episodes)
rng = np.random.default_rng(args.seed)
selected = sorted(map(int, rng.choice(candidates, size=args.episodes, replace=False)))
clip_counts = np.maximum(lengths - 19, 0)
clip_offsets = np.concatenate(([0], np.cumsum(clip_counts)[:-1]))
windows = []
for episode in selected:
    starts = np.arange(0, int(lengths[episode]) - 19, 20, dtype=np.int64)
    chosen = sorted(map(int, rng.choice(starts, size=min(4, len(starts)), replace=False)))
    assert all(b - a >= 20 for a, b in zip(chosen, chosen[1:]))
    for start in chosen:
        idx = int(clip_offsets[episode] + start)
        assert int(arrays['clip_episode_ids'][idx]) == episode
        assert int(arrays['clip_start_steps'][idx]) == start
        windows.append({'dataset_index': idx, 'episode_id': episode, 'start': start})
assert not (set(selected) & forbidden)
assert len({w['dataset_index'] for w in windows}) == len(windows)
train_indices = set(map(int, arrays['train_window_indices']))
validation_indices = set(map(int, arrays['validation_window_indices']))
assert not ({w['dataset_index'] for w in windows} & (train_indices | validation_indices))
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {
    'name': 'unseen-predictors-20261106-01',
    'selection_seed': args.seed,
    'exposure_pool': pool_key,
    'selection_policy': '512 episodes sampled uniformly without replacement from the reconstructed unused pool; up to 4 non-overlapping 20-step windows per episode, selected before evaluating losses.',
    'dataset_path': '/home/adam/lewm/storage/datasets/tworoom.h5',
    'history_size': 3, 'frameskip': 5, 'batch_size': 8,
    'eligible_episode_count': len(candidates), 'selected_episode_count': len(selected),
    'test_window_count': len(windows), 'predicted_transition_count': 3 * len(windows),
    'selected_episode_ids': selected,
    'forbidden_episode_ids': sorted(forbidden),
    'test_windows': windows,
    'exposure_provenance': {'json_path': str(args.exposure_json), 'json_sha256': sha(args.exposure_json),
                            'npz_path': str(args.exposure_npz), 'npz_sha256': sha(args.exposure_npz),
                            'report': exposure},
    'preprocessing_note': 'Existing models used action mean/std computed over the full dataset. These test episodes are excluded from reconstructed training forwards and previous validation/probes, but their actions contributed to those fixed normalization statistics.',
    'scope': 'Only next-embedding MSE; BF16 matching training validation is primary, FP32 inference is a separately labeled sensitivity check. No retraining or model selection on these test results.'
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(manifest, indent=2))
print(json.dumps({k: manifest[k] for k in ('name','eligible_episode_count','selected_episode_count','test_window_count','predicted_transition_count')}))
