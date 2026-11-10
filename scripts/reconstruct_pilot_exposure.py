"""Reconstruct pilot window exposure with an index-only CPU scheduling replay.

No world-model weights, images, or CUDA computation are used. A zero-learning-rate
scalar fixture exercises installed Lightning/stable-pretraining loader control flow.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import traceback

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import lightning as pl
import stable_pretraining as spt

REPO = Path(__file__).resolve().parents[1]


def state_digest(generator):
    return hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()


class IndexOnlyDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.fetched = []

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        self.fetched.append(int(index))
        return {'window_index': int(index)}


class TracedLoader(DataLoader):
    def __iter__(self):
        before = state_digest(self.generator) if self.generator is not None else None
        iterator = super().__iter__()
        self.trace.append({'role': self.role, 'generator_before': before,
                           'generator_after_creation': state_digest(self.generator) if self.generator is not None else None,
                           'base_seed': iterator._base_seed,
                           'caller_stack': [{'file': frame.filename, 'line': frame.lineno, 'function': frame.name}
                                            for frame in traceback.extract_stack(limit=9)[:-1]]})
        return iterator


def split_indices(size):
    generator = torch.Generator().manual_seed(3072)
    train, val = spt.data.random_split(IndexOnlyDataset(size), [0.9, 1 - 0.9], generator=generator)
    return generator, train, val


def replay(size, label, output_root, steps=1000, smoke=False):
    pl.seed_everything(3072, workers=True)
    generator, train_set, val_set = split_indices(size)
    train_data, val_data = IndexOnlyDataset(size), IndexOnlyDataset(size)
    train_set.dataset, val_set.dataset = train_data, val_data
    trace = []
    kwargs = dict(batch_size=8, num_workers=0, persistent_workers=False, prefetch_factor=None, pin_memory=True)
    train = TracedLoader(train_set, **kwargs, shuffle=True, drop_last=True, generator=generator)
    val = TracedLoader(val_set, **kwargs, shuffle=False, drop_last=False)
    for loader, role in ((train, 'train'), (val, 'validation')):
        loader.trace, loader.role = trace, role
    optimized, validated, sanity = [], [], []
    def fixture_forward(self, batch, stage):
        index = batch['window_index'].tolist()
        if stage == 'fit':
            optimized.extend(index)
        elif self.trainer.sanity_checking:
            sanity.extend(index)
        else:
            validated.extend(index)
        loss = 0.0 * (self.model.weight.sum() + self.model.bias.sum())
        self.log_dict({stage + '/loss': loss.detach()}, on_step=True, sync_dist=True, batch_size=len(index))
        return {'loss': loss}
    model = spt.Module(model=nn.Linear(1, 1), forward=fixture_forward,
                       optim={'model_opt': {'modules': 'model', 'optimizer': {'type': 'AdamW', 'lr': 0.0},
                                            'scheduler': {'type': 'LinearWarmupCosineAnnealingLR'}, 'interval': 'step'}})
    spt.set(cache_dir=str(output_root / label), requeue_checkpoint_every_n_steps=0 if smoke else 250)
    trainer = pl.Trainer(max_epochs=1, max_steps=-1 if smoke else steps,
                         accelerator='cpu', devices=1, precision='32-true',
                         gradient_clip_val=1.0, limit_train_batches=steps,
                         limit_val_batches=1 if smoke else 100,
                         val_check_interval=1.0 if smoke else 250, num_sanity_val_steps=1,
                         log_every_n_steps=1 if smoke else 10, enable_progress_bar=False,
                         logger=None, enable_checkpointing=True,
                         default_root_dir=str(output_root / label))
    manager = spt.Manager(trainer=trainer, module=model,
                          data=spt.data.DataModule(train=train, val=val), seed=3072,
                          ckpt_path=None, weights_only=False)
    manager()
    assert trainer.global_step == steps
    assert len(optimized) == steps * 8
    return {'optimized': np.asarray(optimized, dtype=np.int64),
            'fetched': np.asarray(train_data.fetched, dtype=np.int64),
            'validated': np.asarray(validated, dtype=np.int64),
            'sanity': np.asarray(sanity, dtype=np.int64),
            'validation_fetched': np.asarray(val_data.fetched, dtype=np.int64),
            'iterator_trace': trace, 'global_step': trainer.global_step,
            'train_generator_final_sha256': state_digest(generator)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/home/adam/lewm/logs/pilot-exposure.json'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    storage = Path(os.environ.get('STABLEWM_HOME', '/home/adam/lewm/storage'))
    with h5py.File(storage / 'datasets/tworoom.h5', 'r') as dataset:
        lengths, offsets = dataset['ep_len'][:].astype(np.int64), dataset['ep_offset'][:].astype(np.int64)
    clips_per_episode = np.maximum(lengths - 19, 0)
    size = int(clips_per_episode.sum())
    clip_episode = np.repeat(np.arange(len(lengths), dtype=np.int64), clips_per_episode)
    clip_starts = np.concatenate([np.arange(count, dtype=np.int64) for count in clips_per_episode])
    generator, train_set, val_set = split_indices(size)
    train_split, val_split = np.asarray(train_set.indices), np.asarray(val_set.indices)
    split_state = state_digest(generator)
    expected_base_seed = int(torch.empty((), dtype=torch.int64).random_(generator=generator))
    order = torch.randperm(len(train_split), generator=generator).numpy()
    expected_optimized = train_split[order[:8000]]
    expected_val = val_split[:800]
    replay_root = args.output.parent / 'pilot-exposure-replay'
    pilot = replay(size, 'pilot', replay_root)
    smoke = replay(size, 'smoke', replay_root, steps=2, smoke=True)
    np.testing.assert_array_equal(pilot['optimized'], expected_optimized)
    np.testing.assert_array_equal(np.unique(pilot['validated']), np.unique(expected_val))
    np.testing.assert_array_equal(smoke['optimized'], expected_optimized[:16])
    assert len([event for event in pilot['iterator_trace'] if event['role'] == 'train']) == 1
    assert next(event for event in pilot['iterator_trace'] if event['role'] == 'train')['base_seed'] == expected_base_seed
    probes = np.unique(np.r_[np.linspace(0, size - 1, 32, dtype=np.float32).astype(np.int64),
                              torch.linspace(0, size - 1, 32).long().numpy(), 0]).astype(np.int64)
    # Preserve exact torch.linspace probe IDs; also exclude NumPy variants conservatively.
    seen_windows = np.unique(np.concatenate([pilot['optimized'], pilot['fetched'], pilot['validation_fetched'],
                                             smoke['fetched'], smoke['validation_fetched'], probes]))
    optimized_episodes = np.unique(clip_episode[pilot['optimized']])
    seen_episodes = np.unique(clip_episode[seen_windows])
    unused_episodes = np.setdiff1d(np.arange(len(lengths)), seen_episodes)
    conservative_candidates = []
    for base_seed_draws in (0, 1, 2):
        candidate_generator, candidate_train, _ = split_indices(size)
        for _ in range(base_seed_draws):
            torch.empty((), dtype=torch.int64).random_(generator=candidate_generator)
        candidate_order = torch.randperm(len(candidate_train), generator=candidate_generator).numpy()
        conservative_candidates.append(np.asarray(candidate_train.indices)[candidate_order[:8008]])
    conservative_candidates = np.unique(np.concatenate(conservative_candidates))
    conservative_forbidden = np.unique(np.r_[seen_episodes, clip_episode[conservative_candidates]])
    conservative_unused = np.setdiff1d(np.arange(len(lengths)), conservative_forbidden)
    optimized_base_rows = offsets[clip_episode[pilot['optimized']]] + clip_starts[pilot['optimized']]
    observed_frames = np.unique((optimized_base_rows[:, None] + np.array([0, 5, 10, 15])).ravel())
    fetched_extra = np.setdiff1d(pilot['fetched'], pilot['optimized'])
    fetched_extra_rows = offsets[clip_episode[fetched_extra]] + clip_starts[fetched_extra]
    extra_frames = np.unique((fetched_extra_rows[:, None] + np.array([0, 5, 10, 15])).ravel())
    action_rows = np.unique((optimized_base_rows[:, None] + np.arange(15)).ravel())
    output_npz = args.output.with_suffix('.npz')
    np.savez_compressed(output_npz,
        train_window_indices=pilot['optimized'], train_fetched_window_indices=pilot['fetched'],
        train_fetch_only_window_indices=fetched_extra, validation_window_indices=expected_val,
        validation_fetched_window_indices=np.unique(pilot['validation_fetched']),
        smoke_train_window_indices=smoke['optimized'], smoke_fetched_window_indices=smoke['fetched'],
        startup_sanity_window_indices=pilot['sanity'], excluded_probe_window_indices=probes,
        training_episode_ids=optimized_episodes, all_seen_episode_ids=seen_episodes, unused_episode_ids=unused_episodes,
        all_seen_window_indices=seen_windows, clip_episode_ids=clip_episode, clip_start_steps=clip_starts,
        conservative_training_candidate_window_indices=conservative_candidates,
        conservative_forbidden_episode_ids=conservative_forbidden, conservative_unused_episode_ids=conservative_unused,
        episode_lengths=lengths, episode_offsets=offsets, train_observed_frame_rows=observed_frames,
        train_fetch_only_frame_rows=extra_frames, train_used_action_rows=action_rows,
        train_split_indices=train_split, val_split_indices=val_split)
    dependency_paths = [Path(inspect.getfile(DataLoader)), Path(inspect.getfile(torch.utils.data.RandomSampler)),
                        Path(inspect.getfile(spt.data.random_split)), Path(inspect.getfile(spt.data.DataModule)),
                        Path(inspect.getfile(spt.Manager))]
    configs = {}
    from omegaconf import OmegaConf
    for name in ('tworoom-pilot-20261106-01', 'tworoom-action-pilot-20261106-01'):
        path = storage / 'checkpoints' / name / 'config.yaml'
        cfg = OmegaConf.load(path)
        configs[name] = {'seed': cfg.seed, 'train_split': cfg.train_split,
                         'history_size': cfg.history_size, 'num_preds': cfg.num_preds,
                         'frameskip': cfg.data.dataset.frameskip,
                         'loader': OmegaConf.to_container(cfg.loader, resolve=True),
                         'max_steps': cfg.trainer.max_steps, 'limit_train_batches': cfg.trainer.limit_train_batches,
                         'limit_val_batches': cfg.trainer.limit_val_batches, 'val_check_interval': cfg.trainer.val_check_interval}
    assert configs['tworoom-pilot-20261106-01'] == configs['tworoom-action-pilot-20261106-01']
    result = {
        'passed': True, 'created_utc': datetime.now(timezone.utc).isoformat(), 'arrays': str(output_npz),
        'dataset': {'episodes': len(lengths), 'rows': int(lengths.sum()), 'valid_windows': size,
                    'window_raw_span': 20, 'observation_offsets': [0, 5, 10, 15]},
        'split': {'train_windows': len(train_split), 'validation_windows': len(val_split), 'seed': 3072,
                  'post_split_generator_sha256': split_state, 'train_iterator_base_seed': expected_base_seed},
        'training_coverage': {'optimizer_steps_per_run': 1000, 'windows_per_run': 8000,
                             'unique_windows_per_run': len(np.unique(pilot['optimized'])),
                             'fraction_all_valid_windows': len(np.unique(pilot['optimized'])) / size,
                             'fraction_train_split_windows': len(np.unique(pilot['optimized'])) / len(train_split),
                             'unique_episodes': len(optimized_episodes),
                             'unique_observed_frames': len(observed_frames),
                             'fraction_dataset_frames_observed': len(observed_frames) / int(lengths.sum()),
                             'frame_presentations': 32000, 'unique_raw_action_rows_in_objective': len(action_rows),
                             'fetched_windows': len(pilot['fetched']), 'fetch_only_windows': len(fetched_extra),
                             'fetch_only_unique_observed_frames': len(extra_frames),
                             'two_runs_share_same_optimized_window_order': True},
        'validation_exposure': {'unique_evaluated_windows': len(np.unique(pilot['validated'])),
                                'forward_window_presentations': len(pilot['validated']),
                                'startup_sanity_windows': len(pilot['sanity']),
                                'unique_fetched_windows_including_sanity': len(np.unique(pilot['validation_fetched'])),
                                'unique_episodes_first_800': len(np.unique(clip_episode[expected_val]))},
        'retrospective_episode_holdout': {'excluded_probe_windows': probes.tolist(),
                                          'seen_episode_count_union': len(seen_episodes),
                                          'unused_whole_episode_count': len(unused_episodes),
                                          'conservative_unused_whole_episode_count': len(conservative_unused),
                                          'conservative_forbidden_episode_count': len(conservative_forbidden),
                                          'conservative_additional_exclusion': 'Union first8008candidate training windows under0/1/2base-seed draws after split, plus all actual replay and known probes; exact replay count remains separate.',
                                          'exclusion_basis': 'All replayed optimization and fetched windows, all validation/sanity fetches, smoke fetches, historical first-sample and 32-window probes; exclude each entire containing episode.'},
        'replay': {'kind': 'Installed Lightning + stable-pretraining Manager/Module/DataModule with index-only dataset and fixed zero-LR scalar fixture; CPU only, no world model or image loading.',
                   'pilot_global_step': pilot['global_step'], 'smoke_global_step': smoke['global_step'],
                   'direct_sampler_reconstruction_matches_replay_exactly': True,
                   'smoke_matches_first_16_pilot_windows': True,
                   'pilot_iterator_trace': pilot['iterator_trace'], 'smoke_iterator_trace': smoke['iterator_trace']},
        'matching_original_run_configs': configs,
        'versions': {name: importlib.metadata.version(name) for name in ('torch', 'lightning', 'stable-pretraining', 'stable-worldmodel')},
        'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in dependency_paths},
        'limitations': ['Historical runs did not log sample IDs; this is a deterministic reconstruction verified against index-only loader/loop replay, not a historical access trace.',
                        'CPU replay replaces accelerator, numeric precision, model forward, and export callback; these do not consume the dedicated DataLoader generator. Installed callback/data-loader control flow remains in use.',
                        'Action/proprio normalization statistics used the full dataset before the split, so unseen episodes are not statistically pristine external test data.',
                        'Episode holdout is retrospective and excludes observed training, validation, sanity, smoke, and known diagnostic windows; no external test dataset exists in this local file.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2), encoding='utf-8')
    temporary.replace(args.output)
    print(json.dumps({key: result[key] for key in ('passed', 'arrays', 'training_coverage', 'validation_exposure', 'retrospective_episode_holdout')}, indent=2), flush=True)


if __name__ == '__main__':
    main()