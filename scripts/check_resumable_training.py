"""CPU-only, stochastic interruption/resume verification for the native driver helpers.

Uses a disposable tiny model and actual AdamW/scheduler; never loads observations
or runs the research model. RNG-dependent inputs exercise Python, NumPy, Torch,
and dropout state across object destruction and checkpoint reload.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path
import sys
import tempfile
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from stable_pretraining.optim.lr_scheduler import LinearWarmupCosineAnnealingLR
from resumable_training import (
    CheckpointManager, DurableTrainingTrace, FixedOrderBatchSampler,
    capture_rng_state, make_fixed_order, pause_requested, restore_rng_state,
)


class TinyDataset(Dataset):
    def __len__(self):
        return 41

    def __getitem__(self, idx):
        return {'_window_index': int(idx),
                'x': torch.tensor([idx / 41, (idx % 7) / 7, (idx % 3) / 3], dtype=torch.float32)}


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 11), nn.Tanh(), nn.Dropout(.3), nn.Linear(11, 2))

    def forward(self, x):
        return self.net(x)


class TinyAuxiliary(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('calls', torch.zeros((), dtype=torch.int64))

    def forward(self, x):
        self.calls.add_(1)
        return x.square().mean() * (1 + torch.rand(()))


class FakeClock:
    def __init__(self):
        self.now = 1000.

    def __call__(self):
        return self.now


def assert_equal(a, b, where='root'):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and torch.equal(a, b), where
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b, err_msg=where)
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), where
        for key in a:
            assert_equal(a[key], b[key], f'{where}.{key}')
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b), where
        for i, (left, right) in enumerate(zip(a, b)):
            assert_equal(left, right, f'{where}[{i}]')
    else:
        assert a == b, where


def seed_all():
    random.seed(901)
    np.random.seed(902)
    torch.manual_seed(903)


def make_parts(root, order, *, resume=None):
    model = TinyModel()
    auxiliary = TinyAuxiliary()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.001)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_steps=1, max_steps=6)
    sampler = FixedOrderBatchSampler(order, batch_size=8)
    trace = DurableTrainingTrace(root, order, resume=resume)
    manager = CheckpointManager(root, manifest_sha256='m' * 64, config_sha256='c' * 64,
                                batch_size=8, order=order, interval_seconds=300)
    loader = DataLoader(TinyDataset(), batch_sampler=sampler, num_workers=0,
                        generator=torch.Generator().manual_seed(3072))
    return model, auxiliary, optimizer, scheduler, sampler, trace, manager, loader


def train_batch(parts, batch):
    model, auxiliary, optimizer, scheduler, sampler, trace, _, _ = parts
    optimizer.zero_grad(set_to_none=True)
    out = model(batch['x'])
    random_offset = random.random() + float(np.random.uniform()) + float(torch.rand(()))
    target = torch.stack((batch['x'][:, 0] + random_offset, batch['x'][:, 1] - random_offset), dim=1)
    loss = (out - target).square().mean() + .09 * auxiliary(out)
    assert torch.isfinite(loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    scheduler.step()
    ids = batch['_window_index'].numpy()
    sampler.commit(ids)
    trace.append(ids)
    return float(loss.detach())


def state_of(parts):
    model, auxiliary, optimizer, scheduler, sampler, trace, _, _ = parts
    return {'model': model.state_dict(), 'auxiliary': auxiliary.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'cursor': sampler.cursor, 'global_step': sampler.global_step,
            'rng': capture_rng_state()}


def run_checks(work):
    order = make_fixed_order(np.arange(41, dtype=np.int64), seed=3072)
    independent = torch.Generator().manual_seed(3072)
    torch.empty((), dtype=torch.int64).random_(generator=independent)
    expected = torch.randperm(41, generator=independent).numpy()
    np.testing.assert_array_equal(order, expected)

    # A prefetched iterator must not mutate the durable committed cursor.
    sampler = FixedOrderBatchSampler(order, batch_size=8)
    it = iter(sampler)
    first, second = next(it), next(it)
    assert sampler.cursor == 0 and sampler.global_step == 0
    sampler.commit(first)
    assert sampler.cursor == 8 and sampler.global_step == 1
    np.testing.assert_array_equal(next(iter(FixedOrderBatchSampler(order, 8, cursor=8))), second)
    before = sampler.cursor
    try:
        sampler.commit(order[16:24])
    except (ValueError, AssertionError, RuntimeError):
        pass
    else:
        raise AssertionError('Wrong-order commit was accepted')
    assert sampler.cursor == before

    seed_all()
    uninterrupted = make_parts(work / 'uninterrupted', order)
    reference_losses = [train_batch(uninterrupted, batch) for batch in uninterrupted[-1]]
    reference = state_of(uninterrupted)
    uninterrupted[5].close()

    seed_all()
    first_run = make_parts(work / 'resumed', order)
    loader_it = iter(first_run[-1])
    partial_losses = [train_batch(first_run, next(loader_it)) for _ in range(3)]
    manager = first_run[6]
    saved = manager.save(model=first_run[0], optimizer=first_run[2], scheduler=first_run[3],
                         sigreg=first_run[1], sampler=first_run[4], trace=first_run[5], status='paused')
    checkpoint_path = Path(saved) if isinstance(saved, (str, Path)) else work / 'resumed' / 'checkpoints' / 'last.ckpt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    assert checkpoint['global_step'] == 3 and checkpoint['resumable']['cursor'] == 24
    # Simulate an update and trace record after the most recent durable checkpoint.
    train_batch(first_run, next(loader_it))
    first_run[5].close()
    del first_run, loader_it
    random.random(); np.random.uniform(); torch.rand(13)

    resumed = make_parts(work / 'resumed', order, resume=checkpoint['resumable'])
    loaded = resumed[6].load(checkpoint_path, model=resumed[0], optimizer=resumed[2], scheduler=resumed[3],
                             sigreg=resumed[1], sampler=resumed[4], restore_rng=False)
    resumed_it = iter(resumed[-1])
    restore_rng_state(loaded['resumable']['rng'])
    resumed_losses = [train_batch(resumed, batch) for batch in resumed_it]
    final = state_of(resumed)
    assert_equal(reference, final)
    assert reference_losses == partial_losses + resumed_losses
    final_path = resumed[6].save(model=resumed[0], optimizer=resumed[2], scheduler=resumed[3],
                                 sigreg=resumed[1], sampler=resumed[4], trace=resumed[5],
                                 status='complete', final=True)
    assert Path(final_path).name == 'final.ckpt'
    assert Path(final_path).samefile(Path(final_path).parent / 'last.ckpt')
    assert_equal(reference, state_of(resumed), 'checkpoint_save_preserves_state')
    wrong_identity = CheckpointManager(work / 'resumed', manifest_sha256='m' * 64,
                                        config_sha256='wrong', batch_size=8, order=order)
    try:
        wrong_identity.load(final_path, model=resumed[0], optimizer=resumed[2], scheduler=resumed[3],
                            sigreg=resumed[1], sampler=resumed[4])
    except ValueError as exc:
        assert 'config_sha256' in str(exc)
    else:
        raise AssertionError('Changed experiment identity was accepted')
    final_payload = torch.load(final_path, map_location='cpu', weights_only=False)
    sidecar = Path(final_path).parent / (final_payload['resumable']['generation_id'] + '.json')
    integrity = json.loads(sidecar.read_text())
    integrity['sha256'] = '0' * 64
    sidecar.write_text(json.dumps(integrity))
    try:
        resumed[6].load(final_path, model=resumed[0], optimizer=resumed[2], scheduler=resumed[3],
                        sigreg=resumed[1], sampler=resumed[4])
    except ValueError as exc:
        assert 'checksum' in str(exc)
    else:
        raise AssertionError('Incorrect checkpoint checksum was accepted')
    resumed[5].close()
    actual = np.fromfile(work / 'resumed' / 'training-indices.i64', dtype='<i8')
    np.testing.assert_array_equal(actual, order)
    assert len(actual) == len(set(actual.tolist())) == 41

    # Fake elapsed wall time: checkpoint deadline is 300 seconds, reset by a save.
    clock = FakeClock()
    timer = CheckpointManager(work / 'timing', manifest_sha256='m' * 64, config_sha256='c' * 64,
                              batch_size=8, order=order, interval_seconds=300, clock=clock)
    assert timer.due(), 'An initial step-zero checkpoint is required'
    timing_parts = make_parts(work / 'timing', order)
    def timed_save(status):
        return timer.save(model=timing_parts[0], optimizer=timing_parts[2], scheduler=timing_parts[3],
                          sigreg=timing_parts[1], sampler=timing_parts[4], trace=timing_parts[5], status=status)
    timed_save('running')
    assert not timer.due()
    clock.now += 299.9
    assert not timer.due()
    clock.now += .1
    assert timer.due()
    timed_save('running')
    assert not timer.due()
    clock.now += 1
    assert not timer.due()
    forced_pause = timed_save('paused')
    assert torch.load(forced_pause, map_location='cpu', weights_only=False)['resumable']['status'] == 'paused'
    timing_parts[5].close()
    # Pausing must be recognized from control state, independent of the timer.
    control = work / 'control.json'
    control.write_text(json.dumps({'request': 'pause'}))
    assert pause_requested(control)
    control.write_text(json.dumps({'request': 'run'}))
    assert not pause_requested(control)
    assert not torch.cuda.is_initialized(), 'CPU verification unexpectedly initialized CUDA'
    return {'cpu_only': True, 'windows': 41, 'batch_size': 8, 'steps': 6, 'tail': 1,
            'resume_step': 3, 'prefetch_does_not_commit': True,
            'wrong_order_commit_rejected': True, 'crash_trace_tail_reconciled': True,
            'sample_order_identical': True, 'model_optimizer_scheduler_auxiliary_rng_bitwise_identical': True,
            'loss_trajectory_bitwise_identical': True, 'checkpoint_300_second_deadline': True,
            'pause_control_recognized': True, 'final_alias_verified': True,
            'changed_config_rejected': True, 'invalid_checkpoint_checksum_rejected': True,
            'checkpoint_save_preserves_rng_and_state': True,
            'order_sha256': hashlib.sha256(np.asarray(order, dtype='<i8').tobytes()).hexdigest()}


class DriverDataset(Dataset):
    def __init__(self):
        self.transform = None

    def __len__(self):
        return 41

    def get_dim(self, name):
        return 2

    def __getitem__(self, index):
        return {'pixels': torch.arange(12, dtype=torch.float32).reshape(4, 3) / 13 + index / 41,
                'action': torch.arange(8, dtype=torch.float32).reshape(4, 2) / 9 + index / 41}


class DriverWorldModel(nn.Module):
    def __init__(self, action_head):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(3, 3), nn.Dropout(.2))
        self.action_encoder = nn.Linear(2, 3)
        self.predictor = nn.Sequential(nn.Linear(6, 3), nn.Dropout(.2))
        self.action_predictor = nn.Sequential(nn.Linear(3, 3), nn.Dropout(.2)) if action_head else None

    def encode(self, batch):
        return {'emb': self.encoder(batch['pixels']), 'act_emb': self.action_encoder(batch['action'])}

    def predict(self, emb, act):
        return self.predictor(torch.cat((emb, act), dim=-1))


def run_driver_checks(work):
    """Exercise actual _train control flow/forwards while replacing only external I/O/model size."""
    import csv
    from unittest.mock import patch
    from omegaconf import OmegaConf
    import train_resumable as driver

    work.mkdir(parents=True, exist_ok=True)
    bundle = {'manifest_path': str(work / 'manifest.json'),
              'manifest': {'manifest_sha256': 'd' * 64,
                           'training_order': {'batch_size': 8, 'steps_per_epoch': 6, 'seed': 3072}},
              'arrays': {'train_clip_indices': np.arange(41, dtype=np.int64)}}
    export_root = work / 'exports'
    def export(model, run_name, config, filename):
        destination = export_root / run_name
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), destination / filename)

    reports = []
    for objective in ('sigreg', 'action_prediction'):
        def configuration(label):
            return OmegaConf.create({
                'episode_manifest': bundle['manifest_path'], 'run_metadata_dir': str(work / objective / label),
                'resume_from': None, 'seed': 3072, 'cpu_threads': 1, 'compile_training': False,
                'data': {'dataset': {'name': 'tiny_fixture', 'keys_to_load': ['pixels', 'action'], 'frameskip': 1}},
                'img_size': 3, 'history_size': 3, 'num_preds': 1,
                'model': {'_target_': 'cpu_fixture.DriverWorldModel', 'encoder': {'pretrained': False},
                          'action_encoder': {'input_dim': 2}},
                'loader': {'batch_size': 8, 'num_workers': 0},
                'trainer': {'max_epochs': 1, 'max_steps': -1, 'accelerator': 'cpu',
                            'precision': '32-true', 'gradient_clip_val': 1.},
                'loss': {'type': objective, 'sigreg': {'weight': .09, 'kwargs': {'knots': 5, 'num_proj': 8}},
                         'action_prediction': {'weight': .09}},
                'optimizer': {'type': 'AdamW', 'lr': .003, 'weight_decay': .001},
                'checkpoint_every_seconds': 300, 'output_model_name': f'{objective}-{label}'})

        with patch.object(driver.swm.data, 'load_dataset', side_effect=lambda *a, **kw: DriverDataset()), \
             patch.object(driver, 'load_episode_experiment', return_value=bundle), \
             patch.object(driver, 'get_img_preprocessor', return_value=lambda x: x), \
             patch.object(driver, 'get_episode_normalizer', return_value=lambda x: x), \
             patch.object(driver.spt.data.transforms, 'Compose', side_effect=lambda *args: lambda x: x), \
             patch.object(driver.hydra.utils, 'instantiate', side_effect=lambda cfg: DriverWorldModel(objective == 'action_prediction')), \
             patch.object(driver.swm.data.utils, 'get_cache_dir', return_value=str(export_root)), \
             patch.object(driver.swm.wm.utils, 'save_pretrained', side_effect=export):
            reference_cfg = configuration('uninterrupted')
            with patch.object(driver, 'pause_requested', return_value=False):
                assert driver._train(reference_cfg) == 0
            resumed_cfg = configuration('resumed')
            checks = {'calls': 0}
            def pause_after_three(_):
                checks['calls'] += 1
                return checks['calls'] == 4  # Initial check, then completed batches1,2,3.
            with patch.object(driver, 'pause_requested', side_effect=pause_after_three):
                assert driver._train(resumed_cfg) == 75
            resumed_dir = Path(resumed_cfg.run_metadata_dir)
            paused = torch.load(resumed_dir / 'checkpoints' / 'last.ckpt', map_location='cpu', weights_only=False)
            assert paused['global_step'] == 3 and paused['resumable']['status'] == 'paused'
            resumed_cfg.resume_from = str(resumed_dir / 'checkpoints' / 'last.ckpt')
            with patch.object(driver, 'pause_requested', return_value=False):
                assert driver._train(resumed_cfg) == 0

        reference_dir = Path(reference_cfg.run_metadata_dir)
        reference = torch.load(reference_dir / 'checkpoints' / 'final.ckpt', map_location='cpu', weights_only=False)
        resumed = torch.load(resumed_dir / 'checkpoints' / 'final.ckpt', map_location='cpu', weights_only=False)
        for key in ('state_dict', 'optimizer_states', 'lr_schedulers', 'global_step'):
            assert_equal(reference[key], resumed[key], f'{objective}.{key}')
        assert_equal(reference['resumable']['rng'], resumed['resumable']['rng'], f'{objective}.rng')
        for directory in (reference_dir, resumed_dir):
            trace = np.fromfile(directory / 'training-indices.i64', dtype='<i8')
            np.testing.assert_array_equal(trace, make_fixed_order(np.arange(41, dtype=np.int64)))
            completion = json.loads((directory / 'runcompletion.json').read_text())
            assert completion['passed'] and completion['global_step'] == 6
            assert completion['resume_count'] == (1 if directory == resumed_dir else 0)
            exported = torch.load(completion['export_path'], map_location='cpu', weights_only=True)
            ckpt = torch.load(completion['final_ckpt_path'], map_location='cpu', weights_only=False)
            assert_equal(exported, {k.removeprefix('model.'): v for k, v in ckpt['state_dict'].items() if k.startswith('model.')})
        reference_rows = list(csv.DictReader((reference_dir / 'metrics.csv').open()))
        resumed_rows = list(csv.DictReader((resumed_dir / 'metrics.csv').open()))
        assert reference_rows == resumed_rows and len(reference_rows) == 6
        steps = [int(row['step']) for row in reference_rows]
        assert steps in (list(range(6)), list(range(1, 7)))
        original_cfg = OmegaConf.load(resumed_dir / 'config.yaml')
        assert original_cfg.resume_from is None
        reports.append({'objective': objective, 'steps': 6, 'resume_step': 3, 'windows': 41,
                        'exit75_on_pause': True, 'exit0_after_resume': True,
                        'state_optimizer_scheduler_rng_and_csv_exact_match': True,
                        'all_samples_once': True, 'export_matches_final_checkpoint': True,
                        'fresh_config_preserved': True, 'resume_count': 1,
                        'csv_steps': steps})
    assert not torch.cuda.is_initialized()
    return {'cpu_only': True, 'actual_native_driver_control_flow': True,
            'actual_training_objective_functions': True, 'fixture_model_and_data': True,
            'objectives': reports}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--driver-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = {'passed': False}
    try:
        with tempfile.TemporaryDirectory(prefix='lewm-resume-check-') as directory:
            report['checks'] = run_driver_checks(Path(directory)) if args.driver_only else run_checks(Path(directory))
        report['passed'] = True
    except Exception as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())