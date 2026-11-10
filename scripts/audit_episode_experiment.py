"""Audit a completed one-pass episode experiment without evaluating held-out data.

Required evidence lives in --run-dir: initialization.json, training-indices.i64,
data-order.json, runcompletion.json. The final checkpoint is loaded on CPU only.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from episode_data import load_episode_experiment
from experiment_tracking import state_dict_sha256
from resumable_training import EpochOrder


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def expected_order(bundle, epoch=0):
    spec = bundle['manifest']['training_order']
    generator = torch.Generator().manual_seed(spec['seed'] + epoch)
    base_seed = int(torch.empty((), dtype=torch.int64).random_(generator=generator))
    indices = bundle['arrays']['train_clip_indices']
    order = indices[torch.randperm(len(indices), generator=generator).numpy()]
    return np.asarray(order, dtype='<i8'), base_seed


def audit_native_checkpoint(checkpoint, checkpoint_path, run_dir, manifest_sha,
                            expected, observed_hash, batch_size, expected_steps):
    """Check durable recovery provenance without changing checkpoint or RNG state."""
    metadata = checkpoint['resumable']
    scheduled = isinstance(expected, EpochOrder)
    assert metadata['version'] == (2 if scheduled else 1) and metadata['backend'] == 'native_pytorch'
    assert metadata['status'] == 'complete'
    assert metadata['manifest_sha256'] == manifest_sha
    assert metadata['batch_size'] == batch_size
    assert metadata['cursor'] == metadata['total_windows'] == len(expected)
    assert metadata['expected_steps'] == checkpoint['global_step'] == expected_steps
    if scheduled:
        assert 'order' not in metadata and metadata['order_protocol'] == expected.protocol
        assert checkpoint['epoch'] == expected.epochs - 1
    else:
        np.testing.assert_array_equal(metadata['order'].cpu().numpy(), expected)
    assert metadata['order_sha256'] == metadata['trace_sha256'] == observed_hash
    assert metadata['trace_bytes'] == len(expected) * np.dtype('<i8').itemsize
    assert len(metadata['config_sha256']) == 64
    generation_id = metadata['generation_id']
    assert Path(generation_id).name == generation_id
    generation = checkpoint_path.parent / f'{generation_id}.ckpt'
    sidecar = read_json(checkpoint_path.parent / f'{generation_id}.json')
    assert generation.is_file() and generation.samefile(checkpoint_path)
    assert sidecar['generation_id'] == generation_id
    assert sidecar['global_step'] == expected_steps and sidecar['windows_seen'] == len(expected)
    digest = file_sha256(checkpoint_path)
    assert sidecar['size_bytes'] == checkpoint_path.stat().st_size
    assert sidecar['sha256'] == digest
    latest = read_json(run_dir / 'checkpoint.json')
    assert latest['generation_id'] == generation_id and latest['status'] == 'complete'
    assert latest['sha256'] == digest and latest['global_step'] == expected_steps
    assert Path(latest['path']).resolve() == checkpoint_path.resolve()
    rng = metadata['rng']
    assert set(rng) == {'python', 'numpy', 'torch_cpu', 'torch_cuda'}
    assert isinstance(rng['python'], tuple) and isinstance(rng['numpy'], tuple)
    assert rng['torch_cpu'].dtype == torch.uint8 and rng['torch_cpu'].ndim == 1
    assert rng['torch_cuda'] is not None and len(rng['torch_cuda']) == 1
    assert all(item.dtype == torch.uint8 and item.ndim == 1 for item in rng['torch_cuda'])
    generations = [read_json(path) for path in checkpoint_path.parent.glob('step-*.json')]
    assert generations and any(item['global_step'] == 0 for item in generations)
    assert all(0 <= item['global_step'] <= expected_steps for item in generations)
    retained = list(checkpoint_path.parent.glob('step-*.ckpt'))
    if scheduled:
        assert 2 <= len(retained) <= 3 and any(p.name.startswith('step-00000000-') for p in retained)
        assert file_sha256(run_dir / 'first-step-gradients.json') == metadata['extra']['first_step_gradient_sha256']
    return {'backend': 'native_pytorch', 'protocol_version': metadata['version'], 'retained_generation_count': len(retained),
            'committed_cursor': metadata['cursor'], 'config_sha256': metadata['config_sha256'],
            'generation_id': generation_id, 'checksum_verified': True, 'sha256': digest,
            'rng_states_present': ['python', 'numpy', 'torch_cpu', 'torch_cuda'],
            'generation_count': len(generations), 'initial_step_zero_checkpoint_present': True}

def audit(args):
    run_dir = args.run_dir.resolve(strict=True)
    bundle = load_episode_experiment(args.manifest)
    manifest, arrays = bundle['manifest'], bundle['arrays']
    manifest_sha = manifest['manifest_sha256']
    epochs = getattr(args, 'expected_epochs', 1)
    scheduled = args.expected_objective == 'delta_jepa'
    assert isinstance(epochs, int) and epochs >= 1 and (scheduled or epochs == 1)
    windows_per_epoch = manifest['counts']['train']['windows']
    expected_windows = epochs * windows_per_epoch
    batch_size = manifest['training_order']['batch_size']
    assert isinstance(batch_size, int) and not isinstance(batch_size, bool) and batch_size > 0
    epoch_steps = math.ceil(windows_per_epoch / batch_size)
    expected_steps = epochs * epoch_steps
    expected_tail = windows_per_epoch - (epoch_steps - 1) * batch_size
    assert epoch_steps == manifest['training_order']['steps_per_epoch']
    assert 1 <= expected_tail <= batch_size
    assert manifest['training_order']['drop_last'] is False
    assert [manifest['counts'][name]['episodes'] for name in ('train', 'test', 'unused')] == [5000, 3000, 2000]
    completion = read_json(run_dir / 'runcompletion.json')
    initialization = read_json(run_dir / 'initialization.json')
    order_record = read_json(run_dir / 'data-order.json')
    assert completion['passed'] is True
    assert completion['global_step'] == expected_steps
    assert completion['objective'] == args.expected_objective
    assert all(item['manifest_sha256'] == manifest_sha for item in (completion, initialization, order_record))
    process_exit_code = None
    for filename in ('training-exit.code', 'exit.code'):
        if (run_dir / filename).is_file():
            process_exit_code = int((run_dir / filename).read_text().strip())
            assert process_exit_code == 0, f'Training process exit code was {process_exit_code}'
            break
    assert order_record['status'] == 'complete' and order_record['counts_match'] is True
    assert order_record['expected_steps'] == order_record['optimizer_steps'] == order_record['batches_seen'] == expected_steps
    assert order_record['expected_windows'] == order_record['windows_seen'] == expected_windows
    assert order_record['last_batch_size'] == expected_tail
    assert order_record['trace_dtype'] == '<i8'
    assert order_record.get('nonfinite_loss_name') is None
    assert math.isfinite(order_record['last_train_loss'])
    trace_path = (run_dir / order_record['trace_file']).resolve(strict=True)
    assert trace_path.parent == run_dir, 'Trace must remain inside its run directory'
    assert trace_path.stat().st_size == order_record['trace_bytes'] == expected_windows * np.dtype('<i8').itemsize
    if scheduled:
        assert order_record['epochs'] == completion['epochs'] == epochs
        assert order_record['completed_epochs'] == epochs
    observed_order = np.memmap(trace_path, dtype='<i8', mode='r')
    observed_hash = file_sha256(trace_path)
    assert observed_hash == order_record['order_sha256']
    epoch_hashes = []
    for epoch in range(epochs):
        expected_epoch, epoch_base_seed = expected_order(bundle, epoch)
        observed = observed_order[epoch * windows_per_epoch:(epoch + 1) * windows_per_epoch]
        np.testing.assert_array_equal(observed, expected_epoch)
        # Check the actual train set independently in every epoch, including each short tail.
        np.testing.assert_array_equal(np.sort(observed), arrays['train_clip_indices'])
        epoch_hashes.append(hashlib.sha256(observed).hexdigest())
        if epoch == 0:
            base_seed = epoch_base_seed
    expected = EpochOrder(arrays['train_clip_indices'], manifest['training_order']['seed'], epochs) if scheduled else expected_epoch
    if scheduled:
        assert expected.order_hash == observed_hash and epoch_hashes == expected.protocol['epoch_sha256']
    assert observed_order[:len(order_record['first_ids'])].tolist() == order_record['first_ids']
    assert observed_order[-len(order_record['last_ids']):].tolist() == order_record['last_ids']
    seen_episodes = np.unique(arrays['clip_episode_ids'][observed_order[:windows_per_epoch]])
    np.testing.assert_array_equal(seen_episodes, np.sort(arrays['train_episode_ids']))
    assert not np.intersect1d(seen_episodes, arrays['test_episode_ids']).size
    assert not np.intersect1d(seen_episodes, arrays['unused_episode_ids']).size

    checkpoint_path = Path(completion['final_ckpt_path']).resolve(strict=True)
    export_path = Path(completion['export_path']).resolve(strict=True)
    registry = Path(completion['registry_run']).resolve(strict=True)
    assert checkpoint_path.parent == registry / 'checkpoints'
    assert checkpoint_path.name == 'final.ckpt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    native_checkpoint = checkpoint.get('resumable') is not None
    config_path = export_path.parent / 'config.yaml'
    cfg = OmegaConf.load(config_path)
    assert Path(cfg.episode_manifest).resolve() == args.manifest.resolve()
    assert cfg.episode_manifest_sha256 == manifest_sha
    assert cfg.seed == manifest['training_order']['seed'] == initialization['seed']
    assert cfg.loader.batch_size == batch_size
    workers = int(cfg.num_workers)
    assert workers >= 0 and cfg.loader.num_workers == workers
    persistent_workers = cfg.loader.get('persistent_workers', False)
    prefetch_factor = cfg.loader.get('prefetch_factor')
    assert isinstance(persistent_workers, bool)
    if workers == 0:
        assert persistent_workers is False and prefetch_factor is None
    else:
        assert prefetch_factor is None or (isinstance(prefetch_factor, int)
                                           and not isinstance(prefetch_factor, bool) and prefetch_factor > 0)
    assert cfg.trainer.max_epochs == epochs and cfg.trainer.get('max_steps', -1) == -1
    assert initialization['resume_from'] is None, 'The experiment must originate from a fresh initialization'
    if not native_checkpoint:
        assert cfg.get('resume_from') is None
    assert cfg.model.encoder.pretrained is False and initialization['encoder_pretrained'] is False
    assert cfg.loss.get('type', 'sigreg') == args.expected_objective
    if native_checkpoint:
        assert cfg.get('checkpoint_every_seconds') == 300
    else:
        assert cfg.get('checkpoint_every_n_steps') == 1000
    assert cfg.trainer.precision == 'bf16-mixed'
    assert cfg.trainer.get('limit_val_batches', 0) == 0
    if scheduled:
        assert cfg.loss.ldad.weight == args.expected_ldad_weight and cfg.loss.sigreg.weight == args.expected_sigreg_weight
        assert (cfg.loss.sigreg.weight > 0) == any(k.startswith('sigreg.') for k in checkpoint['state_dict'])
    elif args.expected_objective == 'mse_only':
        assert cfg.loss.get('sigreg') is None and cfg.loss.get('action_prediction') is None
        assert cfg.model.get('action_predictor') is None
    elif args.expected_objective == 'action_prediction':
        assert cfg.loss.sigreg is None and cfg.loss.action_prediction.weight == getattr(args, 'expected_action_weight', 0.09)
        assert not any(key.startswith('sigreg.') for key in checkpoint['state_dict']), 'Unexpected SIGReg checkpoint state'
    else:
        assert cfg.loss.sigreg.weight == getattr(args, 'expected_sigreg_weight', 0.09)

    # Independently reconstruct fresh initialization on CPU; no samples or test data.
    torch.manual_seed(initialization['seed'])
    model = hydra.utils.instantiate(cfg.model)
    fresh_state = model.state_dict()
    fresh_full_hash = state_dict_sha256(fresh_state)
    exclusions = ('action_predictor.', 'ldad.') if scheduled else ('action_predictor.',)
    assert initialization['shared_state_excluded_prefixes'] == list(exclusions)
    fresh_shared_hash = state_dict_sha256(fresh_state, exclude_prefixes=exclusions)
    assert fresh_full_hash == initialization['state_sha256'], 'Recorded initialization was not the declared fresh model'
    assert fresh_shared_hash == initialization['shared_state_sha256']
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert parameter_count == initialization['parameter_count'] == initialization['trainable_parameter_count']
    assert initialization['state_entry_count'] == len(fresh_state)
    assert initialization['shared_state_entry_count'] == sum(not name.startswith(exclusions) for name in fresh_state)

    recovery = (audit_native_checkpoint(checkpoint, checkpoint_path, run_dir, manifest_sha, expected,
                                        observed_hash, batch_size, expected_steps) if native_checkpoint else None)
    resume_count = 0
    if native_checkpoint:
        config_identity = read_json(run_dir / 'config-identity.json')
        assert config_identity['excluded_runtime_keys'] == ['resume_from', 'control_file']
        resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        fingerprint = {key: value for key, value in resolved.items() if key not in ('resume_from', 'control_file')}
        config_hash = hashlib.sha256(json.dumps(fingerprint, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        assert config_hash == config_identity['config_sha256'] == checkpoint['resumable']['config_sha256']
        assert config_identity['manifest_sha256'] == manifest_sha
        assert checkpoint['resumable']['extra']['objective'] == args.expected_objective
        events_file = run_dir / 'resume_events.json'
        events = read_json(events_file) if events_file.exists() else []
        resume_count = len(events)
        assert completion.get('resume_count', 0) == order_record.get('resume_count', 0) == resume_count
        for event in events:
            assert event['config_sha256'] == config_hash
            assert 0 <= event['windows_seen'] <= expected_windows
            completed_epochs, within_epoch = divmod(event['windows_seen'], windows_per_epoch)
            assert within_epoch % batch_size == 0
            assert event['global_step'] == completed_epochs * epoch_steps + math.ceil(within_epoch / batch_size)
        recovery.update(config_identity_verified=True, resume_count=resume_count)
    assert checkpoint['global_step'] == expected_steps
    for name, value in checkpoint['state_dict'].items():
        if torch.is_tensor(value):
            assert torch.isfinite(value).all(), f'Nonfinite checkpoint tensor: {name}'
    assert len(checkpoint['optimizer_states']) == 1
    optimizer = checkpoint['optimizer_states'][0]
    parameter_steps, nonzero_moments = [], 0
    for state in optimizer['state'].values():
        for value in state.values():
            if torch.is_tensor(value):
                assert torch.isfinite(value).all(), 'Nonfinite optimizer state'
        if 'step' in state:
            parameter_steps.append(int(state['step']))
        if 'exp_avg' in state:
            nonzero_moments += int(bool(torch.count_nonzero(state['exp_avg'])))
    decoder_disabled = scheduled and cfg.loss.ldad.weight == 0
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if scheduled:
        assert len(optimizer['param_groups']) == 1
        ids = optimizer['param_groups'][0]['params']
        assert len(ids) == len(names)
        expected_active = {pid for pid, name in zip(ids, names) if not (decoder_disabled and name.startswith('ldad.'))}
        assert set(optimizer['state']) == expected_active
    assert len(parameter_steps) == sum(not (decoder_disabled and n.startswith('ldad.')) for n in names)
    assert min(parameter_steps) == max(parameter_steps) == expected_steps
    assert nonzero_moments > 0
    assert len(checkpoint['lr_schedulers']) == 1
    scheduler = checkpoint['lr_schedulers'][0]
    assert scheduler['max_steps'] == expected_steps
    assert scheduler['warmup_steps'] == max(1, int(0.01 * expected_steps))
    assert scheduler['last_epoch'] == expected_steps
    assert all(group['lr'] == 0.0 for group in optimizer['param_groups'])
    exported = torch.load(export_path, map_location='cpu', weights_only=True)
    checkpoint_model = {name.removeprefix('model.'): value for name, value in checkpoint['state_dict'].items() if name.startswith('model.')}
    assert set(exported) == set(checkpoint_model) == set(fresh_state)
    for name, value in exported.items():
        assert torch.isfinite(value).all() and torch.equal(value, checkpoint_model[name]), name
    final_state_hash = state_dict_sha256(exported)
    assert final_state_hash != fresh_full_hash, 'Final model is unchanged from initialization'
    if scheduled:
        decoder_changed = any(not torch.equal(fresh_state[n], v) for n, v in exported.items() if n.startswith('ldad.'))
        assert decoder_changed != decoder_disabled
    model.load_state_dict(exported, strict=True)

    metrics_path = registry / 'metrics.csv'
    if native_checkpoint:
        metrics_metadata = checkpoint['resumable']['extra']
        assert metrics_path.stat().st_size == metrics_metadata['metrics_bytes']
        assert file_sha256(metrics_path) == metrics_metadata['metrics_sha256']
    metrics = {}
    loss_fields = ['fit/loss', 'fit/pred_loss']
    if scheduled:
        loss_fields.append('fit/ldad_loss')
        if cfg.loss.sigreg.weight > 0:
            loss_fields.append('fit/sigreg_loss')
    elif args.expected_objective != 'mse_only':
        loss_fields.append('fit/sigreg_loss' if args.expected_objective == 'sigreg' else 'fit/action_loss')
    row_count = 0
    with metrics_path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames and not any(n.startswith('validate/') for n in reader.fieldnames)
        for row_count, row in enumerate(reader, 1):
            if native_checkpoint:
                assert int(row['step']) == row_count and int(row['epoch']) == (row_count - 1) // epoch_steps
            for key in loss_fields:
                if not row.get(key):
                    assert not native_checkpoint
                    continue
                value = float(row[key])
                assert math.isfinite(value), key
                item = metrics.setdefault(key, {'logged_count': 0, 'first': value, 'minimum': value, 'maximum': value})
                item['logged_count'] += 1
                item['last'] = value
                item['minimum'] = min(item['minimum'], value)
                item['maximum'] = max(item['maximum'], value)
    assert row_count > 0 and set(metrics) == set(loss_fields)
    if native_checkpoint:
        assert row_count == expected_steps
        recovery['metrics_prefix_checksum_and_exact_steps_verified'] = True
    pair = None
    if args.paired_run_dir:
        other = args.paired_run_dir.resolve(strict=True)
        other_initialization = read_json(other / 'initialization.json')
        other_order = read_json(other / 'data-order.json')
        other_audit = read_json(other / 'final-audit.json')
        assert other_audit['passed'] is True
        assert other_initialization['manifest_sha256'] == other_order['manifest_sha256'] == manifest_sha
        assert other_initialization['seed'] == initialization['seed']
        assert other_initialization['shared_state_sha256'] == fresh_shared_hash
        assert other_order['status'] == 'complete' and other_order['order_sha256'] == observed_hash
        assert other_order['windows_seen'] == expected_windows and other_order['optimizer_steps'] == expected_steps
        pair = {'paired_run': str(other), 'paired_audit_passed': True,
                'shared_initial_weights_identical': True, 'actual_consumed_order_identical': True,
                'manifest_and_train_only_normalization_identical': True}
    return {
        'run_dir': str(run_dir), 'objective': args.expected_objective, 'process_exit_code': process_exit_code,
        'manifest': str(args.manifest.resolve()), 'manifest_sha256': manifest_sha,
        'data_provenance': {'counts': manifest['counts'], 'normalization': manifest['normalization'],
                            'all_train_windows_consumed_exactly_once': epochs == 1, 'all_train_windows_once_per_epoch': True, 'test_and_unused_episodes_not_consumed': True,
                            'expected_steps': expected_steps, 'completed_steps': checkpoint['global_step'],
                            'windows': expected_windows, 'epochs': epochs, 'steps_per_epoch': epoch_steps, 'windows_per_epoch': windows_per_epoch,
                            'epoch_order_sha256': epoch_hashes, 'batch_size': batch_size, 'last_batch_size': expected_tail,
                            'order_sha256': observed_hash, 'loader_base_seed': base_seed,
                            'training_elapsed_seconds': order_record['elapsed_seconds'], 'validation_during_training': False,
                            'loader': {'num_workers': workers, 'persistent_workers': persistent_workers,
                                       'prefetch_factor': prefetch_factor,
                                       'effective_prefetch_factor': (prefetch_factor or 2) if workers else None}},
        'initialization': {'fresh_seeded_model_verified': True, 'seed': initialization['seed'],
                            'parameter_count': parameter_count, 'shared_state_sha256': fresh_shared_hash,
                            'state_sha256': fresh_full_hash, 'pretrained': False,
                            'originated_from_fresh_initialization': True,
                            'resume_count': resume_count},
        'checkpoint': {'path': str(checkpoint_path), 'global_step': checkpoint['global_step'],
                       'optimizer_parameter_states': len(parameter_steps), 'nonzero_moment_states': nonzero_moments,
                       'all_optimizer_and_model_tensors_finite': True, 'scheduler': scheduler,
                       'recovery_protocol': recovery},
        'export': {'path': str(export_path), 'matches_final_checkpoint_exactly': True,
                   'strict_load_passed': True, 'state_sha256': final_state_hash, 'tensor_count': len(exported)},
        'training_metrics': metrics, 'paired_run_checks': pair,
        'scope': 'Lifecycle/data/weights audit only. No held-out observations or model evaluation are performed here.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--expected-objective', choices=('sigreg', 'action_prediction', 'mse_only'), required=True)
    parser.add_argument('--paired-run-dir', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = {'passed': False, 'started_utc': datetime.now(timezone.utc).isoformat()}
    try:
        result['audit'] = audit(args)
        result['passed'] = True
    except Exception as exc:
        result.update(error=str(exc), traceback=traceback.format_exc())
    result['finished_utc'] = datetime.now(timezone.utc).isoformat()
    output = args.output or args.run_dir / 'final-audit.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(output)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())