"""Focused CPU checks and optional post-training audit for action-prediction LeWM.

Run ./lewm.sh python scripts/check_action_predictor.py --output /path/checks.json
Audit only after training exits: add --audit-log-dir LOG --registry-run RUN --device cuda.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import inspect
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
from torch import nn
import torch.nn.functional as F

from jepa import JEPA
from module import ActionPredictor, ARPredictor, Block, ConditionalBlock, Embedder, SIGReg
from train import action_prediction_forward, lejepa_forward


class Capture:
    def __init__(self, model, sigreg=None):
        self.model = model
        self.sigreg = sigreg
        self.logged = {}

    def log_dict(self, values, **kwargs):
        self.logged.update(values)


class ProbeLinear(nn.Linear):
    def forward(self, delta):
        self.last_delta = delta
        return super().forward(delta)


class AlignmentFixture(nn.Module):
    """Independent tensors allow isolation of gradients into learned targets."""
    def __init__(self):
        super().__init__()
        self.emb = nn.Parameter(torch.arange(32, dtype=torch.float32).reshape(2, 4, 4) / 19)
        self.act = nn.Parameter(torch.arange(32, dtype=torch.float32).reshape(2, 4, 4).flip(1) / 23)
        self.pred = nn.Parameter(torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 17 - 0.4)
        self.action_predictor = ProbeLinear(4, 4)
        with torch.no_grad():
            self.action_predictor.weight.copy_(torch.eye(4) * 0.7)
            self.action_predictor.bias.copy_(torch.arange(4) / 13)

    def encode(self, batch):
        assert torch.isfinite(batch['action']).all(), 'NaN actions were not sanitized'
        return {'emb': self.emb, 'act_emb': self.act}

    def predict(self, context, actions):
        self.seen_context, self.seen_actions = context, actions
        return self.pred


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 8)

    def forward(self, pixels, **kwargs):
        return SimpleNamespace(last_hidden_state=self.proj(pixels).unsqueeze(1))


def objective_cfg():
    return OmegaConf.create({'history_size': 3, 'num_preds': 1,
                             'loss': {'type': 'action_prediction', 'action_prediction': {'weight': 0.09}}})


def check_objective():
    fixture = AlignmentFixture()
    capture = Capture(fixture)
    cfg = objective_cfg()
    raw = torch.zeros(2, 4, 2)
    raw[0, 0, 0] = float('nan')
    result = action_prediction_forward(capture, {'action': raw}, 'fit', cfg)
    torch.testing.assert_close(fixture.seen_context, fixture.emb[:, :3])
    torch.testing.assert_close(fixture.seen_actions, fixture.act[:, :3])
    expected_delta = fixture.pred - fixture.emb[:, :3]
    torch.testing.assert_close(fixture.action_predictor.last_delta, expected_delta)
    expected_action = F.linear(expected_delta, fixture.action_predictor.weight, fixture.action_predictor.bias)
    expected_pred_loss = (fixture.pred - fixture.emb[:, 1:4]).square().mean()
    expected_action_loss = (expected_action - fixture.act[:, :3]).square().mean()
    expected_total = expected_pred_loss + 0.09 * expected_action_loss
    for key, expected in [('pred_loss', expected_pred_loss), ('action_loss', expected_action_loss), ('loss', expected_total)]:
        assert result[key].ndim == 0 and torch.isfinite(result[key]), key
        torch.testing.assert_close(result[key], expected)
    assert 'sigreg_loss' not in result and not any('sigreg' in key for key in capture.logged)
    for leaf in (fixture.emb, fixture.pred, fixture.act):
        actual_gradient = torch.autograd.grad(result['action_loss'], leaf, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected_action_loss, leaf, retain_graph=True)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient)
        assert actual_gradient.abs().sum() > 0
    observation_target_gradient = torch.autograd.grad(result['pred_loss'], fixture.emb, retain_graph=True)[0]
    expected_gradient = torch.zeros_like(fixture.emb)
    expected_gradient[:, 1:4] = 2 * (fixture.emb[:, 1:4] - fixture.pred) / fixture.pred.numel()
    torch.testing.assert_close(observation_target_gradient, expected_gradient)
    for key in ('latent_std', 'action_embedding_std', 'predicted_action_std'):
        assert not capture.logged['fit/' + key].requires_grad
    invalid = objective_cfg()
    invalid.num_preds = 2
    try:
        action_prediction_forward(capture, {'action': torch.zeros(2, 4, 2)}, 'fit', invalid)
    except ValueError:
        pass
    else:
        raise AssertionError('Misaligned num_preds=2 was accepted')
    return {'alignment': 'context0:3, next_observation1:4, action0:3',
            'delta_only_head_input': True, 'exact_weighted_loss': True,
            'action_target_context_prediction_gradients_attached': True,
            'observation_target_gradients_attached': True, 'no_sigreg': True,
            'nonunit_prediction_offset_rejected': True}


def check_integrated_gradients():
    dims = dict(num_frames=3, input_dim=8, hidden_dim=8, output_dim=8,
                depth=2, heads=2, dim_head=4, mlp_dim=16, dropout=0.0)
    model = JEPA(encoder=TinyEncoder(), predictor=ARPredictor(**dims),
                 action_encoder=Embedder(input_dim=2, smoothed_dim=4, emb_dim=8, mlp_scale=2),
                 projector=nn.Linear(8, 8), pred_proj=nn.Linear(8, 8),
                 action_predictor=ActionPredictor(**dims))
    output = action_prediction_forward(Capture(model),
                {'pixels': torch.randn(2, 4, 4), 'action': torch.randn(2, 4, 2)}, 'fit', objective_cfg())
    output['loss'].backward()
    report = {}
    for name in ('encoder', 'action_encoder', 'predictor', 'action_predictor', 'projector', 'pred_proj'):
        parameters = list(getattr(model, name).parameters())
        assert parameters and all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters), name
        norm = sum(p.grad.abs().sum().item() for p in parameters)
        assert norm > 0, name
        report[name] = {'all_gradients_present_and_finite': True, 'gradient_l1_sum': norm}
    baseline_cfg = OmegaConf.create({'history_size': 3, 'num_preds': 1, 'loss': {'sigreg': {'weight': 0.09}}})
    baseline_capture = Capture(model, SIGReg(knots=5, num_proj=16))
    baseline = lejepa_forward(baseline_capture,
                {'pixels': torch.randn(2, 4, 4), 'action': torch.randn(2, 4, 2)}, 'fit', baseline_cfg)
    torch.testing.assert_close(baseline['loss'], baseline['pred_loss'] + 0.09 * baseline['sigreg_loss'])
    assert 'action_loss' not in baseline
    return {'components': report, 'baseline_sigreg_objective_unchanged': True}


def check_config_and_causality():
    with initialize_config_dir(config_dir=str(REPO / 'config/train'), version_base=None):
        overrides = ['data=tworoom', 'subdir=verification', 'model.action_encoder.input_dim=10']
        baseline_cfg = compose(config_name='lewm', overrides=overrides)
        variant_cfg = compose(config_name='lewm_action', overrides=overrides)
    assert baseline_cfg.loss.get('type', 'sigreg') == 'sigreg'
    assert variant_cfg.loss.type == 'action_prediction' and variant_cfg.loss.sigreg is None
    assert variant_cfg.loss.action_prediction.weight == 0.09
    torch.manual_seed(3072)
    baseline = hydra.utils.instantiate(baseline_cfg.model)
    torch.manual_seed(3072)
    variant = hydra.utils.instantiate(variant_cfg.model)
    baseline_state, variant_state = baseline.state_dict(), variant.state_dict()
    assert baseline.action_predictor is None
    assert all(not name.startswith('action_predictor.') for name in baseline_state)
    assert all(torch.equal(value, variant_state[name]) for name, value in baseline_state.items()), 'Shared initial weights differ'
    baseline.load_state_dict(baseline_state, strict=True)
    head = variant.action_predictor.eval()
    assert list(inspect.signature(head.forward).parameters) == ['delta']
    assert len(head.transformer.layers) == 6
    assert all(isinstance(layer, Block) and not isinstance(layer, ConditionalBlock) for layer in head.transformer.layers)
    assert tuple(head.pos_embedding.shape) == (1, 3, 192)
    conditioning_calls = []
    hook = head.transformer.register_forward_pre_hook(lambda module, args: conditioning_calls.append(len(args)))
    delta = torch.randn(2, 3, 192)
    with torch.no_grad():
        reference = head(delta)
        assert reference.shape == (2, 3, 192) and torch.isfinite(reference).all()
        prefix_errors = []
        for prefix in (1, 2):
            changed = delta.clone()
            changed[:, prefix:] = torch.randn_like(changed[:, prefix:]) * 10
            perturbed = head(changed)
            error = (reference[:, :prefix] - perturbed[:, :prefix]).abs().max().item()
            torch.testing.assert_close(reference[:, :prefix], perturbed[:, :prefix], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(reference[:, :prefix], head(delta[:, :prefix]), rtol=1e-5, atol=1e-6)
            assert not torch.allclose(reference[:, prefix:], perturbed[:, prefix:])
            prefix_errors.append(error)
    hook.remove()
    assert conditioning_calls and all(count == 1 for count in conditioning_calls)
    return {'baseline_parameters': sum(p.numel() for p in baseline.parameters()),
            'variant_parameters': sum(p.numel() for p in variant.parameters()),
            'head_parameters': sum(p.numel() for p in head.parameters()),
            'shared_initial_tensors_equal': len(baseline_state),
            'same_initialization_seed': 3072, 'causal_prefix_max_errors': prefix_errors,
            'head_shape': list(reference.shape), 'head_has_no_action_argument_or_conditional_blocks': True}


def tensor_distribution(value):
    value = value.float().reshape(-1, value.shape[-1]).cpu()
    assert torch.isfinite(value).all()
    std = value.std(dim=0, unbiased=False)
    centered = value - value.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    mass = singular.square()
    probabilities = mass / mass.sum().clamp_min(1e-30)
    effective_rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-30).log()).sum()) if mass.sum() > 0 else torch.tensor(0.)
    return {'sample_count': value.shape[0], 'width': value.shape[1], 'mean_coordinate_std': std.mean().item(),
            'mean_coordinate_variance': std.square().mean().item(), 'minimum_coordinate_std': std.min().item(),
            'maximum_coordinate_std': std.max().item(), 'activation_rms': value.square().mean().sqrt().item(),
            'centered_rms': centered.square().mean().sqrt().item(), 'effective_rank': effective_rank.item(),
            'fraction_coordinates_std_below_1e_minus_3': (std < 1e-3).float().mean().item()}


def compare_bf16(model, batches, fp32_features, device):
    if device != 'cuda':
        return {'skipped': 'BF16 autocast comparison requires the requested CUDA audit'}
    collected = {'latent': [], 'encoded_action_target': []}
    if model.action_predictor is not None:
        collected['predicted_action'] = []
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for saved_batch in batches:
            batch = {name: value.to(device) for name, value in saved_batch.items()}
            encoded = model.encode(batch)
            context, target = encoded['emb'][:, :3], encoded['act_emb'][:, :3]
            collected['latent'].append(encoded['emb'].cpu())
            collected['encoded_action_target'].append(target.cpu())
            if model.action_predictor is not None:
                predicted = model.predict(context, target)
                collected['predicted_action'].append(model.action_predictor(predicted - context).cpu())
    report = {'precision': 'CUDA BF16 autocast, statistics computed in FP32',
              'distributions': {name: tensor_distribution(torch.cat(values)) for name, values in collected.items()},
              'rms_output_difference_from_fp32': {
                  name: (torch.cat(values).float() - torch.cat(fp32_features[name]).float()).square().mean().sqrt().item()
                  for name, values in collected.items() if name in fp32_features}}
    if 'predicted_action' in collected:
        targets = torch.cat(collected['encoded_action_target']).float()
        predicted = torch.cat(collected['predicted_action']).float()
        head_mse = (predicted - targets).square().mean().item()
        constant_mse = (targets - targets.mean(dim=(0, 1), keepdim=True)).square().mean().item()
        report['inverse_prediction_comparison'] = {
            'action_head_mse': head_mse, 'constant_target_mean_mse': constant_mse,
            'head_to_constant_mse_ratio': head_mse / constant_mse if constant_mse else None}
    return report


def audit_run(args):
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from utils import get_column_normalizer, get_img_preprocessor
    from smoke_wsl import exercise
    log_dir, registry = args.audit_log_dir.resolve(), args.registry_run.resolve()
    assert int((log_dir / 'exit.code').read_text()) == 0, 'Training must finish successfully before audit'
    checkpoint_path = registry / 'checkpoints/last.ckpt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    assert checkpoint['global_step'] == args.expected_steps
    optimizer = checkpoint['optimizer_states'][0]
    steps = [int(state['step']) for state in optimizer['state'].values() if 'step' in state]
    assert steps and min(steps) == max(steps) == args.expected_steps
    moments_nonzero = 0
    for state in optimizer['state'].values():
        for value in state.values():
            if torch.is_tensor(value):
                assert torch.isfinite(value).all()
        moments_nonzero += int(bool(torch.count_nonzero(state['exp_avg']))) if 'exp_avg' in state else 0
    assert moments_nonzero > 0
    storage = Path(os.environ['STABLEWM_HOME'])
    cfg = OmegaConf.load(storage / 'checkpoints' / log_dir.name / 'config.yaml')
    assert cfg.loss.type == 'action_prediction' and cfg.loss.sigreg is None
    export = storage / 'checkpoints' / log_dir.name / 'weights_epoch_1.pt'
    weights = torch.load(export, map_location='cpu', weights_only=True)
    assert set(weights) == {name.removeprefix('model.') for name in checkpoint['state_dict'] if name.startswith('model.')}
    for name, value in weights.items():
        assert torch.isfinite(value).all() and torch.equal(value, checkpoint['state_dict']['model.' + name]), name
    model = swm.wm.utils.load_pretrained(f'{log_dir.name}/weights_epoch_1.pt').to(args.device).eval().requires_grad_(False)
    model.load_state_dict(weights, strict=True)
    rows = list(csv.DictReader((registry / 'metrics.csv').open()))
    assert not any('sigreg_loss' in name for name in rows[0]), 'Unexpected SIGReg metric'
    metrics = {}
    for stage, suffix in (('fit', ''), ('validate', '_epoch')):
        keys = [f'{stage}/{name}{suffix}' for name in ('loss', 'pred_loss', 'action_loss')]
        metrics[stage] = []
        for row in rows:
            if row.get(keys[0]):
                values = [float(row[key]) for key in keys]
                assert all(math.isfinite(value) for value in values)
                assert math.isclose(values[0], values[1] + 0.09 * values[2], rel_tol=1e-5, abs_tol=1e-6)
                item = {'completed_steps': int(row['step']) + 1, **dict(zip(('loss', 'pred_loss', 'action_loss'), values))}
                for name in ('latent_std', 'action_embedding_std', 'predicted_action_std'):
                    key = f'{stage}/{name}{suffix}'
                    if row.get(key):
                        item[name] = float(row[key])
                metrics[stage].append(item)
    assert len(metrics['validate']) == 4
    dataset = swm.data.load_dataset('tworoom.h5', cache_dir=str(storage), num_steps=4, frameskip=5,
                                   keys_to_load=['pixels', 'action'], keys_to_cache=['action'], transform=None)
    dataset.transform = spt.data.transforms.Compose(get_img_preprocessor('pixels', 'pixels', 224),
                                                    get_column_normalizer(dataset, 'action', 'action'))
    indices = torch.linspace(0, len(dataset) - 1, 32).long().tolist()
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=8, num_workers=0)
    distributions = {name: [] for name in ('latent', 'encoded_action_target', 'predicted_action', 'predicted_delta', 'normalized_action_input')}
    first_batch = None
    diagnostic_batches = []
    with torch.inference_mode():
        for batch in loader:
            batch = {name: value.to(args.device) for name, value in batch.items() if torch.is_tensor(value)}
            batch['action'] = torch.nan_to_num(batch['action'])
            diagnostic_batches.append({name: value.detach().cpu() for name, value in batch.items()})
            encoded = model.encode(batch)
            ctx, action = encoded['emb'][:, :3], encoded['act_emb'][:, :3]
            predicted = model.predict(ctx, action)
            delta = predicted - ctx
            predicted_action = model.action_predictor(delta)
            for name, value in [('latent', encoded['emb']), ('encoded_action_target', action),
                                ('predicted_action', predicted_action), ('predicted_delta', delta),
                                ('normalized_action_input', batch['action'][:, :3])]:
                distributions[name].append(value.cpu())
            if first_batch is None:
                first_batch = batch
        inference = exercise(model, first_batch['pixels'][:1], first_batch['action'][:1], 3)
        encoded = model.encode({'pixels': first_batch['pixels'], 'action': first_batch['action']})
        ctx, action = encoded['emb'][:, :3], encoded['act_emb'][:, :3]
        actual = model.predict(ctx, action)
        zero = model.predict(ctx, model.action_encoder(torch.zeros_like(first_batch['action'][:, :3])))
        difference = actual - zero
        assert torch.isfinite(difference).all()
        sensitivity = {'rms_difference': difference.square().mean().sqrt().item(),
                       'maximum_absolute_difference': difference.abs().max().item(),
                       'prediction_changed': not torch.equal(actual, zero),
                       'note': 'Same context, recorded versus zero normalized actions; diagnostic only. Normalized zero means dataset-average action.'}
    targets = torch.cat(distributions['encoded_action_target']).float()
    action_estimates = torch.cat(distributions['predicted_action']).float()
    constant_target_mean = targets.mean(dim=(0, 1), keepdim=True)
    action_head_mse = (action_estimates - targets).square().mean().item()
    constant_mean_mse = (constant_target_mean - targets).square().mean().item()
    inverse_comparison = {
        'windows': 32, 'transitions': int(targets.shape[0] * targets.shape[1]),
        'action_head_mse': action_head_mse,
        'constant_target_mean_mse': constant_mean_mse,
        'head_to_constant_mse_ratio': action_head_mse / constant_mean_mse if constant_mean_mse > 0 else None,
        'note': 'Same 32 evaluation windows; the constant baseline predicts their per-feature mean learned action embedding. This is an in-sample diagnostic, not a fitted or held-out inverse-dynamics benchmark.'
    }
    diagnostic_trends = {
        stage: {'first': entries[0], 'last': entries[-1]}
        for stage, entries in metrics.items() if entries
    }
    parameter_count = sum(p.numel() for p in model.parameters())
    distribution_stats = {name: tensor_distribution(torch.cat(values)) for name, values in distributions.items()}
    bf16_comparison = compare_bf16(model, diagnostic_batches, distributions, args.device)
    baseline_comparison = None
    if args.compare_baseline:
        # Load models sequentially; the comparison reuses identical normalized images/actions.
        model.cpu()
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        baseline_model = swm.wm.utils.load_pretrained(args.compare_baseline).to(args.device).eval().requires_grad_(False)
        baseline_features = {'latent': [], 'encoded_action_target': []}
        with torch.inference_mode():
            for saved_batch in diagnostic_batches:
                baseline_batch = {name: value.to(args.device) for name, value in saved_batch.items()}
                baseline_encoded = baseline_model.encode(baseline_batch)
                baseline_features['latent'].append(baseline_encoded['emb'].cpu())
                baseline_features['encoded_action_target'].append(baseline_encoded['act_emb'][:, :3].cpu())
        baseline_stats = {name: tensor_distribution(torch.cat(values)) for name, values in baseline_features.items()}
        baseline_bf16 = compare_bf16(baseline_model, diagnostic_batches, baseline_features, args.device)
        baseline_comparison = {'checkpoint': args.compare_baseline, 'same_normalized_windows': True,
                               'bf16_comparison': baseline_bf16,
                               'window_indices': indices, 'baseline_distributions': baseline_stats,
                               'variant_to_baseline_std_ratio': {
                                   name: distribution_stats[name]['mean_coordinate_std'] / stats['mean_coordinate_std']
                                   if stats['mean_coordinate_std'] > 0 else None
                                   for name, stats in baseline_stats.items()},
                               'note': 'Matched input diagnostic at the two 1000-step endpoints, not a generalization or planning comparison.'}
    console = (log_dir / 'console.log').read_text().splitlines()
    start = next((line.split(': ', 1)[1] for line in console if line.startswith('Started (New York): ')), None)
    elapsed = ((log_dir / 'exit.code').stat().st_mtime - datetime.fromisoformat(start).timestamp()) if start else None
    return {'training_exit_code': 0, 'global_step': checkpoint['global_step'],
            'wall_seconds_including_startup': elapsed, 'model_parameters': parameter_count,
            'optimizer_states': len(steps), 'nonzero_moment_states': moments_nonzero, 'optimizer_state_finite': True,
            'export_matches_checkpoint_exactly': True, 'checkpoint': str(checkpoint_path), 'export': str(export),
            'scheduler_states': checkpoint['lr_schedulers'], 'metrics': metrics,
            'real_data_inference': inference, 'action_sensitivity': sensitivity,
            'inverse_prediction_comparison': inverse_comparison, 'first_last_metric_diagnostics': diagnostic_trends,
            'representation_diagnostics': distribution_stats, 'baseline_comparison': baseline_comparison,
            'bf16_comparison': bf16_comparison,
            'diagnostic_note': '32 windows spread across the dataset. Variance/rank have no performance threshold; low loss with vanishing representation or target variance can indicate collapse. Window-level validation overlaps training episodes.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--audit-log-dir', type=Path)
    parser.add_argument('--registry-run', type=Path)
    parser.add_argument('--expected-steps', type=int, default=1000)
    parser.add_argument('--compare-baseline', help='Optional saved model name/path for matched-input embedding diagnostics')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if bool(args.audit_log_dir) != bool(args.registry_run):
        parser.error('--audit-log-dir and --registry-run must be provided together')
    torch.set_num_threads(4)
    torch.manual_seed(3072)
    result = {'passed': False, 'started_utc': datetime.now(timezone.utc).isoformat()}
    try:
        if args.audit_log_dir:
            result['audit'] = audit_run(args)
        else:
            assert args.device == 'cpu', 'Preflight checks run on CPU only'
            result['objective'] = check_objective()
            result['gradient_flow'] = check_integrated_gradients()
            result['configuration_and_causality'] = check_config_and_causality()
        result['passed'] = True
    except Exception as exc:
        result.update(error=str(exc), traceback=traceback.format_exc())
    result['finished_utc'] = datetime.now(timezone.utc).isoformat()
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp = args.output.with_suffix(args.output.suffix + '.tmp')
        temp.write_text(text, encoding='utf-8')
        temp.replace(args.output)
    print(text, flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())