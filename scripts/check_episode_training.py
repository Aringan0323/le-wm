"""Check sealed episode data/order and the real final training batch.

Default checks use CPU only. --forward-backward --device cuda explicitly enables
one forward/backward per objective on the actual final batch; no optimizer step.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import h5py
import numpy as np
import torch
import hydra
from hydra import compose, initialize_config_dir
import stable_pretraining as spt
import stable_worldmodel as swm
from episode_data import load_episode_experiment, get_episode_normalizer, _array_hash
from module import SIGReg
from train import action_prediction_forward, lejepa_forward
from utils import get_img_preprocessor


class Capture:
    def __init__(self, model, sigreg=None):
        self.model, self.sigreg = model, sigreg

    def log_dict(self, values, **kwargs):
        pass


def expected_order(bundle):
    spec = bundle['manifest']['training_order']
    generator = torch.Generator().manual_seed(spec['seed'])
    base_seed = int(torch.empty((), dtype=torch.int64).random_(generator=generator))
    train_indices = bundle['arrays']['train_clip_indices']
    permutation = torch.randperm(len(train_indices), generator=generator).numpy()
    order = np.asarray(train_indices[permutation], dtype='<i8')
    return order, base_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--forward-backward', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = {'passed': False, 'started_utc': datetime.now(timezone.utc).isoformat()}
    try:
        bundle = load_episode_experiment(args.manifest)
        manifest, arrays = bundle['manifest'], bundle['arrays']
        assert [manifest['counts'][name]['episodes'] for name in ('train', 'test', 'unused')] == [5000, 3000, 2000]
        assert manifest['training_order']['drop_last'] is False
        mask = np.repeat(np.isin(np.arange(len(arrays['episode_lengths'])), arrays['train_episode_ids']), arrays['episode_lengths'])
        normalization = {}
        with h5py.File(manifest['dataset']['path'], 'r') as h5:
            for name in ('action', 'proprio'):
                selected = np.asarray(h5[name])[mask]
                selected = selected[~np.isnan(selected).any(axis=1)]
                spec = manifest['normalization'][name]
                assert _array_hash(selected) == spec['selected_train_values_sha256']
                values = torch.from_numpy(selected)
                mean, std = values.mean(0, keepdim=True), values.std(0, keepdim=True, correction=1)
                torch.testing.assert_close(mean, torch.tensor(spec['mean'], dtype=values.dtype), rtol=2e-6, atol=2e-6)
                torch.testing.assert_close(std, torch.tensor(spec['std'], dtype=values.dtype), rtol=2e-6, atol=2e-6)
                normalization[name] = {'train_rows_verified': len(values), 'mean_std_match_train_only_data': True}
        order, base_seed = expected_order(bundle)
        batch_size = manifest['training_order']['batch_size']
        assert isinstance(batch_size, int) and not isinstance(batch_size, bool) and batch_size > 0
        steps = (len(order) + batch_size - 1) // batch_size
        last_batch_size = len(order) - (steps - 1) * batch_size
        assert steps == manifest['training_order']['steps_per_epoch']
        assert 1 <= last_batch_size <= batch_size
        final_batch_indices = order[-last_batch_size:]
        assert len(np.unique(order)) == len(order)
        np.testing.assert_array_equal(np.sort(order), arrays['train_clip_indices'])
        report.update(manifest_sha256=manifest['manifest_sha256'], normalization=normalization,
                      train_windows=len(order), batch_size=batch_size, expected_steps=steps, last_batch_size=last_batch_size,
                      expected_order_sha256=hashlib.sha256(order.tobytes()).hexdigest(), loader_base_seed=base_seed,
                      final_window_index=int(order[-1]), final_window_episode=int(arrays['clip_episode_ids'][order[-1]]),
                      final_batch_window_indices=final_batch_indices.tolist(),
                      final_batch_episode_ids=arrays['clip_episode_ids'][final_batch_indices].tolist())
        with initialize_config_dir(config_dir=str(REPO / 'config/train'), version_base=None):
            overrides = ['data=tworoom', 'subdir=episode-preflight', 'model.action_encoder.input_dim=10']
            configs = {'sigreg': compose(config_name='lewm', overrides=overrides),
                       'action_prediction': compose(config_name='lewm_action', overrides=overrides)}
        models = {}
        for objective, cfg in configs.items():
            torch.manual_seed(3072)
            models[objective] = hydra.utils.instantiate(cfg.model)
        baseline_state = models['sigreg'].state_dict()
        variant_state = models['action_prediction'].state_dict()
        assert all(torch.equal(value, variant_state[name]) for name, value in baseline_state.items())
        report['same_initial_shared_tensors'] = len(baseline_state)
        report['model_parameters'] = {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()}
        if args.forward_backward:
            dataset = swm.data.load_dataset('tworoom.h5', cache_dir=str(Path(manifest['dataset']['path']).parent.parent),
                                           num_steps=4, frameskip=5, keys_to_load=['pixels', 'action', 'proprio'],
                                           keys_to_cache=['action', 'proprio'], transform=None)
            load_episode_experiment(args.manifest, dataset)
            dataset.transform = spt.data.transforms.Compose(get_img_preprocessor('pixels', 'pixels', 224),
                                                            get_episode_normalizer(bundle, 'action'),
                                                            get_episode_normalizer(bundle, 'proprio'))
            cpu_batch = next(iter(torch.utils.data.DataLoader(
                torch.utils.data.Subset(dataset, final_batch_indices.tolist()),
                batch_size=batch_size, drop_last=False, num_workers=0)))
            assert cpu_batch['pixels'].shape[0] == last_batch_size
            checks = {}
            for objective, model in models.items():
                model.to(args.device).train()
                sigreg = SIGReg(**configs['sigreg'].loss.sigreg.kwargs).to(args.device) if objective == 'sigreg' else None
                batch = {name: value.to(args.device).clone() for name, value in cpu_batch.items() if torch.is_tensor(value)}
                bn_shapes = []
                handles = [module.register_forward_pre_hook(lambda _, inputs: bn_shapes.append(list(inputs[0].shape)))
                           for module in model.modules() if isinstance(module, torch.nn.BatchNorm1d)]
                function = lejepa_forward if objective == 'sigreg' else action_prediction_forward
                with torch.autocast(device_type=args.device, dtype=torch.bfloat16, enabled=args.device == 'cuda'):
                    output = function(Capture(model, sigreg), batch, 'fit', configs[objective])
                assert torch.isfinite(output['loss'])
                output['loss'].backward()
                parameters = list(model.parameters())
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
                assert [shape[0] for shape in bn_shapes] == [
                    last_batch_size * (configs[objective].history_size + configs[objective].num_preds),
                    last_batch_size * configs[objective].history_size]
                checks[objective] = {'loss': output['loss'].item(), 'finite_gradients_for_parameter_tensors': len(parameters),
                                     'actual_batch_size': last_batch_size,
                                     'batchnorm_input_shapes': bn_shapes, 'no_optimizer_step': True,
                                     'precision': 'bf16-mixed' if args.device == 'cuda' else 'fp32'}
                for handle in handles:
                    handle.remove()
                model.zero_grad(set_to_none=True)
                model.cpu()
                if args.device == 'cuda':
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            report['final_batch_forward_backward'] = checks
        report['passed'] = True
    except Exception as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    output = json.dumps(report, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(output, encoding='utf-8')
    temporary.replace(args.output)
    print(output, flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())