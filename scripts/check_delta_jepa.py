"""Verify the LDAD experiment's training inputs, architecture, initialization and gradients on CPU."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import hydra
import stable_pretraining as spt
import stable_worldmodel as swm
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from episode_data import get_episode_normalizer, load_episode_experiment
from experiment_tracking import state_dict_sha256
from train import configure_training_objective
from utils import get_img_preprocessor, PreserveExecutedActions


def training_batch(manifest_path):
    bundle = load_episode_experiment(manifest_path)
    path = bundle["manifest"]["dataset"]["path"]
    dataset = swm.data.load_dataset(path, frameskip=5, num_steps=4,
                                    keys_to_load=["action", "pixels"], keys_to_cache=["action"])
    load_episode_experiment(manifest_path, dataset)
    image_transform = get_img_preprocessor("pixels", "pixels", 224)
    dataset.transform = spt.data.transforms.Compose(PreserveExecutedActions(5), image_transform,
                                                    get_episode_normalizer(bundle, "action"))
    arrays = bundle["arrays"]
    ids = arrays["train_clip_indices"]
    first, last = int(ids[0]), int(ids[-1])
    items = []
    with h5py.File(path, "r", swmr=True) as h5:
        for index in (first, last):
            ep, local = dataset.clip_indices[index]
            start = int(arrays["episode_offsets"][ep] + local)
            item = dataset[index]
            raw = torch.from_numpy(h5["action"][start:start + 20])
            expected_target = raw[:15].reshape(3, 5, 2)
            torch.testing.assert_close(item["executed_actions"], expected_target, rtol=0, atol=0)
            assert torch.isfinite(expected_target).all()
            if index == last:
                assert raw[-1].isnan().all(), "Last training clip should reach its episode boundary"
            normed = get_episode_normalizer(bundle, "action")({"action": raw})["action"]
            torch.testing.assert_close(item["action"], normed.reshape(4, 10), rtol=0, atol=0, equal_nan=True)
            pixels = torch.from_numpy(h5["pixels"][start:start + 16:5]).permute(0, 3, 1, 2)
            torch.testing.assert_close(item["pixels"], image_transform({"pixels": pixels})["pixels"], rtol=0, atol=0)
            items.append(item)
    return torch.utils.data.default_collate(items), [first, last]


def check_query_dependencies(ldad):
    displacement = torch.randn(1, 2, 192)
    output = ldad(displacement)
    # The first output query must see the last query, which is impossible under a causal mask.
    grad = torch.autograd.grad(output[0, 0, 0].sum(), ldad.queries)[0]
    assert grad[0, -1].abs().max() > 0
    changed = displacement.clone()
    changed[:, 1] += 1
    torch.testing.assert_close(ldad(changed)[:, 0], output[:, 0], rtol=0, atol=0)


def check_config(config_name, weights, batch):
    with hydra.initialize_config_dir(config_dir=str(REPO / "config/train"), version_base=None):
        cfg = hydra.compose(config_name=config_name)
    cfg.model.action_encoder.input_dim = 10
    torch.manual_seed(cfg.seed)
    model = hydra.utils.instantiate(cfg.model).train()
    assert cfg.trainer.max_epochs == 50 and cfg.optimizer.lr == 5e-5
    assert (cfg.loss.ldad.weight, cfg.loss.sigreg.weight) == weights
    forward, sigreg = configure_training_objective(cfg, model)
    assert (sigreg is None) == (weights[1] == 0)
    ldad = model.ldad
    assert ldad.queries.shape == (1, 5, 192) and len(ldad.layers) == 3 and ldad.readout.out_features == 2
    for layer in ldad.layers:
        assert layer.attn.heads == 8 and layer.attn.to_qkv.out_features == 3 * 8 * 64
        assert layer.mlp.net[1].out_features == 512
    initial = state_dict_sha256(model.state_dict())
    shared = state_dict_sha256(model.state_dict(), exclude_prefixes=("ldad.",))
    check_query_dependencies(ldad)
    holder = SimpleNamespace(model=model, sigreg=sigreg)
    logged = {}
    holder.log_dict = lambda metrics, **kwargs: logged.update(metrics)
    seen = []
    def capture_inputs(module, inputs, result):
        seen.append((inputs, result))
    handle = ldad.register_forward_hook(capture_inputs)
    torch.manual_seed(271)
    with patch.object(torch.nn.functional, "scaled_dot_product_attention",
                      wraps=torch.nn.functional.scaled_dot_product_attention) as attention:
        out = forward(holder, batch, "fit", cfg)
    handle.remove()
    ldad_attention = [call for call in attention.call_args_list if tuple(call.args[0].shape[1:]) == (8, 5, 64)]
    assert len(ldad_attention) == 3 and all(call.kwargs["is_causal"] is False for call in ldad_attention)
    (decoder_inputs, actions), = seen
    (displacement,) = decoder_inputs
    torch.testing.assert_close(displacement, out["emb"][:, 1:] - out["emb"][:, :-1])
    assert actions.shape == (2, 3, 5, 2)
    torch.testing.assert_close(out["ldad_loss"], (actions - batch["executed_actions"]).square().mean())
    prediction_mse = (out["pred_emb"] - out["emb"][:, 1:]).square().mean()
    total = prediction_mse
    if weights[1]:
        total = total + weights[1] * out["sigreg_loss"]
    if weights[0]:
        total = total + weights[0] * out["ldad_loss"]
        endpoint, prediction, action_input = torch.autograd.grad(out["ldad_loss"],
            (out["emb"], out["pred_emb"], out["act_emb"]), retain_graph=True, allow_unused=True)
        assert endpoint[:, 0].abs().max() > 0 and endpoint[:, -1].abs().max() > 0
        assert prediction is None and action_input is None
    else:
        assert not out["ldad_loss"].requires_grad
    torch.testing.assert_close(out["loss"], total, rtol=0, atol=0)
    out["loss"].backward()
    for name, parameter in model.named_parameters():
        if name.startswith("ldad.") and weights[0] == 0:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert ("fit/sigreg_loss" in logged) == bool(weights[1])
    assert "fit/ldad_loss" in logged and "fit/latent_cross_example_rms_std" in logged
    return {"initial_sha256": initial, "shared_sha256": shared, "parameters": sum(p.numel() for p in model.parameters()),
            "ldad_parameters": sum(p.numel() for p in ldad.parameters()),
            "losses": {k: float(v) for k, v in logged.items() if "loss" in k}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    batch, ids = training_batch(args.manifest)
    variants = {"lewm_delta_l0_s09": (0., .09), "lewm_delta_l50_s09": (50., .09),
                "lewm_delta_l50_s0": (50., 0.)}
    checks = {name: check_config(name, weights, batch) for name, weights in variants.items()}
    assert len({v["initial_sha256"] for v in checks.values()}) == 1
    # Verified seed-3072 initialization of the shared historical LeWM.
    assert {v["shared_sha256"] for v in checks.values()} == {
        "7318cb4ee7a40d3183e14fc43d0b851809ed49ac09bbe6a344619284bca081fd"}
    print(json.dumps({"passed": True, "cpu_training_windows": ids, "variants": checks}, indent=2))


if __name__ == "__main__":
    main()
