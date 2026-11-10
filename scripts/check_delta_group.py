"""CPU equivalence, single-load and interrupted checkpoint checks for the shared Delta trainer."""

import csv
import hashlib
import json
import random
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import train_delta_group as grouped
import train_resumable as native
from check_delta_lifecycle import TinyDataset, TinyLDAD, compose
from check_resumable_training import DriverWorldModel, assert_equal
from delta_sweep_config import VARIANTS
from experiment_tracking import state_dict_sha256
from resumable_training import EpochOrder
from utils import PreserveExecutedActions


class World(DriverWorldModel):
    def __init__(self, cfg):
        super().__init__(False)  # Real dropout in the encoder and forward predictor.
        self.normalization = nn.BatchNorm1d(3)
        self.ldad = TinyLDAD(3, 2) if cfg.get("ldad") is not None else None

    def encode(self, batch):
        result = super().encode(batch)
        latent = result["emb"]
        latent = self.normalization(latent.flatten(0, 1)).reshape_as(latent)
        result["emb"] = latent + (random.random() - np.random.uniform()) * .01
        return result


class CountedDataset(TinyDataset):
    def __init__(self):
        super().__init__()
        self.ids = []

    def __getitem__(self, index):
        self.ids.append(int(index))
        return super().__getitem__(index)


def normalizer(sample):
    return {**sample, "action": sample["action"] * .5}


def configs(root, manifest):
    root.mkdir()
    (root / "control.json").write_text(json.dumps({"request":"run"}))
    (root / "manifest.json").write_text(json.dumps(manifest))
    result = {}
    for key, variant in VARIANTS.items():
        member = root / "members" / key
        member.mkdir(parents=True)
        (member / "manifest.json").write_text(json.dumps(manifest))
        result[key] = OmegaConf.create({
            "episode_manifest": str(member / "manifest.json"), "run_metadata_dir": str(member / "runs" / key),
            "control_file": str(root / "control.json"), "resume_from": None, "seed": 3072, "cpu_threads": 1,
            "data": {"dataset": {"name": "tiny", "frameskip": 1, "keys_to_load": ["pixels", "action"]}},
            "history_size": 3, "num_preds": 1, "img_size": 3, "track_cross_example_latent_spread": True,
            "model": {"_target_": "fixture.World", "encoder": {"pretrained": False}, "action_encoder": {"input_dim": 2},
                      "ldad": {"horizon": 1} if variant.ldad_weight else None},
            "loader": {"batch_size": 4, "num_workers": 0},
            "trainer": {"max_epochs": 3, "accelerator": "cpu", "precision": "bf16-mixed", "gradient_clip_val": 1.},
            "optimizer": {"type": "AdamW", "lr": .003, "weight_decay": .001},
            "loss": {"type": "delta_jepa", "sigreg": {"weight": variant.sigreg_weight, "kwargs": {"knots": 5, "num_proj": 8}},
                     "ldad": {"weight": variant.ldad_weight}},
            "checkpoint_every_seconds": 300, "output_model_name": f"fixture-{root.name}-{key}"})
    return result


def read(path):
    return json.loads(path.read_text())


def output_checks(ref, candidate, expected_trace):
    a = torch.load(ref/"checkpoints/final.ckpt", map_location="cpu", weights_only=False)
    b = torch.load(candidate/"checkpoints/final.ckpt", map_location="cpu", weights_only=False)
    for name in ("global_step", "epoch", "state_dict", "optimizer_states", "lr_schedulers"):
        assert_equal(a[name], b[name], name)
    assert_equal(a["resumable"]["rng"], b["resumable"]["rng"], "member RNG")
    assert (ref/"metrics.csv").read_bytes() == (candidate/"metrics.csv").read_bytes()
    assert (candidate/"training-indices.i64").read_bytes() == expected_trace
    complete = read(candidate/"runcompletion.json")
    assert complete["passed"] and complete["global_step"] == 9 and complete["epochs"] == 3
    exported = torch.load(complete["export_path"], map_location="cpu", weights_only=True)
    assert_equal(exported, {k[6:]:v for k,v in a["state_dict"].items() if k.startswith("model.")}, "export")
    with (candidate/"metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    has_ldad = any(n.startswith("model.ldad.") for n in b["state_dict"])
    assert ("fit/ldad_loss" in rows[0]) == has_ldad
    assert [float(row["fit/latent_cross_example_count"]) for row in rows] == [4, 4, 3] * 3
    return state_dict_sha256(exported)


def run(work):
    torch.set_num_threads(1)
    indices = np.arange(11, dtype=np.int64)
    manifest = {"manifest_sha256":"f"*64, "training_order":{"seed":3072,"steps_per_epoch":3,"batch_size":4},
                "counts":{"train":{"windows":11}}}
    order = EpochOrder(indices, 3072, 3)
    visits = np.concatenate([order.epoch(i) for i in range(3)]).astype("<i8")
    exports = work / "exports"
    exports.mkdir()
    def bundle(path):
        return {"manifest_path":str(path), "manifest":manifest, "arrays":{"train_clip_indices":indices}}
    def export(model, run_name, config, filename):
        directory = exports / run_name
        directory.mkdir(exist_ok=True)
        torch.save(model.state_dict(), directory/filename)
    baseline = configs(work / "standalone", manifest)
    with ExitStack() as patches:
        def replace(obj, attr, **kwargs):
            patches.enter_context(patch.object(obj, attr, **kwargs))
        replace(native.swm.data.utils, "get_cache_dir", return_value=str(exports))
        replace(native.swm.wm.utils, "save_pretrained", side_effect=export)
        with ExitStack() as native_only:
            def stub(obj, attr, **kwargs):
                native_only.enter_context(patch.object(obj, attr, **kwargs))
            stub(native.swm.data, "load_dataset", side_effect=lambda *args,**kw:TinyDataset())
            stub(native, "load_episode_experiment", side_effect=lambda path,dataset:bundle(path))
            stub(native, "get_img_preprocessor", return_value=lambda s:s)
            stub(native, "get_episode_normalizer", return_value=normalizer)
            stub(native.spt.data.transforms, "Compose", side_effect=compose)
            stub(native.hydra.utils, "instantiate", side_effect=World)
            for cfg in baseline.values():
                assert native._train(cfg) == 0
        datasets = []
        def data(cfg, manifest_root):
            dataset = CountedDataset()
            dataset.transform = compose(PreserveExecutedActions(1), normalizer)
            datasets.append(dataset)
            return dataset, bundle(Path(manifest_root)/"manifest.json")
        replace(grouped, "load_shared_data", side_effect=data)
        replace(grouped, "build_model", side_effect=lambda cfg:World(cfg.model))
        transfer = grouped.transfer_batch
        reports = []
        for mode in ("uninterrupted", "pause", "member-stage-crash", "group-marker-crash", "group-promotion-crash"):
            root = work / mode
            models = configs(root, manifest)
            batches = []
            def transfer_once(batch, device):
                batches.append(batch["_window_index"].tolist())
                if len(batches) == 4 and mode != "uninterrupted":
                    (root/"control.json").write_text(json.dumps({"request":"pause"}))
                return transfer(batch, device)
            with ExitStack() as fault:
                fault.enter_context(patch.object(grouped, "transfer_batch", side_effect=transfer_once))
                if mode == "member-stage-crash":
                    stage = grouped.CheckpointManager.stage
                    staged = []
                    def fail_second(manager, *args, **kwargs):
                        if len(batches) == 4:
                            staged.append(manager.run_dir)
                            if len(staged) == 2:
                                raise RuntimeError("simulated member stage failure")
                        return stage(manager, *args, **kwargs)
                    fault.enter_context(patch.object(grouped.CheckpointManager, "stage", fail_second))
                if mode in ("group-marker-crash", "group-promotion-crash"):
                    commit = grouped.commit_group
                    def fail_commit(path, record):
                        if record["global_step"] == 4 and mode == "group-marker-crash":
                            raise RuntimeError("simulated group marker failure")
                        committed = commit(path, record)
                        if record["global_step"] == 4 and mode == "group-promotion-crash":
                            raise RuntimeError("simulated member alias promotion failure")
                        return committed
                    fault.enter_context(patch.object(grouped, "commit_group", side_effect=fail_commit))
                try:
                    result = grouped.train_group(root, configs=models)
                except RuntimeError as error:
                    if "simulated " not in str(error) or not mode.endswith("crash"):
                        raise
                else:
                    assert result == (75 if mode == "pause" else 0) and not mode.endswith("crash")
                assert datasets[-1].ids == sum(batches, [])
            if mode != "uninterrupted":
                saved = read(root/"group-checkpoint.json")
                restored_step = 4 if mode in ("pause", "group-promotion-crash") else 0
                assert saved["global_step"] == restored_step and saved["cursor"] == (15 if restored_step else 0)
                for entry in saved["members"].values():
                    assert entry["global_step"] == restored_step and Path(entry["path"]).is_file()
                (root/"control.json").write_text(json.dumps({"request":"run"}))
                with patch.object(grouped, "transfer_batch", wraps=transfer) as resumed_transfers:
                    assert grouped.train_group(root, configs=models) == 0
                assert resumed_transfers.call_count == 9 - restored_step
                assert datasets[-1].ids == visits[saved["cursor"]:].tolist()
            else:
                assert len(batches) == 9 and datasets[-1].ids == visits.tolist()
            digests = {key:output_checks(Path(baseline[key].run_metadata_dir), Path(models[key].run_metadata_dir),
                                         visits.tobytes()) for key in VARIANTS}
            initial = {key:read(Path(cfg.run_metadata_dir)/"initialization.json") for key,cfg in models.items()}
            common = {v["shared_state_sha256"] for v in initial.values()}
            assert len(common) == 1
            pos = [key for key,v in VARIANTS.items() if v.ldad_weight]
            assert initial[pos[0]]["state_sha256"] == initial[pos[1]]["state_sha256"] != initial["delta_l0_s09"]["state_sha256"]
            assert len(set(digests.values())) == 3
            final = read(root/"group-checkpoint.json")
            assert final["global_step"] == 9 and final["cursor"] == 33
            reports.append({"case":mode,"unique_batches":9,"first_pass_transfer_calls":len(batches),
                            "member_state_sha256":digests,"trace_sha256":hashlib.sha256(visits.tobytes()).hexdigest()})
    assert not torch.cuda.is_initialized()
    return {"passed":True,"device":"cpu","models":3,"epochs":3,"windows_per_epoch":11,"checks":reports}


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="delta-shared-check-") as temp:
        print(json.dumps(run(Path(temp)), indent=2))
