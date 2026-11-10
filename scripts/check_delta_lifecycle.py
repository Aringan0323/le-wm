"""CPU fixture for real multi-epoch driver pause/resume, tails, objective and retention."""
import csv
import hashlib
import json
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
import train_resumable as driver
from audit_delta_experiment import verify_delta_metrics
from check_resumable_training import DriverDataset, DriverWorldModel, assert_equal
from delta_sweep_config import VARIANTS
from resumable_training import EpochOrder


class TinyDataset(DriverDataset):
    def __len__(self):
        return 11

    def __getitem__(self, index):
        return self.transform(super().__getitem__(index))


class TinyLDAD(nn.Linear):
    horizon = 1
    def forward(self, delta):
        return super().forward(delta).unsqueeze(-2)


class TinyWorldModel(DriverWorldModel):
    def __init__(self):
        super().__init__(False)
        self.ldad = TinyLDAD(3, 2)


def compose(*transforms):
    def apply(sample):
        for transform in transforms:
            sample = transform(sample)
        return sample
    return apply


def run(work):
    torch.set_num_threads(1)
    read = lambda p: json.loads(p.read_text())
    bundle = {"manifest_path":str(work/"manifest.json"), "manifest":{"manifest_sha256":"f"*64,
              "training_order":{"batch_size":4,"steps_per_epoch":3,"seed":3072}},
              "arrays":{"train_clip_indices":np.arange(11,dtype=np.int64)}}
    def export(model, run_name, config, filename):
        torch.save(model.state_dict(), work/"exports"/run_name/filename)
    reports = []
    full_initializations = set()
    for index, variant in enumerate(VARIANTS.values()):
        cut = (3, 5, 8)[index]
        def cfg(label):
            return OmegaConf.create({
                "episode_manifest":bundle["manifest_path"],"run_metadata_dir":str(work/variant.label/label),
                "control_file":str(work/"control.json"),"resume_from":None,"seed":3072,"cpu_threads":1,
                "data":{"dataset":{"name":"fixture","keys_to_load":["pixels","action"],"frameskip":1}},
                "history_size":3,"num_preds":1,"img_size":3,"track_cross_example_latent_spread":True,
                "model":{"_target_":"fixture.TinyWorldModel","encoder":{"pretrained":False},"action_encoder":{"input_dim":2}},
                "loader":{"batch_size":4,"num_workers":0},
                "trainer":{"max_epochs":3,"accelerator":"cpu","precision":"bf16-mixed","gradient_clip_val":1.},
                "loss":{"type":"delta_jepa","sigreg":{"weight":variant.sigreg_weight,"kwargs":{"knots":5,"num_proj":8}},
                        "ldad":{"weight":variant.ldad_weight}},
                "optimizer":{"type":"AdamW","lr":.003,"weight_decay":.001},
                "checkpoint_every_seconds":300,"output_model_name":variant.label+"-"+label})
        with ExitStack() as stack:
            def mock(obj, attr, **kwargs):
                stack.enter_context(patch.object(obj,attr,**kwargs))
            mock(driver.swm.data,"load_dataset",side_effect=lambda *a,**kw:TinyDataset())
            mock(driver,"load_episode_experiment",return_value=bundle)
            mock(driver,"get_img_preprocessor",return_value=lambda v:v)
            mock(driver,"get_episode_normalizer",return_value=lambda v:{**v, "action":v["action"]*.5})
            mock(driver.spt.data.transforms,"Compose",side_effect=compose)
            mock(driver.hydra.utils,"instantiate",side_effect=lambda _:TinyWorldModel())
            mock(driver.swm.data.utils,"get_cache_dir",return_value=str(work/"exports"))
            mock(driver.swm.wm.utils,"save_pretrained",side_effect=export)
            mock(driver.CheckpointManager,"due",return_value=True)
            reference, resumed = cfg("reference"),cfg("resumed")
            assert driver._train(reference) == 0
            polls = iter([False]*cut+[True])
            with patch.object(driver,"pause_requested",side_effect=lambda _:next(polls)):
                assert driver._train(resumed) == 75
            base = Path(resumed.run_metadata_dir)
            last = base/'checkpoints/last.ckpt'
            paused = torch.load(last,map_location='cpu',weights_only=False)
            assert paused["global_step"] == cut and "order" not in paused["resumable"]
            with (base/'training-indices.i64').open('ab') as f: f.write(b'abandoned')
            with (base/'metrics.csv').open('ab') as f: f.write(b'abandoned-metric')
            resumed.resume_from = str(last)
            assert driver._train(resumed) == 0
        ref = Path(reference.run_metadata_dir)
        def load_final(folder):return torch.load(folder/'checkpoints/final.ckpt',map_location='cpu',weights_only=False)
        a,b=load_final(ref),load_final(base)
        for k in ("state_dict","optimizer_states","lr_schedulers","global_step","epoch"):
            assert_equal(a[k],b[k])
        assert_equal(a["resumable"]["rng"],b["resumable"]["rng"])
        assert (base/"metrics.csv").read_bytes()==(ref/"metrics.csv").read_bytes()
        assert (base/"training-indices.i64").read_bytes()==(ref/"training-indices.i64").read_bytes()
        for folder in (ref,base):
            record=read(folder/"data-order.json")
            assert record["counts_match"] and record["epochs"]==record["completed_epochs"]==3
            assert record["optimizer_steps"]==9 and record["windows_seen"]==33 and record["last_batch_size"]==3
            trace = np.fromfile(folder/"training-indices.i64",dtype="<i8")
            for e in range(3):
                g=torch.Generator().manual_seed(3072+e)
                torch.empty((),dtype=torch.int64).random_(generator=g)
                np.testing.assert_array_equal(trace[e*11:(e+1)*11],torch.randperm(11,generator=g).numpy())
            verify_delta_metrics(folder/"metrics.csv",variant,3,3,4,3)
            retained=list((folder/"checkpoints").glob("step-*.ckpt"))
            assert len(retained)==3 and any(p.name.startswith("step-00000000-") for p in retained)
            assert read(folder/"runcompletion.json")["export_path"].endswith("weights_epoch_3.pt")
            grad=read(folder/"first-step-gradients.json")
            assert bool(grad["ldad"]["gradient_tensors"])==bool(variant.ldad_weight)
            full_initializations.add(read(folder/"initialization.json")["state_sha256"])
        assert len(list(base.glob("training-indices.abandoned-*.i64")))==len(list(base.glob("metrics.abandoned-*.csv")))==1
        reports.append({"variant":variant.label,"resumed_after_step":cut,"three_epochs_bitwise_equal":True})
    assert len(full_initializations)==1 and not torch.cuda.is_initialized()
    original = Path("/home/adam/lewm/experiments/episode50-30-20261106-03")
    if not (original / "arrays.npz").is_file():
        return {"passed":True,"cpu_only":True,"variants":reports,"historical_full_schedule_fixture_available":False}
    with np.load(original/"arrays.npz") as a:
        order=EpochOrder(a["train_clip_indices"],3072,50)
    assert order.protocol["epoch_sha256"][0]==read(original/"runs/sigreg/data-order.json")["order_sha256"]
    assert len(set(order.protocol["epoch_sha256"]))==50 and len(order)==18282850
    return {"passed":True,"cpu_only":True,"variants":reports,"real_50_epoch_order_sha256":order.order_hash,
            "real_epoch_one_sha256":order.protocol["epoch_sha256"][0],"real_50_epoch_windows":len(order)}


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="delta-resume-check-") as directory:
        print(json.dumps(run(Path(directory)),indent=2),flush=True)
