"""Train three independent Delta-JEPA models on one shared minibatch stream."""

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import random
import sys
import time
import uuid
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf, open_dict

from episode_data import get_episode_normalizer, load_episode_experiment
from experiment_tracking import initialization_record
from resumable_training import (CheckpointManager, DurableTrainingTrace, EpochOrder, FixedOrderBatchSampler,
    IndexedWindowDataset, atomic_json, capture_rng_state, file_sha256, json_hash, pause_requested, restore_rng_state, utc_now)
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from delta_sweep_config import VARIANTS
from train import CROSS_EXAMPLE_FIELDS, configure_training_objective
from train_resumable import MetricsWriter, delta_gradient_evidence
from utils import PreserveExecutedActions, get_img_preprocessor


def document(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_shared_data(cfg, manifest_root):
    options = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    name = options.pop("name")
    dataset = swm.data.load_dataset(name, transform=None, cache_dir=os.environ.get("LOCAL_DATASET_DIR"), **options)
    bundle = load_episode_experiment(manifest_root / "manifest.json", dataset)
    transforms = [PreserveExecutedActions(cfg.data.dataset.frameskip), get_img_preprocessor("pixels", "pixels", cfg.img_size)]
    for column in cfg.data.dataset.keys_to_load:
        if not column.startswith("pixels"):
            transforms.append(get_episode_normalizer(bundle, column))
    dataset.transform = spt.data.transforms.Compose(*transforms)
    return dataset, bundle


def build_model(cfg):
    return hydra.utils.instantiate(cfg.model)


def transfer_batch(batch, device):
    """One device-batch construction for all members; window IDs stay on CPU."""
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) and key != "_window_index" else value
            for key, value in batch.items()}


def save_export(member):
    swm.wm.utils.save_pretrained(member.model, run_name=member.cfg.output_model_name,
        config=member.cfg.model, filename=f"weights_epoch_{member.cfg.trainer.max_epochs}.pt")


def compose_members(root):
    configs = {}
    with hydra.initialize_config_dir(config_dir=str(Path(__file__).resolve().parent / "config/train"), version_base=None):
        for key, variant in VARIANTS.items():
            plan = document(root / "members" / key / "plan.json")
            command = plan["training_commands"][key]
            pos = command.index("--config-name")
            if command[pos + 1] != variant.config:
                raise ValueError(f"Unexpected training configuration for {key}")
            cfg = hydra.compose(config_name=variant.config, overrides=command[pos + 2:])
            if cfg.output_model_name != plan["model_names"][key] or plan["epochs"] != cfg.trainer.max_epochs:
                raise ValueError(f"Member plan/config mismatch for {key}")
            configs[key] = cfg
    return configs


def checked_configs(root, configs, dataset, bundle, strict):
    if set(configs) != set(VARIANTS):
        raise ValueError("Group must contain exactly the three registered Delta variants")
    shared = None
    decoder_spec = None
    hashes, output_names = {}, set()
    manifest = bundle["manifest"]
    for key, variant in VARIANTS.items():
        cfg = configs[key]
        member = root / "members" / key
        own_manifest = (member / "manifest.json").resolve(strict=True)
        own = document(own_manifest)
        if own["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError(f"Manifest differs for {key}")
        if strict:
            if json_hash({k:v for k,v in own.items() if k != "manifest_sha256"}) != own["manifest_sha256"]:
                raise ValueError(f"Invalid member manifest checksum: {key}")
            archive = (own_manifest.parent / own["arrays_file"]).resolve(strict=True)
            canonical = (root / manifest["arrays_file"]).resolve(strict=True)
            if not archive.samefile(canonical) and file_sha256(archive) != manifest["arrays_sha256"]:
                raise ValueError(f"Invalid member split archive: {key}")
        if (Path(cfg.episode_manifest).resolve() != own_manifest or
                Path(cfg.run_metadata_dir).resolve() != (member / "runs" / key).resolve() or
                Path(cfg.control_file).resolve() != root / "control.json"):
            raise ValueError(f"Invalid group-owned manifest, run, or control path for {key}")
        if cfg.get("resume_from") or cfg.get("compile_training") or cfg.trainer.get("max_steps",-1) != -1:
            raise ValueError("Grouped resume uses only its durable group journal")
        if (cfg.loss.type != variant.objective or cfg.loss.ldad.weight != variant.ldad_weight
                or cfg.loss.sigreg.weight != variant.sigreg_weight or not cfg.track_cross_example_latent_spread):
            raise ValueError(f"Delta objective/diagnostics differ from {key}")
        if not isinstance(cfg.trainer.max_epochs, int) or cfg.trainer.max_epochs <= 0 or cfg.seed != manifest["training_order"]["seed"]:
            raise ValueError("Invalid epoch count or training order seed")
        if cfg.loader.batch_size != manifest["training_order"]["batch_size"] or cfg.get("checkpoint_every_seconds",300) != 300:
            raise ValueError("Expected the sealed batch size and 300-second checkpoint interval")
        name = str(cfg.output_model_name)
        if not name or Path(name).name != name or name in output_names:
            raise ValueError("Each member requires its own simple export name")
        output_names.add(name)
        with open_dict(cfg):
            cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")
            cfg.episode_manifest = str(own_manifest)
            cfg.episode_manifest_sha256 = manifest["manifest_sha256"]
        resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        identity = {k:v for k,v in resolved.items() if k not in ("resume_from","control_file")}
        hashes[key] = json_hash(identity)
        matched = copy.deepcopy(identity)
        for k in ("subdir","output_model_name","run_metadata_dir","episode_manifest","wandb"):
            matched.pop(k,None)
        matched["trainer"].pop("default_root_dir",None)
        candidate_decoder = matched["model"].pop("ldad",None)
        if variant.ldad_weight == 0 and candidate_decoder is not None:
            raise ValueError("The grouped control must omit the LDAD module entirely")
        if variant.ldad_weight > 0:
            if strict and not candidate_decoder:
                raise ValueError("The LDAD-enabled group members require a decoder config")
            if decoder_spec is not None and decoder_spec != candidate_decoder:
                raise ValueError("The two LDAD-enabled group members require identical decoder settings")
            decoder_spec = candidate_decoder
        for term in ("ldad","sigreg"):
            matched["loss"][term].pop("weight")
        if shared is not None and matched != shared:
            raise ValueError("Member model, training, or data settings differ beyond the allowed objectives and paths")
        shared = matched
    return hashes


def export_directory(cfg):
    return Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), cfg.output_model_name).resolve()


def bootstrap(root, configs, hashes, order, commit_exists):
    """A failed first save can restart from the shared seed after preserving partial artifacts."""
    path = root / "group-identity.json"
    expected = {"version":1, "order_sha256":order.order_hash, "configs":hashes,
                "exports":{k:str(export_directory(c)) for k,c in configs.items()}}
    if path.exists():
        if document(path) != expected:
            raise ValueError("Group configuration, order, or export namespace changed")
        if commit_exists:
            return
    elif commit_exists:
        raise ValueError("Group checkpoint exists without its bootstrap identity")
    reserved = ["initialization.json","order-protocol.json","config-identity.json","config.yaml","resume_events.json",
                "training-indices.i64","metrics.csv","progress.json","checkpoints","checkpoint.json","first-step-gradients.json",
                "data-order.json","runcompletion.json"]
    for cfg in configs.values():
        run_dir = Path(cfg.run_metadata_dir)
        existing = [run_dir / name for name in reserved if (run_dir / name).exists()]
        output = export_directory(cfg)
        if output.exists():
            contents = list(output.iterdir())
            if contents and (not path.exists() or any(p.name != "config.yaml" for p in contents)):
                raise FileExistsError(f"Export directory already exists for a fresh grouped member: {output}")
        if any((run_dir / name).is_file() for name in ("runcompletion.json","data-order.json")):
            raise RuntimeError(f"Completed member lacks a committed group journal: {run_dir}")
        if any(p.name.split("-")[1] != "00000000" for p in (run_dir/"checkpoints").glob("step-*.ckpt")):
            raise RuntimeError(f"Noninitial generation lacks a committed group journal: {run_dir}")
        if existing:
            if not path.exists():
                raise FileExistsError(f"Unowned run artifacts at {run_dir}")
            backup = run_dir / ("abandoned-bootstrap-" + uuid.uuid4().hex)
            backup.mkdir()
            for artifact in existing:
                artifact.rename(backup / artifact.name)
    atomic_json(path, expected)


class Member:
    def __init__(self, key, cfg, config_hash, bundle, order, sampler, device, resuming):
        self.key,self.cfg,self.config_hash = key,cfg,config_hash
        self.run_dir = Path(cfg.run_metadata_dir).resolve()
        self.run_dir.mkdir(parents=True,exist_ok=True)
        self.device,self.use_bf16 = device, str(cfg.trainer.precision) in ("bf16","bf16-mixed")
        seed_all(int(cfg.seed))
        self.model = build_model(cfg)
        if (self.model.ldad is not None) != (cfg.loss.ldad.weight > 0):
            raise ValueError("The control must have no decoder, and both positive objectives must have one")
        if not resuming:
            initialization_record(self.model, cfg, bundle, self.run_dir)
        self.init = document(self.run_dir / "initialization.json")
        self.model.to(device).train()
        self.forward,self.sigreg = configure_training_objective(cfg,self.model)
        if self.sigreg is not None:
            self.sigreg.to(device)
        opts = OmegaConf.to_container(cfg.optimizer,resolve=True)
        if opts.pop("type") != "AdamW":
            raise ValueError("Expected AdamW")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **opts)
        from stable_pretraining.optim.lr_scheduler import LinearWarmupCosineAnnealingLR
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer,warmup_steps=max(1,int(.01*sampler.total_steps)),
                                                      max_steps=sampler.total_steps,warmup_start_lr=0.,eta_min=0.)
        self.rng = capture_rng_state(synchronize=False)
        self.manager = CheckpointManager(self.run_dir,bundle["manifest"]["manifest_sha256"],config_hash,
                    sampler.batch_size,order,interval_seconds=300)
        schedule = {**order.protocol,"manifest_sha256":self.manager.manifest_sha256,"order_sha256":order.order_hash,
                    "batch_size":sampler.batch_size,"steps_per_epoch":sampler.steps_per_epoch,"expected_steps":sampler.total_steps}
        self.export_dir = export_directory(cfg)
        if resuming:
            if (document(self.run_dir / "order-protocol.json") != schedule or
                    document(self.run_dir / "config-identity.json")["config_sha256"] != config_hash):
                raise ValueError(f"Incompatible saved identity for {key}")
        else:
            self.export_dir.mkdir(parents=True,exist_ok=True)
            OmegaConf.save(cfg,self.export_dir / "config.yaml")
            OmegaConf.save(cfg,self.run_dir / "config.yaml")
            atomic_json(self.run_dir/"config-identity.json",{"config_sha256":config_hash,"manifest_sha256":self.manager.manifest_sha256,
                                                      "excluded_runtime_keys":["resume_from","control_file"]})
            atomic_json(self.run_dir/"order-protocol.json",schedule)
            atomic_json(self.run_dir/"resume_events.json",[])
        self.holder=SimpleNamespace(model=self.model,sigreg=self.sigreg,logged={})
        self.holder.log_dict=lambda values,**kwargs:setattr(self.holder,"logged",values)
        self.first_gradient_hash=None
        self.last_values={}
        self.events=document(self.run_dir/"resume_events.json")

    def open_journals(self, order, stage, sampler, stack):
        meta=None
        if stage is not None:
            payload=self.manager.load(stage["path"],self.model,self.optimizer,self.scheduler,self.sigreg,sampler=sampler)
            meta=payload["resumable"]
            self.rng=meta["rng"]
            self.last_values=meta["extra"]["last_values"]
            self.first_gradient_hash=meta["extra"]["first_step_gradient_sha256"]
            del payload
        self.trace=DurableTrainingTrace(self.run_dir,order,resume=meta)
        stack.callback(self.trace.close)
        fields=["step","epoch","fit/pred_loss","fit/loss","hparams/lr_model_opt"]
        if self.sigreg is not None:
            fields.append("fit/sigreg_loss")
        if self.model.ldad is not None:
            fields.append("fit/ldad_loss")
        fields.extend("fit/"+k for k in CROSS_EXAMPLE_FIELDS)
        self.metrics=MetricsWriter(self.run_dir/"metrics.csv",fields,resume=meta["extra"] if meta else None)
        stack.callback(self.metrics.close)
        if stage is not None and stage["status"]!="complete":
            self.events.append({"resumed_utc":utc_now(),"checkpoint":stage["path"],"global_step":sampler.global_step,
                                "windows_seen":sampler.cursor,"config_sha256":self.config_hash})
            atomic_json(self.run_dir/"resume_events.json",self.events)

    def step(self, batch, previous_step):
        restore_rng_state(self.rng)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device.type,dtype=torch.bfloat16,enabled=self.use_bf16):
            output=self.forward(self.holder,batch,"fit",self.cfg)
        if not torch.isfinite(output["loss"]).item():
            raise FloatingPointError(f"Nonfinite loss in {self.key}")
        output["loss"].backward()
        if previous_step==0:
            evidence=delta_gradient_evidence(self.model,self.cfg.loss.ldad.weight>0)
            dtypes={k:str(v.dtype) for k,v in output.items() if k in ("pred_loss","ldad_loss","sigreg_loss","loss")}
            path=self.run_dir/"first-step-gradients.json"
            atomic_json(path,{"step":1,"config_sha256":self.config_hash,"term_dtypes":dtypes,**evidence})
            self.first_gradient_hash=file_sha256(path)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),float(self.cfg.trainer.gradient_clip_val),error_if_nonfinite=True)
        self.optimizer.step()
        self.scheduler.step()
        metric_keys = list(self.holder.logged)
        values = torch.stack([self.holder.logged[k].detach().float().reshape(()) for k in metric_keys]).cpu().tolist()
        self.last_values = dict(zip(metric_keys, values))
        self.last_values["hparams/lr_model_opt"]=float(self.optimizer.param_groups[0]["lr"])
        if not all(math.isfinite(x) for x in self.last_values.values()):
            raise FloatingPointError(f"Nonfinite diagnostic in {self.key}")
        self.holder.logged={}
        self.optimizer.zero_grad(set_to_none=True)
        self.rng=capture_rng_state(synchronize=False)


def hash_prefix(path,length):
    digest=hashlib.sha256()
    with path.open("rb") as f:
        while length:
            chunk=f.read(min(1024*1024,length))
            if not chunk:
                raise ValueError(f"Truncated committed prefix in {path}")
            digest.update(chunk)
            length-=len(chunk)
    return digest.hexdigest()


def validate_commit(record,members,order,sampler):
    if (record["version"]!=1 or record["record_sha256"]!=json_hash({k:v for k,v in record.items() if k!="record_sha256"})
            or set(record["members"])!=set(members) or record["order_sha256"]!=order.order_hash):
        raise ValueError("Invalid group checkpoint identity/order")
    sampler.restore_cursor(record["cursor"])
    if sampler.global_step!=record["global_step"]:
        raise ValueError("Inconsistent group checkpoint cursor")
    for key,m in members.items():
        entry=record["members"][key]
        path=Path(entry["path"])
        if (path.parent!=m.manager.directory or path.name!=entry["generation_id"]+".ckpt" or
                entry["config_sha256"]!=m.config_hash or record["manifest_sha256"]!=m.manager.manifest_sha256 or
                entry["global_step"]!=sampler.global_step or entry["windows_seen"]!=sampler.cursor or
                entry["status"]!=record["status"] or file_sha256(path)!=entry["sha256"]):
            raise ValueError(f"Unusable group generation for {key}")
        side=document(path.with_suffix(".json"))
        if side["sha256"]!=entry["sha256"] or path.stat().st_size!=side["size_bytes"] or side["generation_id"]!=entry["generation_id"]:
            raise ValueError(f"Unusable checksum sidecar for {key}")
        meta=torch.load(path,mmap=True,map_location="cpu",weights_only=False)["resumable"]
        extra=meta["extra"]
        if (meta["generation_id"]!=entry["generation_id"] or meta["config_sha256"]!=m.config_hash or
                meta["order_protocol"]!=order.protocol or meta["manifest_sha256"]!=record["manifest_sha256"] or
                meta["cursor"]!=sampler.cursor or meta["trace_sha256"]!=record["trace_sha256"] or
                hash_prefix(m.run_dir/"training-indices.i64",meta["trace_bytes"])!=meta["trace_sha256"] or
                hash_prefix(m.run_dir/"metrics.csv",extra["metrics_bytes"])!=extra["metrics_sha256"]):
            raise ValueError(f"Committed evidence mismatch for {key}")
        if sampler.cursor and file_sha256(m.run_dir/"first-step-gradients.json")!=extra["first_step_gradient_sha256"]:
            raise ValueError(f"Committed first-step gradients mismatch for {key}")


def recover_commit(root,members,order,sampler):
    errors=[]
    for name in ("group-checkpoint.json","group-checkpoint.previous.json","group-checkpoint.initial.json"):
        path=root/name
        if not path.is_file():
            continue
        try:
            record=document(path)
            validate_commit(record,members,order,sampler)
        except (OSError,ValueError,KeyError,RuntimeError,EOFError) as exc:
            errors.append(f"{name}: {exc}")
            continue
        if name!="group-checkpoint.json":
            atomic_json(root/("group-recovery-"+uuid.uuid4().hex+".json"),{"selected":name,"errors":errors,"utc":utc_now()})
            atomic_json(root/"group-checkpoint.json",record)
            # A newer invalid generation cannot serve as the backup of this recovery.
            atomic_json(root/"group-checkpoint.previous.json",record)
        return record
    raise RuntimeError("No valid committed group generation: "+"; ".join(errors))


def commit_group(root, record):
    """The current group marker is the commit point, after all three stages are durable."""
    path=root/"group-checkpoint.json"
    if path.exists():
        atomic_json(root/"group-checkpoint.previous.json",document(path))
    if not (root/"group-checkpoint.initial.json").exists():
        if record["global_step"]!=0:
            raise ValueError("Initial group generation must be step zero")
        atomic_json(root/"group-checkpoint.initial.json",record)
    atomic_json(path,record)


def promote_group(root,members,record):
    initial=document(root/"group-checkpoint.initial.json")["members"]
    prior_path=root/"group-checkpoint.previous.json"
    prior=document(prior_path)["members"] if prior_path.is_file() else {}
    for key,m in members.items():
        m.manager.publish(record["members"][key], final=record["status"]=="complete",
                          previous_generation=prior.get(key,initial[key])["generation_id"],
                          initial_generation=initial[key]["generation_id"])


def publish_group_checkpoint(root,members,sampler,status,elapsed,started):
    stages={}
    for key,m in members.items():
        extra={"elapsed_seconds":elapsed,"started_utc":started,"objective":"delta_jepa","last_values":m.last_values,
               "first_step_gradient_sha256":m.first_gradient_hash,**m.metrics.checkpoint_metadata()}
        stages[key]=m.manager.stage(m.model,m.optimizer,m.scheduler,m.sigreg,sampler,m.trace,status=status,extra=extra,rng=m.rng)
    traces={m.trace.digest.hexdigest() for m in members.values()}
    if len(traces)!=1:
        raise ValueError("Group traces differ at checkpoint boundary")
    record={"version":1,"generation_id":uuid.uuid4().hex,"saved_utc":utc_now(),"status":status,
            "global_step":sampler.global_step,"cursor":sampler.cursor,"order_sha256":sampler.order.order_hash,
            "manifest_sha256":next(iter(members.values())).manager.manifest_sha256,"trace_sha256":traces.pop(),
            "snapshot_elapsed_seconds":elapsed,"started_utc":started,"members":stages}
    record["record_sha256"]=json_hash(record)
    commit_group(root,record)
    promote_group(root,members,record)
    return record


def train_group(root, configs=None):
    root=Path(root).resolve(strict=True)
    with (root/"group-training.lock").open("a") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _train_group(root,configs)


def _train_group(root,configs):
    strict=configs is None
    configs=compose_members(root) if strict else configs
    if set(configs)!=set(VARIANTS):
        raise ValueError("Three distinct member configurations are required")
    first=configs[next(iter(VARIANTS))]
    seed=int(first.seed)
    seed_all(seed)
    if first.get("cpu_threads") is not None:
        torch.set_num_threads(int(first.cpu_threads))
    accelerator=first.trainer.accelerator
    if accelerator not in ("gpu","cuda","cpu"):
        raise ValueError("Use an explicit GPU or CPU accelerator")
    device=torch.device("cpu" if accelerator=="cpu" else "cuda:0")
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if str(first.trainer.precision) not in ("bf16","bf16-mixed","32","32-true"):
        raise ValueError("Unsupported group precision")
    dataset,bundle=load_shared_data(first,root)
    manifest=bundle["manifest"]
    if document(root/"manifest.json")["manifest_sha256"]!=manifest["manifest_sha256"]:
        raise ValueError("Shared dataset and root manifest differ")
    hashes=checked_configs(root,configs,dataset,bundle,strict)
    epochs=int(first.trainer.max_epochs)
    order=EpochOrder(bundle["arrays"]["train_clip_indices"],seed,epochs)
    sampler=FixedOrderBatchSampler(order,int(first.loader.batch_size))
    if sampler.steps_per_epoch!=manifest["training_order"]["steps_per_epoch"]:
        raise ValueError("Sampler disagrees with sealed manifest")
    resuming=any((root/name).exists() for name in ("group-checkpoint.json","group-checkpoint.previous.json","group-checkpoint.initial.json"))
    bootstrap(root,configs,hashes,order,resuming)
    members={key:Member(key,configs[key],hashes[key],bundle,order,sampler,device,resuming) for key in VARIANTS}
    resident_storage = set()
    for m in members.values():
        tensors = list(m.model.parameters()) + list(m.model.buffers())
        storage = {t.untyped_storage().data_ptr() for t in tensors if t.numel()}
        if resident_storage.intersection(storage):
            raise ValueError("Member models must not alias mutable tensor storage")
        resident_storage.update(storage)
    if len({m.init["shared_state_sha256"] for m in members.values()})!=1:
        raise ValueError("The three base/shared initializations differ")
    positive=[m for m in members.values() if m.cfg.loss.ldad.weight>0]
    if len({m.init["state_sha256"] for m in positive})!=1:
        raise ValueError("LDAD-enabled models must start from identical full weights")
    result=recover_commit(root,members,order,sampler) if resuming else None
    old_elapsed=result["snapshot_elapsed_seconds"] if result else 0.
    started=result["started_utc"] if result else utc_now()
    session_start=time.monotonic()
    def elapsed():return old_elapsed + time.monotonic()-session_start
    transferred=sampler.global_step
    stream_opts=OmegaConf.to_container(first.loader,resolve=True)
    for key in ("batch_size","shuffle","drop_last","sampler","batch_sampler","generator"):
        stream_opts.pop(key,None)
    loader=torch.utils.data.DataLoader(IndexedWindowDataset(dataset),batch_sampler=sampler,
                generator=torch.Generator().manual_seed(seed),**stream_opts)
    log_interval=max(1,int(first.trainer.get("log_every_n_steps",10)))
    control_path=root/"control.json"
    iterator=None

    def progress(status,error=None,write_order=False):
        t=elapsed()
        common={"schema_version":1,"backend":"native_pytorch","status":status,"started_utc":started,
                "updated_utc":utc_now(),"manifest_sha256":manifest["manifest_sha256"],"epochs":epochs,
                "steps_per_epoch":sampler.steps_per_epoch,"windows_per_epoch":sampler.windows_per_epoch,
                "expected_steps":sampler.total_steps,"expected_windows":len(order),"optimizer_steps":sampler.global_step,
                "batches_seen":sampler.global_step,"last_batch_size":sampler.last_batch_size,"elapsed_seconds":t,
                "completed_epochs":sampler.cursor//sampler.windows_per_epoch,
                "current_epoch":min(epochs,sampler.cursor//sampler.windows_per_epoch+1),
                "windows_in_epoch":sampler.cursor%sampler.windows_per_epoch if sampler.cursor<len(order) else sampler.windows_per_epoch,
                "eta_seconds":t*(sampler.total_steps-sampler.global_step)/sampler.global_step if sampler.global_step else None,
                "counts_match":sampler.global_step==sampler.total_steps and sampler.cursor==len(order)}
        if error:
            common.update(exception_type=type(error).__name__,exception_message=str(error))
        models={}
        for key,m in members.items():
            row={**common,**m.trace.summary(),"resume_count":len(m.events),"last_train_loss":m.last_values.get("fit/loss"),
                 "last_losses":{k.split("/",1)[-1]:v for k,v in m.last_values.items() if "loss" in k},
                 "nonfinite_loss_name":None,"shared_batch_group":str(root),
                 "ldad_parameter_tensors":sum(1 for _ in m.model.ldad.parameters()) if m.model.ldad is not None else 0}
            latest=m.manager.latest_record
            if latest:
                row.update(latest_checkpoint=latest["path"],checkpoint_step=latest["global_step"],checkpoint_saved_utc=latest["saved_utc"])
            atomic_json(m.run_dir/"progress.json",row)
            if write_order:
                atomic_json(m.run_dir/"data-order.json",row)
            models[key]=row
        memory={"allocated_mib":None,"reserved_mib":None,"peak_allocated_mib":None,"peak_reserved_mib":None}
        if device.type=="cuda":
            memory=dict(allocated_mib=torch.cuda.memory_allocated(device)/2**20,
                        reserved_mib=torch.cuda.memory_reserved(device)/2**20,
                        peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20,
                        peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20)
        leader=next(iter(members.values()))
        group={**common,"backend":"shared_batch_native_pytorch","members":models,"windows_seen":sampler.cursor,
               "order_sha256":leader.trace.digest.hexdigest(),"shared_batches_transferred":transferred,
               "batch_transfers_per_committed_group_step":1,"resident_model_count":len(members),
               "independent_member_storage_verified":True,"device":str(device),
               "cuda_memory":memory,"aggregate_model_updates":sampler.global_step*len(members),
               "aggregate_window_visits":sampler.cursor*len(members)}
        if result:
            group.update(checkpoint_step=result["global_step"],checkpoint_saved_utc=result["saved_utc"],
                         checkpoint_generation=result["generation_id"])
        atomic_json(root/"group-progress.json",group)
        return group

    with ExitStack() as stack:
        for key,m in members.items():
            m.open_journals(order,result["members"][key] if result else None,sampler,stack)
        if result:
            promote_group(root,members,result)
        try:
            if not result:
                result=publish_group_checkpoint(root,members,sampler,"running",elapsed(),started)
            if sampler.cursor<len(order):
                progress("running")
                if pause_requested(control_path):
                    result=publish_group_checkpoint(root,members,sampler,"paused",elapsed(),started)
                    progress("paused")
                    return 75
                iterator=iter(loader)
                for host_batch in iterator:
                    ids=host_batch["_window_index"]
                    batch=transfer_batch(host_batch,device)
                    transferred+=1
                    for m in members.values():
                        m.step(batch,sampler.global_step)
                    del batch,host_batch
                    sampler.commit(ids)
                    epoch=(sampler.cursor-1)//sampler.windows_per_epoch
                    for m in members.values():
                        m.trace.append(ids)
                        m.metrics.append({"step":sampler.global_step,"epoch":epoch,**m.last_values})
                    if pause_requested(control_path):
                        result=publish_group_checkpoint(root,members,sampler,"paused",elapsed(),started)
                        progress("paused")
                        return 75
                    if next(iter(members.values())).manager.due():
                        result=publish_group_checkpoint(root,members,sampler,"running",elapsed(),started)
                        progress("running")
                    elif sampler.global_step%log_interval==0:
                        for m in members.values():
                            m.metrics.flush()
                            if sampler.global_step%100==0:
                                m.trace.flush()
                        progress("running")
            if sampler.cursor!=len(order) or sampler.global_step!=sampler.total_steps:
                raise RuntimeError("Shared loader stopped before the complete schedule")
            if result["status"]!="complete":
                result=publish_group_checkpoint(root,members,sampler,"complete",elapsed(),started)
            for m in members.values():
                save_export(m)
            progress("complete",write_order=True)
            for key,m in members.items():
                final_path=m.manager.directory/"final.ckpt"
                export_path=m.export_dir / f"weights_epoch_{epochs}.pt"
                if not export_path.is_file():
                    raise FileNotFoundError(export_path)
                atomic_json(m.run_dir/"runcompletion.json",{
                    "passed":True,"backend":"native_pytorch","global_step":sampler.global_step,"expected_steps":sampler.total_steps,
                    "expected_windows":len(order),"epochs":epochs,"steps_per_epoch":sampler.steps_per_epoch,
                    "windows_per_epoch":sampler.windows_per_epoch,"final_ckpt_path":str(final_path),"export_path":str(export_path),
                    "registry_run":str(m.run_dir),"manifest_sha256":manifest["manifest_sha256"],"objective":"delta_jepa",
                    "resume_count":len(m.events),"completed_utc":utc_now(),"group_generation_id":result["generation_id"],
                    "group_record_sha256":result["record_sha256"]})
            atomic_json(root/"group-runcompletion.json",{"passed":True,"group_generation_id":result["generation_id"],
                        "global_step":sampler.global_step,"windows_seen":sampler.cursor,"members":list(members),"completed_utc":utc_now()})
            return 0
        except BaseException as exc:
            progress("failed",error=exc)
            raise
        finally:
            shutdown=getattr(iterator if iterator is not None else getattr(loader,"_iterator",None),"_shutdown_workers",None)
            if shutdown:
                shutdown()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir",required=True,type=Path)
    return train_group(parser.parse_args().experiment_dir)


if __name__=="__main__":
    raise SystemExit(main())
