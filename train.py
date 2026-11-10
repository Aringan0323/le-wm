import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from training_compute import combined_compute, mse_only_compute
from utils import get_column_normalizer, get_img_preprocessor, PreserveExecutedActions, SaveCkptCallback


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

def action_prediction_forward(self, batch, stage, cfg):
    """Predict next latents and infer encoded actions from their predicted change."""
    if cfg.num_preds != 1:
        raise ValueError("Action prediction requires num_preds=1 for aligned transitions")
    ctx_len = cfg.history_size
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)
    emb = output["emb"]
    ctx_emb = emb[:, :ctx_len]
    ctx_act = output["act_emb"][:, :ctx_len]
    tgt_emb = emb[:, 1:ctx_len + 1]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    # Each predicted transition is paired with the actions that produced it.
    # The inverse model sees only the latent delta, never the action embedding.
    delta = pred_emb - ctx_emb
    pred_act_emb = self.model.action_predictor(delta)
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    # Keep both branches attached, matching joint training of learned targets.
    output["action_loss"] = (pred_act_emb - ctx_act).pow(2).mean()
    output["loss"] = output["pred_loss"] + cfg.loss.action_prediction.weight * output["action_loss"]

    # Diagnostics only: these do not contribute to the training objective.
    diagnostics = {
        "latent_std": emb.detach().float().std(dim=(0, 1), unbiased=False).mean(),
        "action_embedding_std": ctx_act.detach().float().std(dim=(0, 1), unbiased=False).mean(),
        "predicted_action_std": pred_act_emb.detach().float().std(dim=(0, 1), unbiased=False).mean(),
    }
    if tracks_cross_example_latents(cfg):
        diagnostics.update(cross_example_latent_metrics(emb))
    metrics = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    metrics.update({f"{stage}/{k}": v for k, v in diagnostics.items()})
    self.log_dict(metrics, on_step=True, sync_dist=True, batch_size=emb.size(0))
    return output



CROSS_EXAMPLE_FIELDS = ("latent_cross_example_count", "latent_cross_example_var",
                        "latent_cross_example_rms_std", "latent_cross_example_max_range",
                        "latent_global_max_range")


def tracks_cross_example_latents(cfg):
    objective = cfg.loss.get("type", "sigreg")
    return objective == "mse_only" or (objective in ("action_prediction", "combined", "delta_jepa")
                                       and cfg.get("track_cross_example_latent_spread", False))


def cross_example_latent_metrics(emb):
    """Describe variability of encoded observations of shape [examples, times, channels].

    Each time/channel contributes its sample variance across examples (B - 1
    denominator). The reported variance averages those values, and rms_std is
    its square root. The cross-example maximum range is zero exactly when
    examples agree at each time/channel. The global maximum range also compares
    times and is zero exactly when every frame has the same latent vector.
    """
    values = emb.detach().float()
    minimum, maximum = torch.aminmax(values, dim=0)
    metrics = {
        "latent_cross_example_count": values.new_tensor(values.shape[0]),
        "latent_global_max_range": (maximum.amax(dim=0) - minimum.amin(dim=0)).amax(),
    }
    # A singleton supplies no pair of different examples to compare.
    if values.shape[0] >= 2:
        variance = values.var(dim=0, correction=1).mean()
        metrics.update({
            "latent_cross_example_var": variance,
            "latent_cross_example_rms_std": variance.sqrt(),
            "latent_cross_example_max_range": (maximum - minimum).amax(),
        })
    return metrics


def _log_computed_output(self, output, stage, cfg, batch_size):
    metrics = {f"{stage}/{name}": value.detach() for name, value in output.items()
               if "loss" in name}
    objective = cfg.loss.get("type", "sigreg")
    if objective in ("action_prediction", "combined"):
        metrics.update({
            f"{stage}/latent_std": output["emb"].detach().float().std(dim=(0, 1), unbiased=False).mean(),
            f"{stage}/action_embedding_std": output["ctx_act"].detach().float().std(dim=(0, 1), unbiased=False).mean(),
            f"{stage}/predicted_action_std": output["pred_act_emb"].detach().float().std(dim=(0, 1), unbiased=False).mean(),
        })
    if tracks_cross_example_latents(cfg):
        metrics.update({f"{stage}/{name}": value
                        for name, value in cross_example_latent_metrics(output["emb"]).items()})
    self.log_dict(metrics, on_step=True, sync_dist=True, batch_size=batch_size)
    return output


def combined_forward(self, batch, stage, cfg):
    output = combined_compute(self.model, self.sigreg, batch["pixels"], batch["action"],
                              history_size=cfg.history_size, num_preds=cfg.num_preds,
                              aux_weight=cfg.loss.action_prediction.weight,
                              sigreg_weight=cfg.loss.sigreg.weight)
    return _log_computed_output(self, output, stage, cfg, batch["pixels"].size(0))


def mse_only_forward(self, batch, stage, cfg):
    output = mse_only_compute(self.model, None, batch["pixels"], batch["action"],
                              history_size=cfg.history_size, num_preds=cfg.num_preds)
    return _log_computed_output(self, output, stage, cfg, batch["pixels"].size(0))


def delta_jepa_forward(self, batch, stage, cfg):
    """Score the forward prediction and decode raw actions from observed displacements."""
    output = mse_only_compute(self.model, None, batch["pixels"], batch["action"],
                              history_size=cfg.history_size, num_preds=1)
    emb = output["emb"]
    target_actions = batch["executed_actions"]
    weight = cfg.loss.ldad.weight
    with torch.set_grad_enabled(torch.is_grad_enabled() and weight > 0):
        predicted_actions = self.model.ldad(emb[:, 1:] - emb[:, :-1])
        if predicted_actions.shape != target_actions.shape:
            raise ValueError("LDAD predictions must match the recorded per-interval action sequence")
        output["ldad_loss"] = torch.nn.functional.mse_loss(predicted_actions.float(), target_actions.float())
    if cfg.loss.sigreg.weight > 0:
        output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
        output["loss"] = output["loss"] + cfg.loss.sigreg.weight * output["sigreg_loss"]
    if weight > 0:
        output["loss"] = output["loss"] + weight * output["ldad_loss"]
    return _log_computed_output(self, output, stage, cfg, batch["pixels"].size(0))


def compiled_training_forward(self, batch, stage, cfg):
    """Run the tensor objective through Inductor and log outside the graph."""
    output = self.compute_objective(self.model, getattr(self, "sigreg", None),
                                    batch["pixels"], batch["action"])
    return _log_computed_output(self, output, stage, cfg, batch["pixels"].size(0))


def configure_training_objective(cfg, model):
    """Select the forward function and auxiliary module for either trainer."""
    objective = cfg.loss.get("type", "sigreg")
    functions = {"sigreg": lejepa_forward, "action_prediction": action_prediction_forward,
                 "combined": combined_forward, "mse_only": mse_only_forward,
                 "delta_jepa": delta_jepa_forward}
    if objective not in functions:
        raise ValueError(f"Unknown loss type: {objective}")
    if objective == "mse_only" and model.action_predictor is not None:
        raise ValueError("The MSE-only experiment uses model=lewm without an inverse action predictor")
    if objective in ("action_prediction", "combined") and (
        cfg.num_preds != 1 or model.action_predictor is None
    ):
        raise ValueError("Action prediction requires num_preds=1 and an action_predictor")
    if objective == "delta_jepa":
        if cfg.num_preds != 1 or model.ldad is None or model.action_predictor is not None:
            raise ValueError("Delta-JEPA requires one-step prediction and the LDAD model")
        if model.ldad.horizon != cfg.data.dataset.frameskip:
            raise ValueError("The LDAD query count must match the action interval between observed frames")
        if min(cfg.loss.ldad.weight, cfg.loss.sigreg.weight) < 0:
            raise ValueError("Delta-JEPA loss weights must be nonnegative")
    use_sigreg = objective in ("sigreg", "combined") or (objective == "delta_jepa" and cfg.loss.sigreg.weight > 0)
    sigreg = SIGReg(**cfg.loss.sigreg.kwargs) if use_sigreg else None
    return functions[objective], sigreg

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    # Seed before model initialization, then keep Manager on the same seed.
    pl.seed_everything(cfg.seed, workers=True)
    if cfg.get("cpu_threads") is not None:
        torch.set_num_threads(int(cfg.cpu_threads))
    spt.set(requeue_checkpoint_every_n_steps=int(cfg.get('checkpoint_every_n_steps', 0)))

    #########################
    ##       dataset       ##
    #########################

    episode_bundle = None
    if cfg.get("episode_manifest"):
        from episode_data import load_episode_experiment, episode_subset, get_episode_normalizer
        from experiment_tracking import IndexedSubsetWrapper, initialization_record, TrainingTraceCallback
        if cfg.get("resume_from"):
            raise ValueError("Episode experiments require explicit data-cursor recovery; automatic mid-epoch resume is disabled")
        if cfg.trainer.max_epochs != 1 or cfg.trainer.get("max_steps", -1) != -1:
            raise ValueError("This episode experiment requires one complete epoch without a step cap")
        if not cfg.get("run_metadata_dir"):
            raise ValueError("Episode experiments require run_metadata_dir")

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    if cfg.get("episode_manifest"):
        episode_bundle = load_episode_experiment(cfg.episode_manifest, dataset)
        with open_dict(cfg):
            cfg.episode_manifest = episode_bundle["manifest_path"]
            cfg.episode_manifest_sha256 = episode_bundle["manifest"]["manifest_sha256"]
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    if cfg.loss.get("type") == "delta_jepa":
        transforms.insert(0, PreserveExecutedActions(cfg.data.dataset.frameskip))
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = (get_episode_normalizer(episode_bundle, col, col, col)
                          if episode_bundle is not None else get_column_normalizer(dataset, col, col))
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    if episode_bundle is not None:
        train_set = IndexedSubsetWrapper(episode_subset(dataset, episode_bundle, "train"))
        train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=False, generator=rnd_gen)
        val = None  # The sealed test episodes are evaluated only after both models finish.
    else:
        train_set, val_set = spt.data.random_split(
            dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
        )
        train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
        val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)
    if episode_bundle is not None:
        initialization_record(world_model, cfg, episode_bundle, Path(cfg.run_metadata_dir))

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            # stable-pretraining 0.1.8 derives this scheduler's length in steps.
            "interval": "step",
        },
    }

    objective = cfg.loss.get("type", "sigreg")
    forward_fn, sigreg = configure_training_objective(cfg, world_model)
    auxiliary_modules = {"sigreg": sigreg} if sigreg is not None else {}

    if cfg.get("compile_training", False):
        from training_compute import build_training_compute
        aux_config = cfg.loss.sigreg if objective == "sigreg" else cfg.loss.get("action_prediction")
        auxiliary_modules["compute_objective"] = build_training_compute(
            objective, history_size=cfg.history_size, num_preds=cfg.num_preds,
            aux_weight=aux_config.weight if aux_config is not None else 0.0,
            sigreg_weight=cfg.loss.sigreg.weight if sigreg is not None else 0.0,
            compile_mode=cfg.get("compile_mode", "reduce-overhead"), dynamic=False,
        )
        forward_fn = compiled_training_forward
    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        forward=partial(forward_fn, cfg=cfg),
        optim=optimizers,
        **auxiliary_modules,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
    )

    callbacks = [object_dump_callback]
    if episode_bundle is not None:
        callbacks.append(TrainingTraceCallback(
            Path(cfg.run_metadata_dir),
            expected_steps=episode_bundle["manifest"]["training_order"]["steps_per_epoch"],
            expected_windows=episode_bundle["manifest"]["counts"]["train"]["windows"],
            manifest_sha=episode_bundle["manifest"]["manifest_sha256"],
        ))
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=0 if episode_bundle is not None else 1,
        logger=logger,
        enable_checkpointing=True,
    )

    resume_from = cfg.get("resume_from")
    ckpt_path = Path(resume_from).expanduser().resolve() if resume_from else None
    if ckpt_path is not None and not ckpt_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {ckpt_path}")
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path,
        weights_only=False,  # Restore optimizer, scheduler, and loop state on resume.
        seed=cfg.seed,
    )

    manager()
    if episode_bundle is not None:
        # The periodic last.ckpt may lag the last partial batch; save an explicit final state.
        import json
        final_checkpoint = Path(manager._run_dir) / "checkpoints" / "final.ckpt"
        trainer.save_checkpoint(str(final_checkpoint))
        completion = {
            "passed": True,
            "global_step": trainer.global_step,
            "final_ckpt_path": str(final_checkpoint),
            "export_path": str(run_dir / "weights_epoch_1.pt"),
            "registry_run": str(manager._run_dir),
            "manifest_sha256": episode_bundle["manifest"]["manifest_sha256"],
            "objective": objective,
        }
        destination = Path(cfg.run_metadata_dir) / "runcompletion.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(completion, indent=2))
        temporary.replace(destination)
    return


if __name__ == "__main__":
    run()
