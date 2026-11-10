"""Tensor-only training objectives, optionally compiled without wrapping models.

Logging and detached diagnostic statistics belong in the caller. These functions
create their own encode dictionary, so caller batches are never mutated. Normal
training behavior (dropout, SIGReg sampling, and BatchNorm running-state updates)
is retained. Compilation can change RNG implementation and floating-point fusion;
the formulas and attached target-gradient paths are unchanged.
"""

from functools import partial

import torch


def _next_embedding_terms(model, pixels, actions, history_size, num_preds):
    """Encode the clip, condition on recorded actions, and score shifted targets."""
    output = model.encode({"pixels": pixels, "action": torch.nan_to_num(actions, 0.0)})
    emb = output["emb"]
    ctx_emb = emb[:, :history_size]
    ctx_act = output["act_emb"][:, :history_size]
    tgt_emb = emb[:, num_preds:]
    pred_emb = model.predict(ctx_emb, ctx_act)
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["ctx_emb"] = ctx_emb
    output["ctx_act"] = ctx_act
    output["pred_emb"] = pred_emb
    return output


def mse_only_compute(model, sigreg, pixels, actions, *, history_size=3,
                     num_preds=1, aux_weight=0.0):
    """Train the encoder, action encoder, and forward predictor on next-embedding MSE."""
    output = _next_embedding_terms(model, pixels, actions, history_size, num_preds)
    output["loss"] = output["pred_loss"]
    return output


def sigreg_compute(model, sigreg, pixels, actions, *, history_size=3,
                   num_preds=1, aux_weight=0.09):
    """Original next-embedding MSE plus SIGReg, with both target branches attached."""
    if sigreg is None:
        raise ValueError("The SIGReg objective requires a regularizer module")
    output = _next_embedding_terms(model, pixels, actions, history_size, num_preds)
    output["sigreg_loss"] = sigreg(output["emb"].transpose(0, 1))
    output["loss"] = output["pred_loss"] + aux_weight * output["sigreg_loss"]
    return output


def _action_prediction_terms(model, pixels, actions, history_size, num_preds):
    """Compute both prediction errors from one encoding and one dynamics pass."""
    if num_preds != 1:
        raise ValueError("Action prediction requires num_preds=1 for aligned transitions")
    if model.action_predictor is None:
        raise ValueError("The action-prediction objective requires an action predictor")
    output = model.encode({"pixels": pixels, "action": torch.nan_to_num(actions, 0.0)})
    emb = output["emb"]
    ctx_emb = emb[:, :history_size]
    ctx_act = output["act_emb"][:, :history_size]
    tgt_emb = emb[:, 1:history_size + 1]
    pred_emb = model.predict(ctx_emb, ctx_act)
    delta = pred_emb - ctx_emb
    pred_act_emb = model.action_predictor(delta)
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["action_loss"] = (pred_act_emb - ctx_act).pow(2).mean()
    output["ctx_emb"] = ctx_emb
    output["ctx_act"] = ctx_act
    output["pred_emb"] = pred_emb
    output["pred_act_emb"] = pred_act_emb
    return output


def action_prediction_compute(model, sigreg, pixels, actions, *, history_size=3,
                               num_preds=1, aux_weight=0.09):
    """Compute next-embedding MSE plus inverse-action MSE."""
    output = _action_prediction_terms(model, pixels, actions, history_size, num_preds)
    output["loss"] = output["pred_loss"] + aux_weight * output["action_loss"]
    return output


def combined_compute(model, sigreg, pixels, actions, *, history_size=3,
                     num_preds=1, aux_weight=0.09, sigreg_weight=0.09):
    """Use SIGReg on all encoded frames and inverse-action MSE on predicted deltas."""
    if sigreg is None:
        raise ValueError("The combined objective requires a SIGReg module")
    output = _action_prediction_terms(model, pixels, actions, history_size, num_preds)
    output["sigreg_loss"] = sigreg(output["emb"].transpose(0, 1))
    output["loss"] = (output["pred_loss"] + sigreg_weight * output["sigreg_loss"]
                      + aux_weight * output["action_loss"])
    return output


def build_training_compute(objective, *, history_size=3, num_preds=1,
                           aux_weight=0.09, sigreg_weight=0.09, compile_mode=None,
                           backend="inductor", fullgraph=False, dynamic=None):
    """Return ``compute(model, sigreg, pixels, actions)`` for one fixed objective.

    ``compile_mode=None`` uses eager execution. Explicit ``'default'`` and
    ``'reduce-overhead'`` call torch.compile on the function, leaving the model
    object, parameter registration, optimizer, and state_dict names unchanged.
    Compile warmup and any final partial-batch specialization are the caller's
    responsibility. Inputs must already be on their target device; invoke this
    function within the same autocast context used for the eager baseline.
    """
    if history_size < 1 or num_preds < 1:
        raise ValueError("History size and prediction offset must be positive")
    if objective == "sigreg":
        function = sigreg_compute
    elif objective == "mse_only":
        function = mse_only_compute
    elif objective in ("action_prediction", "combined"):
        if num_preds != 1:
            raise ValueError("Action prediction requires num_preds=1")
        function = (action_prediction_compute if objective == "action_prediction"
                    else partial(combined_compute, sigreg_weight=float(sigreg_weight)))
    else:
        raise ValueError(f"Unknown training objective: {objective}")
    compute = partial(function, history_size=int(history_size),
                      num_preds=int(num_preds), aux_weight=float(aux_weight))
    if compile_mode is None:
        return compute
    if compile_mode not in ("default", "reduce-overhead"):
        raise ValueError("compile_mode must be None, 'default', or 'reduce-overhead'")
    return torch.compile(compute, backend=backend, mode=compile_mode,
                         fullgraph=fullgraph, dynamic=dynamic)
