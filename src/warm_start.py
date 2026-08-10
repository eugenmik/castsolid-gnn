"""Warm-start loader: build a v4 (node_in=15, FiLM) SolidGNN from an SP1
(node_in=13) checkpoint with zero learned knowledge lost.

The v4 model adds two trailing per-node input columns (boundary_extraction,
exothermic_input) and FiLM conditioning (per-block gamma/beta from a material
g-vector). Both extensions are zero-init: the new input columns are zeroed in
the node-encoder's first Linear, and FiLM's last layer is zero-init (see
`src.model_gnn.FiLM`) so gamma=0, beta=0 => identity. Together this means the
freshly warm-started v4 model reproduces the SP1 model's output exactly (see
`sanity_gate` / tests/test_warm_start.py), before any FT2b training happens.

torch 2.3 note: `load_state_dict(..., strict=False)` does NOT silently skip
shape-mismatched tensors -- it raises `RuntimeError: size mismatch` even with
strict=False (verified empirically). So we pre-filter the SP1 state_dict to
keys whose shape matches the v4 model before calling load_state_dict, and
splice the one shape-mismatched tensor (the node_encoder's first Linear
weight) by hand.
"""
import logging

import torch

from src.model_gnn import SolidGNN

logger = logging.getLogger(__name__)

_ENC_W = "node_encoder.0.0.weight"
_ENC_B = "node_encoder.0.0.bias"


def splice_node_encoder(v4, sp1_sd, n_old=13):
    """Copy SP1's node-encoder first-Linear weight into v4's leading n_old
    input columns; zero the trailing (new) columns. Bias copied verbatim.

    Args:
        v4: SolidGNN instance (node_in > n_old).
        sp1_sd: SP1 state_dict (or any mapping with the same keys).
        n_old: number of leading input columns coming from SP1 (13).
    """
    with torch.no_grad():
        w_new = v4.node_encoder[0][0].weight
        w_new[:, :n_old] = sp1_sd[_ENC_W]
        w_new[:, n_old:] = 0.0
        v4.node_encoder[0][0].bias.copy_(sp1_sd[_ENC_B])


def build_warm_started_v4(sp1_ckpt_path, hidden=128, n_blocks=12, g_dim=5,
                           node_in_new=15, n_old=13):
    """Build a film=True, two_head=True SolidGNN(node_in=node_in_new) warm-
    started from an SP1 checkpoint (node_in=n_old).

    All shape-matching parameters (edge_encoder, processor blocks, decoder,
    scale_head) are loaded verbatim from SP1. The node_encoder's first Linear
    is spliced (SP1 columns 0:n_old, zeros n_old:). FiLM stays at its
    zero-init (identity) construction.

    Args:
        sp1_ckpt_path: path to the SP1 `{model, norm, cfg}` checkpoint.
        hidden, n_blocks, g_dim: v4 model hyperparameters. If the checkpoint's
            `cfg` carries `hidden`/`n_blocks`/`edge_in`/`two_head`, those take
            precedence (the warm-started model must match SP1's architecture
            to load its weights) -- the arguments here are fallback defaults
            for checkpoints without a `cfg`.
        node_in_new: v4 input feature count (13 v3 features + 2 new BC cols).
        n_old: SP1 input feature count.

    Returns:
        (model, sp1_norm): the warm-started v4 SolidGNN (train mode, as
        constructed) and the SP1 checkpoint's `norm` dict (feature/target
        normalization stats), so callers can decide how to reconcile it with
        v4's own (different) normalization -- not this function's concern.
    """
    ckpt = torch.load(sp1_ckpt_path, map_location="cpu", weights_only=False)
    sp1_sd = ckpt["model"]
    cfg = ckpt.get("cfg", {})
    edge_in = cfg.get("edge_in", 4)
    two_head = cfg.get("two_head", True)
    hidden = cfg.get("hidden", hidden)
    n_blocks = cfg.get("n_blocks", n_blocks)

    v4 = SolidGNN(node_in=node_in_new, edge_in=edge_in, hidden=hidden,
                  n_blocks=n_blocks, two_head=two_head, film=True, g_dim=g_dim)

    v4_sd = v4.state_dict()
    matching = {k: v for k, v in sp1_sd.items()
                if k in v4_sd and v4_sd[k].shape == v.shape}
    load_result = v4.load_state_dict(matching, strict=False)

    # Everything skipped by the shape filter above must be exactly the
    # spliced input-encoder tensors (handled below by splice_node_encoder).
    skipped_sp1_keys = [k for k in sp1_sd if k not in matching]
    unexpected_skips = set(skipped_sp1_keys) - {_ENC_W, _ENC_B}
    if unexpected_skips:
        raise RuntimeError(
            f"warm_start: unexpected SP1 keys skipped by shape filter "
            f"(expected only the node_encoder input layer): {sorted(unexpected_skips)}"
        )

    # Everything load_state_dict reports "missing" in v4 must be exactly the
    # spliced input-encoder tensors plus the new FiLM params (no SP1 counterpart).
    unexpected_missing = [
        k for k in load_result.missing_keys
        if k not in (_ENC_W, _ENC_B) and not k.startswith("films.")
    ]
    if unexpected_missing:
        raise RuntimeError(
            f"warm_start: unexpected missing keys after load "
            f"(expected only node_encoder input layer + films.*): {unexpected_missing}"
        )
    if load_result.unexpected_keys:
        raise RuntimeError(
            f"warm_start: unexpected extra keys in SP1 checkpoint: "
            f"{load_result.unexpected_keys}"
        )

    logger.info("warm_start: SP1 keys spliced by hand (input encoder): %s", skipped_sp1_keys)
    logger.info("warm_start: new v4-only keys left at zero-init (FiLM): %s",
                [k for k in load_result.missing_keys if k.startswith("films.")])

    splice_node_encoder(v4, sp1_sd, n_old=n_old)

    return v4, ckpt["norm"]


def build_or_load_v4(ckpt_path, hidden=128, n_blocks=12, g_dim=5,
                      node_in_new=15, n_old=13):
    """Warm-start entry point used by FT2b's staged-training driver, which
    chains checkpoints across stages (stage 2 warm-starts from stage 1's own
    output, stage 3 from stage 2's) -- NOT always from the original SP1
    checkpoint.

    Dispatches on the checkpoint's own `cfg`:
      - `cfg["node_in"] == node_in_new and cfg["film"]` => this is ALREADY a
        v4 FT2b checkpoint (e.g. a previous stage's `best_gnn.pt`): same
        architecture end to end, so it is loaded directly with a plain
        `strict=True` `load_state_dict` -- no splicing, since there is no
        column-count mismatch to bridge.
      - otherwise => this is an SP1-shaped checkpoint (`node_in=n_old`, no
        FiLM), so it is warm-started via `build_warm_started_v4` exactly as
        before (the node-encoder splice + zero-init FiLM path).

    `build_warm_started_v4` itself is intentionally left untouched -- it always
    assumes an SP1-shaped source and unconditionally splices, which would
    corrupt (or crash on, due to the column-count mismatch) an
    already-15-column source encoder. This function is the one place that
    decides which path applies.

    Returns:
        (model, norm): same contract as `build_warm_started_v4` -- the
        `norm` is the SOURCE checkpoint's own `norm` dict, for callers to
        reconcile against the current training data's normalization (as
        `train()` already does for the build_warm_started_v4 path).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {})
    if cfg.get("node_in") == node_in_new and cfg.get("film"):
        edge_in = cfg.get("edge_in", 4)
        two_head = cfg.get("two_head", True)
        hidden = cfg.get("hidden", hidden)
        n_blocks = cfg.get("n_blocks", n_blocks)
        model = SolidGNN(node_in=node_in_new, edge_in=edge_in, hidden=hidden,
                          n_blocks=n_blocks, two_head=two_head, film=True, g_dim=g_dim)
        model.load_state_dict(ckpt["model"], strict=True)
        logger.info("build_or_load_v4: %s is already a v4 FT2b checkpoint "
                    "(node_in=%d, film=True) -- loaded directly, no splice.",
                    ckpt_path, node_in_new)
        return model, ckpt["norm"]
    return build_warm_started_v4(ckpt_path, hidden=hidden, n_blocks=n_blocks,
                                  g_dim=g_dim, node_in_new=node_in_new, n_old=n_old)


def build_or_load_v5(ckpt_path, hidden=128, n_blocks=12, g_dim=5,
                      node_in_new=20, n_old=15):
    """bc-reach warm-start entry point: rebuild the node encoder for the 5 new
    BC-reach columns (node_in 15 -> 20), or load an already-v5 checkpoint
    directly.

    `build_or_load_v4`'s dispatch logic (splice-from-a-smaller-node_in-source
    vs. direct-load-an-already-this-shape checkpoint, `torch 2.3` shape-filter
    pre-check, `load_state_dict(..., strict=False)` caveat) is already fully
    generic in its `node_in_new`/`n_old` arguments -- it makes no assumption
    that the source is specifically SP1-shaped, only that its node_encoder's
    first-Linear weight is the sole shape mismatch. This function is a thin,
    v5-flavored entry point over it: given the production `ckpt_ep021.pt`
    (node_in=15, film=True -- itself already a fine-tuned FT2b/FT3 checkpoint,
    NOT a fresh warm-start, so its FiLM/blocks/decoder/scale_head weights are
    real learned values, not zero-init), it:
      - copies `weight[:, :15]` (the node encoder's first Linear) verbatim,
      - zero-inits the 5 new trailing columns (`weight[:, 15:20]`),
      - copies the bias verbatim (unaffected by the input-width change --
        Linear bias only depends on `hidden`, never on `node_in`),
      - strict-copies everything else (edge_encoder, processor blocks, FiLM,
        decoder, scale_head) verbatim, since `hidden`/`n_blocks`/`g_dim` are
        unchanged from v4 to v5.
    Dispatches to a direct `strict=True` load instead (no splice) when
    `ckpt_path`'s own `cfg` already says `node_in==node_in_new and film` --
    e.g. a later bc-reach stage chaining off an earlier bc-reach checkpoint.

    Returns:
        (model, norm): same contract as `build_or_load_v4` -- `norm` is the
        SOURCE checkpoint's own `norm` dict (15-wide for a genuine v4 source),
        for callers to reconcile against the CURRENT training data's own
        (fresh, 20-wide) normalization -- `src.train_ft2b.run_bcr` does NOT
        forward this as a `norm_override` (unlike FT3's frozen-norm
        fine-tune): bc-reach's v5 feature cache gets its own fresh norm_stats
        (Task 2, `build_v5`), so `train()` standardizes against that cache's
        own stats, not `ckpt_ep021`'s 15-wide ones.
    """
    return build_or_load_v4(ckpt_path, hidden=hidden, n_blocks=n_blocks, g_dim=g_dim,
                             node_in_new=node_in_new, n_old=n_old)


def sanity_gate(model_v4, sp1_model, sample):
    """Run both models (eval, no_grad) on the same reference sample; return
    the max abs diff between their outputs (y_nodal and y_global combined).

    `sample` is a dict with keys:
        x_v4:       (N, node_in_v4) input features for the v4 model.
        x_sp1:      (N, node_in_sp1) input features for the SP1 model.
        edge_index: (2, E) shared graph connectivity.
        edge_attr:  (E, edge_in) shared edge features.
        batch:      (N,) graph-membership vector, shared.
        g:          (B, g_dim) material g-vector for v4's FiLM, optional.

    This function makes no assumption about how x_v4/x_sp1 relate (e.g.
    whether x_v4 is x_sp1 zero-padded, or real v4-normalized features) --
    that reconciliation is the caller's job.
    """
    model_v4.eval()
    sp1_model.eval()
    with torch.no_grad():
        yn_sp1, yg_sp1 = sp1_model(sample["x_sp1"], sample["edge_index"],
                                    sample["edge_attr"], sample["batch"])
        yn_v4, yg_v4 = model_v4(sample["x_v4"], sample["edge_index"],
                                 sample["edge_attr"], sample["batch"],
                                 g=sample.get("g"))
    node_diff = (yn_v4 - yn_sp1).abs().max().item()
    global_diff = (yg_v4 - yg_sp1).abs().max().item()
    return max(node_diff, global_diff)
