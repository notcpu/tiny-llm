"""
Checkpoint save/load, plus config normalisation so checkpoints from older
versions of the trainer (different key names) still load correctly.
"""
import os

import torch

from tiny_llm.logging_utils import append_debug
from tiny_llm.tokenizer import CharTokenizer


def unwrap(model):
    return getattr(model, "_orig_mod", model)


def save_ckpt(path, model, optimizer, cfg, epoch, step, loss_history):
    m = unwrap(model)
    state_dict = None
    if m is not None:
        state_dict = m.state_dict()
        # cast back to storage dtype so the checkpoint stays compact
        store = cfg.get("dtype", "fp32") if cfg else "fp32"
        if store == "fp16":
            state_dict = {k: v.half() if v.is_floating_point() else v
                          for k, v in state_dict.items()}
    torch.save({
        "model_state":     state_dict,
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "cfg":             cfg,
        "epoch":           epoch,
        "global_step":     step,
        "loss_history":    loss_history,
    }, path)
    append_debug(f"checkpoint saved → {path} (weights as {cfg.get('dtype','fp32') if cfg else 'fp32'})")


def load_ckpt(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:    return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(path, map_location="cpu")


def _resolve_cfg(raw_cfg: dict) -> dict:
    """
    Normalise checkpoint cfg regardless of which trainer version created it.
    Old trainer used:  n_embd, (no block_size), (no ff_mult)
    New trainer uses:  embed_dim, block_size, ff_mult
    """
    cfg = dict(raw_cfg)  # don't mutate original
    # embed_dim  ←  n_embd (old key)
    if "embed_dim" not in cfg and "n_embd" in cfg:
        cfg["embed_dim"] = cfg["n_embd"]
    # block_size fallback
    if "block_size" not in cfg:
        cfg["block_size"] = cfg.get("seq_len", 128)
    # n_layer fallback — try common old key names
    if "n_layer" not in cfg:
        cfg["n_layer"] = cfg.get("num_layers", cfg.get("n_layers", cfg.get("num_hidden_layers", 6)))
    # n_head fallback
    if "n_head" not in cfg:
        embed_dim = cfg.get("embed_dim", 256)
        # pick largest power-of-2 head count that divides embed_dim, max 16
        for nh in [16, 8, 4, 2, 1]:
            if embed_dim % nh == 0:
                cfg["n_head"] = nh
                break
    # ff_mult fallback
    if "ff_mult" not in cfg:
        cfg["ff_mult"] = 4
    # dropout fallback
    if "dropout" not in cfg:
        cfg["dropout"] = 0.1
    # vocab_size fallback (infer from embedding weight if missing)
    if "vocab_size" not in cfg:
        cfg["vocab_size"] = CharTokenizer().vocab_size
    # dtype fallback — old checkpoints didn't store this, assume fp32
    if "dtype" not in cfg:
        cfg["dtype"] = "fp32"
    return cfg
