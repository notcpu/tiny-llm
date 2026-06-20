"""
Learning-rate schedule helpers: warmup + cosine decay, plus a heuristic
for auto-computing a sensible peak LR from model/batch scale.
"""
import math


def get_lr(step, total_steps, lr_max, warmup_frac=0.05, cosine=True):
    if not cosine:
        return lr_max
    total_steps  = max(1, int(total_steps))
    warmup_steps = max(1, int(total_steps * max(0.0, warmup_frac)))
    if step < warmup_steps:
        return lr_max * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_max * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def optimal_lr(n_params: int, batch_size: int, block_size: int) -> float:
    """
    Heuristic optimal peak LR based on model scale + effective batch tokens.
    Derived from empirical scaling observations (GPT-3, Chinchilla, nanoGPT):
      - Base LR ~ 6e-4 for ~1B params, scales as N^-0.3
      - Then linearly scaled up by sqrt(effective_batch / reference_batch)
      - Clipped to [1e-5, 1e-2] for safety
    """
    ref_batch = 512 * 128           # nanoGPT reference: 512 seqs * 128 ctx
    eff_batch = batch_size * block_size
    # base LR at reference scale (1B params)
    base_lr   = 6e-4
    # scale down for larger models, scale up for smaller ones
    n_ref     = 1_000_000_000
    n_params  = max(n_params, 1_000)
    scale_n   = (n_ref / n_params) ** 0.3
    # square-root batch scaling
    scale_b   = math.sqrt(eff_batch / ref_batch)
    lr        = base_lr * scale_n * scale_b
    return float(max(1e-5, min(1e-2, lr)))
