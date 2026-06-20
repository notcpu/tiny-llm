"""
Device selection + AMP (automatic mixed precision) helpers.
"""
import torch

from tiny_llm.logging_utils import append_debug

try:
    from torch.amp import GradScaler as _GradScaler
except ImportError:
    _GradScaler = None


def best_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        append_debug(f"CUDA device: {name}")
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        append_debug("Apple MPS device")
        return torch.device("mps")
    append_debug("CPU device")
    return torch.device("cpu")


def autocast_ctx(device, enabled=True):
    if not enabled:
        return torch.amp.autocast(device_type="cpu", enabled=False)
    dt = torch.device(device).type
    if dt == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return torch.amp.autocast(device_type="cpu", enabled=False)


def make_scaler(device, enabled=True):
    amp_on = bool(enabled and torch.device(device).type == "cuda")
    if _GradScaler is not None:
        try:
            return _GradScaler("cuda", enabled=amp_on)
        except TypeError:
            return _GradScaler(enabled=amp_on)
    return torch.cuda.amp.GradScaler(enabled=amp_on)
