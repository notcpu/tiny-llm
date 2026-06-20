"""
Thread-safe shared state, read by the TUI rendering thread and written by
the training worker thread. Also holds the param-diagnostics snapshot
helper, since its output (rows) is what populates state.param_stats
"""
import threading
from collections import deque
from datetime import datetime

from tiny_llm.logging_utils import append_debug


def _now():
    return datetime.now().strftime("%H:%M:%S")


class SharedState:
    def __init__(self):
        self.lock          = threading.Lock()
        self.epoch         = 0
        self.iter_in_epoch = 0
        self.iters_per_epoch = 0
        self.global_step   = 0
        self.loss          = 0.0
        self.loss_history  = []
        self.logs          = deque(maxlen=2000)
        self.sample_text   = "no sample yet"
        self.param_stats   = []
        self.lr            = 0.0
        self.sps           = 0.0
        self.tps           = 0.0
        self.status        = "idle"
        self.cache_lens    = []
        self.gpu_util      = 0
        self.cpu_util      = 0.0
        # timing / progress
        self.train_start   = None   # wall time training began
        self.total_steps   = 0      # total optimizer updates planned
        self.best_loss     = float("inf")
        self.loss_smoothed = 0.0    # EMA loss
        self.grad_norm     = 0.0
        self.total_epochs  = 0
        self.total_tokens  = 0      # tokens seen since start

    def log(self, msg):
        s = f"[{_now()}] {msg}"
        try:
            with self.lock:
                self.logs.append(s)
        except Exception: pass
        append_debug(msg)


# Module-level singleton imported as `from tiny_llm.state import state`
# everywhere that needs to read/write shared training state.
state = SharedState()


def snapshot_params(model, max_items=12):
    rows = []
    for n, p in model.named_parameters():
        try: pn = float(p.data.norm().cpu())
        except Exception: pn = float("nan")
        try: gn = float(p.grad.norm().cpu()) if p.grad is not None else 0.0
        except Exception: gn = float("nan")
        try:
            flat = p.data.view(-1)[:8].cpu().numpy()
            snip = ",".join(f"{float(v):.3f}" for v in flat)
            if p.data.numel() > 8: snip += ",…"
        except Exception: snip = "n/a"
        rows.append((n, pn, gn, snip))
    return sorted(rows, key=lambda r: r[2], reverse=True)[:max_items]
