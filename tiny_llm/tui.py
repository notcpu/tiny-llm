"""
Rich-based live TUI: header (progress/ETA), logs panel, live sample panel,
param-diagnostics + system stats panel, and a footer with a loss sparkline.
"""
import time
import queue

import torch
from rich.live    import Live
from rich.panel   import Panel
from rich.table   import Table
from rich.text    import Text
from rich.console import Console
from rich.layout  import Layout

try:
    import psutil
except ImportError:
    psutil = None

from tiny_llm import logging_utils
from tiny_llm.logging_utils import append_error, append_traceback
from tiny_llm.state import state

console = Console()


def _sparkline(values, width=80):
    if not values: return "no data"
    arr    = list(values[-width:])
    lo, hi = min(arr), max(arr)
    span   = hi - lo if hi != lo else 1.0
    blocks = "▁▂▃▄▅▆▇█"
    return "".join(blocks[int((v-lo)/span*(len(blocks)-1))] for v in arr)


def _make_layout():
    lay = Layout()
    lay.split_column(
        Layout(name="header", size=4),
        Layout(name="main",   ratio=1),
        Layout(name="footer", size=6),
    )
    lay["main"].split_row(
        Layout(name="left",   ratio=30),
        Layout(name="center", ratio=45),
        Layout(name="right",  ratio=25),
    )
    return lay


def _fmt_duration(seconds):
    """Format seconds into human-readable elapsed/ETA string."""
    seconds = max(0, int(seconds))
    h, rem  = divmod(seconds, 3600)
    m, s    = divmod(rem, 60)
    if h:   return f"{h}h {m:02d}m {s:02d}s"
    if m:   return f"{m}m {s:02d}s"
    return  f"{s}s"


def _progress_bar(done, total, width=28):
    """Unicode block progress bar."""
    frac   = min(1.0, done / max(1, total))
    filled = int(frac * width)
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {frac*100:.1f}%"


def _render_header():
    with state.lock:
        step        = state.global_step
        epoch       = state.epoch
        total_ep    = state.total_epochs
        it          = state.iter_in_epoch
        iters       = state.iters_per_epoch
        status      = state.status
        total_steps = state.total_steps
        t_start     = state.train_start
        sps         = state.sps

    now     = time.time()
    elapsed = now - t_start if t_start else 0.0
    eta     = (total_steps - step) / sps if sps > 0 and step < total_steps else 0.0
    step_bar = _progress_bar(step, max(1, total_steps))
    ep_bar   = _progress_bar(it,   max(1, iters))

    g = Table.grid(expand=True, padding=(0,1))
    g.add_column(ratio=5)
    g.add_column(ratio=5, justify="right")
    g.add_row(
        f"[bold green]tiny-llm[/]  [yellow]{status}[/]  "
        f"epoch [bold cyan]{epoch}[/]/[cyan]{total_ep}[/]  "
        f"step [bold]{step}[/]/[dim]{total_steps}[/]",
        f"[bold]elapsed[/] [green]{_fmt_duration(elapsed)}[/]  "
        f"[bold]ETA[/] [yellow]{_fmt_duration(eta)}[/]",
    )
    g.add_row(f"overall {step_bar}", f"epoch   {ep_bar}")
    return Panel(g, style="white on black", padding=(0,1))


def _render_logs():
    # Rich's Live layout reserves ~6 rows for header/footer borders/padding.
    # We compute how many log lines fit based on terminal height so the panel
    # always shows the latest logs without overflowing or wasting space.
    try:
        term_h = console.size.height
    except Exception:
        term_h = 40
    # header=4, footer=6, panel borders+padding ~4, safety margin 2
    available = max(4, term_h - 4 - 6 - 6)
    with state.lock:
        lines = list(state.logs)
    # always show the most recent `available` lines  → true auto-scroll
    visible = lines[-available:]
    text = Text("\n".join(visible))
    return Panel(text, title=f"Logs  [{len(lines)} total]", border_style="cyan", padding=(0,1))


def _render_sample():
    with state.lock:
        sample = state.sample_text
    return Panel(Text(sample), title="Live Sample", border_style="magenta", padding=(1,1))


def _render_right():
    # param table
    tbl = Table(expand=True, show_lines=False)
    tbl.add_column("param",   style="bold", no_wrap=True)
    tbl.add_column("‖p‖",    justify="right")
    tbl.add_column("‖g‖",    justify="right", style="yellow")
    tbl.add_column("vals",    overflow="fold", style="dim")
    with state.lock:
        rows = list(state.param_stats)[:8]
    if not rows:
        tbl.add_row("-","-","-","-")
    else:
        for n, pn, gn, snip in rows:
            tbl.add_row(n, f"{pn:.3f}", f"{gn:.3f}", snip)

    # system and training stats
    with state.lock:
        t_start     = state.train_start
        total_steps = state.total_steps
        step_now    = state.global_step
        sps         = state.sps
        tps         = state.tps
        lr          = state.lr
        loss        = state.loss
        smoothed    = state.loss_smoothed
        best_loss   = state.best_loss
        grad_norm   = state.grad_norm
        total_tokens= state.total_tokens

    elapsed = time.time() - t_start if t_start else 0.0
    eta     = (total_steps - step_now) / sps if sps > 0 and step_now < total_steps else 0.0
    best_tag = " ★" if abs(loss - best_loss) < 1e-6 else ""

    sys_lines = ["── training ─────────────────"]
    sys_lines.append(f"loss     {loss:.4f}{best_tag}")
    sys_lines.append(f"smooth   {smoothed:.4f}")
    sys_lines.append(f"best     {best_loss:.4f}")
    sys_lines.append(f"‖grad‖   {grad_norm:.4f}")
    sys_lines.append(f"lr       {lr:.2e}")
    sys_lines.append(f"tok/s    {int(tps):,}")
    sys_lines.append(f"tokens   {_fmt_tokens(total_tokens)}")
    sys_lines.append(f"elapsed  {_fmt_duration(elapsed)}")
    sys_lines.append(f"ETA      {_fmt_duration(eta)}")
    sys_lines.append(f"steps/s  {sps:.1f}")
    sys_lines.append("── hardware ─────────────────")
    if torch.cuda.is_available():
        try:
            alloc      = torch.cuda.memory_allocated() / (1024**2)
            res        = torch.cuda.memory_reserved()  / (1024**2)
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            used_pct   = 100 * alloc / max(1, total_vram)
            vbar = "█" * int(used_pct / 5) + "░" * (20 - int(used_pct / 5))
            sys_lines.append(f"GPU   {torch.cuda.get_device_name(0)[:22]}")
            sys_lines.append(f"VRAM  {alloc:.0f}/{total_vram:.0f}MB ({used_pct:.0f}%)")
            sys_lines.append(f"[{vbar}]")
            sys_lines.append(f"res   {res:.0f}MB reserved")
        except Exception: pass
    if psutil:
        try:
            with state.lock: cpu_pct = state.cpu_util
            vm       = psutil.virtual_memory()
            ram_used = vm.used  / (1024**3)
            ram_tot  = vm.total / (1024**3)
            cpu_bar  = "█" * int(cpu_pct / 5) + "░" * (20 - int(cpu_pct / 5))
            sys_lines.append(f"CPU   [{cpu_bar}] {cpu_pct:.0f}%")
            sys_lines.append(f"RAM   {ram_used:.1f}/{ram_tot:.1f}GB")
        except Exception: pass

    right = Layout()
    right.split_column(
        Layout(Panel(tbl, title="param diag", border_style="red",   padding=(0,1)), size=14),
        Layout(Panel(Text("\n".join(sys_lines)), title="system", border_style="green", padding=(0,1)), ratio=1),
    )
    return right


def _fmt_tokens(n):
    """Human-readable token count."""
    if n >= 1_000_000_000: return f"{n/1e9:.2f}B"
    if n >= 1_000_000:     return f"{n/1e6:.2f}M"
    if n >= 1_000:         return f"{n/1e3:.1f}K"
    return str(n)


def _render_footer():
    with state.lock:
        hist        = list(state.loss_history[-120:])
        last_loss   = state.loss
        smoothed    = state.loss_smoothed
        best_loss   = state.best_loss
        t_start     = state.train_start
        total_steps = state.total_steps
        step        = state.global_step
        total_ep    = state.total_epochs
        epoch       = state.epoch
        total_tok   = state.total_tokens

    spark   = _sparkline(hist)
    elapsed = time.time() - t_start if t_start else 0.0
    pct     = 100 * step / max(1, total_steps)

    txt = Text()
    txt.append(f"loss  curr={last_loss:.4f}  smooth={smoothed:.4f}  best={best_loss:.4f}  "
               f"│  {pct:.1f}% done  │  tokens seen={_fmt_tokens(total_tok)}  "
               f"│  elapsed={_fmt_duration(elapsed)}\n", style="bold green")
    txt.append(spark + "\n", style="white")
    txt.append("ctrl+c → graceful save & exit   •   error log: tiny_llm_error.log",
               style="dim yellow")
    return Panel(txt, title="loss curve ▁▂▃▄▅▆▇█", border_style="green")


def tui_runner(stop_event, exception_queue):
    lay = _make_layout()
    with Live(lay, refresh_per_second=6, screen=True):
        while not stop_event.is_set() and not logging_utils.shutdown_requested:
            lay["header"].update(_render_header())
            lay["left"].update(_render_logs())
            lay["center"].update(_render_sample())
            lay["right"].update(_render_right())
            lay["footer"].update(_render_footer())
            if torch.cuda.is_available():
                try:
                    with state.lock:
                        state.gpu_util = 0   # pynvml hook point
                except Exception: pass
            if psutil:
                with state.lock:
                    state.cpu_util = psutil.cpu_percent(interval=None)
            try:
                tb = exception_queue.get_nowait()
                append_error("worker crash — see tiny_llm_error.log")
                append_traceback(tb)
            except queue.Empty: pass
            time.sleep(0.15)
