"""
CLI entry point: argument parsing, signal handling, and the main() that
dispatches to create/chat/gen or spins up the training worker + TUI threads.
"""
import os
import signal
import threading
import time
import queue
import argparse

from tiny_llm import logging_utils
from tiny_llm.logging_utils import append_error, append_traceback
from tiny_llm.arch import PRESETS
from tiny_llm.commands import cmd_create, cmd_chat, cmd_gen
from tiny_llm.training import training_worker
from tiny_llm.tui import tui_runner, console


# signal handling
def _handle_interrupt(signum, frame):
    logging_utils.shutdown_requested = True

# signals registered in main() to avoid re-running on windows worker re-import


def parse_args():
    p = argparse.ArgumentParser(prog="tiny_llm_trainer")
    sub = p.add_subparsers(dest="cmd", required=True)

    # create
    c = sub.add_parser("create", help="initialise a new checkpoint")
    c.add_argument("--params",     type=str, default=None,
                   help="param target e.g. 500k, 7m, 1b, 7b")
    c.add_argument("--preset",     choices=PRESETS.keys(), default=None)
    c.add_argument("--model",      type=str, default=None)
    c.add_argument("--block-size", dest="block_size", type=int, default=None,
                   help="context length override (auto-chosen based on scale if omitted)")
    c.add_argument("--fp32",       dest="fp32",       action="store_true",
                   help="init weights in fp32 instead of the default fp16")
    c.add_argument("--layers",     dest="layers",     type=int, default=None,
                   help="force a specific number of transformer layers (embed_dim auto-fitted)")

    # train
    t = sub.add_parser("train", help="train with live TUI")
    t.add_argument("--model",             type=str,   required=True)
    t.add_argument("--data-dir",          dest="data_dir", type=str, default="data")
    t.add_argument("--epochs",            type=int,   default=10)
    t.add_argument("--batch-size",        dest="batch_size", type=int, default=32)
    t.add_argument("--lr",                type=float, default=3e-3)
    t.add_argument("--warmup-frac",       dest="warmup_frac", type=float, default=0.05)
    t.add_argument("--no-cosine-lr",      dest="no_cosine_lr", action="store_true")
    t.add_argument("--weight-decay",      dest="weight_decay", type=float, default=0.1)
    t.add_argument("--accum",             type=int,   default=1,  help="gradient accumulation steps")
    t.add_argument("--max-grad-norm",     dest="max_grad_norm", type=float, default=1.0)
    t.add_argument("--fp16",              action="store_true")
    t.add_argument("--compile",           action="store_true")
    t.add_argument("--save-every",        dest="save_every",   type=int, default=500)
    t.add_argument("--sample-every",      dest="sample_every", type=int, default=300)
    t.add_argument("--print-every",       dest="print_every",  type=int, default=50)
    t.add_argument("--param-stats-every", dest="param_stats_every", type=int, default=20)
    t.add_argument("--gen-len",           dest="gen_len",      type=int, default=80)
    t.add_argument("--sample-temp",       dest="sample_temp",  type=float, default=1.0)
    t.add_argument("--sample-top-k",      dest="sample_top_k", type=int,   default=40)
    t.add_argument("--gpu-data",          dest="gpu_data",     action="store_true",
                   help="pin dataset tensor to VRAM")
    t.add_argument("--gpu-data-max-mb",   dest="gpu_data_max_mb", type=float, default=512.0)
    t.add_argument("--strict-clean",      dest="strict_clean", action="store_true")
    # hf dataset args
    t.add_argument("--hf-dataset",    dest="hf_dataset",  type=str, default=None,
                   help="HuggingFace dataset id, e.g. tatsu-lab/alpaca")
    t.add_argument("--hf-split",      dest="hf_split",    type=str, default="train",
                   help="Dataset split to use (default: train)")
    t.add_argument("--hf-subset",     dest="hf_subset",   type=str, default=None,
                   help="HuggingFace dataset config/subset name, e.g. 'wikitext-2-raw-v1'")
    t.add_argument("--hf-field",      dest="hf_field",    type=str, default=None,
                   help="Column name to use, or 'inst_col,input_col,out_col' for instruct. "
                        "Auto-detected if omitted.")
    t.add_argument("--hf-max-rows",   dest="hf_max_rows", type=int, default=None,
                   help="Cap dataset at N rows (useful for quick tests)")
    t.add_argument("--auto-lr",       dest="auto_lr",     action="store_true",
                   help="Auto-compute optimal peak LR based on model size and batch (ignores --lr)")
    t.add_argument("--no-tui",            dest="no_tui",       action="store_true")

    # chat
    ch = sub.add_parser("chat")
    ch.add_argument("--model",   type=str, required=True)
    ch.add_argument("--gen-len", dest="gen_len", type=int,   default=200)
    ch.add_argument("--temp",    type=float, default=1.0)
    ch.add_argument("--top-k",   dest="top_k", type=int, default=40)

    # gen
    g2 = sub.add_parser("gen")
    g2.add_argument("--model",   type=str, required=True)
    g2.add_argument("--prompt",  type=str, default="The")
    g2.add_argument("--gen-len", dest="gen_len", type=int,   default=200)
    g2.add_argument("--temp",    type=float, default=0.8)
    g2.add_argument("--top-k",   dest="top_k", type=int, default=40)

    return p.parse_args()


def main():
    # register here so windows multiprocessing re-imports dont double-fire
    signal.signal(signal.SIGINT,  _handle_interrupt)
    signal.signal(signal.SIGTERM, _handle_interrupt)
    args = parse_args()

    if args.cmd == "create":
        cmd_create(args); return
    if args.cmd == "chat":
        cmd_chat(args);   return
    if args.cmd == "gen":
        cmd_gen(args);    return

    # train
    if not os.path.exists(args.model):
        console.print(f"[red]model {args.model} not found — run 'create' first[/]")
        return

    exc_q = queue.Queue()

    if args.no_tui:
        logging_utils.TUI_ACTIVE = False
        try:
            training_worker(args, args.model, exc_q)
        finally:
            try:
                tb = exc_q.get_nowait()
                append_error("worker crash — see tiny_llm_error.log")
                append_traceback(tb)
            except queue.Empty: pass
        return

    logging_utils.TUI_ACTIVE = True
    stop_evt   = threading.Event()

    worker = threading.Thread(target=training_worker,
                               args=(args, args.model, exc_q), daemon=True)
    tui    = threading.Thread(target=tui_runner,
                               args=(stop_evt, exc_q), daemon=True)
    worker.start()
    tui.start()

    try:
        while worker.is_alive():
            try:
                tb = exc_q.get_nowait()
                append_error("worker crash — see tiny_llm_error.log")
                append_traceback(tb)
                break
            except queue.Empty: pass
            time.sleep(0.5)
    except KeyboardInterrupt:
        logging_utils.shutdown_requested = True
    finally:
        stop_evt.set()
        tui.join(timeout=5.0)
        worker.join(timeout=10.0)
        logging_utils.TUI_ACTIVE = False
