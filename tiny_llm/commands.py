"""
The three "simple" subcommands: create (initialise a checkpoint), chat
(interactive REPL), and gen (one-shot generation). Training lives in
training.py since it's the one that needs the TUI machinery.
"""
import torch
from rich.console import Console
from rich.panel import Panel

from tiny_llm.tokenizer import CharTokenizer
from tiny_llm.arch import choose_arch, parse_params, PRESETS
from tiny_llm.model import TinyLLM
from tiny_llm.checkpoint import load_ckpt, _resolve_cfg
from tiny_llm.device_utils import best_device

console = Console()


def _auto_block_size(target):
    """Pick a sensible context length based on model scale."""
    if target <    10_000_000: return 128
    if target <   100_000_000: return 512
    if target <   500_000_000: return 1024
    if target < 5_000_000_000: return 2048
    return 4096


def cmd_create(args):
    param_str = args.params or PRESETS.get(args.preset)
    if not param_str:
        raise ValueError("pass --params (e.g. 3000k, 1b) or --preset nano|micro|small|medium|large|xl")
    tok        = CharTokenizer()
    target     = parse_params(param_str)
    block_size = getattr(args, "block_size", None) or _auto_block_size(target)
    force_layers = getattr(args, "layers", None)
    arch       = choose_arch(target, vocab_size=tok.vocab_size, block_size=block_size,
                             force_n_layer=force_layers)
    if arch is None:
        raise RuntimeError("couldn't find a matching architecture")
    cfg = {
        "vocab_size": tok.vocab_size,
        "block_size": block_size,
        "n_layer":    arch["n_layer"],
        "n_head":     arch["n_head"],
        "embed_dim":  arch["embed_dim"],
        "ff_mult":    4,
        "dropout":    0.1,
        "tokenizer":  tok.to_config(),
    }
    use_fp32  = getattr(args, "fp32", False)
    dtype     = torch.float32 if use_fp32 else torch.float16
    dtype_str = "fp32" if use_fp32 else "fp16"
    cfg["dtype"] = dtype_str

    model = TinyLLM(**{k: cfg[k] for k in
                       ("vocab_size","block_size","n_layer","n_head","embed_dim","ff_mult","dropout")})
    model = model.to(dtype)
    total = model.count_params()
    tag   = "".join(c for c in param_str.lower() if c.isalnum())
    fname = args.model if getattr(args, "model", None) else f"ckpt_{tag}.pt"
    torch.save({"model_state": model.state_dict(), "cfg": cfg, "epoch": 1,
                "global_step": 0, "loss_history": []}, fname)
    err_pct = abs(total - target) / target * 100
    mem_mb  = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    layers_tag = f" [yellow](forced)[/]" if force_layers else ""
    console.print(f"[green]✓ created {fname}[/]")
    console.print(f"  target={target:,}  actual=[bold]{total:,}[/]  err={err_pct:.1f}%")
    console.print(f"  layers={arch['n_layer']}{layers_tag}  embed={arch['embed_dim']}  heads={arch['n_head']}  block_size={block_size}")
    console.print(f"  dtype=[bold cyan]{dtype_str}[/]  weight size≈[bold]{mem_mb:.1f}MB[/]")


def cmd_chat(args):
    ckpt = load_ckpt(args.model)
    cfg  = _resolve_cfg(ckpt.get("cfg", {}))
    tok  = CharTokenizer.from_config(cfg.get("tokenizer"))
    dev  = best_device()
    model = TinyLLM(vocab_size=cfg["vocab_size"], block_size=cfg["block_size"],
                    n_layer=cfg["n_layer"], n_head=cfg["n_head"],
                    embed_dim=cfg["embed_dim"], ff_mult=cfg["ff_mult"],
                    dropout=cfg["dropout"])
    model.load_state_dict(ckpt.get("model_state", ckpt))
    ckpt_dtype = {"fp16": torch.float16, "fp32": torch.float32}.get(cfg.get("dtype","fp32"), torch.float32)
    model.eval().to(device=dev, dtype=ckpt_dtype)
    console.print(f"  dtype={cfg.get('dtype','fp32')}")
    console.print(f"[green]model loaded ({model.count_params():,} params)  •  type 'quit' to exit[/]")
    while True:
        try: prompt = input("\n>>> ")
        except (EOFError, KeyboardInterrupt): break
        if prompt.strip().lower() in ("quit","exit","q"): break
        with torch.no_grad():
            ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=dev)
            if ids.size(1) > model.block_size:
                ids = ids[:, -model.block_size:]
            out    = model.generate(ids, args.gen_len, temperature=args.temp, top_k=args.top_k)
            suffix = out[0].tolist()[ids.size(1):]
            console.print(Panel(tok.decode(suffix), title="model", border_style="magenta"))


def cmd_gen(args):
    ckpt  = load_ckpt(args.model)
    cfg   = _resolve_cfg(ckpt.get("cfg", {}))
    tok   = CharTokenizer.from_config(cfg.get("tokenizer"))
    dev   = best_device()
    model = TinyLLM(vocab_size=cfg["vocab_size"], block_size=cfg["block_size"],
                    n_layer=cfg["n_layer"], n_head=cfg["n_head"],
                    embed_dim=cfg["embed_dim"], ff_mult=cfg["ff_mult"],
                    dropout=cfg["dropout"])
    model.load_state_dict(ckpt.get("model_state", ckpt))
    ckpt_dtype = {"fp16": torch.float16, "fp32": torch.float32}.get(cfg.get("dtype","fp32"), torch.float32)
    model.eval().to(device=dev, dtype=ckpt_dtype)
    with torch.no_grad():
        ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=dev)
        out = model.generate(ids, args.gen_len, temperature=args.temp, top_k=args.top_k)
    print(tok.decode(out[0].tolist()))
