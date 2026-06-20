"""
The actual training loop, run on a background thread so the TUI thread
can render concurrently. Pushes progress into state for the TUI to read,
and writes any crash traceback into exception_queue for the main thread.
"""
import math
import random
import time
import traceback

import torch
import torch.nn as nn

from tiny_llm import logging_utils
from tiny_llm.logging_utils import append_error, append_traceback
from tiny_llm.device_utils import best_device, autocast_ctx, make_scaler
from tiny_llm.lr_schedule import get_lr, optimal_lr
from tiny_llm.tokenizer import CharTokenizer
from tiny_llm.dataset import (
    read_texts_from_dir, load_hf_dataset, maybe_pin_to_device, get_batch,
)
from tiny_llm.model import TinyLLM
from tiny_llm.checkpoint import unwrap, save_ckpt, load_ckpt, _resolve_cfg
from tiny_llm.state import state, snapshot_params


def training_worker(args, model_path, exception_queue):
    try:
        if getattr(args, "hf_dataset", None):
            state.log(f"loading HF dataset: {args.hf_dataset}  split={args.hf_split}" +
                      (f"  subset={args.hf_subset}" if getattr(args, "hf_subset", None) else ""))
            raw = load_hf_dataset(
                args.hf_dataset,
                hf_split=args.hf_split,
                hf_field=getattr(args, "hf_field", None),
                max_rows=getattr(args, "hf_max_rows", None),
                hf_subset=getattr(args, "hf_subset", None),
            )
            state.log(f"{len(raw):,} chars from HF dataset")
        else:
            state.log(f"loading text from {args.data_dir}")
            raw = read_texts_from_dir(args.data_dir, strict=args.strict_clean)
            state.log(f"{len(raw):,} chars loaded")

        ckpt     = load_ckpt(model_path)
        cfg      = _resolve_cfg(ckpt.get("cfg", {}))
        tok      = CharTokenizer.from_config(cfg.get("tokenizer"))
        cfg["tokenizer"]  = tok.to_config()
        cfg["vocab_size"] = tok.vocab_size

        data = torch.tensor(tok.encode(raw), dtype=torch.long)
        unk  = int((data == 0).sum())
        if unk: state.log(f"⚠ {unk:,} <unk> tokens — consider cleaning data")
        state.log(f"{data.size(0):,} tokens")

        device = best_device()

        # build model
        model = TinyLLM(
            vocab_size=cfg["vocab_size"],
            block_size=cfg["block_size"],
            n_layer=cfg["n_layer"],
            n_head=cfg["n_head"],
            embed_dim=cfg["embed_dim"],
            ff_mult=cfg["ff_mult"],
            dropout=cfg["dropout"],
        )
        # Always train in fp32. GradScaler requires fp32 params.
        # fp16 is the *storage* dtype (checkpoint on disk); AMP handles fp16 compute.
        store_dtype = cfg.get("dtype", "fp32")
        raw_state   = ckpt.get("model_state", ckpt)
        # upcast fp16 weights to fp32 for training
        if store_dtype == "fp16":
            raw_state = {k: v.float() if v.dtype == torch.float16 else v
                         for k, v in raw_state.items()}
        model.load_state_dict(raw_state)
        model = model.to(device=device)   # fp32 on device
        state.log(f"model: {model.count_params():,} params → {device} "
                  f"(train=fp32, store={store_dtype})")

        # optional: pin dataset to VRAM
        data = maybe_pin_to_device(data, device, enabled=args.gpu_data,
                                   max_mb=args.gpu_data_max_mb)

        # dummy forward (surface errors early)
        model.eval()
        with torch.no_grad():
            seq = min(4, cfg.get("block_size", 128))
            tx  = torch.randint(0, cfg["vocab_size"], (1,seq), dtype=torch.long, device=device)
            model(tx, past_kv=None)
        model.train()
        state.log("dummy fwd OK")

        if args.compile:
            try:
                model = torch.compile(model)
                state.log("torch.compile ON")
            except Exception as e:
                state.log(f"torch.compile failed: {e}")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr,
            betas=(0.9, 0.95), weight_decay=args.weight_decay,
        )
        # optimal LR heuristic
        if getattr(args, "auto_lr", False):
            n_params   = unwrap(model).count_params()
            computed_lr = optimal_lr(n_params, args.batch_size, cfg.get("block_size", 128))
            args.lr    = computed_lr
            for pg in optimizer.param_groups:
                pg["lr"] = computed_lr
            state.log(f"auto LR: {computed_lr:.2e}  (model={n_params:,} params, "
                      f"batch={args.batch_size}, ctx={cfg.get('block_size',128)})")
        # AMP auto-on when: model stored as fp16 OR user passed --fp16, AND on CUDA
        want_amp = args.fp16 or (cfg.get("dtype", "fp32") == "fp16")
        amp_on   = bool(want_amp and device.type == "cuda")
        if want_amp and not amp_on:
            state.log(f"AMP unavailable on {device.type} — training in fp32")
        state.log(f"AMP={'ON (fp16 autocast)' if amp_on else 'off'}")
        scaler = make_scaler(device, enabled=amp_on)

        start_epoch    = ckpt.get("epoch", 1)
        global_step    = ckpt.get("global_step", 0)
        loss_history   = ckpt.get("loss_history", [])

        block_size      = cfg.get("block_size", 128)
        iters_per_epoch = max(1, math.floor(data.size(0) / (args.batch_size * block_size)))
        total_updates   = max(1, args.epochs * math.ceil(iters_per_epoch / max(1, args.accum)))
        opt_step        = global_step // max(1, args.accum)

        with state.lock:
            state.iters_per_epoch = iters_per_epoch
            state.epoch           = start_epoch
            state.total_steps     = total_updates
            state.total_epochs    = args.epochs

        state.log(f"iters/epoch ≈ {iters_per_epoch}  total_updates ≈ {total_updates}")

        wall_start = time.time()
        with state.lock:
            state.train_start = wall_start
        optimizer.zero_grad(set_to_none=True)
        ema_alpha  = 0.05   # smoothing factor for loss EMA

        for epoch in range(start_epoch, args.epochs + 1):
            if logging_utils.shutdown_requested:
                state.log("shutdown: saving & exiting")
                save_ckpt(model_path, model, optimizer, cfg, epoch, global_step, loss_history)
                return

            epoch_loss = 0.0
            for it in range(iters_per_epoch):
                if logging_utils.shutdown_requested:
                    state.log("shutdown: saving & exiting")
                    save_ckpt(model_path, model, optimizer, cfg, epoch, global_step, loss_history)
                    return

                x, y = get_batch(data, block_size, args.batch_size, device)

                with autocast_ctx(device, enabled=amp_on):
                    _, loss, _ = model(x, targets=y, past_kv=None)
                    loss = loss / max(1, args.accum)

                scaler.scale(loss).backward()

                update_due = ((it+1) % max(1,args.accum) == 0) or ((it+1) == iters_per_epoch)
                if update_due:
                    new_lr = get_lr(opt_step, total_updates, args.lr,
                                    warmup_frac=args.warmup_frac,
                                    cosine=not args.no_cosine_lr)
                    for pg in optimizer.param_groups:
                        pg["lr"] = new_lr
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    opt_step += 1

                loss_val     = float(loss.item() * max(1, args.accum))
                epoch_loss  += loss_val
                loss_history.append(loss_val)
                global_step += 1

                # update shared state for TUI
                with state.lock:
                    elapsed               = time.time() - wall_start
                    state.global_step     = global_step
                    state.iter_in_epoch   = it + 1
                    state.epoch           = epoch
                    state.loss            = loss_val
                    state.loss_history    = loss_history[-2000:]
                    state.sps             = global_step / max(1e-8, elapsed)
                    state.tps             = state.sps * args.batch_size * block_size
                    state.lr              = optimizer.param_groups[0]["lr"]
                    state.status          = "training"
                    # EMA smoothed loss
                    if state.loss_smoothed == 0.0:
                        state.loss_smoothed = loss_val
                    else:
                        state.loss_smoothed = (1 - ema_alpha) * state.loss_smoothed + ema_alpha * loss_val
                    # best loss
                    if loss_val < state.best_loss:
                        state.best_loss = loss_val
                    # grad norm (captured after unscale if update happened)
                    try:
                        gn = sum(p.grad.data.norm(2).item()**2
                                 for p in model.parameters() if p.grad is not None) ** 0.5
                        state.grad_norm = gn
                    except Exception: pass
                    # total tokens
                    state.total_tokens += args.batch_size * block_size
                    if global_step % max(1, args.param_stats_every) == 0:
                        try: state.param_stats = snapshot_params(model)
                        except Exception: pass

                if global_step % max(1, args.print_every) == 0:
                    state.log(f"ep {epoch}/{args.epochs} it {it+1}/{iters_per_epoch} "
                              f"step {global_step} loss {loss_val:.4f}")

                # live sample gen
                if args.sample_every > 0 and global_step % args.sample_every == 0:
                    try:
                        model.eval()
                        gm      = unwrap(model)
                        prompt  = random.choice(["The","Once upon","Hello","In the","def "])
                        p_ids   = tok.encode(prompt)
                        idx     = torch.tensor([p_ids], dtype=torch.long, device=device)
                        out     = gm.generate(idx, args.gen_len,
                                              temperature=args.sample_temp,
                                              top_k=args.sample_top_k)
                        reply   = tok.decode(out[0].tolist()[len(p_ids):])
                        with state.lock:
                            state.sample_text = f"[{prompt}]\n{reply}"
                        # capture KV-cache lens for display
                        try:
                            _, _, pkv = gm(idx, past_kv=None)
                            with state.lock:
                                state.cache_lens = [kv[0].size(2) if kv[0] is not None else 0
                                                    for kv in pkv]
                        except Exception: pass
                        model.train()
                    except Exception as e:
                        append_error(f"sample gen failed: {e}")
                        append_traceback(traceback.format_exc())

                if args.save_every > 0 and global_step % args.save_every == 0:
                    save_ckpt(model_path, model, optimizer, cfg,
                              epoch, global_step, loss_history)

            # end of epoch
            avg = epoch_loss / max(1, iters_per_epoch)
            state.log(f"── epoch {epoch} done  avg_loss={avg:.4f}")
            save_ckpt(model_path, model, optimizer, cfg, epoch+1, global_step, loss_history)

        state.log("training complete")
        save_ckpt(model_path, model, optimizer, cfg, args.epochs+1, global_step, loss_history)
        with state.lock:
            state.status = "done"

    except Exception as e:
        tb = traceback.format_exc()
        append_error(f"training_worker crash: {e}")
        append_traceback(tb)
        try:
            if "model" in dir():
                save_ckpt(model_path,
                          locals().get("model"), locals().get("optimizer"), {},
                          locals().get("epoch", -1), locals().get("global_step", -1),
                          locals().get("loss_history", []))
        except Exception: pass
        exception_queue.put(tb)
