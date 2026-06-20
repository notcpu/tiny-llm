# tiny_llm_trainer
```
python tiny_llm_trainer.py create --preset micro
python tiny_llm_trainer.py train --model ckpt_micro.pt --data_dir data --epochs 10 --fp16
python tiny_llm_trainer.py chat --model ckpt_micro.pt
python tiny_llm_trainer.py gen --model ckpt_micro.pt --prompt "Once upon"
```

## Module map

| File                      | What's in it                                                              |
|----------------------------|----------------------------------------------------------------------------|
| `logging_utils.py`         | `append_debug/error/traceback`, log file paths, `shutdown_requested` / `TUI_ACTIVE` flags |
| `device_utils.py`          | `best_device`, `autocast_ctx`, `make_scaler` CUDA/MPS/CPU + AMP handling |
| `lr_schedule.py`           | `get_lr` (warmup+cosine), `optimal_lr` (auto-LR heuristic)                |
| `tokenizer.py`             | `CharTokenizer` the character-level vocab                               |
| `dataset.py`               | local `.txt` loading/cleaning, HuggingFace dataset loading, `get_batch`   |
| `model.py`                 | `TinyLLM` and its building blocks (attention, MLP, transformer block)     |
| `arch.py`                  | `choose_arch` picks layer/dim/head counts to hit a param target; `PRESETS` |
| `checkpoint.py`            | `save_ckpt` / `load_ckpt` / `_resolve_cfg` (back-compat for old checkpoints) |
| `state.py`                 | `SharedState` (thread-safe state shared between training + TUI), `snapshot_params` |
| `tui.py`                   | All the Rich `Live` rendering like header/logs/sample/param-table/footer    |
| `training.py`              | `training_worker` is the actual training loop, runs on its own thread     |
| `commands.py`              | `cmd_create`, `cmd_chat`, `cmd_gen` are the non-training subcommands        |
| `cli.py`                   | `argparse` wiring, signal handling, `main()` spins up worker + TUI threads |

## Why it's split this way

Basically, my first 3 versions (i had LLM (failed), LLMv2, LLMkvcache (kvcache test, technically v3), and LLMv4 (which this is based off of)
Originally it was made as one single file because that was my original "religion", i thought it would be easier... I guess not

The original file already had numbered section headers (`# 1. LOGGING`,
`# 2. DEVICE + AMP HELPERS`, etc.) — turns out those were basically already
the module boundaries. I followed them as-is rather than inventing a new
structure, so anything you remember from the original file should map
directly to a same-named file here.

## A couple of things worth knowing

- **`state` is a singleton.** `tiny_llm/state.py` creates one `SharedState()`
  instance at import time, and everything that needs it does
  `from tiny_llm.state import state`. It's the same object everywhere —
  that's what lets the training thread write progress and the TUI thread
  read it concurrently (guarded by `state.lock`).

- **`shutdown_requested` / `TUI_ACTIVE` live in `logging_utils.py`** and get
  *mutated through the module* (`logging_utils.shutdown_requested = True`)
  rather than imported by value, since `from x import y` would freeze a
  stale copy. If you add new code that needs to check these flags, import
  the module (`from tiny_llm import logging_utils`) and read
  `logging_utils.shutdown_requested`, not a bare imported name.

- **Bug fix included:** the original had an fp16 overflow crash in
  `generate()`'s top-k masking (`torch.full_like(logits, -1e10)` blows up
  in fp16 — `-1e10` doesn't fit in a half). Reproduced it on the original
  file too, so it wasn't a refactor artifact. Fixed by swapping to
  `float("-inf")`, which is fp16-safe and behaves identically for masking
  purposes. That's the **only** behavioral change in this refactor —
  everything else was verified to produce byte-identical output (seeded
  weight init and seeded training steps were diffed against the original
  file and matched exactly).

## Verification performed

- Every module imports cleanly, no circular imports.
- All four subcommands (`create`, `train`, `chat`, `gen`) tested via `--help`
  and real runs.
- `create` with a fixed seed produces **byte-identical weights** to the
  original single-file script.
- `train` with a fixed seed produces **identical loss values and
  byte-identical post-training weights** vs. the original.
- Full TUI run (training thread + Rich `Live` thread + locking) completed
  50 epochs with zero crashes and a correctly progressing checkpoint.
