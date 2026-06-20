#!/usr/bin/env python3
"""

Subcommands:
  create  — initialise a new checkpoint from a param target
  train   — train with a live Rich TUI
  chat    — interactive chat with a saved model
  gen     — one-shot generation

Quick start:
  pip install torch rich psutil
  make sure also to install the correct torch if u want cuda sometimes it doesnt work

  python tiny_llm_trainer.py create --preset micro
  python tiny_llm_trainer.py train  --model ckpt_micro.pt --data_dir data --epochs 10 --fp16
  python tiny_llm_trainer.py chat   --model ckpt_micro.pt

  # no TUI / plain text:
  python tiny_llm_trainer.py train --model ckpt_micro.pt --data_dir data --no-tui

  # GPU data pin (loads whole dataset onto VRAM if small enough):
  python tiny_llm_trainer.py train --model ckpt_micro.pt --data_dir data --gpu-data
"""
from tiny_llm.cli import main

if __name__ == "__main__":
    # Required on Windows: prevents recursive subprocess spawning
    import multiprocessing
    multiprocessing.freeze_support()
    main()
