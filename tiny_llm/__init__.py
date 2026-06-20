"""
Module map:
  logging_utils   — debug/error logging to file + console
  device_utils    — device selection, autocast, grad scaler
  lr_schedule     — warmup+cosine LR schedule, optimal-LR heuristic
  tokenizer       — CharTokenizer
  dataset         — local .txt loading, HuggingFace loading, batching
  model           — TinyLLM transformer (KV-cache aware)
  arch            — architecture search / param-count targeting
  checkpoint      — save/load/resolve checkpoint dicts
  state           — SharedState (thread-safe state shared with the TUI)
  tui             — Rich Live TUI rendering
  training        — training_worker (the actual training loop)
  commands        — cmd_create / cmd_chat / cmd_gen
  cli             — argparse wiring + main()
"""
