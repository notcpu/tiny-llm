"""
Dataset loading + batching.

Covers:
  - reading/cleaning plain-text .txt files from a local directory
  - loading + auto-formatting a HuggingFace dataset (text or instruct-style)
  - vectorised random batch sampling, with optional GPU pinning
"""
import os
import glob

import torch

from tiny_llm.logging_utils import append_debug
from tiny_llm.tokenizer import _DEFAULT_CHARS

_SKIP_HINTS = ("debug","error","log","checkpoint","ckpt","tokenizer",
               "cache","__pycache__",".git","wandb")
_CODE_HINTS  = ("import ","def ","class ","torch.","#include","<?php","<script")


def _decode_file(path):
    for enc in ("utf-8","latin-1","utf-8-sig"):
        try:
            with open(path, encoding=enc) as f:
                return f.read()
        except Exception:
            pass
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="replace")


def _clean_text(text):
    allowed = set(_DEFAULT_CHARS)
    out, prev_blank = [], 0
    for ch in text:
        if ch in allowed:        s = ch
        elif ch in ("\n","\r"):  s = "\n"
        elif ch.isspace():       s = " "
        else:                    s = " "
        if s == "\n":
            prev_blank += 1
            if prev_blank > 2: continue
        elif s not in (" ","\t"):
            prev_blank = 0
        out.append(s)
    return "".join(out)


def _looks_clean(text):
    s = text.strip()
    if len(s) < 32: return False, "too short"
    vis = sum(1 for c in s if c in _DEFAULT_CHARS or c.isspace())
    if vis / max(1, len(s)) < 0.95: return False, "too many non-text chars"
    hits = sum(1 for h in _CODE_HINTS if h in s[:5000].lower())
    if hits >= 3: return False, "looks like code/log"
    return True, "ok"


def read_texts_from_dir(data_dir, strict=False):
    paths = glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True)
    if not paths:
        raise FileNotFoundError(f"no .txt files in {data_dir}")
    parts, skipped = [], []
    for p in paths:
        rel = os.path.relpath(p, data_dir)
        if any(h in rel.lower() for h in _SKIP_HINTS):
            skipped.append((rel, "name looks like artifact")); continue
        try:
            text = _clean_text(_decode_file(p))
            ok, reason = _looks_clean(text)
            if not ok:
                skipped.append((rel, reason))
                if strict: raise ValueError(f"dirty file {rel}: {reason}")
                continue
            parts.append(text)
        except Exception as e:
            skipped.append((rel, str(e)))
            if strict: raise
    for rel, r in skipped[:20]:
        append_debug(f"skipped {rel}: {r}")
    if not parts:
        raise FileNotFoundError(f"no clean .txt files in {data_dir}")
    return "\n\n".join(parts)

# hf dataset loader

_TEXT_FIELD_PRIORITY = ["text", "content", "document", "passage", "article",
                        "body", "story", "sentence", "abstract"]
_INSTRUCT_FIELDS = [
    ("instruction", "input",  "output"),
    ("prompt",      None,     "response"),
    ("question",    None,     "answer"),
    ("human",       None,     "assistant"),
    ("user",        None,     "bot"),
]


def _hf_detect_mode(column_names):
    cols = set(c.lower() for c in column_names)
    for inst, inp, out in _INSTRUCT_FIELDS:
        if inst in cols and out in cols:
            return "instruct"
    return "text"


def _hf_detect_text_field(column_names):
    cols_lower = {c.lower(): c for c in column_names}
    for candidate in _TEXT_FIELD_PRIORITY:
        if candidate in cols_lower:
            return cols_lower[candidate]
    return column_names[0]


def _hf_detect_instruct_fields(column_names):
    cols_lower = {c.lower(): c for c in column_names}
    for inst, inp, out in _INSTRUCT_FIELDS:
        if inst in cols_lower and out in cols_lower:
            return (cols_lower[inst],
                    cols_lower[inp] if inp and inp in cols_lower else None,
                    cols_lower[out])
    raise RuntimeError(f"no instruction/output columns found in {column_names}")


def _format_instruct_row(row, inst_col, inp_col, out_col):
    inst = str(row.get(inst_col, "") or "").strip()
    inp  = str(row.get(inp_col,  "") or "").strip() if inp_col else ""
    out  = str(row.get(out_col,  "") or "").strip()
    if inp:
        return f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
    return f"### Instruction:\n{inst}\n\n### Response:\n{out}"


def load_hf_dataset(hf_dataset, hf_split="train", hf_field=None, max_rows=None, hf_subset=None):
    """Load a HuggingFace dataset and return a single cleaned training string."""
    try:
        from datasets import load_dataset as _load
    except ImportError:
        raise ImportError("Run: pip install datasets")
    if hf_subset:
        append_debug(f"loading HF dataset {hf_dataset!r}  subset={hf_subset!r}  split={hf_split}")
        ds = _load(hf_dataset, hf_subset, split=hf_split)
    else:
        append_debug(f"loading HF dataset {hf_dataset!r}  split={hf_split}")
        ds = _load(hf_dataset, split=hf_split)
    if max_rows:
        ds = ds.select(range(min(max_rows, len(ds))))
    cols = ds.column_names
    append_debug(f"columns: {cols}")
    if hf_field:
        parts = [p.strip() for p in hf_field.split(",")]
        if len(parts) == 3 and parts[0] and parts[2]:
            inst_col, inp_col, out_col = parts[0], parts[1] or None, parts[2]
            append_debug(f"instruct mode (manual): {inst_col}/{inp_col}/{out_col}")
            rows = [_format_instruct_row(r, inst_col, inp_col, out_col) for r in ds]
        else:
            field = parts[0]
            append_debug(f"text mode (manual field={field!r})")
            rows = [str(r.get(field, "") or "") for r in ds]
        return _clean_text("\n\n".join(rows))
    mode = _hf_detect_mode(cols)
    append_debug(f"auto-detected mode: {mode}")
    if mode == "instruct":
        inst_col, inp_col, out_col = _hf_detect_instruct_fields(cols)
        append_debug(f"fields: {inst_col!r} / {inp_col!r} / {out_col!r}")
        rows = [_format_instruct_row(r, inst_col, inp_col, out_col) for r in ds]
    else:
        field = _hf_detect_text_field(cols)
        append_debug(f"text field: {field!r}")
        rows = [str(r.get(field, "") or "") for r in ds]
    result = _clean_text("\n\n".join(rows))
    append_debug(f"HF → {len(result):,} chars from {len(ds):,} rows")
    return result


def maybe_pin_to_device(data, device, enabled=True, max_mb=512.0):
    device = torch.device(device)
    if not enabled or device.type not in ("cuda","mps"):
        return data
    mb = data.numel() * data.element_size() / (1024*1024)
    if mb > max_mb:
        append_debug(f"dataset too large ({mb:.1f}MB) to pin to {device}"); return data
    try:
        moved = data.to(device)
        append_debug(f"dataset pinned to {device} ({mb:.1f}MB)")
        return moved
    except Exception as e:
        append_debug(f"pin failed: {e}"); return data


def get_batch(data, block_size, batch_size, device):
    """Vectorised random batch no py loop"""
    max_s  = max(1, data.size(0) - block_size)
    src    = data.device
    starts = torch.randint(0, max_s, (batch_size,), device=src)
    offs   = torch.arange(block_size, device=src)
    ix     = starts[:, None] + offs[None, :]
    x, y   = data[ix], data[ix + 1]
    if src != torch.device(device):
        nb = torch.device(device).type == "cuda"
        x  = x.to(device, non_blocking=nb)
        y  = y.to(device, non_blocking=nb)
    return x, y
