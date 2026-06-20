"""
Logging helpers — writes to debug/error log files, and to stdout/stderr
when the TUI isn't active (the TUI takes over the screen, so plain prints
would corrupt the display).

Note: this module is imported by tiny_llm.state (SharedState.log() calls
append_debug), so it must NOT import tiny_llm.state at module load time —
that would create a circular import. We import it lazily inside the
functions that need it instead.
"""
from datetime import datetime
import sys

DEBUG_LOG = "tiny_llm_debug.log"
ERROR_LOG = "tiny_llm_error.log"

# Mirrors the original module-level flag. cli.main() flips this when the
# TUI thread starts/stops so logging knows whether stdout is "owned" by Rich.
TUI_ACTIVE = False

# Set by cli._handle_interrupt on SIGINT/SIGTERM; checked by the training loop.
shutdown_requested = False


def _now():
    return datetime.now().strftime("%H:%M:%S")


def _write_file(path, msg):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def append_debug(msg: str):
    s = f"[{_now()}] {msg}"
    _write_file(DEBUG_LOG, s)
    if not TUI_ACTIVE:
        try:
            print(s)
        except Exception:
            pass


def append_error(msg: str):
    s = f"[{_now()}] {msg}"
    _write_file(ERROR_LOG, s)
    if TUI_ACTIVE:
        try:
            from tiny_llm.state import state
            with state.lock:
                state.logs.append("[ERR] " + s)
        except Exception:
            pass
    else:
        try:
            print(s, file=sys.stderr)
        except Exception:
            pass


def append_traceback(tb: str):
    _write_file(ERROR_LOG, f"[{_now()}] TRACEBACK:\n{tb}\n")
    if TUI_ACTIVE:
        try:
            from tiny_llm.state import state
            short = tb.splitlines()[-1] if tb else "traceback"
            with state.lock:
                state.logs.append("[TB] " + short)
        except Exception:
            pass
