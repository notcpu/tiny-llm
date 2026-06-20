"""
Architecture search: given a target parameter count, finds a
(n_layer, embed_dim, n_head) combo that gets as close as possible.
Also handles parsing of param-count strings (e.g. "500k", "7m", "1b")
and the named presets used by `create --preset ...`.
"""


def _est_params(vocab_size, block_size, n_layer, embed_dim, ff_mult):
    return int(2*vocab_size*embed_dim + block_size*embed_dim + n_layer*(4+2*ff_mult)*embed_dim**2)


def choose_arch(target, vocab_size=96, block_size=128, ff_mult=4, force_n_layer=None):
    """
    Find (n_layer, embed_dim, n_head) closest to target total params.
    If force_n_layer is set, only that layer count is searched. embed_dim/n_head
    are then chosen to get as close to target as possible.
    Uses tiered search grids so it scales from 1K all the way to 70B+.
    """
    if target < 10_000_000:
        dims   = list(range(64, 512, 16))
        layers = list(range(2, 20))
        heads  = [2, 4, 8]
    elif target < 100_000_000:
        dims   = list(range(128, 1025, 64))
        layers = list(range(4, 32))
        heads  = [4, 8, 16]
    elif target < 500_000_000:
        dims   = list(range(512, 4097, 128))
        layers = list(range(8, 64))
        heads  = [8, 16, 32]
    else:
        dims   = list(range(1024, 32769, 256))
        layers = list(range(12, 128))
        heads  = [16, 32, 64]

    if force_n_layer is not None:
        layers = [force_n_layer]

    best, best_err = None, 1e18
    for n_layer in layers:
        for embed_dim in dims:
            for n_head in heads:
                if embed_dim % n_head: continue
                est = _est_params(vocab_size, block_size, n_layer, embed_dim, ff_mult)
                err = abs(est - target)
                if err < best_err:
                    best_err = err
                    best = {"n_layer": n_layer, "embed_dim": embed_dim,
                            "n_head": n_head, "est": est}
                if est > target * 4:
                    break
    return best


def parse_params(s):
    s = s.strip().lower()
    if s.endswith("b"): return int(float(s[:-1]) * 1_000_000_000)
    if s.endswith("m"): return int(float(s[:-1]) * 1_000_000)
    if s.endswith("k"): return int(float(s[:-1]) * 1_000)
    if s.isdigit():     return int(s)
    raise ValueError(f"can't parse param count: {s!r}  (use e.g. 500k, 7m, 1b)")


PRESETS = {"nano": "50k", "micro": "500k", "small": "5m", "medium": "125m", "large": "1b", "xl": "2b"}
