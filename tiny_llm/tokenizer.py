"""
A minimal character-level tokenizer. Vocabulary is built from a fixed
printable-ASCII charset (plus newline/tab) so it round-trips across any
plain-text training corpus, with an <unk> token for anything outside it.
"""

_DEFAULT_CHARS = "\n\t" + "".join(chr(i) for i in range(32, 127))


class CharTokenizer:
    def __init__(self, chars=None):
        chars = chars or _DEFAULT_CHARS
        seen, ordered = set(), []
        for ch in chars:
            if ch not in seen:
                ordered.append(ch); seen.add(ch)
        self.unk   = "<unk>"
        self.itos  = [self.unk] + ordered
        self.stoi  = {ch: i for i, ch in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

    def encode(self, s):
        return [self.stoi.get(c, 0) for c in s]

    def decode(self, ids):
        return "".join("?" if self.itos[int(i)] == self.unk
                       else self.itos[int(i)]
                       for i in ids if 0 <= int(i) < len(self.itos))

    def to_config(self):
        return {"type": "char", "chars": "".join(self.itos[1:])}

    @classmethod
    def from_config(cls, cfg=None):
        if not cfg:
            return cls()
        return cls(cfg.get("chars") or _DEFAULT_CHARS)


def make_tokenizer(cfg=None):
    return CharTokenizer.from_config((cfg or {}).get("tokenizer"))
