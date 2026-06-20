"""
The model itself: a small KV-cache aware decoder-only transformer.
Uses flash attention (scaled_dot_product_attention) when no cache is in
play, and falls back to a manual masked-attention path when generating
with a KV cache.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_head, dropout):
        super().__init__()
        assert embed_dim % n_head == 0
        self.n_head   = n_head
        self.head_dim = embed_dim // n_head
        self.scale    = self.head_dim ** -0.5
        self.qkv      = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out      = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.dropout    = dropout
        self.use_flash  = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x, past_k=None, past_v=None, past_len=0):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
        q = qkv[:,:,0].permute(0,2,1,3)
        k = qkv[:,:,1].permute(0,2,1,3)
        v = qkv[:,:,2].permute(0,2,1,3)

        no_cache = past_k is None and past_v is None and past_len == 0
        if self.use_flash and no_cache:
            dp = self.dropout if self.training else 0.0
            y  = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                dropout_p=dp, is_causal=True)
            y  = y.transpose(1,2).contiguous().view(B, T, C)
            return self.resid_drop(self.out(y)), k, v

        if past_k is not None:
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        S   = k.size(2)
        att = (q @ k.transpose(-2,-1)) * self.scale
        ik  = torch.arange(S, device=x.device).unsqueeze(0)
        iq  = torch.arange(past_len, past_len+T, device=x.device).unsqueeze(1)
        att = att.masked_fill(~(ik <= iq).unsqueeze(0).unsqueeze(0), float("-inf"))
        att = self.attn_drop(F.softmax(att, dim=-1))
        y   = (att @ v).transpose(1,2).contiguous().view(B, T, C)
        return self.resid_drop(self.out(y)), k, v


class MLP(nn.Module):
    def __init__(self, embed_dim, ff_mult, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_mult * embed_dim, bias=False),
            nn.GELU(),
            nn.Linear(ff_mult * embed_dim, embed_dim, bias=False),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)


class Block(nn.Module):
    def __init__(self, embed_dim, n_head, ff_mult, dropout):
        super().__init__()
        self.ln1  = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, n_head, dropout)
        self.ln2  = nn.LayerNorm(embed_dim)
        self.mlp  = MLP(embed_dim, ff_mult, dropout)

    def forward(self, x, past_k=None, past_v=None, past_len=0):
        a, k, v = self.attn(self.ln1(x), past_k=past_k, past_v=past_v, past_len=past_len)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, k, v


class TinyLLM(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, embed_dim, ff_mult=4, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.n_layer    = n_layer
        self.tok_emb    = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb    = nn.Embedding(block_size, embed_dim)
        self.drop       = nn.Dropout(dropout)
        self.blocks     = nn.ModuleList([Block(embed_dim, n_head, ff_mult, dropout) for _ in range(n_layer)])
        self.ln_f       = nn.LayerNorm(embed_dim)
        self.head       = nn.Linear(embed_dim, vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias); nn.init.ones_(m.weight)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, past_kv=None, past_len=0):
        B, T = idx.shape
        assert T <= self.block_size
        pos    = torch.arange(past_len, past_len+T, device=idx.device).unsqueeze(0) % self.block_size
        x      = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        past_kv = past_kv or [(None, None)] * self.n_layer
        new_kv  = []
        for blk, (pk, pv) in zip(self.blocks, past_kv):
            x, k, v = blk(x, past_k=pk, past_v=pv, past_len=past_len)
            new_kv.append((k, v))
        logits = self.head(self.ln_f(x))
        loss   = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) \
                 if targets is not None else None
        return logits, loss, new_kv

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        device = next(self.parameters()).device
        idx    = idx.to(device)
        B, cur = idx.shape
        if cur == 0:
            idx = torch.zeros((B,1), dtype=torch.long, device=device); cur = 1
        if cur > self.block_size:
            idx = idx[:, -self.block_size:]; cur = idx.size(1)
        _, _, past_kv = self.forward(idx, past_kv=None, past_len=0)
        for _ in range(max_new_tokens):
            last    = idx[:, -1:]
            logits, _, past_kv = self.forward(last, past_kv=past_kv, past_len=idx.size(1)-1)
            logits  = logits[:, -1, :] / max(1e-8, temperature)
            if top_k:
                k  = min(int(top_k), logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits = torch.where(logits < v[:,-1:], torch.full_like(logits, float("-inf")), logits)
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits,-1), 1)], dim=1)
        return idx
