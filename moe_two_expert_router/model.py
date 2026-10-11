"""Tiny char-level transformer (2 layers, d_model 96) with MoE or dense FFN."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe import MoELayer, DenseFFN


class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention."""

    def __init__(self, d_model: int, n_heads: int, max_seq: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        mask = torch.tril(torch.ones(max_seq, max_seq, dtype=torch.bool))
        self.register_buffer("causal", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        hd = d // self.n_heads
        q = q.view(B, S, self.n_heads, hd).transpose(1, 2)
        k = k.view(B, S, self.n_heads, hd).transpose(1, 2)
        v = v.view(B, S, self.n_heads, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / hd**0.5
        att = att.masked_fill(~self.causal[:S, :S], float("-inf"))
        y = F.softmax(att, dim=-1) @ v
        y = y.transpose(1, 2).contiguous().view(B, S, d)
        return self.proj(y)


class Block(nn.Module):
    """Pre-LN transformer block; FFN is either a MoELayer or a DenseFFN."""

    def __init__(self, d_model, n_heads, d_ff, max_seq, use_moe, k, capacity_factor):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq)
        self.ln2 = nn.LayerNorm(d_model)
        self.use_moe = use_moe
        if use_moe:
            self.ffn = MoELayer(d_model, d_ff, n_experts=2, k=k, capacity_factor=capacity_factor)
        else:
            self.ffn = DenseFFN(d_model, d_ff)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln1(x))
        if self.use_moe:
            y, aux, stats = self.ffn(self.ln2(x))
            return x + y, aux, stats
        return x + self.ffn(self.ln2(x)), None, None


class CharTransformer(nn.Module):
    """Char LM: embeddings + 2 blocks + LM head.

    forward(idx) -> (logits, aux_total, per_layer_stats). aux_total is the mean
    of the per-layer aux losses for MoE models, else a zero tensor.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 96,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 192,
        max_seq: int = 128,
        use_moe: bool = True,
        k: int = 1,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        self.max_seq = max_seq
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, d_ff, max_seq, use_moe, k, capacity_factor)
            for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor):
        B, S = idx.shape
        pos = torch.arange(S, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None]
        aux_terms, per_layer_stats = [], []
        for block in self.blocks:
            x, aux, stats = block(x)
            if aux is not None:
                aux_terms.append(aux)
                per_layer_stats.append(stats)
        x = self.ln_f(x)
        logits = self.head(x)
        aux_total = torch.stack(aux_terms).mean() if aux_terms else logits.new_zeros(())
        return logits, aux_total, per_layer_stats

    def active_ffn_params_per_token(self) -> int:
        """FFN parameters touched per token (router included for MoE)."""
        block = self.blocks[0]
        if block.use_moe:
            moe = block.ffn
            one_expert = sum(p.numel() for p in moe.experts[0].parameters())
            router = sum(p.numel() for p in moe.router.parameters())
            return len(self.blocks) * (moe.k * one_expert + router)
        dense = sum(p.numel() for p in block.ffn.parameters())
        return len(self.blocks) * dense
