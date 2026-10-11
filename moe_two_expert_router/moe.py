"""Two-expert MoE layer: expert FFNs, dispatch/combine, overflow accounting."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .router import Router, top_k_assignment, apply_capacity, load_balancing_loss, router_entropy_bits


class ExpertFFN(nn.Module):
    """One expert: position-wise FFN d_model -> d_ff -> d_model with GELU."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class DenseFFN(ExpertFFN):
    """Dense baseline FFN -- identical shape to ONE expert, so a top-1 MoE and
    this baseline activate the same number of FFN parameters per token."""


class MoELayer(nn.Module):
    """Mixture-of-Experts FFN with softmax gate, top-1/top-2 routing, capacity.

    forward returns (output, aux_loss, stats). Output is ONLY the expert
    contribution: tokens dropped by capacity contribute zero, so the standard
    residual connection (x + moe(ln(x))) passes them through unchanged.

    stats (all detached floats / tensors):
        load_fractions: (E,) fraction of top-1 tokens per expert (pre-drop)
        entropy_bits: mean router entropy
        dropped_frac: fraction of assignments dropped by capacity
        aux: aux loss value
        top1_expert: (T,) top-1 expert id per token (for specialization maps)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 2,
        k: int = 1,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        assert k in (1, 2), "only top-1 and top-2 routing are implemented"
        self.n_experts = n_experts
        self.k = k
        self.capacity_factor = capacity_factor
        self.router = Router(d_model, n_experts)
        self.experts = nn.ModuleList(ExpertFFN(d_model, d_ff) for _ in range(n_experts))

    def capacity_for(self, n_tokens: int) -> int:
        """capacity = ceil(capacity_factor * k * tokens / n_experts), at least 1.

        Defined over total assignments (k * tokens) so that with perfectly
        balanced routing, capacity_factor 1.0 drops nothing for any k.
        """
        return max(1, math.ceil(self.capacity_factor * self.k * n_tokens / self.n_experts))

    def forward(self, x: torch.Tensor):
        B, S, d = x.shape
        flat = x.reshape(-1, d)
        T = flat.shape[0]

        logits, probs = self.router(flat)
        expert_idx, gates = top_k_assignment(probs, self.k)
        capacity = self.capacity_for(T)
        keep, _ = apply_capacity(expert_idx, self.n_experts, capacity)
        aux = load_balancing_loss(probs, expert_idx)

        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            mask = (expert_idx == e) & keep  # (T, k)
            token_mask = mask.any(dim=1)
            if not token_mask.any():
                continue
            rows = token_mask.nonzero(as_tuple=True)[0]
            # combine weight: sum of gate weights for this expert on kept slots
            w = (gates * mask.to(gates.dtype)).sum(dim=1)[rows].unsqueeze(1)
            out[rows] = out[rows] + w * expert(flat[rows])

        with torch.no_grad():
            counts = torch.nn.functional.one_hot(expert_idx[:, 0], self.n_experts).sum(0)
            stats = {
                "load_fractions": (counts.float() / T).cpu(),
                "entropy_bits": float(router_entropy_bits(probs)),
                "dropped_frac": float((~keep).float().mean()),
                "aux": float(aux.detach()),
                "capacity": capacity,
                "top1_expert": expert_idx[:, 0].detach().cpu(),
            }
        return out.reshape(B, S, d), aux, stats
