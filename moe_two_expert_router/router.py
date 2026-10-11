"""Router primitives: softmax gate, top-k assignment, capacity dropping, aux loss.

All functions operate on a flattened token axis (T = batch * seq_len).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    """Linear softmax gate: d_model -> n_experts.

    forward(x) with x of shape (T, d) returns (logits, probs), both (T, E).
    probs is a valid distribution over experts per token.
    """

    def __init__(self, d_model: int, n_experts: int):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=0.02)

    def forward(self, x: torch.Tensor):
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        return logits, probs


def top_k_assignment(probs: torch.Tensor, k: int):
    """Select the top-k experts per token.

    Args:
        probs: (T, E) router probabilities.
        k: 1 (Switch-style) or 2.

    Returns:
        expert_idx: (T, k) long -- chosen expert ids, ordered by descending prob.
        gates: (T, k) -- combine weights. For k=1 this is the raw chosen
            probability (Switch scales expert output by p_i). For k>1 the
            selected probabilities are renormalized to sum to 1 (Mixtral-style).
    """
    top_p, expert_idx = torch.topk(probs, k, dim=-1)
    if k == 1:
        gates = top_p
    else:
        gates = top_p / top_p.sum(dim=-1, keepdim=True)
    return expert_idx, gates


def apply_capacity(expert_idx: torch.Tensor, n_experts: int, capacity: int):
    """First-come-first-served capacity dropping.

    Assignments are processed in flat order (token 0 slot 0, token 0 slot 1,
    token 1 slot 0, ...). Each expert keeps at most `capacity` assignments;
    later ones overflow and are dropped.

    Args:
        expert_idx: (T, k) long expert ids.
        n_experts: number of experts E.
        capacity: max tokens each expert may process.

    Returns:
        keep: (T, k) bool -- True where the assignment fits under capacity.
        position_in_expert: (T, k) long -- 0-based arrival rank at that expert.
    """
    T, k = expert_idx.shape
    flat = expert_idx.reshape(-1)  # (T*k,) in arrival order
    one_hot = F.one_hot(flat, n_experts)  # (T*k, E)
    position = one_hot.cumsum(dim=0) - one_hot  # arrivals before this one
    position_in_expert = position.gather(1, flat.unsqueeze(1)).reshape(T, k)
    keep = position_in_expert < capacity
    return keep, position_in_expert


def load_balancing_loss(probs: torch.Tensor, expert_idx: torch.Tensor) -> torch.Tensor:
    """Switch Transformer auxiliary load-balancing loss.

    L_aux = E * sum_e f_e * P_e, where over the T tokens:
        f_e = fraction of routed assignments sent to expert e (pre-drop),
        P_e = mean router probability mass on expert e.
    For perfectly balanced uniform routing L_aux = 1; imbalance raises it.
    f is non-differentiable; the gradient flows through P.

    Args:
        probs: (T, E) router probabilities.
        expert_idx: (T, k) chosen experts (before capacity dropping).
    """
    T, E = probs.shape
    k = expert_idx.shape[1]
    counts = F.one_hot(expert_idx.reshape(-1), E).sum(dim=0).to(probs.dtype)
    f = counts / (T * k)
    P = probs.mean(dim=0)
    return E * torch.sum(f * P)


def router_entropy_bits(probs: torch.Tensor) -> torch.Tensor:
    """Mean per-token entropy of the gate distribution, in bits.

    Max is log2(E) (uniform gate); 0 means a fully confident router.
    """
    ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    return ent.mean() / math.log(2.0)
