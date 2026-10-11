"""Correctness tests for the two-expert MoE router, capacity, and aux loss."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from moe_two_expert_router import (
    Router,
    top_k_assignment,
    apply_capacity,
    load_balancing_loss,
    router_entropy_bits,
    MoELayer,
    CharTransformer,
    build_corpus,
)

torch.manual_seed(0)


# ---------------------------------------------------------------- router gate
def test_router_probs_are_valid_distribution():
    router = Router(d_model=16, n_experts=2)
    x = torch.randn(64, 16)
    _, probs = router(x)
    assert probs.shape == (64, 2)
    assert torch.all(probs >= 0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(64), atol=1e-6)


def test_top1_and_top2_masks_select_exact_expert_counts():
    probs = F.softmax(torch.randn(128, 2), dim=-1)
    idx1, gates1 = top_k_assignment(probs, k=1)
    assert idx1.shape == (128, 1)
    # top-1 gate is the raw chosen probability (Switch scaling)
    assert torch.allclose(gates1.squeeze(1), probs.max(dim=-1).values)

    idx2, gates2 = top_k_assignment(probs, k=2)
    assert idx2.shape == (128, 2)
    # exactly 2 DISTINCT experts per token, and gates renormalize to 1
    assert all(len(set(row.tolist())) == 2 for row in idx2)
    assert torch.allclose(gates2.sum(dim=-1), torch.ones(128), atol=1e-6)


# ------------------------------------------------------------------- aux loss
def test_aux_loss_matches_hand_computed_value_on_4_tokens():
    # 4 tokens, 2 experts, fixed probabilities; top-1 assignment.
    probs = torch.tensor(
        [[0.9, 0.1], [0.8, 0.2], [0.6, 0.4], [0.3, 0.7]]
    )
    expert_idx = torch.tensor([[0], [0], [0], [1]])
    # f = (3/4, 1/4); P = ((0.9+0.8+0.6+0.3)/4, (0.1+0.2+0.4+0.7)/4) = (0.65, 0.35)
    # L = 2 * (0.75*0.65 + 0.25*0.35) = 2 * (0.4875 + 0.0875) = 1.15
    loss = load_balancing_loss(probs, expert_idx)
    assert math.isclose(float(loss), 1.15, rel_tol=1e-6)


def test_balanced_routing_minimizes_aux_loss():
    # Perfect balance with uniform gates -> loss exactly 1.0.
    T = 100
    uniform = torch.full((T, 2), 0.5)
    balanced_idx = torch.tensor([[i % 2] for i in range(T)])
    balanced = float(load_balancing_loss(uniform, balanced_idx))
    assert math.isclose(balanced, 1.0, rel_tol=1e-6)

    # Property: in a family where a fraction phi of tokens gates (0.8, 0.2)
    # and the rest (0.2, 0.8), the loss is minimized exactly at balance
    # (phi = 0.5) and rises with skew on either side.
    def family_loss(phi: float) -> float:
        n0 = int(round(T * phi))
        probs = torch.cat(
            [torch.tensor([[0.8, 0.2]]).expand(n0, 2),
             torch.tensor([[0.2, 0.8]]).expand(T - n0, 2)]
        )
        idx, _ = top_k_assignment(probs, k=1)
        return float(load_balancing_loss(probs, idx))

    l_bal = family_loss(0.5)
    for phi in (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0):
        assert family_loss(phi) > l_bal + 1e-6, f"phi={phi} should score worse than balance"

    # Full collapse with a confident router approaches the max (2.0 for E=2)...
    all_zero = torch.zeros(T, 1, dtype=torch.long)
    peaked = torch.tensor([[0.999, 0.001]]).expand(T, 2)
    assert float(load_balancing_loss(peaked, all_zero)) > 1.9
    # ...and no configuration can beat the theoretical infimum 7/8 for E=2
    # (f from argmax can anti-correlate with P, so values slightly below 1.0
    # are legal; 0.875 = min over f of 2f^2 - f + 1 is not).
    g = torch.Generator().manual_seed(1)
    for _ in range(50):
        probs = F.softmax(torch.randn(T, 2, generator=g) * 2, dim=-1)
        idx, _ = top_k_assignment(probs, k=1)
        assert float(load_balancing_loss(probs, idx)) >= 0.875 - 1e-6


# ------------------------------------------------------------------- capacity
def test_capacity_keeps_first_c_per_expert_and_drops_right_overflow():
    # 8 tokens all wanting expert 0, capacity 3 -> tokens 0,1,2 kept, 3..7 dropped.
    idx = torch.zeros(8, 1, dtype=torch.long)
    keep, pos = apply_capacity(idx, n_experts=2, capacity=3)
    assert keep.squeeze(1).tolist() == [True] * 3 + [False] * 5
    assert pos.squeeze(1).tolist() == list(range(8))

    # Mixed pattern: E0 at tokens 0,2,4,6; E1 at 1,3,5,7; capacity 2 keeps the
    # first two arrivals of EACH expert.
    idx = torch.tensor([[0], [1], [0], [1], [0], [1], [0], [1]])
    keep, _ = apply_capacity(idx, n_experts=2, capacity=2)
    assert keep.squeeze(1).tolist() == [True, True, True, True, False, False, False, False]
    # exactly `capacity` kept per expert
    for e in (0, 1):
        assert int(((idx.squeeze(1) == e) & keep.squeeze(1)).sum()) == 2


def test_moe_overflow_accounting_matches_manual_count():
    torch.manual_seed(3)
    layer = MoELayer(d_model=8, d_ff=16, n_experts=2, k=1, capacity_factor=0.5)
    x = torch.randn(2, 16, 8)  # 32 tokens -> capacity = ceil(0.5*32/2) = 8 per expert
    _, _, stats = layer(x)
    assert stats["capacity"] == 8
    kept_max = 2 * 8  # both experts full at most
    dropped_min = (32 - kept_max) / 32
    assert stats["dropped_frac"] >= dropped_min - 1e-6
    assert 0.0 <= stats["dropped_frac"] < 1.0


def test_dropped_tokens_pass_through_residual_unchanged():
    torch.manual_seed(4)
    layer = MoELayer(d_model=8, d_ff=16, n_experts=2, k=1, capacity_factor=0.25)
    x = torch.randn(1, 32, 8)
    out, _, stats = layer(x)
    flat_out = out.reshape(-1, 8)
    _, probs = layer.router(x.reshape(-1, 8))
    idx, _ = top_k_assignment(probs, k=1)
    keep, _ = apply_capacity(idx, 2, layer.capacity_for(32))
    dropped = ~keep.squeeze(1)
    assert dropped.any(), "capacity 0.25 must drop tokens"
    # MoE contribution is zero for dropped tokens -> residual x + 0 = x.
    assert torch.allclose(flat_out[dropped], torch.zeros_like(flat_out[dropped]))
    residual = x + out
    assert torch.allclose(residual.reshape(-1, 8)[dropped], x.reshape(-1, 8)[dropped])


# ------------------------------------------------------------------ gradients
def test_gradient_flows_to_both_experts_and_router_under_top2():
    torch.manual_seed(5)
    layer = MoELayer(d_model=8, d_ff=16, n_experts=2, k=2, capacity_factor=2.0)
    x = torch.randn(1, 10, 8, requires_grad=True)
    out, aux, _ = layer(x)
    (out.sum() + aux).backward()
    for e, expert in enumerate(layer.experts):
        g = expert.fc1.weight.grad
        assert g is not None and float(g.abs().sum()) > 0, f"expert {e} got no gradient"
    rg = layer.router.gate.weight.grad
    assert rg is not None and float(rg.abs().sum()) > 0, "router got no gradient"
    assert x.grad is not None and float(x.grad.abs().sum()) > 0


# ---------------------------------------------------------------- determinism
def test_determinism_same_seed_same_output():
    def run():
        torch.manual_seed(11)
        m = CharTransformer(vocab_size=20, d_model=32, n_layers=2, n_heads=2,
                            d_ff=64, max_seq=32, use_moe=True, k=1)
        x = torch.arange(64).remainder(20).reshape(2, 32)
        logits, aux, _ = m(x)
        return logits.detach().clone(), float(aux.detach())

    l1, a1 = run()
    l2, a2 = run()
    assert torch.equal(l1, l2)
    assert a1 == a2


# ------------------------------------------------------------------- plumbing
def test_entropy_bits_bounds_and_uniform_max():
    uniform = torch.full((10, 2), 0.5)
    assert math.isclose(float(router_entropy_bits(uniform)), 1.0, rel_tol=1e-6)
    peaked = torch.tensor([[1.0 - 1e-9, 1e-9]]).expand(10, 2)
    assert float(router_entropy_bits(peaked)) < 1e-6


def test_active_ffn_params_match_dense_baseline_for_top1():
    moe = CharTransformer(30, use_moe=True, k=1)
    dense = CharTransformer(30, use_moe=False)
    router_params = 2 * 96 * 2  # 2 layers x (96 -> 2 gate)
    assert moe.active_ffn_params_per_token() - router_params == dense.active_ffn_params_per_token()
    # total params: MoE holds ~2x the FFN weights of dense
    n_moe = sum(p.numel() for p in moe.parameters())
    n_dense = sum(p.numel() for p in dense.parameters())
    assert n_moe > n_dense


def test_corpus_is_mixed_labeled_and_deterministic():
    t1, l1, v1 = build_corpus(seed=0)
    t2, l2, v2 = build_corpus(seed=0)
    assert t1 == t2 and np.array_equal(l1, l2)
    assert len(t1) >= 40_000
    frac_prose = float((l1 == 0).mean())
    assert 0.3 < frac_prose < 0.7, "both sublanguages must be well represented"
    assert "def " in t1 and "return" in t1  # code side
    assert ". " in t1  # prose side
    ids = v1.encode(t1)
    assert v1.decode(ids[:100]) == t1[:100]
