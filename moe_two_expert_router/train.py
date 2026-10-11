"""Training loop with routing diagnostics + domain-specialization measurement."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .corpus import PROSE_LABEL, CODE_LABEL
from .model import CharTransformer


def _batch(ids: np.ndarray, block: int, batch: int, rng: np.random.Generator, device):
    starts = rng.integers(0, len(ids) - block - 1, size=batch)
    x = np.stack([ids[s : s + block] for s in starts])
    y = np.stack([ids[s + 1 : s + block + 1] for s in starts])
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.no_grad()
def eval_loss(model, ids, block, batch, n_batches, seed=123, device="cpu"):
    """Mean cross-entropy over deterministic random val batches."""
    rng = np.random.default_rng(seed)
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = _batch(ids, block, batch, rng, device)
        logits, _, _ = model(x)
        losses.append(float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))))
    model.train()
    return float(np.mean(losses))


def train_model(
    model: CharTransformer,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    steps: int = 600,
    block: int = 128,
    batch: int = 24,
    lr: float = 3e-3,
    aux_weight: float = 0.01,
    eval_every: int = 25,
    seed: int = 0,
    device: str = "cpu",
):
    """Train and log routing diagnostics.

    Returns a history dict with per-eval-step lists:
        step, train_loss, val_loss, aux,
        load_e0 (n_layers lists: expert-0 top-1 fraction per layer),
        entropy_bits, dropped_frac.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n_layers = len(model.blocks)
    hist = {
        "step": [],
        "train_loss": [],
        "val_loss": [],
        "aux": [],
        "load_e0": [[] for _ in range(n_layers)],
        "entropy_bits": [],
        "dropped_frac": [],
    }
    model.train()
    for step in range(1, steps + 1):
        x, y = _batch(train_ids, block, batch, rng, device)
        logits, aux, stats = model(x)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss = ce + aux_weight * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % eval_every == 0 or step == 1:
            hist["step"].append(step)
            hist["train_loss"].append(float(ce.detach()))
            hist["val_loss"].append(eval_loss(model, val_ids, block, batch, 4, device=device))
            hist["aux"].append(float(aux.detach()))
            if stats:
                for li, s in enumerate(stats):
                    hist["load_e0"][li].append(float(s["load_fractions"][0]))
                hist["entropy_bits"].append(float(np.mean([s["entropy_bits"] for s in stats])))
                hist["dropped_frac"].append(float(np.mean([s["dropped_frac"] for s in stats])))
            else:
                hist["entropy_bits"].append(float("nan"))
                hist["dropped_frac"].append(0.0)
    return hist


@torch.no_grad()
def domain_routing_fractions(
    model: CharTransformer,
    ids: np.ndarray,
    labels: np.ndarray,
    block: int = 128,
    batch: int = 24,
    n_batches: int = 8,
    seed: int = 7,
    device: str = "cpu",
):
    """Fraction of prose vs code tokens routed (top-1) to each expert, per layer.

    Returns array of shape (n_layers, 2 domains, 2 experts); rows sum to 1.
    """
    rng = np.random.default_rng(seed)
    n_layers = len(model.blocks)
    counts = np.zeros((n_layers, 2, 2), dtype=np.int64)
    model.eval()
    for _ in range(n_batches):
        starts = rng.integers(0, len(ids) - block - 1, size=batch)
        x = torch.from_numpy(np.stack([ids[s : s + block] for s in starts])).to(device)
        lab = np.stack([labels[s : s + block] for s in starts]).reshape(-1)
        _, _, stats = model(x)
        for li, s in enumerate(stats):
            expert = s["top1_expert"].numpy()
            for dom in (PROSE_LABEL, CODE_LABEL):
                sel = lab == dom
                for e in (0, 1):
                    counts[li, dom, e] += int(np.sum(expert[sel] == e))
    fr = counts.astype(np.float64)
    fr /= np.maximum(fr.sum(axis=2, keepdims=True), 1)
    return fr
