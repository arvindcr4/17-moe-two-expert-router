"""Run all MoE router experiments and write plots into plots/.

Ten training runs, executed sequentially in one process with 2 torch threads
(each independently seeded, so results are deterministic; ~3-4 min on CPU):

    moe_top1_aux      -- headline MoE: top-1 (Switch), cf 1.5, aux 0.01
    moe_top1_noaux    -- ablation: aux weight 0 -> router collapse
    moe_top1_aux100x  -- break-it: aux weight 100x (1.0) -> balance != quality
    moe_top2_aux      -- top-2 routing for the val-loss comparison
    dense             -- dense baseline with matched ACTIVE params
    cf025/05/10/20    -- capacity-factor sweep (0.25 is the break-it point)
    collapse_forced   -- break-it: aux weight -0.1 (rewards imbalance), cf 2.0
                         so nothing is dropped -> total router collapse

Usage: python demo.py
"""

from __future__ import annotations

import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from moe_two_expert_router import build_corpus, CharTransformer
from moe_two_expert_router.train import train_model, domain_routing_fractions, eval_loss

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, "plots")

STEPS = 600
BLOCK = 64
BATCH = 16
SEED = 0

# --- palette (dataviz reference palette, light mode; first 3 slots) ---------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"

CONFIGS = {
    "moe_top1_aux": dict(use_moe=True, k=1, capacity_factor=1.5, aux_weight=0.01),
    "moe_top1_noaux": dict(use_moe=True, k=1, capacity_factor=1.5, aux_weight=0.0),
    "moe_top1_aux100x": dict(use_moe=True, k=1, capacity_factor=1.5, aux_weight=1.0),
    "moe_top2_aux": dict(use_moe=True, k=2, capacity_factor=1.5, aux_weight=0.01),
    "dense": dict(use_moe=False, k=1, capacity_factor=1.5, aux_weight=0.0),
    "cf025": dict(use_moe=True, k=1, capacity_factor=0.25, aux_weight=0.01),
    "cf05": dict(use_moe=True, k=1, capacity_factor=0.5, aux_weight=0.01),
    "cf10": dict(use_moe=True, k=1, capacity_factor=1.0, aux_weight=0.01),
    "cf20": dict(use_moe=True, k=1, capacity_factor=2.0, aux_weight=0.01),
    "collapse_forced": dict(use_moe=True, k=1, capacity_factor=2.0, aux_weight=-0.1),
}


def run_experiment(name: str):
    """Train one configuration; returns a results dict."""
    cfg = CONFIGS[name]
    text, labels, vocab = build_corpus(seed=0)
    ids = vocab.encode(text)
    split = int(0.9 * len(ids))
    train_ids, val_ids = ids[:split], ids[split:]
    val_labels = labels[split:]

    torch.manual_seed(SEED)
    model = CharTransformer(
        len(vocab),
        d_model=96,
        n_layers=2,
        n_heads=4,
        d_ff=192,
        max_seq=BLOCK,
        use_moe=cfg["use_moe"],
        k=cfg["k"],
        capacity_factor=cfg["capacity_factor"],
    )
    t0 = time.time()
    hist = train_model(
        model, train_ids, val_ids,
        steps=STEPS, block=BLOCK, batch=BATCH,
        aux_weight=cfg["aux_weight"], eval_every=20, seed=SEED,
    )
    out = {
        "name": name,
        "hist": hist,
        "seconds": time.time() - t0,
        "final_val": eval_loss(model, val_ids, BLOCK, BATCH, 8, seed=999),
        "n_params": sum(p.numel() for p in model.parameters()),
        "active_ffn": model.active_ffn_params_per_token(),
    }
    if cfg["use_moe"]:
        out["domain_fractions"] = domain_routing_fractions(
            model, val_ids, val_labels, block=BLOCK, batch=BATCH, n_batches=8
        ).tolist()
        out["final_dropped"] = float(np.mean(hist["dropped_frac"][-5:]))
    return out


# ------------------------------------------------------------------ plotting
def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _save(fig, fname):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    path = os.path.join(PLOTS, fname)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("wrote", path)


def plot_load_fractions(res):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, key, title in [
        (axes[0], "moe_top1_aux", "WITH aux loss (0.01)"),
        (axes[1], "moe_top1_noaux", "WITHOUT aux loss (0.0)"),
    ]:
        h = res[key]["hist"]
        ax.axhline(0.5, color=INK2, linewidth=1, linestyle="--", zorder=1)
        ax.plot(h["step"], h["load_e0"][0], color=BLUE, linewidth=1.8, label="layer 1", zorder=3)
        ax.plot(h["step"], h["load_e0"][1], color=ORANGE, linewidth=1.8, label="layer 2", zorder=3)
        _style(ax, title, "training step", "fraction of tokens -> expert 0")
        ax.set_ylim(-0.03, 1.03)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle("Expert load over training: top-1 routing, 2 experts", color=INK, fontsize=12, x=0.01, ha="left")
    fig.subplots_adjust(top=0.82)
    _save(fig, "load_fractions_aux_vs_noaux.png")


def plot_entropy(res):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axhline(1.0, color=INK2, linewidth=1, linestyle="--", zorder=1)
    ax.text(res["moe_top1_aux"]["hist"]["step"][-1], 1.01, "uniform gate (max)", color=INK2,
            fontsize=7, ha="right", va="bottom")
    for key, color, label in [
        ("moe_top1_aux", BLUE, "with aux loss"),
        ("moe_top1_noaux", ORANGE, "without aux loss"),
    ]:
        h = res[key]["hist"]
        ax.plot(h["step"], h["entropy_bits"], color=color, linewidth=1.8, label=label, zorder=3)
    _style(ax, "Router entropy over training (mean over tokens & layers)", "training step", "gate entropy (bits)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    _save(fig, "router_entropy.png")


def plot_specialization(res):
    fr = np.array(res["moe_top1_aux"]["domain_fractions"])  # (layers, dom, expert)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    for li, ax in enumerate(axes):
        m = fr[li]
        im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{m[r, c]:.2f}", ha="center", va="center",
                        color=INK if m[r, c] < 0.6 else SURFACE, fontsize=11)
        ax.set_xticks([0, 1], ["expert 0", "expert 1"], color=INK2, fontsize=9)
        ax.set_yticks([0, 1], ["prose", "code"], color=INK2, fontsize=9)
        ax.set_title(f"layer {li + 1}", color=INK, fontsize=10, loc="left")
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Where does each domain go? Fraction of tokens routed per expert (rows sum to 1)",
                 color=INK, fontsize=11, x=0.02, ha="left")
    fig.colorbar(im, ax=axes, fraction=0.04, pad=0.03).ax.tick_params(colors=INK2, labelsize=7)
    fig.patch.set_facecolor(SURFACE)
    path = os.path.join(PLOTS, "specialization_heatmap.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def plot_val_loss(res):
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for key, color, label in [
        ("moe_top1_aux", BLUE, "MoE top-1, aux on"),
        ("moe_top2_aux", AQUA, "MoE top-2, aux on"),
        ("dense", ORANGE, "dense (matched active params)"),
    ]:
        h = res[key]["hist"]
        ax.plot(h["step"], h["val_loss"], color=color, linewidth=1.8, label=label, zorder=3)
        ax.annotate(f'{res[key]["final_val"]:.3f}', (h["step"][-1], h["val_loss"][-1]),
                    textcoords="offset points", xytext=(4, 0), color=INK2, fontsize=7)
    _style(ax, "Validation loss: 2-expert MoE vs dense with matched active FFN params",
           "training step", "val cross-entropy (nats/char)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    _save(fig, "val_loss_moe_vs_dense.png")


def plot_capacity_sweep(res):
    cfs = [0.25, 0.5, 1.0, 1.5, 2.0]
    keys = ["cf025", "cf05", "cf10", "moe_top1_aux", "cf20"]
    dropped = [100 * res[k]["final_dropped"] for k in keys]
    vals = [res[k]["final_val"] for k in keys]
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.4), sharex=True)
    axes[0].plot(cfs, dropped, color=BLUE, linewidth=1.8, marker="o", markersize=5, zorder=3)
    for x, y in zip(cfs, dropped):
        axes[0].annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 6),
                         ha="center", color=INK2, fontsize=8)
    _style(axes[0], "Dropped-token share vs capacity factor (top-1, aux on)",
           "", "dropped assignments (%)")
    axes[1].plot(cfs, vals, color=BLUE, linewidth=1.8, marker="o", markersize=5, zorder=3)
    for x, y in zip(cfs, vals):
        axes[1].annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 6),
                         ha="center", color=INK2, fontsize=8)
    _style(axes[1], "Final validation loss vs capacity factor",
           "capacity factor", "val cross-entropy")
    axes[1].annotate("break-it: cf 0.25", (0.25, vals[0]), textcoords="offset points",
                     xytext=(8, -12), color=INK2, fontsize=8)
    _save(fig, "capacity_sweep.png")


def plot_breakit_aux100x(res):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax = axes[0]
    ax.axhline(0.5, color=INK2, linewidth=1, linestyle="--", zorder=1)
    for key, color, label in [("moe_top1_aux", BLUE, "aux 0.01"),
                              ("moe_top1_aux100x", ORANGE, "aux 1.00 (100x)")]:
        h = res[key]["hist"]
        ax.plot(h["step"], h["load_e0"][0], color=color, linewidth=1.8, label=label, zorder=3)
        ax.plot(h["step"], h["load_e0"][1], color=color, linewidth=1.2, alpha=0.55, zorder=3)
    _style(ax, "Balance: expert-0 load (bold: layer 1, thin: layer 2)",
           "training step", "fraction of tokens -> expert 0")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)

    ax = axes[1]
    for key, color, label in [("moe_top1_aux", BLUE, "aux 0.01"),
                              ("moe_top1_aux100x", ORANGE, "aux 1.00 (100x)")]:
        h = res[key]["hist"]
        ax.plot(h["step"], h["val_loss"], color=color, linewidth=1.8, label=label, zorder=3)
    _style(ax, "Validation loss", "training step", "val cross-entropy")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle("Break-it: 100x aux weight (1.0) vs default (0.01)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.subplots_adjust(top=0.82)
    _save(fig, "breakit_aux100x.png")


def plot_breakit_cf025(res):
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for key, color, label in [
        ("moe_top1_aux", BLUE, f'cf 1.5 (drops {100 * res["moe_top1_aux"]["final_dropped"]:.0f}%)'),
        ("cf025", ORANGE, f'cf 0.25 (drops {100 * res["cf025"]["final_dropped"]:.0f}%)'),
    ]:
        h = res[key]["hist"]
        ax.plot(h["step"], h["val_loss"], color=color, linewidth=1.8, label=label, zorder=3)
        ax.annotate(f'{res[key]["final_val"]:.3f}', (h["step"][-1], h["val_loss"][-1]),
                    textcoords="offset points", xytext=(4, 0), color=INK2, fontsize=7)
    _style(ax, "Break-it: capacity 0.25 starves the experts (top-1, aux on)",
           "training step", "val cross-entropy")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    _save(fig, "breakit_capacity_025.png")


def plot_breakit_collapse(res):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax = axes[0]
    ax.axhline(0.5, color=INK2, linewidth=1, linestyle="--", zorder=1)
    for key, color, label in [("moe_top1_aux", BLUE, "aux +0.01"),
                              ("moe_top1_noaux", AQUA, "aux 0"),
                              ("collapse_forced", ORANGE, "aux -0.1 (forced)")]:
        h = res[key]["hist"]
        ax.plot(h["step"], h["load_e0"][0], color=color, linewidth=1.8, label=label, zorder=3)
        ax.plot(h["step"], h["load_e0"][1], color=color, linewidth=1.2, alpha=0.55, zorder=3)
    _style(ax, "Expert-0 load (bold: layer 1, thin: layer 2)",
           "training step", "fraction of tokens -> expert 0")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)

    ax = axes[1]
    for key, color, label in [("moe_top1_aux", BLUE, "MoE aux +0.01"),
                              ("collapse_forced", ORANGE, "MoE collapsed"),
                              ("dense", INK2, "dense baseline")]:
        h = res[key]["hist"]
        ax.plot(h["step"], h["val_loss"], color=color, linewidth=1.8,
                label=f'{label} (final {res[key]["final_val"]:.3f})', zorder=3)
    _style(ax, "A collapsed MoE is just a dense model", "training step", "val cross-entropy")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle("Break-it: router collapse on purpose (negative aux weight)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.subplots_adjust(top=0.82)
    _save(fig, "breakit_router_collapse.png")


def main():
    torch.set_num_threads(2)
    os.makedirs(PLOTS, exist_ok=True)
    t0 = time.time()
    names = list(CONFIGS)
    results = {}
    for name in names:
        results[name] = run_experiment(name)
        print(f"{name:18s} val {results[name]['final_val']:.4f}  ({results[name]['seconds']:.0f}s)", flush=True)
    wall = time.time() - t0
    print(f"\nall {len(names)} runs done in {wall:.0f}s wall "
          f"({sum(r['seconds'] for r in results.values()):.0f}s cpu-run total)")

    plot_load_fractions(results)
    plot_entropy(results)
    plot_specialization(results)
    plot_val_loss(results)
    plot_capacity_sweep(results)
    plot_breakit_aux100x(results)
    plot_breakit_cf025(results)
    plot_breakit_collapse(results)

    summary = {
        "wall_seconds": round(wall, 1),
        "runs": {
            name: {
                "final_val": round(r["final_val"], 4),
                "final_dropped_pct": round(100 * r.get("final_dropped", 0.0), 1),
                "final_load_e0_per_layer": [round(l[-1], 3) for l in r["hist"]["load_e0"]] if r["hist"]["load_e0"][0] else None,
                "final_entropy_bits": (round(r["hist"]["entropy_bits"][-1], 3)
                                       if not np.isnan(r["hist"]["entropy_bits"][-1]) else None),
                "n_params": r["n_params"],
                "active_ffn_per_token": r["active_ffn"],
                "seconds": round(r["seconds"], 1),
            }
            for name, r in results.items()
        },
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
