"""Two-expert Mixture-of-Experts layer + router, from scratch.

Modules:
    corpus  -- procedural two-sublanguage char corpus (English prose vs Python-like code)
    router  -- softmax gate, top-1/top-2 assignment, capacity dropping, aux load-balancing loss
    moe     -- MoELayer (dispatch/combine, overflow accounting) and expert FFNs
    model   -- tiny 2-layer char transformer with MoE or dense FFN blocks
    train   -- training loop with routing diagnostics
"""

from .corpus import PROSE_LABEL, CODE_LABEL, build_corpus, CharVocab
from .router import (
    Router,
    top_k_assignment,
    apply_capacity,
    load_balancing_loss,
    router_entropy_bits,
)
from .moe import ExpertFFN, MoELayer, DenseFFN
from .model import CharTransformer
from .train import train_model, domain_routing_fractions

__all__ = [
    "PROSE_LABEL",
    "CODE_LABEL",
    "build_corpus",
    "CharVocab",
    "Router",
    "top_k_assignment",
    "apply_capacity",
    "load_balancing_loss",
    "router_entropy_bits",
    "ExpertFFN",
    "MoELayer",
    "DenseFFN",
    "CharTransformer",
    "train_model",
    "domain_routing_fractions",
]
