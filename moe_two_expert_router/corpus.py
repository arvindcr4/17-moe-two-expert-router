"""Self-written ~40KB char corpus mixing two visibly different sublanguages.

The corpus alternates blocks of English prose with blocks of Python-like code,
and carries a per-character domain label (PROSE_LABEL / CODE_LABEL) so routing
diagnostics can ask "which expert sees prose chars vs code chars?".

Everything is generated procedurally from hand-written sentence/code templates
with a seeded RNG -- no downloads, fully deterministic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

PROSE_LABEL = 0
CODE_LABEL = 1

# ---------------------------------------------------------------- prose bank
_PROSE_SENTENCES = [
    "The river turned silver in the late afternoon light.",
    "She kept a small notebook of questions she could not yet answer.",
    "Every winter the town forgot the smell of rain and had to learn it again.",
    "A good map tells you where you are, but a good story tells you why you left.",
    "The lighthouse keeper counted ships the way other people count sheep.",
    "Nobody remembered who planted the orchard, yet everyone ate the apples.",
    "He walked to the market slowly, rehearsing an apology he never gave.",
    "The library smelled of dust, glue, and patient thought.",
    "In the morning the valley filled with fog like a bowl with milk.",
    "Her grandmother spoke of storms as if they were difficult relatives.",
    "The train was late, and so the whole day arrived a little out of order.",
    "A single crow argued with the wind from the top of the bell tower.",
    "They measured the summer in bicycle rides and borrowed books.",
    "The old bridge creaked, but it had creaked for a hundred years.",
    "By candlelight the letters looked braver than the person writing them.",
    "The gardener claimed the roses listened, and the roses did not object.",
    "Snow erased the road, and for one quiet hour nobody minded.",
    "He collected words the way sailors collect knots, for weather yet to come.",
    "The bakery opened before dawn, and the whole street woke up sweeter.",
    "On clear nights the children traded names for the same three stars.",
    "The clockmaker said that time was patient, but customers were not.",
    "A letter arrived forty years late and was somehow still on time.",
    "The sea returned everything eventually, except the things that mattered.",
    "She painted the door blue so the house would remember the sky.",
    "The professor lost his umbrella twice a week and his temper never.",
    "Autumn negotiated with the maples one leaf at a time.",
    "The choir practiced in the empty church, and the stone walls sang back.",
    "His father taught him to whittle, to whistle, and to wait.",
    "The ferry crossed the strait as if stitching two shores together.",
    "At the end of the lane stood a gate that opened only for the curious.",
]

# ------------------------------------------------------------ code templates
_FN_NAMES = ["scan", "merge", "fold", "score", "clip", "hash_id", "pack", "route",
             "norm", "split_at", "gather", "decode", "window", "bucket"]
_VAR_NAMES = ["x", "y", "acc", "buf", "idx", "tok", "val", "left", "right", "total"]


def _make_function(rng: random.Random) -> str:
    """Emit one small Python-like function from templates."""
    fn = rng.choice(_FN_NAMES)
    a, b = rng.sample(_VAR_NAMES, 2)
    n1, n2 = rng.randint(0, 9), rng.randint(2, 16)
    body_kind = rng.randrange(4)
    if body_kind == 0:
        return (
            f"def {fn}({a}, {b}={n1}):\n"
            f"    acc = 0\n"
            f"    for i in range(len({a})):\n"
            f"        acc += {a}[i] * {b} + {n2}\n"
            f"    return acc % {n2}\n"
        )
    if body_kind == 1:
        return (
            f"def {fn}({a}):\n"
            f"    if not {a}:\n"
            f"        return []\n"
            f"    out = [{a}[0]]\n"
            f"    for v in {a}[1:]:\n"
            f"        out.append(v + {n1})\n"
            f"    return out\n"
        )
    if body_kind == 2:
        return (
            f"def {fn}({a}, {b}):\n"
            f"    lo, hi = min({a}, {b}), max({a}, {b})\n"
            f"    while lo < hi:\n"
            f"        lo, hi = lo + 1, hi - 1\n"
            f"    return (lo + hi) // 2\n"
        )
    return (
        f"def {fn}({a}, {b}=None):\n"
        f"    {b} = {b} or {{}}\n"
        f"    for k, v in {b}.items():\n"
        f"        if v > {n2}:\n"
        f"            {a}[k] = v - {n1}\n"
        f"    return {a}\n"
    )


def _make_prose_block(rng: random.Random, n_sentences: int) -> str:
    return " ".join(rng.choice(_PROSE_SENTENCES) for _ in range(n_sentences)) + "\n"


@dataclass
class CharVocab:
    """Char-level vocabulary with encode/decode."""

    chars: list[str]

    @classmethod
    def from_text(cls, text: str) -> "CharVocab":
        return cls(sorted(set(text)))

    def __len__(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> np.ndarray:
        stoi = {c: i for i, c in enumerate(self.chars)}
        return np.array([stoi[c] for c in text], dtype=np.int64)

    def decode(self, ids) -> str:
        return "".join(self.chars[int(i)] for i in ids)


def build_corpus(target_bytes: int = 40_000, seed: int = 0):
    """Build the mixed corpus.

    Returns (text, labels, vocab): labels is an int array aligned char-for-char
    with text (PROSE_LABEL or CODE_LABEL). Blocks alternate prose/code so both
    domains appear in most training windows' neighborhoods.
    """
    rng = random.Random(seed)
    parts: list[str] = []
    labels: list[np.ndarray] = []
    total = 0
    prose_turn = True
    while total < target_bytes:
        if prose_turn:
            block = _make_prose_block(rng, rng.randint(3, 6))
            lab = PROSE_LABEL
        else:
            block = "".join(_make_function(rng) for _ in range(rng.randint(2, 3)))
            lab = CODE_LABEL
        parts.append(block)
        labels.append(np.full(len(block), lab, dtype=np.int64))
        total += len(block)
        prose_turn = not prose_turn
    text = "".join(parts)
    label_arr = np.concatenate(labels)
    assert len(text) == len(label_arr)
    return text, label_arr, CharVocab.from_text(text)
