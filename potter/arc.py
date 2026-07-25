"""Narrative arc extraction via semantic axis projection.

Stage 5, and the least obvious method here, so the rationale matters.

The usual way to plot a story's emotional arc is a sentiment lexicon (VADER,
AFINN). On fiction that performs badly: it is tuned for product reviews, it has no
entry for most of an invented world's vocabulary, and it scores "he drew his wand"
as neutral.

Instead we define an axis in embedding space. Embed a set of positive-pole probe
sentences and a set of negative-pole probes, take the difference of their centroids,
and project each narrative window onto that unit vector. This gives a continuous
score that is:

  - lexicon-free, so invented vocabulary still lands somewhere sensible;
  - defined by intent - you choose the poles, so the same machinery yields a
    tension axis, a wonder axis, or a domesticity axis by swapping probes;
  - cheap once embeddings are cached, which they are (stage 6 shares them).

Caveat, stated honestly: this measures semantic similarity to the poles, not
authorial sentiment. It is a directional signal, not a calibrated emotion score.
"""

from __future__ import annotations

import numpy as np

AXES: dict[str, tuple[list[str], list[str]]] = {
    "tension": (
        [
            "a sudden danger, and everyone was afraid for their lives",
            "a desperate fight against a terrible enemy",
            "darkness, threat, and the fear of what was coming",
            "they were trapped with no way out and no hope of rescue",
        ],
        [
            "a calm, safe afternoon among friends",
            "a warm meal and comfortable conversation at home",
            "peaceful rest with nothing to worry about",
            "a happy celebration full of laughter",
        ],
    ),
    "wonder": (
        [
            "an astonishing magical marvel none of them had ever seen",
            "a strange and beautiful enchantment beyond understanding",
            "impossible creatures and glittering, otherworldly places",
        ],
        [
            "an ordinary, dull errand of no interest",
            "plain practical chores and everyday routine",
            "a mundane conversation about nothing in particular",
        ],
    ),
    "companionship": (
        [
            "loyal friends standing together and helping one another",
            "warmth, trust and affection between companions",
            "a kind promise never to abandon a friend",
        ],
        [
            "bitter isolation, entirely alone and forgotten",
            "cold suspicion and betrayal between enemies",
            "a lonely journey with nobody to turn to",
        ],
    ),
}


def build_axis(model, axis: str) -> np.ndarray:
    """Unit vector pointing from the negative pole to the positive pole."""
    if axis not in AXES:
        raise KeyError(f"Unknown axis '{axis}'. Available: {', '.join(AXES)}")
    pos, neg = AXES[axis]
    pos_vecs = model.encode(pos, normalize_embeddings=True)
    neg_vecs = model.encode(neg, normalize_embeddings=True)
    direction = np.asarray(pos_vecs).mean(axis=0) - np.asarray(neg_vecs).mean(axis=0)
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("Degenerate axis: poles have identical centroids.")
    return direction / norm


def project(embeddings: np.ndarray, axis_vec: np.ndarray) -> np.ndarray:
    """Project unit-normalised embeddings onto the axis, then z-score.

    Z-scoring is what makes the output readable: absolute cosine values against an
    axis sit in a narrow band around zero, so relative movement is the signal.
    """
    raw = np.asarray(embeddings) @ axis_vec
    std = raw.std()
    return (raw - raw.mean()) / std if std > 0 else raw - raw.mean()


def smooth(values: np.ndarray, window: int = 9) -> np.ndarray:
    """Centred moving average. Paragraph-level scores are far too noisy to read as
    an arc; the arc lives at scene scale."""
    values = np.asarray(values, dtype=float)
    if len(values) < window or window < 2:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def sparkline(values: np.ndarray, width: int = 60) -> str:
    """Render a series as a unicode sparkline for terminal output."""
    blocks = "▁▂▃▄▅▆▇█"
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return ""
    # Resample to the requested width by bucket mean.
    if len(values) > width:
        idx = np.array_split(np.arange(len(values)), width)
        values = np.array([values[i].mean() for i in idx if len(i)])
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-9:
        return blocks[len(blocks) // 2] * len(values)
    scaled = (values - lo) / (hi - lo) * (len(blocks) - 1)
    return "".join(blocks[int(round(v))] for v in scaled)
