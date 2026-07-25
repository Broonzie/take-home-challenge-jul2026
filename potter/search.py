"""Hybrid retrieval: dense semantic search + lexical concordance.

Stage 6. Two retrieval modes because they answer different questions, and shipping
only one would be a worse tool:

  - Dense (bi-encoder, cosine over normalised embeddings) answers "where does
    something like this happen?" It finds a scene from a paraphrase.
  - Lexical KWIC (keyword in context) answers "where exactly is this word used?"
    It is exhaustive and verifiable, which dense retrieval is not.

Fused mode combines them with Reciprocal Rank Fusion. RRF is used rather than score
averaging because cosine similarities and lexical counts are not on a comparable
scale, and RRF only needs the ranks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .corpus import Passage


@dataclass
class Hit:
    passage: Passage
    score: float
    why: str

    def to_dict(self) -> dict:
        return {
            "ref": self.passage.ref,
            "book": self.passage.book,
            "chapter": self.passage.chapter,
            "chapter_title": self.passage.chapter_title,
            "score": round(float(self.score), 4),
            "why": self.why,
            "text": self.passage.text,
        }


def embed_passages(model, passages: list[Passage], batch_size: int = 128) -> np.ndarray:
    """Encode every passage once. Normalised so cosine reduces to a dot product."""
    texts = [p.text for p in passages]
    return np.asarray(
        model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )


def dense_search(model, query: str, embeddings: np.ndarray, passages: list[Passage], k: int = 8) -> list[Hit]:
    q = np.asarray(model.encode([query], normalize_embeddings=True), dtype=np.float32)[0]
    sims = embeddings @ q
    top = np.argsort(-sims)[:k]
    return [Hit(passages[i], float(sims[i]), "semantic") for i in top]


def kwic(term: str, passages: list[Passage], k: int = 8, width: int = 60) -> list[Hit]:
    """Keyword-in-context concordance lines. Case-insensitive, whole-word."""
    pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    hits: list[Hit] = []
    for passage in passages:
        matches = list(pattern.finditer(passage.text))
        if not matches:
            continue
        m = matches[0]
        left = passage.text[max(0, m.start() - width) : m.start()]
        right = passage.text[m.end() : m.end() + width]
        snippet = f"...{left}[{m.group(0)}]{right}..."
        hits.append(Hit(passage, float(len(matches)), snippet))
    hits.sort(key=lambda h: -h.score)
    return hits[:k]


def fused_search(
    model,
    query: str,
    embeddings: np.ndarray,
    passages: list[Passage],
    k: int = 8,
    rrf_k: int = 60,
) -> list[Hit]:
    """Reciprocal Rank Fusion over the dense and lexical rankings."""
    dense = dense_search(model, query, embeddings, passages, k=k * 3)
    # Use the longest query token as the lexical probe.
    tokens = [t for t in re.findall(r"[A-Za-z][A-Za-z'\-]{2,}", query)]
    lexical: list[Hit] = []
    if tokens:
        probe = max(tokens, key=len)
        lexical = kwic(probe, passages, k=k * 3)

    scores: dict[str, float] = {}
    sources: dict[str, set[str]] = {}
    by_ref: dict[str, Passage] = {}
    for ranking, label in ((dense, "semantic"), (lexical, "lexical")):
        for rank, hit in enumerate(ranking):
            ref = hit.passage.ref
            scores[ref] = scores.get(ref, 0.0) + 1.0 / (rrf_k + rank + 1)
            sources.setdefault(ref, set()).add(label)
            by_ref[ref] = hit.passage

    ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
    return [Hit(by_ref[ref], score, "+".join(sorted(sources[ref]))) for ref, score in ordered]
