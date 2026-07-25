"""Lexical statistics and distinctive-vocabulary extraction.

Stage 4. The interesting question is not "which words are common in book 3" - that
is always "the", "and", "said". It is "which words are over-represented in book 3
relative to the rest of the series".

Raw log-odds is unusable here: a word appearing 3 times in one book and 0 elsewhere
gets an enormous score. We use the log-odds ratio with an informative Dirichlet
prior (Monroe, Colaresi & Quinn 2008), where the prior is the corpus-wide frequency
distribution, and divide by the estimated standard deviation. That shrinks
low-count words toward zero, so the ranking surfaces genuinely distinctive
vocabulary rather than rare accidents.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .corpus import Passage

WORD_RE = re.compile(r"[a-z][a-z'\-]*")

FUNCTION_WORDS = {
    "the", "and", "a", "to", "of", "in", "it", "i", "you", "he", "she", "they", "we",
    "was", "is", "were", "are", "be", "been", "had", "has", "have", "that", "this",
    "for", "with", "as", "at", "on", "but", "not", "so", "if", "then", "than", "there",
    "which", "who", "what", "when", "how", "all", "no", "yes", "very", "up", "out",
    "his", "her", "its", "their", "our", "my", "your", "him", "them", "us", "me",
    "said", "did", "do", "would", "could", "should", "will", "shall", "may", "might",
    "one", "two", "now", "here", "from", "by", "or", "an", "any", "some", "more",
    "into", "about", "over", "after", "before", "again", "too", "just", "only", "own",
}


def tokenise(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def basic_stats(passages: list[Passage]) -> dict:
    """Readability-adjacent and lexical-richness measures."""
    tokens: list[str] = []
    sentences = 0
    for p in passages:
        tokens.extend(tokenise(p.text))
        sentences += len(re.findall(r"[.!?]+", p.text)) or 1

    counts = Counter(tokens)
    n = len(tokens)
    if n == 0:
        return {}
    hapax = sum(1 for w, c in counts.items() if c == 1)
    return {
        "tokens": n,
        "types": len(counts),
        "type_token_ratio": len(counts) / n,
        "hapax_legomena": hapax,
        "hapax_pct": hapax / len(counts),
        "sentences": sentences,
        "mean_sentence_len": n / max(sentences, 1),
        "dialogue_share": _dialogue_share(passages),
    }


def _dialogue_share(passages: list[Passage]) -> float:
    """Fraction of characters inside double quotes. A proxy for pace: dialogue-heavy
    stretches read faster than description-heavy ones."""
    total = quoted = 0
    for p in passages:
        total += len(p.text)
        inside = False
        for ch in p.text:
            if ch == '"':
                inside = not inside
            elif inside:
                quoted += 1
    return quoted / total if total else 0.0


def distinctive_terms(
    group_tokens: list[str],
    corpus_counts: Counter,
    corpus_total: int,
    top_n: int = 15,
    min_count: int = 5,
    alpha: float = 0.01,
) -> list[tuple[str, float]]:
    """Log-odds ratio with an informative Dirichlet prior, z-scored.

    Returns (term, z) descending. Positive z = over-represented in this group.
    """
    group_counts = Counter(group_tokens)
    n_group = sum(group_counts.values())
    if n_group == 0 or corpus_total == 0:
        return []

    scores: list[tuple[str, float]] = []
    for word, y_i in group_counts.items():
        if y_i < min_count or word in FUNCTION_WORDS or len(word) < 3:
            continue
        a_i = alpha * corpus_total * (corpus_counts[word] / corpus_total)
        rest = corpus_counts[word] - y_i
        n_rest = corpus_total - n_group
        if rest < 0 or n_rest <= 0:
            continue
        # log-odds in group vs rest, both smoothed by the corpus-derived prior
        num = (y_i + a_i) / (n_group + alpha * corpus_total - y_i - a_i)
        den = (rest + a_i) / (n_rest + alpha * corpus_total - rest - a_i)
        if num <= 0 or den <= 0:
            continue
        delta = math.log(num) - math.log(den)
        var = 1.0 / (y_i + a_i) + 1.0 / (rest + a_i)
        scores.append((word, delta / math.sqrt(var)))

    scores.sort(key=lambda kv: -kv[1])
    return scores[:top_n]


def per_book_distinctive(books_passages: dict[str, list[Passage]], top_n: int = 15) -> dict[str, list[tuple[str, float]]]:
    """Distinctive vocabulary for each book against the rest of the corpus."""
    tokenised = {title: tokenise(" ".join(p.text for p in ps)) for title, ps in books_passages.items()}
    corpus_counts: Counter = Counter()
    for toks in tokenised.values():
        corpus_counts.update(toks)
    corpus_total = sum(corpus_counts.values())
    return {
        title: distinctive_terms(toks, corpus_counts, corpus_total, top_n=top_n)
        for title, toks in tokenised.items()
    }
