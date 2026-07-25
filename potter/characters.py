"""Unsupervised character extraction and alias resolution.

Stage 2. Deliberately does NOT use a pretrained NER model. Two reasons:

1. Portability. A gazetteer-free, model-free extractor runs on any corpus with no
   downloads, which matters for a reviewer cloning this repo cold.
2. Fiction breaks off-the-shelf NER anyway. Invented names ("Ozma", "Hermione")
   are out-of-vocabulary, and NER gives you no alias resolution - it will happily
   emit Ron, Ronald and Weasley as three unrelated people.

Method:
  a. Candidate generation - runs of capitalised tokens, with honorifics attached.
  b. Common-word suppression - a real name is almost never seen lowercased. We use
     the corpus itself as its own stopword list, which is language- and
     domain-independent.
  c. Alias clustering - fuzzy match (rapidfuzz) plus token containment, unioned
     into identities via union-find, then labelled by the most frequent surface form.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from .corpus import Passage

HONORIFICS = {
    "mr", "mrs", "miss", "ms", "dr", "professor", "prof", "sir", "madam", "madame",
    "lord", "lady", "aunt", "uncle", "captain", "colonel", "general", "king", "queen",
    "prince", "princess", "master", "mistress", "saint", "st",
}

# Capitalised tokens that are structural or temporal rather than personal.
STRUCTURAL_STOPS = {
    "chapter", "book", "volume", "part", "the", "and", "but", "for", "yet", "so",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "god", "heaven", "earth", "christmas", "english", "america", "american",
    "i", "you", "he", "she", "it", "we", "they", "there", "then", "when", "what",
    "why", "how", "who", "oh", "ah", "well", "yes", "no", "now", "if", "as", "at",
    "on", "in", "to", "of", "by", "up", "or", "an", "a", "his", "her", "my", "your",
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
# A capitalised run, optionally linked by lowercase particles (van, de, of, the).
NAME_RUN_RE = re.compile(
    r"\b((?:[A-Z][a-z'\-]+)(?:\s+(?:of|the|de|van|von|le|la)\s+[A-Z][a-z'\-]+|\s+[A-Z][a-z'\-]+){0,3})\b"
)


@dataclass
class Character:
    """A resolved identity: one person, many surface forms."""

    name: str
    aliases: set[str] = field(default_factory=set)
    count: int = 0

    def to_dict(self) -> dict:
        return {"name": self.name, "aliases": sorted(self.aliases), "count": self.count}


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _lowercase_profile(passages: list[Passage]) -> Counter:
    """How often each word appears lowercased. Our corpus-derived stopword signal."""
    lower = Counter()
    for p in passages:
        for tok in TOKEN_RE.findall(p.text):
            if tok[0].islower():
                lower[tok.lower()] += 1
    return lower


def _candidate_counts(passages: list[Passage]) -> Counter:
    counts = Counter()
    for p in passages:
        for run in NAME_RUN_RE.findall(p.text):
            counts[run.strip()] += 1
    return counts


def _strip_honorific(surface: str) -> tuple[str, bool]:
    parts = surface.split()
    had = False
    while parts and parts[0].lower().strip(".") in HONORIFICS:
        parts = parts[1:]
        had = True
    return " ".join(parts), had


def extract_characters(
    passages: list[Passage],
    min_count: int = 8,
    lowercase_ratio: float = 0.25,
    fuzz_threshold: int = 88,
    top_n: int = 60,
) -> list[Character]:
    """Extract and resolve character identities from passages.

    `lowercase_ratio` is the key filter: if a candidate's word appears lowercased
    more than this fraction of the time, it is ordinary vocabulary that happened to
    start a sentence, not a name.
    """
    lower = _lowercase_profile(passages)
    raw = _candidate_counts(passages)

    # --- (b) suppression -------------------------------------------------------
    surface_counts: Counter = Counter()
    for surface, count in raw.items():
        stripped, _ = _strip_honorific(surface)
        if not stripped:
            continue
        head = stripped.split()[0].lower()
        if head in STRUCTURAL_STOPS or len(head) < 3:
            continue
        # Single-token candidates face the lowercase test; multi-token names
        # (e.g. "Aunt Em") are already strong evidence of a named entity.
        if len(stripped.split()) == 1:
            lc = lower.get(head, 0)
            if lc and lc / (lc + count) > lowercase_ratio:
                continue
        surface_counts[stripped] += count

    candidates = [s for s, c in surface_counts.items() if c >= min_count]
    if not candidates:
        return []

    # --- (c) alias clustering --------------------------------------------------
    uf = _UnionFind()
    # Longer surfaces first so "Dorothy Gale" tends to become the cluster root.
    ordered = sorted(candidates, key=lambda s: (-len(s.split()), -surface_counts[s]))
    for i, a in enumerate(ordered):
        uf.find(a)
        a_tokens = set(a.lower().split())
        for b in ordered[i + 1 :]:
            b_tokens = set(b.lower().split())
            # Containment: "Dorothy" inside "Dorothy Gale".
            if a_tokens & b_tokens and (a_tokens <= b_tokens or b_tokens <= a_tokens):
                uf.union(a, b)
                continue
            # Fuzzy: spelling variants and transcription noise.
            if fuzz.token_set_ratio(a, b) >= fuzz_threshold:
                uf.union(a, b)

    clusters: dict[str, list[str]] = defaultdict(list)
    for s in candidates:
        clusters[uf.find(s)].append(s)

    characters: list[Character] = []
    for members in clusters.values():
        total = sum(surface_counts[m] for m in members)
        # Label with the most frequent surface form - that is what readers call them.
        label = max(members, key=lambda m: (surface_counts[m], -len(m)))
        characters.append(Character(name=label, aliases=set(members), count=total))

    characters.sort(key=lambda c: -c.count)
    return characters[:top_n]


def build_alias_index(characters: list[Character]) -> dict[str, str]:
    """Map every surface form (lowercased) to its canonical character name."""
    index: dict[str, str] = {}
    for ch in characters:
        for alias in ch.aliases:
            index[alias.lower()] = ch.name
    return index


def find_mentions(text: str, alias_index: dict[str, str]) -> set[str]:
    """Canonical characters mentioned in a span of text."""
    hits: set[str] = set()
    for run in NAME_RUN_RE.findall(text):
        stripped, _ = _strip_honorific(run.strip())
        if not stripped:
            continue
        canon = alias_index.get(stripped.lower())
        if canon:
            hits.add(canon)
            continue
        # Fall back to the head token ("Dorothy" from "Dorothy's").
        head = stripped.split()[0]
        canon = alias_index.get(head.lower())
        if canon:
            hits.add(canon)
    return hits
