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

# Reported-speech verbs. The discriminator for person-vs-place: a castle never
# says anything, so adjacency to one of these is strong evidence of a speaker.
SPEECH_VERBS = {
    "said", "asked", "replied", "shouted", "whispered", "muttered", "yelled", "cried",
    "murmured", "added", "answered", "called", "screamed", "hissed", "growled",
    "snapped", "gasped", "breathed", "told", "repeated", "continued", "agreed",
    "admitted", "announced", "demanded", "explained", "groaned", "grinned", "laughed",
    "sighed", "spoke", "stammered", "suggested", "warned", "sobbed", "roared",
    "barked", "sneered", "squeaked", "bellowed", "insisted", "interrupted", "urged",
    "stuttered", "spluttered", "chuckled", "snarled", "pleaded", "protested",
    "corrected", "confessed", "concluded", "observed", "remarked", "recited",
}
# Deliberately excludes generic action verbs (looked, turned, walked). Including
# them collapses the discriminator: "Gryffindor Tower looked..." reads as a speaker.

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")

_VERBS_ALT = "|".join(sorted(SPEECH_VERBS))
# "Harry said" and "said Harry" - both orders occur in English narration.
NAME_THEN_VERB_RE = re.compile(rf"\b([A-Z][a-z'\-]+)\s+(?:{_VERBS_ALT})\b")
VERB_THEN_NAME_RE = re.compile(rf"\b(?:{_VERBS_ALT})\s+([A-Z][a-z'\-]+)\b")
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


def _person_evidence(passages: list[Passage]) -> Counter:
    """Count, per capitalised head token, how often it sits next to a speech verb."""
    evidence: Counter = Counter()
    for p in passages:
        for match in NAME_THEN_VERB_RE.finditer(p.text):
            evidence[match.group(1).lower()] += 1
        for match in VERB_THEN_NAME_RE.finditer(p.text):
            evidence[match.group(1).lower()] += 1
    return evidence


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


# Apostrophe-suffixes that mark a possessive or a contraction rather than a name.
_POSSESSIVE = {"s", "'s"}
_CONTRACTIONS = {"m", "d", "ll", "re", "ve", "t"}


def _normalise_surface(surface: str) -> str | None:
    """Fold a raw candidate to its canonical form, or reject it outright.

    Three failure modes seen in real wrapped text, all of which produce phantom
    characters if left alone:
      "Harry's"  possessive        -> fold to "Harry"
      "I'm"      contraction       -> reject, it is not a name
      "You-"     hyphen fragment   -> reject, an artefact of de-hyphenating a wrap
    """
    tokens: list[str] = []
    for tok in surface.split():
        if tok.startswith("-") or tok.endswith("-"):
            return None
        if "'" in tok:
            head, _, tail = tok.partition("'")
            tail_l = tail.lower()
            if tail_l in _CONTRACTIONS:
                return None
            if tail_l in _POSSESSIVE:
                tok = head
        if not tok:
            return None
        tokens.append(tok)
    return " ".join(tokens) if tokens else None


def extract_characters(
    passages: list[Passage],
    min_count: int = 8,
    lowercase_ratio: float = 0.25,
    fuzz_threshold: int = 88,
    top_n: int = 60,
    min_person_evidence: int = 2,
) -> list[Character]:
    """Extract and resolve character identities from passages.

    `lowercase_ratio` is the key filter: if a candidate's word appears lowercased
    more than this fraction of the time, it is ordinary vocabulary that happened to
    start a sentence, not a name.
    """
    lower = _lowercase_profile(passages)
    raw = _candidate_counts(passages)
    evidence = _person_evidence(passages)

    # --- (b) suppression -------------------------------------------------------
    surface_counts: Counter = Counter()
    honorific_bearing: set[str] = set()
    for surface, count in raw.items():
        stripped, had_honorific = _strip_honorific(surface)
        if not stripped:
            continue
        normalised = _normalise_surface(stripped)
        if not normalised:
            continue
        stripped = normalised
        head = stripped.split()[0].lower()
        if head in STRUCTURAL_STOPS or len(head) < 3:
            continue
        # Single-token candidates face the lowercase test; multi-token names
        # (e.g. "Aunt Em") are already strong evidence of a named entity.
        if len(stripped.split()) == 1:
            lc = lower.get(head, 0)
            if lc and lc / (lc + count) > lowercase_ratio:
                continue
        if had_honorific:
            honorific_bearing.add(stripped)
        surface_counts[stripped] += count

    # --- (b2) person-vs-place ---------------------------------------------------
    # An honorific already proves personhood. Otherwise require the name to have
    # been observed next to a speech or action verb at least twice, which drops
    # institutions and locations ("Hogwarts", "Gryffindor") without a gazetteer.
    def _is_person(surface: str) -> bool:
        if surface in honorific_bearing:
            return True
        return evidence.get(surface.split()[0].lower(), 0) >= min_person_evidence

    candidates = [
        s for s, c in surface_counts.items() if c >= min_count and _is_person(s)
    ]
    if not candidates:
        return []

    # --- (c) alias clustering --------------------------------------------------
    # A shared token is NOT enough to merge two names. "Fred Weasley" and
    # "Ron Weasley" share a surname and are different people, so naive containment
    # collapses an entire family into one identity. We merge only when the evidence
    # points at one person:
    #   - same given name      "Harry" + "Harry Potter"
    #   - unambiguous surname  "Granger" + "Hermione Granger", but only because no
    #                          other full name in the corpus ends in Granger
    multi_token = [c for c in candidates if len(c.split()) > 1]
    surname_owners: dict[str, set[str]] = defaultdict(set)
    for full in multi_token:
        for tok in full.lower().split()[1:]:
            surname_owners[tok].add(full)

    def _mergeable(a: str, b: str) -> bool:
        a_toks, b_toks = a.lower().split(), b.lower().split()
        if a_toks[0] == b_toks[0]:
            return True
        shorter, longer = (a_toks, b_toks) if len(a_toks) <= len(b_toks) else (b_toks, a_toks)
        if len(shorter) == 1 and shorter[0] in longer[1:]:
            # Only if this surname belongs to exactly one full name.
            return len(surname_owners.get(shorter[0], set())) == 1
        return False

    uf = _UnionFind()
    # Longer surfaces first so the fullest form tends to become the cluster root.
    ordered = sorted(candidates, key=lambda s: (-len(s.split()), -surface_counts[s]))
    for i, a in enumerate(ordered):
        uf.find(a)
        for b in ordered[i + 1 :]:
            if _mergeable(a, b):
                uf.union(a, b)
                continue
            # Fuzzy match catches spelling variants, but still requires the given
            # name to agree so it cannot re-introduce the family-merge bug.
            if (
                fuzz.token_set_ratio(a, b) >= fuzz_threshold
                and a.lower().split()[0] == b.lower().split()[0]
            ):
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
        stripped = _normalise_surface(stripped) or ""
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
