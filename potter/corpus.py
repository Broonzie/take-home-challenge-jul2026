"""Corpus ingest and structural segmentation.

Stage 1 of the pipeline. Turns plain text into a flat list of addressable `Passage`
records (book -> chapter -> paragraph). Everything downstream operates on passages,
which is what keeps the rest of the pipeline corpus-agnostic: no module below this
one knows or cares which novels were loaded.

Three input shapes are handled, because real corpora arrive in all three:
  1. one file per book               (filename order = series order)
  2. one file containing a series    (book boundary = chapter-ordinal reset)
  3. one file, one book              (no reset, single book)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DERIVED_DIR = ROOT / "data" / "derived"

# --- chapter ordinals ------------------------------------------------------------
_UNITS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50}
_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def parse_ordinal(token: str) -> int | None:
    """Parse '7', 'VII', 'Seven' or 'THIRTY-EIGHT' into an integer."""
    t = token.strip().lower().strip(".")
    if not t:
        return None
    if t.isdigit():
        return int(t)
    if t in _UNITS:
        return _UNITS[t]
    if t in _TENS:
        return _TENS[t]
    if "-" in t or " " in t:
        parts = re.split(r"[-\s]+", t)
        if len(parts) == 2 and parts[0] in _TENS and parts[1] in _UNITS:
            return _TENS[parts[0]] + _UNITS[parts[1]]
        return None
    if all(c in _ROMAN for c in t):
        total = 0
        prev = 0
        for c in reversed(t):
            val = _ROMAN[c]
            total += -val if val < prev else val
            prev = max(prev, val)
        return total or None
    return None


# Heading forms differ between editions, so try several. Matched on a whole line.
CHAPTER_PATTERNS = [
    re.compile(r"^\s*chapter\s+([A-Za-z\-]+|[IVXLCDM]+|\d{1,3})\b\.?\s*(.*)$", re.IGNORECASE),
    re.compile(r"^\s*([IVXLCDM]{1,7})\.\s*(.*)$"),
    re.compile(r"^\s*(\d{1,3})\.\s+([A-Z].*)$"),
]


@dataclass
class Passage:
    """One paragraph, addressable within the corpus."""

    book: str
    book_index: int
    chapter: int
    chapter_title: str
    para_index: int
    text: str

    @property
    def ref(self) -> str:
        return f"{self.book_index}.{self.chapter}.{self.para_index}"

    @property
    def n_words(self) -> int:
        return len(self.text.split())


@dataclass
class Book:
    title: str
    index: int
    chapters: list[tuple[int, str]] = field(default_factory=list)
    passages: list[Passage] = field(default_factory=list)


@dataclass
class _Unit:
    """A chapter's worth of text, before books are assembled."""

    ordinal: int | None
    heading: str
    title: str
    paragraphs: list[str] = field(default_factory=list)


# --- normalisation ---------------------------------------------------------------

_PARA = "\x00P\x00"


def _unwrap(text: str) -> str:
    """Rejoin hard-wrapped lines into true paragraphs.

    Getting this wrong silently destroys the paragraph structure that every
    downstream stage depends on, and plain-text editions disagree on convention.
    Three cases are handled, in two steps.

    Step 1 - normalise double-spaced files. Some exports (common from Project
    Gutenberg) put a blank line between every WRAPPED line, using two or more blank
    lines for a real paragraph break. Collapse those to the conventional form.

    Step 2 - recover paragraphs from hard wrapping. Many editions do not mark every
    paragraph with a blank line, and use no indentation either. But in hard-wrapped
    text the geometry gives it away: lines inside a paragraph all run close to the
    wrap width, and only the LAST line of a paragraph falls short. So a short line
    marks a paragraph end. This is only applied when the file actually looks
    hard-wrapped, so files with one long line per paragraph pass through untouched.
    """
    para_breaks = len(re.findall(r"\n{3,}", text))
    soft_singles = len(re.findall(r"(?<!\n)\n(?!\n)", text))
    if para_breaks >= 10 and soft_singles <= para_breaks:
        text = re.sub(r"\n{3,}", _PARA, text)
        text = text.replace("\n\n", "\n")
        text = text.replace(_PARA, "\n\n")

    lines = text.split("\n")
    widths = sorted(len(ln.rstrip()) for ln in lines if ln.strip())
    if not widths:
        return text
    # 90th-percentile width approximates the wrap column without being skewed by
    # the many short paragraph-final lines.
    wrap_width = widths[int(len(widths) * 0.9)]
    hard_wrapped = 40 <= wrap_width <= 110
    threshold = wrap_width * 0.75

    paragraphs: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            paragraphs.append(re.sub(r"\s+", " ", " ".join(buffer)).strip())
            buffer.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        buffer.append(stripped)
        if hard_wrapped and len(stripped) < threshold:
            flush()
    flush()

    return "\n\n".join(p for p in paragraphs if p)


def _normalise(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"[–—]", " - ", text)
    return _unwrap(text)


def _clean_title(path: Path) -> str:
    stem = re.sub(r"^\d+[-_ ]*", "", path.stem)
    return stem.replace("-", " ").replace("_", " ").strip().title()


def _match_chapter(line: str, patterns: list[re.Pattern] | None = None) -> tuple[int | None, str, str] | None:
    """If this line is a chapter heading, return (ordinal, heading, trailing title)."""
    stripped = line.strip()
    if not stripped or len(stripped) > 90:
        return None
    for pat in patterns or CHAPTER_PATTERNS:
        m = pat.match(stripped)
        if m:
            ordinal = parse_ordinal(m.group(1))
            if ordinal is None:
                continue
            return ordinal, stripped, (m.group(2) or "").strip(" .")
    return None


def _select_chapter_pattern(blocks: list[str]) -> list[re.Pattern]:
    """Pick the single heading form this file actually uses.

    Trying all patterns at once is how you get false positives: a numbered list or a
    line of dialogue beginning "I." reads as a chapter heading and, worse, as a
    chapter-ordinal reset that invents a whole extra book. An edition uses one
    heading convention, so identify the dominant one and trust only it.
    """
    counts: list[int] = []
    for pat in CHAPTER_PATTERNS:
        n = 0
        for block in blocks:
            stripped = block.strip()
            if not stripped or len(stripped) > 90:
                continue
            m = pat.match(stripped)
            if m and parse_ordinal(m.group(1)) is not None:
                n += 1
        counts.append(n)
    best = max(range(len(counts)), key=lambda i: counts[i])
    return [CHAPTER_PATTERNS[best]] if counts[best] >= 5 else CHAPTER_PATTERNS


# --- parsing ---------------------------------------------------------------------


def _parse_units(text: str) -> list[_Unit]:
    """Split normalised text into chapter units, preserving front matter."""
    blocks = text.split("\n\n")
    patterns = _select_chapter_pattern(blocks)
    units: list[_Unit] = [_Unit(ordinal=None, heading="(front matter)", title="(front matter)")]
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        matched = _match_chapter(block, patterns) if len(block) <= 90 else None
        if matched:
            ordinal, heading, trailing = matched
            units.append(_Unit(ordinal=ordinal, heading=heading, title=trailing))
            continue
        para = re.sub(r"\s+", " ", block)
        if len(para) < 2:
            continue
        # An untitled heading takes the next short block as its title.
        current = units[-1]
        if current.ordinal is not None and not current.title and len(para) <= 70 and not current.paragraphs:
            current.title = para
            continue
        current.paragraphs.append(para)
    return units


def _group_into_books(units: list[_Unit], fallback_title: str, start_index: int) -> list[Book]:
    """Group chapter units into books, splitting where the chapter ordinal resets."""
    groups: list[list[_Unit]] = []
    current: list[_Unit] = []
    last_ordinal = 0
    for unit in units:
        # A new volume starts at chapter one. Requiring exactly 1, plus at least two
        # chapters already banked, keeps a stray low-numbered heading from splitting
        # a book in half.
        chapters_so_far = sum(1 for u in current if u.ordinal is not None)
        if unit.ordinal == 1 and last_ordinal >= 1 and chapters_so_far >= 2:
            groups.append(current)
            current = []
            last_ordinal = 0
        current.append(unit)
        if unit.ordinal is not None:
            last_ordinal = unit.ordinal
    if current:
        groups.append(current)

    # Drop a leading group that is pure front matter with no chapters.
    groups = [g for g in groups if any(u.ordinal is not None for u in g) or any(u.paragraphs for u in g)]

    books: list[Book] = []
    multi = len(groups) > 1
    for offset, group in enumerate(groups):
        index = start_index + offset
        title = _book_title(group, fallback_title, index) if multi else fallback_title
        book = Book(title=title, index=index)
        for unit in group:
            chapter_no = unit.ordinal if unit.ordinal is not None else 0
            chapter_title = unit.title or unit.heading
            if unit.ordinal is not None:
                book.chapters.append((chapter_no, chapter_title))
            for para_no, para in enumerate(unit.paragraphs, 1):
                book.passages.append(
                    Passage(
                        book=title,
                        book_index=index,
                        chapter=chapter_no,
                        chapter_title=chapter_title,
                        para_index=para_no,
                        text=para,
                    )
                )
        if book.passages:
            books.append(book)
    return books


def _book_title(group: list[_Unit], fallback: str, index: int) -> str:
    """Name a book from its front matter, else fall back to an ordinal label.

    In a single-file series the volume title usually sits in the front matter just
    before chapter one, so we take the last short front-matter line.
    """
    front = group[0].paragraphs if group and group[0].ordinal is None else []
    for para in reversed(front[-6:]):
        candidate = para.strip(" *#-")
        if 3 <= len(candidate) <= 70 and re.search(r"[A-Za-z]{3}", candidate):
            words = candidate.split()
            if len(words) <= 12 and not candidate.endswith((".", ",", "!", "?", '"')):
                return candidate.title() if candidate.isupper() else candidate
    return f"{fallback} - Book {index}"


def load_source(path: Path, start_index: int) -> list[Book]:
    """Parse one file into one or more books."""
    text = _normalise(path.read_text(encoding="utf-8", errors="replace"))
    units = _parse_units(text)
    return _group_into_books(units, _clean_title(path), start_index)


def load_corpus(raw_dir: Path | None = None) -> list[Book]:
    """Load every .txt in the raw directory, in filename order."""
    raw_dir = raw_dir or RAW_DIR
    paths = sorted(p for p in raw_dir.glob("*.txt") if p.stat().st_size > 1000)
    if not paths:
        raise FileNotFoundError(
            f"No .txt files found in {raw_dir}. Drop your .txt files there and re-run "
            "`potter build` (see README, 'Corpus and copyright')."
        )
    books: list[Book] = []
    for path in paths:
        books.extend(load_source(path, start_index=len(books) + 1))
    return books


def all_passages(books: list[Book]) -> list[Passage]:
    return [p for b in books for p in b.passages]


def corpus_stats(books: list[Book]) -> dict:
    passages = all_passages(books)
    return {
        "books": len(books),
        "chapters": sum(len(b.chapters) for b in books),
        "passages": len(passages),
        "words": sum(p.n_words for p in passages),
    }
