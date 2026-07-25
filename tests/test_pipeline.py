"""Tests for the extraction heuristics and the interactive layer.

The segmentation and alias-resolution tests use small synthetic fixtures rather than
the real corpus, for three reasons: they run in milliseconds, they need no corpus
present, and they pin the exact failure modes that bit during the build (silent
over-splitting, silent under-splitting, and family merging).

The integration tests skip cleanly when no artefact cache exists.
"""

from __future__ import annotations

import pytest

from potter import characters as char_mod
from potter import corpus as corpus_mod
from potter import lexicon as lex_mod
from potter.corpus import Passage, parse_ordinal

# --------------------------------------------------------------------- ordinals


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("7", 7),
        ("VII", 7),
        ("XIV", 14),
        ("One", 1),
        ("ONE", 1),
        ("THIRTY-EIGHT", 38),
        ("twenty one", 21),
        ("banana", None),
        ("", None),
    ],
)
def test_parse_ordinal(token, expected):
    assert parse_ordinal(token) == expected


# ----------------------------------------------------------------- segmentation


def _wrapped(paragraphs: list[str], width: int = 60) -> str:
    """Hard-wrap paragraphs at `width` with NO blank line between them.

    This is the hostile case: the paragraph break is signalled only by the last
    line of each paragraph falling short of the wrap column.
    """
    out: list[str] = []
    for para in paragraphs:
        words = para.split()
        line = ""
        for word in words:
            if len(line) + len(word) + 1 > width:
                out.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        out.append(line)
    return "\n".join(out)


def test_unwrap_recovers_paragraphs_without_blank_lines():
    paras = [
        "This is the first paragraph and it is deliberately long enough that it must "
        "wrap across several lines before it finally ends here.",
        "The second paragraph is also long enough to wrap more than once so that the "
        "geometry heuristic has something real to work with in this test case.",
    ]
    text = corpus_mod._normalise(_wrapped(paras))
    recovered = [p for p in text.split("\n\n") if p.strip()]
    assert len(recovered) == 2
    assert recovered[0].startswith("This is the first")
    assert recovered[1].startswith("The second paragraph")


def test_double_spaced_edition_is_not_split_per_line():
    """A blank line between every wrapped line must not become a paragraph break.

    The detector needs enough evidence before it believes a file is double-spaced,
    so the fixture supplies a realistic number of paragraphs rather than two.
    """
    para = "This is one paragraph that has been\n\nwrapped over two lines."
    text = corpus_mod._normalise("\n\n\n\n".join([para] * 12))
    recovered = [p for p in text.split("\n\n") if p.strip()]
    assert len(recovered) == 12
    assert all("wrapped over two lines." in p for p in recovered)


def test_dominant_chapter_pattern_ignores_stray_fallback_matches():
    """A stray 'I.' line must not register as a chapter and reset the book ordinal."""
    blocks = [f"CHAPTER {w}" for w in ["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX"]]
    blocks.append("I. said nothing at all")
    patterns = corpus_mod._select_chapter_pattern(blocks)
    assert len(patterns) == 1
    assert corpus_mod._match_chapter("I. said nothing at all", patterns) is None
    assert corpus_mod._match_chapter("CHAPTER THREE", patterns)[0] == 3


def test_series_splits_on_chapter_ordinal_reset(tmp_path):
    body = "Sentence filler for a paragraph that is long enough to survive parsing. "
    text_parts = []
    for book in range(2):
        for chapter in ["ONE", "TWO", "THREE"]:
            text_parts.append(f"CHAPTER {chapter}\n\n{body * 3}\n")
        del book
    path = tmp_path / "series.txt"
    path.write_text("\n".join(text_parts), encoding="utf-8")

    books = corpus_mod.load_source(path, start_index=1)
    assert len(books) == 2
    assert [b.index for b in books] == [1, 2]
    assert all(len(b.chapters) == 3 for b in books)


# ------------------------------------------------------------------- characters


def _passages(texts: list[str]) -> list[Passage]:
    return [
        Passage(book="B", book_index=1, chapter=1, chapter_title="t", para_index=i + 1, text=t)
        for i, t in enumerate(texts)
    ]


def test_possessives_fold_and_contractions_are_rejected():
    assert char_mod._normalise_surface("Harry's") == "Harry"
    assert char_mod._normalise_surface("I'm") is None
    assert char_mod._normalise_surface("You-") is None
    assert char_mod._normalise_surface("O'Brien") == "O'Brien"


def test_shared_surname_does_not_merge_siblings():
    """The family-merge bug: Fred and Ron must stay distinct identities."""
    texts = [
        'Fred Weasley said something clever to the room.',
        'Ron Weasley said something else entirely to nobody.',
        'Fred said it again, and Ron replied at once.',
    ] * 6
    chars = char_mod.extract_characters(_passages(texts), min_count=3, min_person_evidence=1)
    names = {c.name for c in chars}
    fred = next((c for c in chars if "Fred" in c.name), None)
    ron = next((c for c in chars if "Ron" in c.name), None)
    assert fred is not None and ron is not None
    assert fred.name != ron.name
    assert not (fred.aliases & ron.aliases)
    assert "Ron" not in " ".join(fred.aliases)


def test_given_name_merges_with_full_name():
    texts = ['Hermione Granger said hello.', 'Hermione replied to that remark.'] * 8
    chars = char_mod.extract_characters(_passages(texts), min_count=3, min_person_evidence=1)
    assert len(chars) >= 1
    top = chars[0]
    assert {"Hermione", "Hermione Granger"} <= top.aliases


def test_place_without_speech_is_filtered_out():
    """A location that never speaks should not become a character."""
    texts = [
        "Hogwarts stood cold and silent above the lake that year.",
        "Harry said the castle looked enormous from the water.",
    ] * 8
    chars = char_mod.extract_characters(_passages(texts), min_count=3, min_person_evidence=2)
    names = {c.name for c in chars}
    assert "Harry" in names
    assert "Hogwarts" not in names


def test_find_mentions_resolves_possessive_form():
    chars = [char_mod.Character(name="Harry", aliases={"Harry", "Harry Potter"}, count=10)]
    index = char_mod.build_alias_index(chars)
    assert char_mod.find_mentions("Then Harry's owl arrived.", index) == {"Harry"}


# ---------------------------------------------------------------------- lexicon


def test_distinctive_terms_signs_over_and_under_representation():
    """Over-represented terms score positive, under-represented strongly negative."""
    from collections import Counter

    group = ["dragon"] * 30 + ["filler"] * 200
    corpus = Counter(group) + Counter(["filler"] * 4000 + ["elsewhere"] * 300)
    scores = dict(
        lex_mod.distinctive_terms(group, corpus, sum(corpus.values()), top_n=10, min_count=5)
    )
    assert scores["dragon"] > 0
    assert scores["filler"] < 0
    # A term absent from the group is never reported as distinctive to it.
    assert "elsewhere" not in scores


def test_distinctive_terms_respects_min_count():
    """Rare accidents must not reach the ranking at all."""
    from collections import Counter

    group = ["dragon"] * 30 + ["fluke"] * 2
    corpus = Counter(group) + Counter(["filler"] * 4000)
    scores = dict(
        lex_mod.distinctive_terms(group, corpus, sum(corpus.values()), top_n=10, min_count=5)
    )
    assert "dragon" in scores
    assert "fluke" not in scores


def test_dialogue_share_detects_quoted_speech():
    quoted = lex_mod._dialogue_share(_passages(['"All of this is spoken aloud here."']))
    plain = lex_mod._dialogue_share(_passages(["None of this is spoken aloud at all."]))
    assert quoted > 0.8
    assert plain == 0.0


# ------------------------------------------------------------------ integration


@pytest.fixture(scope="module")
def artefacts():
    from potter import build as build_mod

    try:
        return build_mod.load()
    except FileNotFoundError:
        pytest.skip("no artefact cache; run `potter build` first")


def test_cache_is_internally_consistent(artefacts):
    assert artefacts.embeddings.shape[0] == len(artefacts.passages)
    assert artefacts.stats["passages"] == len(artefacts.passages)
    for axis, series in artefacts.arcs.items():
        assert len(series) == len(artefacts.passages), axis


def test_lexical_and_fused_search_return_hits(artefacts):
    from potter import search as search_mod

    lexical = search_mod.kwic("the", artefacts.passages, k=3)
    assert len(lexical) == 3

    fused = search_mod.fused_search(
        artefacts.model(), "a dangerous fight", artefacts.embeddings, artefacts.passages, k=5
    )
    assert len(fused) == 5
    assert all(h.passage.text for h in fused)


def test_web_page_renders_every_view(artefacts):
    from potter.web import _render

    for params in (
        {},
        {"q": ["a dangerous fight"], "mode": ["fused"]},
        {"q": ["wand"], "mode": ["lexical"]},
        {"who": [artefacts.characters[0].name]},
        {"axis": ["wonder"]},
        {"axis": ["nonsense-axis"]},
    ):
        html = _render(artefacts, params)
        assert "corpus explorer" in html
        assert "Narrative arcs" in html
        assert len(html) > 4000
