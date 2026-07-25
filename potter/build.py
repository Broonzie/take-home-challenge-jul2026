"""Pipeline orchestration and artefact cache.

`potter build` runs every extraction stage once and writes the results to
data/derived/. Every interactive command then loads that cache, so exploring the
corpus is instant instead of re-embedding on each query.

The cache holds derived data AND passage text (the concordance and search results
need the text to display). data/ is gitignored in its entirety - see README,
"Corpus and copyright".
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import arc as arc_mod
from . import characters as char_mod
from . import lexicon as lex_mod
from . import network as net_mod
from .corpus import DERIVED_DIR, RAW_DIR, Passage, all_passages, corpus_stats, load_corpus

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class Artefacts:
    """Everything the interactive layer needs, loaded from cache."""

    passages: list[Passage]
    characters: list[char_mod.Character]
    alias_index: dict[str, str]
    graph_json: dict
    graph_analysis: dict
    embeddings: np.ndarray
    arcs: dict[str, list[float]]
    stats: dict
    distinctive: dict
    meta: dict

    _model = None

    def model(self):
        """Lazily load the encoder - only search needs it, and it costs ~2s."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(MODEL_NAME)
        return self._model

    @property
    def books(self) -> list[str]:
        seen: list[str] = []
        for p in self.passages:
            if p.book not in seen:
                seen.append(p.book)
        return seen


def _load_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def build(raw_dir: Path | None = None, derived_dir: Path | None = None, log=print) -> dict:
    """Run the full pipeline and write the artefact cache."""
    raw_dir = raw_dir or RAW_DIR
    derived_dir = derived_dir or DERIVED_DIR
    derived_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    log("[1/6] ingest + segment")
    books = load_corpus(raw_dir)
    passages = all_passages(books)
    stats = corpus_stats(books)
    log(f"      {stats['books']} books, {stats['chapters']} chapters, "
        f"{stats['passages']:,} passages, {stats['words']:,} words")

    log("[2/6] character extraction + alias resolution")
    characters = char_mod.extract_characters(passages)
    alias_index = char_mod.build_alias_index(characters)
    log(f"      {len(characters)} identities from {len(alias_index)} surface forms")

    log("[3/6] co-occurrence network")
    graph = net_mod.build_graph(passages, alias_index)
    analysis = net_mod.analyse(graph)
    log(f"      {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, "
        f"{len(analysis['communities'])} communities")

    log("[4/6] lexical statistics")
    basic = lex_mod.basic_stats(passages)
    by_book: dict[str, list[Passage]] = {}
    for p in passages:
        by_book.setdefault(p.book, []).append(p)
    distinctive = lex_mod.per_book_distinctive(by_book)
    log(f"      TTR {basic['type_token_ratio']:.4f}, "
        f"dialogue {basic['dialogue_share'] * 100:.1f}% of characters")

    log(f"[5/6] embedding {len(passages):,} passages")
    model = _load_model()
    from .search import embed_passages

    embeddings = embed_passages(model, passages)
    log(f"      {embeddings.shape[0]:,} x {embeddings.shape[1]} float32")

    log("[6/6] narrative arcs")
    arcs: dict[str, list[float]] = {}
    for axis in arc_mod.AXES:
        vec = arc_mod.build_axis(model, axis)
        scores = arc_mod.project(embeddings, vec)
        arcs[axis] = arc_mod.smooth(scores).tolist()
    log(f"      axes: {', '.join(arcs)}")

    payload = {
        "passages": [
            {
                "book": p.book,
                "book_index": p.book_index,
                "chapter": p.chapter,
                "chapter_title": p.chapter_title,
                "para_index": p.para_index,
                "text": p.text,
            }
            for p in passages
        ],
        "characters": [c.to_dict() for c in characters],
        "graph": net_mod.to_json(graph),
        "graph_analysis": analysis,
        "arcs": arcs,
        "stats": {**stats, **basic},
        "distinctive": {k: [[w, round(z, 3)] for w, z in v] for k, v in distinctive.items()},
        "meta": {
            "model": MODEL_NAME,
            "build_seconds": round(time.time() - started, 1),
            "source_files": sorted(p.name for p in raw_dir.glob("*.txt")),
        },
    }
    (derived_dir / "artefacts.json").write_text(json.dumps(payload), encoding="utf-8")
    np.save(derived_dir / "embeddings.npy", embeddings)

    log(f"\ndone in {payload['meta']['build_seconds']}s -> {derived_dir}")
    return payload["stats"]


def load(derived_dir: Path | None = None) -> Artefacts:
    """Load the artefact cache. Raises a clear error if `build` has not been run."""
    derived_dir = derived_dir or DERIVED_DIR
    art_path = derived_dir / "artefacts.json"
    emb_path = derived_dir / "embeddings.npy"
    if not art_path.exists() or not emb_path.exists():
        raise FileNotFoundError("No artefact cache found. Run `potter build` first.")

    payload = json.loads(art_path.read_text(encoding="utf-8"))
    passages = [Passage(**p) for p in payload["passages"]]
    characters = [
        char_mod.Character(name=c["name"], aliases=set(c["aliases"]), count=c["count"])
        for c in payload["characters"]
    ]
    return Artefacts(
        passages=passages,
        characters=characters,
        alias_index=char_mod.build_alias_index(characters),
        graph_json=payload["graph"],
        graph_analysis=payload["graph_analysis"],
        embeddings=np.load(emb_path),
        arcs=payload["arcs"],
        stats=payload["stats"],
        distinctive=payload["distinctive"],
        meta=payload["meta"],
    )
