"""Fetch the shipped public-domain demo corpus.

The challenge brief asks for Harry Potter text. That text is under copyright, so
this repository never contains it (see README, "Corpus and copyright"). Instead the
pipeline is corpus-agnostic and ships with two public-domain Project Gutenberg
novels that share the structural properties we care about: a multi-book children's
fantasy series with recurring named characters and numbered chapters.

Point the same pipeline at any plain-text corpus by dropping .txt files into
data/raw/ and re-running `potter build`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

# Project Gutenberg ids -> local filename. Ordered as a series.
DEMO_BOOKS: list[tuple[int, str]] = [
    (55, "01-the-wonderful-wizard-of-oz.txt"),
    (54, "02-the-marvelous-land-of-oz.txt"),
    (33361, "03-ozma-of-oz.txt"),
]

URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove the Gutenberg licence header/footer, keeping only the work itself."""
    start = text.find("*** START OF")
    if start != -1:
        start = text.find("\n", start) + 1
    else:
        start = 0
    end = text.find("*** END OF")
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    ok = 0
    for book_id, filename in DEMO_BOOKS:
        dest = RAW / filename
        if dest.exists() and dest.stat().st_size > 10_000:
            print(f"  cached  {filename}")
            ok += 1
            continue
        url = URL.format(id=book_id)
        try:
            resp = httpx.get(url, timeout=30.0, follow_redirects=True)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - surface the real reason, never silently skip
            print(f"  FAILED  {filename}: {exc}", file=sys.stderr)
            continue
        body = strip_gutenberg_boilerplate(resp.text)
        if len(body) < 10_000:
            print(f"  FAILED  {filename}: suspiciously short ({len(body)} chars)", file=sys.stderr)
            continue
        dest.write_text(body, encoding="utf-8")
        print(f"  fetched {filename}  ({len(body):,} chars)")
        ok += 1

    if ok == 0:
        print("No books fetched. Check your network, or drop .txt files into data/raw/ manually.", file=sys.stderr)
        return 1
    print(f"\n{ok}/{len(DEMO_BOOKS)} books available in {RAW}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
