"""Guided tour of the corpus: `potter demo`.

One command that walks through the most interesting findings in a fixed order,
with a headline explaining why each one matters. Exists for two reasons: it makes
the tool demonstrable in one take, and it is the direct answer to "what is
interesting about this corpus" - each stop is a claim the pipeline discovered,
not a canned fact.

Nothing here is hardcoded to Harry Potter. Every stop queries the artefact cache,
so the demo works on any corpus, it just narrates whatever it finds.
"""

from __future__ import annotations

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import arc as arc_mod
from . import network as net_mod


def _pause(console: Console, interactive: bool) -> None:
    if interactive:
        console.input("\n[dim]press Enter for the next finding...[/dim]")
    else:
        console.print()


def run_demo(art, console: Console, interactive: bool = True) -> None:
    s = art.stats

    console.print(Panel(
        f"[bold]A guided tour of the corpus[/bold]\n\n"
        f"{s['books']} books, {s['chapters']:,} chapters, {s['passages']:,} passages, "
        f"{s['words']:,} words.\n"
        f"Everything below was extracted by the pipeline - no facts are hardcoded.",
        border_style="cyan", title="potter demo",
    ))
    _pause(console, interactive)

    # ---- 1. who the story is about ------------------------------------------
    console.print(Panel(
        "[bold]1. Who the story is about[/bold]\n"
        "[dim]Names found with no NER model and no name list: the corpus's own "
        "lowercase profile filters vocabulary, speech verbs filter places, and "
        "union-find stitches aliases into identities.[/dim]",
        border_style="cyan",
    ))
    table = Table(show_header=True, border_style="dim")
    table.add_column("Character", style="bold")
    table.add_column("Mentions", justify="right")
    table.add_column("Aliases resolved", style="dim")
    for c in art.characters[:8]:
        table.add_row(c.name, f"{c.count:,}", ", ".join(sorted(c.aliases - {c.name})[:3]) or "-")
    console.print(table)
    _pause(console, interactive)

    # ---- 2. the social structure --------------------------------------------
    brokers = [
        n for n in art.graph_analysis["betweenness"]
        if n not in list(art.graph_analysis["centrality"])[:3]
    ][:5]
    console.print(Panel(
        "[bold]2. The social structure has brokers, not just stars[/bold]\n"
        "[dim]Betweenness centrality finds characters who BRIDGE groups rather than "
        "dominate scenes. High-brokerage names sit between communities the "
        "protagonist connects to.[/dim]",
        border_style="cyan",
    ))
    console.print(f"  Top brokers after the leads: [bold]{', '.join(brokers)}[/bold]")
    comms = art.graph_analysis["communities"]
    console.print(f"  The graph splits into [bold]{len(comms)}[/bold] communities "
                  f"of sizes {', '.join(str(len(c)) for c in comms)}")
    for i, comm in enumerate(comms[:2], 1):
        console.print(f"  [dim]community {i}: {', '.join(comm[:8])}...[/dim]")
    _pause(console, interactive)

    # ---- 3. a relationship appearing over time ------------------------------
    a, b = _most_asymmetric_pair(art)
    console.print(Panel(
        f"[bold]3. Watch a relationship start existing: {a} + {b}[/bold]\n"
        "[dim]Shared passages per chapter. A flat line means the relationship "
        "does not exist yet; the shape shows exactly when it begins.[/dim]",
        border_style="cyan",
    ))
    mentioned = art.mentions()
    for book in art.books:
        chapters: dict[int, int] = {}
        for i, p in enumerate(art.passages):
            if p.book != book:
                continue
            chapters.setdefault(p.chapter, 0)
            if a in mentioned[i] and b in mentioned[i]:
                chapters[p.chapter] += 1
        counts = np.asarray([chapters[c] for c in sorted(chapters)], dtype=float)
        total = int(counts.sum())
        spark = arc_mod.sparkline(counts, width=56) if total else "▁" * min(len(counts), 56)
        console.print(f"  [bold]{book[:44]:<44}[/bold] {total:>4}  [cyan]{spark}[/cyan]")
    _pause(console, interactive)

    # ---- 4. the emotional shape ---------------------------------------------
    console.print(Panel(
        "[bold]4. The shape of each book[/bold]\n"
        "[dim]Each passage projected onto a 'tension' axis built from probe "
        "sentences in embedding space - no sentiment lexicon. Watch every book "
        "climb into its climax.[/dim]",
        border_style="cyan",
    ))
    series = np.asarray(art.arcs["tension"])
    for book in art.books:
        idx = [i for i, p in enumerate(art.passages) if p.book == book]
        vals = series[idx]
        peak = art.passages[idx[int(np.argmax(vals))]]
        console.print(f"  [bold]{book[:44]:<44}[/bold] [cyan]{arc_mod.sparkline(vals, width=56)}[/cyan]")
        console.print(f"  [dim]{'':44} peak: ch.{peak.chapter} '{peak.chapter_title}'[/dim]")
    _pause(console, interactive)

    # ---- 5. each book's fingerprint -----------------------------------------
    console.print(Panel(
        "[bold]5. Each book has a vocabulary fingerprint[/bold]\n"
        "[dim]Log-odds with a Dirichlet prior: words over-represented in one book "
        "against the rest of the series. The plot of each volume is visible in "
        "its word list alone.[/dim]",
        border_style="cyan",
    ))
    for book, terms in art.distinctive.items():
        words = ", ".join(w for w, _ in terms[:6])
        console.print(f"  [bold]{book[:44]:<44}[/bold] [dim]{words}[/dim]")

    console.print(Panel(
        "[bold]Explore it yourself[/bold]\n"
        "  potter explore    REPL: find / kwic / who / pair / arc / top / net / words\n"
        "  potter web        the same, in a browser",
        border_style="green",
    ))


def _most_asymmetric_pair(art) -> tuple[str, str]:
    """Pick the pair whose interaction is most concentrated late in the series.

    That asymmetry is what makes a relationship timeline interesting to look at.
    Falls back to the two most-mentioned characters if the graph is tiny.
    """
    graph = net_mod.from_json(art.graph_json)
    mentioned = art.mentions()
    n = len(art.passages)
    if n == 0 or graph.number_of_edges() == 0:
        names = [c.name for c in art.characters[:2]]
        return (names + ["?", "?"])[0], (names + ["?", "?"])[1]

    top = [c.name for c in art.characters[:12] if c.name in graph]
    best, best_score = None, -1.0
    for i, a in enumerate(top):
        for b in top[i + 1:]:
            if not graph.has_edge(a, b) or graph[a][b]["weight"] < 30:
                continue
            positions = [
                k / n for k in range(n) if a in mentioned[k] and b in mentioned[k]
            ]
            if len(positions) < 20:
                continue
            # High mean position = the relationship lives late in the series.
            score = float(np.mean(positions))
            if score > best_score:
                best, best_score = (a, b), score
    return best if best else (top[0], top[1])
