"""Interactive terminal tool.

Two ways in:
  potter <command>   one-shot, scriptable, pipe-friendly
  potter explore     stateful REPL - loads the cache and the encoder once, so
                     follow-up queries are instant

Design note: the REPL exists because exploration is iterative. Paying model load
plus cache load on every single query makes a reviewer give up after two questions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import arc as arc_mod
from . import build as build_mod
from . import network as net_mod
from . import search as search_mod

app = typer.Typer(add_completion=False, help="Corpus intelligence for a book series.")
console = Console()

_CACHE: build_mod.Artefacts | None = None


def _art() -> build_mod.Artefacts:
    global _CACHE
    if _CACHE is None:
        try:
            _CACHE = build_mod.load()
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
    return _CACHE


# --------------------------------------------------------------------------- build


@app.command()
def build(
    raw: Path = typer.Option(None, help="Directory of .txt files (default data/raw)"),
) -> None:
    """Run the full extraction pipeline and cache the artefacts."""
    build_mod.build(raw_dir=raw, log=lambda m: console.print(m))


# ------------------------------------------------------------------------ overview


@app.command()
def overview() -> None:
    """Corpus-level statistics: size, structure, lexical richness, pace."""
    art = _art()
    s = art.stats
    table = Table(show_header=False, box=None, padding=(0, 2))
    rows = [
        ("Books", f"{s['books']}"),
        ("Chapters", f"{s['chapters']:,}"),
        ("Passages", f"{s['passages']:,}"),
        ("Words", f"{s['words']:,}"),
        ("Distinct word types", f"{s['types']:,}"),
        ("Type/token ratio", f"{s['type_token_ratio']:.4f}"),
        ("Hapax legomena", f"{s['hapax_legomena']:,} ({s['hapax_pct'] * 100:.1f}% of types)"),
        ("Mean sentence length", f"{s['mean_sentence_len']:.1f} words"),
        ("Dialogue share", f"{s['dialogue_share'] * 100:.1f}% of characters inside quotes"),
        ("Encoder", art.meta["model"]),
        ("Build time", f"{art.meta['build_seconds']}s"),
    ]
    for label, value in rows:
        table.add_row(f"[dim]{label}[/dim]", f"[bold]{value}[/bold]")
    console.print(Panel(table, title="Corpus overview", border_style="cyan"))
    console.print("[dim]Source files:[/dim] " + ", ".join(art.meta["source_files"]))


# ---------------------------------------------------------------------- characters


@app.command()
def characters(top: int = typer.Option(20, help="How many identities to show")) -> None:
    """Extracted character identities with their resolved aliases."""
    art = _art()
    table = Table(title=f"Top {top} characters by mention volume", border_style="cyan")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Character", style="bold")
    table.add_column("Mentions", justify="right")
    table.add_column("Resolved aliases", style="dim")
    for i, ch in enumerate(art.characters[:top], 1):
        aliases = ", ".join(sorted(ch.aliases - {ch.name})[:6]) or "-"
        table.add_row(str(i), ch.name, f"{ch.count:,}", aliases)
    console.print(table)


# ------------------------------------------------------------------------- network


@app.command()
def network(
    character: str = typer.Option(None, help="Show this character's strongest ties instead"),
    top: int = typer.Option(15),
) -> None:
    """Character co-occurrence network: centrality, brokers, communities."""
    art = _art()
    graph = net_mod.from_json(art.graph_json)
    analysis = art.graph_analysis

    if character:
        match = _resolve_character(art, character)
        if match is None:
            return
        if match not in graph:
            console.print(f"[yellow]{match} has no surviving edges in the pruned graph.[/yellow]")
            return
        ties = sorted(graph[match].items(), key=lambda kv: -kv[1]["weight"])[:top]
        table = Table(title=f"Strongest ties: {match}", border_style="cyan")
        table.add_column("Character", style="bold")
        table.add_column("Shared scenes", justify="right")
        for other, data in ties:
            table.add_row(other, f"{data['weight']:,}")
        console.print(table)
        return

    table = Table(title="Network structure", border_style="cyan")
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("By interaction volume", style="bold")
    table.add_column("Vol", justify="right")
    table.add_column("By brokerage (betweenness)", style="bold")
    table.add_column("Score", justify="right")

    cent = list(analysis["centrality"].items())[:top]
    betw = list(analysis["betweenness"].items())[:top]
    for i in range(max(len(cent), len(betw))):
        c = cent[i] if i < len(cent) else ("", 0)
        b = betw[i] if i < len(betw) else ("", 0)
        table.add_row(str(i + 1), c[0], f"{int(c[1]):,}" if c[0] else "", b[0], f"{b[1]:.3f}" if b[0] else "")
    console.print(table)
    console.print(f"[dim]Graph density:[/dim] {analysis['density']:.3f}")

    for i, comm in enumerate(analysis["communities"][:6], 1):
        console.print(f"[bold cyan]Community {i}[/bold cyan] ({len(comm)}): [dim]{', '.join(comm[:12])}[/dim]")


def _resolve_character(art: build_mod.Artefacts, query: str) -> str | None:
    """Fuzzy-resolve a user-typed name to a canonical character."""
    from rapidfuzz import process

    names = [c.name for c in art.characters]
    canon = art.alias_index.get(query.lower())
    if canon:
        return canon
    match = process.extractOne(query, names, score_cutoff=60)
    if match is None:
        console.print(f"[yellow]No character matching '{query}'. Try `potter characters`.[/yellow]")
        return None
    if match[0].lower() != query.lower():
        console.print(f"[dim]interpreting '{query}' as {match[0]}[/dim]")
    return match[0]


# ----------------------------------------------------------------------------- arc


@app.command()
def arc(
    axis: str = typer.Option("tension", help=f"One of: {', '.join(arc_mod.AXES)}"),
    width: int = typer.Option(64),
    character: str = typer.Option(None, help="Restrict to passages mentioning this character"),
) -> None:
    """Narrative arc along a semantic axis, per book - whole corpus or one character."""
    art = _art()
    if axis not in art.arcs:
        console.print(f"[red]Unknown axis '{axis}'. Available: {', '.join(art.arcs)}[/red]")
        raise typer.Exit(code=1)

    who = None
    if character:
        who = _resolve_character(art, character)
        if who is None:
            return
        mentioned = art.mentions()

    series = np.asarray(art.arcs[axis])
    scope = f" - restricted to passages mentioning [bold]{who}[/bold]" if who else ""
    console.print(Panel(
        f"[bold]{axis}[/bold] axis - z-scored projection onto the "
        f"{axis}-pole direction in embedding space, smoothed over 9 passages{scope}.\n"
        "[dim]Low ▁▁ to high ██. Measures semantic similarity to the axis poles, "
        "not calibrated emotion.[/dim]",
        border_style="cyan",
    ))

    for book in art.books:
        idx = [
            i for i, p in enumerate(art.passages)
            if p.book == book and (who is None or who in mentioned[i])
        ]
        if len(idx) < 5:
            console.print(f"\n[bold]{book}[/bold]  [dim]too few passages ({len(idx)})[/dim]")
            continue
        vals = series[idx]
        console.print(f"\n[bold]{book}[/bold]  [dim]{len(idx):,} passages, "
                      f"mean {vals.mean():+.2f}, peak {vals.max():+.2f}[/dim]")
        console.print(f"  [cyan]{arc_mod.sparkline(vals, width=width)}[/cyan]")
        # Name the peak scene so the number is anchored to something readable.
        peak_local = int(np.argmax(vals))
        peak = art.passages[idx[peak_local]]
        console.print(f"  [dim]peak at ch.{peak.chapter} '{peak.chapter_title}': "
                      f"{peak.text[:110]}...[/dim]")


@app.command()
def pair(
    first: str = typer.Argument(..., help="First character"),
    second: str = typer.Argument(..., help="Second character"),
    width: int = typer.Option(64),
) -> None:
    """Relationship timeline: how often two characters share a passage, per book."""
    art = _art()
    a = _resolve_character(art, first)
    if a is None:
        return
    b = _resolve_character(art, second)
    if b is None:
        return
    if a == b:
        console.print("[yellow]That is the same character twice.[/yellow]")
        return

    mentioned = art.mentions()
    console.print(Panel(
        f"[bold]{a}[/bold] + [bold]{b}[/bold] - shared passages per chapter.\n"
        "[dim]The shape shows when a relationship exists at all, not just how strong it is.[/dim]",
        border_style="cyan",
    ))

    grand_total = 0
    for book in art.books:
        # Count shared passages per chapter, keeping chapter order.
        chapters: dict[int, int] = {}
        for i, p in enumerate(art.passages):
            if p.book != book:
                continue
            chapters.setdefault(p.chapter, 0)
            if a in mentioned[i] and b in mentioned[i]:
                chapters[p.chapter] += 1
        counts = np.asarray([chapters[c] for c in sorted(chapters)], dtype=float)
        total = int(counts.sum())
        grand_total += total
        spark = arc_mod.sparkline(counts, width=width) if total else "▁" * min(len(counts), width)
        console.print(f"\n[bold]{book}[/bold]  [dim]{total:,} shared passages[/dim]")
        console.print(f"  [cyan]{spark}[/cyan]")
        if total:
            peak_ch = max(sorted(chapters), key=lambda c: chapters[c])
            title = next(
                (p.chapter_title for p in art.passages if p.book == book and p.chapter == peak_ch),
                "",
            )
            console.print(f"  [dim]peak ch.{peak_ch} '{title}' with {chapters[peak_ch]} shared passages[/dim]")
    console.print(f"\n[dim]Series total:[/dim] [bold]{grand_total:,}[/bold] shared passages")


# --------------------------------------------------------------------- distinctive


@app.command()
def distinctive(top: int = typer.Option(12)) -> None:
    """Vocabulary that is over-represented in each book vs the rest of the corpus."""
    art = _art()
    console.print(Panel(
        "Log-odds ratio with an informative Dirichlet prior, z-scored "
        "(Monroe et al. 2008).\n[dim]Higher z = more distinctive to that book. "
        "Function words excluded.[/dim]",
        border_style="cyan",
    ))
    for book, terms in art.distinctive.items():
        console.print(f"\n[bold]{book}[/bold]")
        console.print("  " + ", ".join(f"{w} [dim]({z:.1f})[/dim]" for w, z in terms[:top]))


# -------------------------------------------------------------------------- search


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural-language description of a scene"),
    k: int = typer.Option(6),
    mode: str = typer.Option("fused", help="fused | semantic | lexical"),
) -> None:
    """Find passages. Semantic, lexical, or Reciprocal Rank Fusion of both."""
    art = _art()
    _render_hits(_run_search(art, query, k, mode), query, mode)


def _run_search(art: build_mod.Artefacts, query: str, k: int, mode: str) -> list[search_mod.Hit]:
    if mode == "lexical":
        return search_mod.kwic(query.split()[0], art.passages, k=k)
    if mode == "semantic":
        return search_mod.dense_search(art.model(), query, art.embeddings, art.passages, k=k)
    return search_mod.fused_search(art.model(), query, art.embeddings, art.passages, k=k)


def _render_hits(hits: list[search_mod.Hit], query: str, mode: str) -> None:
    if not hits:
        console.print("[yellow]No matches.[/yellow]")
        return
    console.print(f"\n[bold]{mode}[/bold] search: [italic]{query}[/italic]\n")
    for i, hit in enumerate(hits, 1):
        p = hit.passage
        head = f"[bold cyan]{i}.[/bold cyan] {p.book} ch.{p.chapter} '{p.chapter_title}' [dim]({p.ref})[/dim]"
        console.print(head)
        console.print(f"   [dim]score {hit.score:.4f} via {hit.why if hit.why in ('semantic', 'lexical') or '+' in hit.why else 'lexical'}[/dim]")
        text = hit.passage.text
        console.print(f"   {text[:400]}{'...' if len(text) > 400 else ''}\n")


@app.command()
def kwic(
    term: str = typer.Argument(..., help="Word or name to concordance"),
    k: int = typer.Option(10),
) -> None:
    """Keyword-in-context concordance lines for an exact term."""
    art = _art()
    hits = search_mod.kwic(term, art.passages, k=k)
    if not hits:
        console.print(f"[yellow]'{term}' does not appear in the corpus.[/yellow]")
        return
    total = sum(int(h.score) for h in hits)
    console.print(f"\n[bold]{term}[/bold] - {total} occurrences in the top {len(hits)} passages\n")
    for hit in hits:
        p = hit.passage
        console.print(f"[dim]{p.book[:22]:<22} ch.{p.chapter:>2}[/dim]  {hit.why}")


# ------------------------------------------------------------------------- explore


@app.command()
def explore() -> None:
    """Stateful REPL. Loads the cache and encoder once, then answers instantly."""
    art = _art()
    art.model()  # warm the encoder up front so the first query is not slow
    console.print(Panel(
        "[bold]potter explore[/bold]\n\n"
        "  [cyan]find[/cyan] <description>   semantic + lexical scene search\n"
        "  [cyan]kwic[/cyan] <term>          exact concordance lines\n"
        "  [cyan]who[/cyan] <name>           a character's strongest ties\n"
        "  [cyan]pair[/cyan] <a> <b>         relationship timeline for two characters\n"
        "  [cyan]arc[/cyan] [axis] [name]    narrative arc, whole corpus or one character\n"
        "  [cyan]top[/cyan]                  characters by mention volume\n"
        "  [cyan]net[/cyan]                  network structure and communities\n"
        "  [cyan]words[/cyan]                distinctive vocabulary per book\n"
        "  [cyan]stats[/cyan]                corpus overview\n"
        "  [cyan]quit[/cyan]\n\n"
        f"[dim]{art.stats['books']} books, {art.stats['passages']:,} passages, "
        f"{len(art.characters)} characters resolved[/dim]",
        border_style="cyan",
    ))

    while True:
        try:
            line = console.input("\n[bold cyan]potter[/bold cyan] > ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye")
            return
        if not line:
            continue
        cmd, _, rest = line.partition(" ")
        cmd, rest = cmd.lower(), rest.strip()

        try:
            if cmd in {"quit", "exit", "q"}:
                console.print("bye")
                return
            # Typer's @app.command() returns the undecorated function, so these are
            # ordinary calls. Every argument is passed explicitly because the
            # defaults are typer.Option objects, not plain values.
            if cmd == "find" and rest:
                _render_hits(_run_search(art, rest, 6, "fused"), rest, "fused")
            elif cmd == "kwic" and rest:
                kwic(rest, 10)
            elif cmd == "who" and rest:
                network(rest, 12)
            elif cmd == "pair" and rest:
                names = rest.split()
                if len(names) == 2:
                    pair(names[0], names[1], 64)
                else:
                    console.print("[yellow]usage: pair <first> <second>[/yellow]")
            elif cmd == "arc":
                parts = rest.split(maxsplit=1)
                axis_name = parts[0] if parts and parts[0] in art.arcs else "tension"
                char_name = (
                    parts[1] if len(parts) > 1
                    else (parts[0] if parts and parts[0] not in art.arcs else None)
                )
                arc(axis_name, 64, char_name)
            elif cmd == "top":
                characters(20)
            elif cmd == "net":
                network(None, 15)
            elif cmd == "words":
                distinctive(12)
            elif cmd == "stats":
                overview()
            else:
                console.print("[yellow]Unrecognised. Commands: find, kwic, who, arc, top, net, words, stats, quit[/yellow]")
        except Exception as exc:  # noqa: BLE001 - a REPL must not die on one bad query
            console.print(f"[red]error:[/red] {exc}")


# ---------------------------------------------------------------------------- demo


@app.command()
def demo(
    pause: bool = typer.Option(True, help="Wait for Enter between findings"),
) -> None:
    """Guided tour of the most interesting findings. Made for showing the tool."""
    from .demo import run_demo

    run_demo(_art(), console, interactive=pause)


# ----------------------------------------------------------------------------- web


@app.command()
def web(port: int = typer.Option(8000), host: str = typer.Option("127.0.0.1")) -> None:
    """Serve the browser UI."""
    from .web import serve

    serve(_art(), host=host, port=port, console=console)


if __name__ == "__main__":
    app()
