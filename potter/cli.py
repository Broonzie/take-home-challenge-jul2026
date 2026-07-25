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
) -> None:
    """Narrative arc along a semantic axis, per book."""
    art = _art()
    if axis not in art.arcs:
        console.print(f"[red]Unknown axis '{axis}'. Available: {', '.join(art.arcs)}[/red]")
        raise typer.Exit(code=1)

    series = np.asarray(art.arcs[axis])
    console.print(Panel(
        f"[bold]{axis}[/bold] axis - z-scored projection onto the "
        f"{axis}-pole direction in embedding space, smoothed over 9 passages.\n"
        "[dim]Low ▁▁ to high ██. Measures semantic similarity to the axis poles, "
        "not calibrated emotion.[/dim]",
        border_style="cyan",
    ))

    for book in art.books:
        idx = [i for i, p in enumerate(art.passages) if p.book == book]
        vals = series[idx]
        console.print(f"\n[bold]{book}[/bold]  [dim]mean {vals.mean():+.2f}, peak {vals.max():+.2f}[/dim]")
        console.print(f"  [cyan]{arc_mod.sparkline(vals, width=width)}[/cyan]")
        # Name the peak scene so the number is anchored to something readable.
        peak_local = int(np.argmax(vals))
        peak = art.passages[idx[peak_local]]
        console.print(f"  [dim]peak at ch.{peak.chapter} '{peak.chapter_title}': "
                      f"{peak.text[:110]}...[/dim]")


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
        "  [cyan]arc[/cyan] [axis]           narrative arc sparklines\n"
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
            if cmd == "find" and rest:
                _render_hits(_run_search(art, rest, 6, "fused"), rest, "fused")
            elif cmd == "kwic" and rest:
                kwic.callback(rest, 10)  # type: ignore[misc]
            elif cmd == "who" and rest:
                network.callback(rest, 12)  # type: ignore[misc]
            elif cmd == "arc":
                arc.callback(rest or "tension", 64)  # type: ignore[misc]
            elif cmd == "top":
                characters.callback(20)  # type: ignore[misc]
            elif cmd == "net":
                network.callback(None, 15)  # type: ignore[misc]
            elif cmd == "words":
                distinctive.callback(12)  # type: ignore[misc]
            elif cmd == "stats":
                overview.callback()  # type: ignore[misc]
            else:
                console.print("[yellow]Unrecognised. Commands: find, kwic, who, arc, top, net, words, stats, quit[/yellow]")
        except Exception as exc:  # noqa: BLE001 - a REPL must not die on one bad query
            console.print(f"[red]error:[/red] {exc}")


# ----------------------------------------------------------------------------- web


@app.command()
def web(port: int = typer.Option(8000), host: str = typer.Option("127.0.0.1")) -> None:
    """Serve the browser UI."""
    from .web import serve

    serve(_art(), host=host, port=port, console=console)


if __name__ == "__main__":
    app()
