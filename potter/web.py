"""Browser UI over the artefact cache.

Deliberately dependency-free on the serving side: stdlib http.server plus Jinja2,
which is already required by the toolchain. Adding FastAPI/uvicorn for six read-only
routes would be a heavier install for a reviewer with nothing gained.

Everything is read-only and bound to localhost. The artefact cache and the encoder
are loaded once by the CLI and shared, so queries are instant.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from jinja2 import Template

from . import arc as arc_mod
from . import network as net_mod
from . import search as search_mod

PAGE = Template(
    """
<!doctype html>
<html><head><meta charset="utf-8"><title>potter - corpus explorer</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0f1115;--panel:#171a21;--line:#262b36;--ink:#e8eaf0;--dim:#8b93a7;--acc:#7aa2f7;--warm:#e0af68}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 ui-sans-serif,system-ui,'Segoe UI',sans-serif}
header{padding:22px 28px;border-bottom:1px solid var(--line);display:flex;align-items:baseline;gap:16px;flex-wrap:wrap}
h1{margin:0;font-size:19px;letter-spacing:.3px}
h1 span{color:var(--acc)}
.meta{color:var(--dim);font-size:13px}
main{max-width:1180px;margin:0 auto;padding:24px 20px 60px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}
section{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px 20px;overflow-x:auto}
section.wide{grid-column:1/-1}
h2{margin:0 0 14px;font-size:13px;text-transform:uppercase;letter-spacing:1.1px;color:var(--dim);font-weight:600}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;padding:6px 10px 6px 0;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--dim);font-weight:500;font-size:12px;text-transform:uppercase;letter-spacing:.6px}
td.num{text-align:right;font-variant-numeric:tabular-nums}
td.alias{color:var(--dim);font-size:13px;white-space:normal}
a{color:var(--acc);text-decoration:none}
a:hover{text-decoration:underline}
form{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px}
input[type=text]{flex:1;min-width:240px;background:#0b0d11;border:1px solid var(--line);color:var(--ink);
  padding:11px 13px;border-radius:7px;font:inherit}
select,button{background:#0b0d11;border:1px solid var(--line);color:var(--ink);padding:11px 13px;border-radius:7px;font:inherit;cursor:pointer}
button{background:var(--acc);color:#0b0d11;font-weight:600;border-color:var(--acc)}
.hit{border-left:2px solid var(--line);padding:2px 0 2px 14px;margin:0 0 16px}
.hit .ref{color:var(--warm);font-size:12.5px;letter-spacing:.3px}
.hit .why{color:var(--dim);font-size:12px}
.hit p{margin:6px 0 0;white-space:normal}
.spark{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:19px;color:var(--acc);letter-spacing:1px;white-space:pre}
.bookrow{margin-bottom:14px}
.bookrow .t{font-size:13.5px;margin-bottom:2px}
.bookrow .s{color:var(--dim);font-size:12px}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{background:#0b0d11;border:1px solid var(--line);border-radius:20px;padding:3px 10px;font-size:12.5px;color:var(--dim)}
.chip b{color:var(--ink);font-weight:500}
.comm{margin-bottom:10px;font-size:13px}
.comm .n{color:var(--warm);font-size:12px;display:block;margin-bottom:3px}
.note{color:var(--dim);font-size:12.5px;margin:0 0 12px}
</style></head><body>
<header>
  <h1>potter <span>corpus explorer</span></h1>
  <div class="meta">{{ s.books }} books &middot; {{ '{:,}'.format(s.chapters) }} chapters &middot;
    {{ '{:,}'.format(s.passages) }} passages &middot; {{ '{:,}'.format(s.words) }} words &middot;
    {{ chars }} characters resolved</div>
</header>
<main>

<section class="wide">
  <h2>Search</h2>
  <form method="get" action="/">
    <input type="text" name="q" value="{{ q|e }}" placeholder="Describe a scene, or name a term" autofocus>
    <select name="mode">
      {% for m in ['fused','semantic','lexical'] %}
      <option value="{{ m }}"{% if mode==m %} selected{% endif %}>{{ m }}</option>{% endfor %}
    </select>
    <button type="submit">Search</button>
  </form>
  {% if q %}
    <p class="note">{{ hits|length }} result(s) for <b>{{ q|e }}</b> via {{ mode }}
      {%- if mode=='fused' %} (reciprocal rank fusion of dense + lexical){% endif %}</p>
    {% for h in hits %}
    <div class="hit">
      <div class="ref">{{ h.book }} &middot; ch.{{ h.chapter }} {{ h.chapter_title }}
        <span class="why">[{{ h.ref }}] score {{ '%.4f'|format(h.score) }} &middot; {{ h.why }}</span></div>
      <p>{{ h.text }}</p>
    </div>
    {% endfor %}
    {% if not hits %}<p class="note">No matches.</p>{% endif %}
  {% else %}
    <p class="note">Try <a href="/?q=a+dangerous+duel+with+an+enemy">a dangerous duel with an enemy</a>,
      <a href="/?q=a+warm+meal+among+friends">a warm meal among friends</a>, or
      <a href="/?q=wand&mode=lexical">wand</a> in lexical mode.</p>
  {% endif %}
</section>

<div class="grid">

<section>
  <h2>Characters</h2>
  <table><tr><th>#</th><th>Name</th><th class="num">Mentions</th><th>Aliases</th></tr>
  {% for c in characters %}
  <tr><td style="color:var(--dim)">{{ loop.index }}</td>
      <td><a href="/?who={{ c.name|urlencode }}">{{ c.name }}</a></td>
      <td class="num">{{ '{:,}'.format(c.count) }}</td>
      <td class="alias">{{ c.aliases }}</td></tr>
  {% endfor %}</table>
</section>

<section>
  <h2>{% if who %}Ties &middot; {{ who }}{% else %}Network structure{% endif %}</h2>
  {% if who %}
    <p class="note"><a href="/">&larr; all characters</a> &middot; shared narrative windows</p>
    <table><tr><th>Character</th><th class="num">Shared scenes</th><th></th></tr>
    {% for n,w in ties %}<tr><td><a href="/?who={{ n|urlencode }}">{{ n }}</a></td>
      <td class="num">{{ '{:,}'.format(w) }}</td>
      <td><a href="/?a={{ who|urlencode }}&b={{ n|urlencode }}">timeline &rarr;</a></td></tr>{% endfor %}</table>
    {% if who_arc %}
    <p class="note" style="margin-top:14px">Tension arc across passages mentioning {{ who }}:</p>
    {% for b in who_arc %}
    <div class="bookrow"><div class="s">{{ b.title }}</div>
      <div class="spark" style="font-size:15px">{{ b.spark }}</div></div>
    {% endfor %}
    {% endif %}
  {% else %}
    <p class="note">Weighted degree = interaction volume. Betweenness = brokerage between groups.</p>
    <table><tr><th>Volume</th><th class="num"></th><th>Brokerage</th><th class="num"></th></tr>
    {% for row in netrows %}
    <tr><td>{{ row[0] }}</td><td class="num">{{ row[1] }}</td>
        <td>{{ row[2] }}</td><td class="num">{{ row[3] }}</td></tr>
    {% endfor %}</table>
  {% endif %}
</section>

<section class="wide">
  <h2>Relationship timeline</h2>
  <form method="get" action="/">
    <input type="text" name="a" value="{{ pa|e }}" placeholder="First character (e.g. Harry)">
    <input type="text" name="b" value="{{ pb|e }}" placeholder="Second character (e.g. Ginny)">
    <button type="submit">Chart</button>
  </form>
  {% if pair_rows %}
    <p class="note">Shared passages per chapter for <b>{{ pa }}</b> + <b>{{ pb }}</b>. A flat line
      means the relationship does not exist yet; the shape shows when it begins.</p>
    {% for r in pair_rows %}
    <div class="bookrow"><div class="t">{{ r.title }} <span class="s">&middot; {{ r.total }} shared</span></div>
      <div class="spark">{{ r.spark }}</div>
      {% if r.peak %}<div class="s">peak ch.{{ r.peak_ch }} '{{ r.peak_title }}' with {{ r.peak }} shared passages</div>{% endif %}
    </div>
    {% endfor %}
  {% elif pa or pb %}
    <p class="note">Could not resolve both names. Check the character list on the left.</p>
  {% else %}
    <p class="note">Try <a href="/?a=Harry&b=Ginny">Harry + Ginny</a> or
      <a href="/?a=Snape&b=Voldemort">Snape + Voldemort</a>.</p>
  {% endif %}
</section>

<section class="wide">
  <h2>Narrative arcs</h2>
  <p class="note">Z-scored projection onto a semantic axis defined by probe sentences, smoothed over 9
    passages. Measures similarity to the axis poles, not calibrated emotion.
    {% for a in axes %}<a href="/?axis={{ a }}{% if q %}&q={{ q|urlencode }}&mode={{ mode }}{% endif %}">{{ a }}</a>{% if not loop.last %} &middot; {% endif %}{% endfor %}</p>
  {% for b in arcs %}
  <div class="bookrow"><div class="t">{{ b.title }}</div>
    <div class="spark">{{ b.spark }}</div>
    <div class="s">axis <b>{{ axis }}</b> &middot; mean {{ '%+.2f'|format(b.mean) }} &middot; peak {{ '%+.2f'|format(b.peak) }} at ch.{{ b.peak_ch }}</div>
  </div>
  {% endfor %}
</section>

<section>
  <h2>Communities</h2>
  <p class="note">Greedy modularity over the weighted graph.</p>
  {% for c in communities %}
  <div class="comm"><span class="n">Group {{ loop.index }} &middot; {{ c|length }} members</span>
    <div class="chips">{% for m in c %}<span class="chip"><b>{{ m }}</b></span>{% endfor %}</div></div>
  {% endfor %}
</section>

<section>
  <h2>Distinctive vocabulary</h2>
  <p class="note">Log-odds with an informative Dirichlet prior, z-scored. Over-represented versus the rest of the corpus.</p>
  {% for book, terms in distinctive %}
  <div class="comm"><span class="n">{{ book }}</span>
    <div class="chips">{% for w,z in terms %}<span class="chip"><b>{{ w }}</b> {{ '%.1f'|format(z) }}</span>{% endfor %}</div></div>
  {% endfor %}
</section>

<section>
  <h2>Corpus statistics</h2>
  <table>
    {% for k,v in stats %}<tr><td style="color:var(--dim)">{{ k }}</td><td class="num">{{ v }}</td></tr>{% endfor %}
  </table>
</section>

</div>
</main></body></html>
"""
)


def _render(art, params: dict) -> str:
    q = (params.get("q", [""])[0] or "").strip()
    mode = params.get("mode", ["fused"])[0]
    who = (params.get("who", [""])[0] or "").strip()
    axis = params.get("axis", ["tension"])[0]
    if axis not in art.arcs:
        axis = "tension"

    hits = []
    if q:
        if mode == "lexical":
            raw = search_mod.kwic(q.split()[0], art.passages, k=8)
        elif mode == "semantic":
            raw = search_mod.dense_search(art.model(), q, art.embeddings, art.passages, k=8)
        else:
            raw = search_mod.fused_search(art.model(), q, art.embeddings, art.passages, k=8)
        hits = [h.to_dict() for h in raw]

    def _resolve(name: str) -> str | None:
        if not name:
            return None
        canon = art.alias_index.get(name.strip().lower())
        if canon:
            return canon
        from rapidfuzz import process

        match = process.extractOne(name.strip(), [c.name for c in art.characters], score_cutoff=60)
        return match[0] if match else None

    ties = []
    who_arc = []
    if who:
        resolved = _resolve(who)
        who = resolved or who
        graph = net_mod.from_json(art.graph_json)
        if who in graph:
            ties = [(n, d["weight"]) for n, d in
                    sorted(graph[who].items(), key=lambda kv: -kv[1]["weight"])[:15]]
        if resolved:
            mentioned = art.mentions()
            tension = np.asarray(art.arcs["tension"])
            for book in art.books:
                idx = [i for i, p in enumerate(art.passages)
                       if p.book == book and who in mentioned[i]]
                if len(idx) >= 5:
                    who_arc.append({"title": book, "spark": arc_mod.sparkline(tension[idx], width=64)})

    pa = (params.get("a", [""])[0] or "").strip()
    pb = (params.get("b", [""])[0] or "").strip()
    pair_rows = []
    ra, rb = _resolve(pa), _resolve(pb)
    if ra and rb and ra != rb:
        pa, pb = ra, rb
        mentioned = art.mentions()
        for book in art.books:
            chapters: dict[int, int] = {}
            titles: dict[int, str] = {}
            for i, p in enumerate(art.passages):
                if p.book != book:
                    continue
                chapters.setdefault(p.chapter, 0)
                titles.setdefault(p.chapter, p.chapter_title)
                if ra in mentioned[i] and rb in mentioned[i]:
                    chapters[p.chapter] += 1
            ordered = sorted(chapters)
            counts = np.asarray([chapters[c] for c in ordered], dtype=float)
            total = int(counts.sum())
            peak_ch = max(ordered, key=lambda c: chapters[c]) if total else 0
            pair_rows.append({
                "title": book,
                "total": total,
                "spark": arc_mod.sparkline(counts, width=72) if total else "▁" * min(len(ordered), 72),
                "peak": chapters.get(peak_ch, 0) if total else 0,
                "peak_ch": peak_ch,
                "peak_title": titles.get(peak_ch, ""),
            })

    series = np.asarray(art.arcs[axis])
    arcs = []
    for book in art.books:
        idx = [i for i, p in enumerate(art.passages) if p.book == book]
        if not idx:
            continue
        vals = series[idx]
        peak_local = int(np.argmax(vals))
        arcs.append({
            "title": book,
            "spark": arc_mod.sparkline(vals, width=72),
            "mean": float(vals.mean()),
            "peak": float(vals.max()),
            "peak_ch": art.passages[idx[peak_local]].chapter,
        })

    cent = list(art.graph_analysis["centrality"].items())[:15]
    betw = list(art.graph_analysis["betweenness"].items())[:15]
    netrows = []
    for i in range(max(len(cent), len(betw))):
        c = cent[i] if i < len(cent) else ("", 0)
        b = betw[i] if i < len(betw) else ("", 0)
        netrows.append((c[0], f"{int(c[1]):,}" if c[0] else "",
                        b[0], f"{b[1]:.3f}" if b[0] else ""))

    s = art.stats
    stat_rows = [
        ("word types", f"{s['types']:,}"),
        ("type/token ratio", f"{s['type_token_ratio']:.4f}"),
        ("hapax legomena", f"{s['hapax_legomena']:,}"),
        ("mean sentence length", f"{s['mean_sentence_len']:.1f}"),
        ("dialogue share", f"{s['dialogue_share'] * 100:.1f}%"),
        ("encoder", art.meta["model"].split("/")[-1]),
        ("build time", f"{art.meta['build_seconds']}s"),
    ]

    return PAGE.render(
        s=s,
        stats=stat_rows,
        chars=len(art.characters),
        q=q,
        mode=mode,
        hits=hits,
        who=who,
        ties=ties,
        who_arc=who_arc,
        pa=pa,
        pb=pb,
        pair_rows=pair_rows,
        axis=axis,
        axes=list(art.arcs),
        arcs=arcs,
        netrows=netrows,
        characters=[
            {"name": c.name, "count": c.count,
             "aliases": ", ".join(sorted(c.aliases - {c.name})[:4]) or "-"}
            for c in art.characters[:25]
        ],
        communities=art.graph_analysis["communities"][:5],
        distinctive=[(b, t[:12]) for b, t in list(art.distinctive.items())],
    )


def serve(art, host: str = "127.0.0.1", port: int = 8000, console=None) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            parsed = urlparse(self.path)
            if parsed.path == "/api/graph":
                body = json.dumps(art.graph_json).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path not in ("/", "/index.html"):
                self.send_error(404)
                return
            try:
                body = _render(art, parse_qs(parsed.query)).encode("utf-8")
            except Exception as exc:  # noqa: BLE001 - show the error, never a blank page
                body = f"<pre>render error: {exc}</pre>".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:  # keep the terminal clean
            return

    art.model()  # warm the encoder before accepting requests
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    if console:
        console.print(f"[green]serving[/green] {url}  [dim](ctrl-c to stop)[/dim]")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        if console:
            console.print("\nstopped")
    finally:
        server.server_close()
