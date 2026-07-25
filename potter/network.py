"""Character co-occurrence network and graph analytics.

Stage 3. Two characters sharing a narrative window are treated as interacting.
The window is a sliding run of paragraphs rather than a single paragraph, because
dialogue in fiction alternates across paragraphs - a one-paragraph window would
miss almost every two-person conversation, which is exactly the signal we want.

Edge weight = number of windows in which the pair co-occurs.
"""

from __future__ import annotations

import networkx as nx

from .characters import find_mentions
from .corpus import Passage


def build_graph(
    passages: list[Passage],
    alias_index: dict[str, str],
    window: int = 3,
    min_weight: int = 2,
) -> nx.Graph:
    """Build a weighted, undirected co-occurrence graph over resolved characters."""
    mentions_per_passage = [find_mentions(p.text, alias_index) for p in passages]

    graph = nx.Graph()
    for i, passage in enumerate(passages):
        # Only slide the window within a single chapter - a scene break should not
        # create a spurious interaction across a chapter boundary.
        window_sets = []
        for j in range(i, min(i + window, len(passages))):
            if (passages[j].book_index, passages[j].chapter) != (passage.book_index, passage.chapter):
                break
            window_sets.append(mentions_per_passage[j])

        present = sorted(set().union(*window_sets)) if window_sets else []
        for name in present:
            if graph.has_node(name):
                graph.nodes[name]["mentions"] += 1
            else:
                graph.add_node(name, mentions=1)
        for a_idx, a in enumerate(present):
            for b in present[a_idx + 1 :]:
                if graph.has_edge(a, b):
                    graph[a][b]["weight"] += 1
                else:
                    graph.add_edge(a, b, weight=1)

    # Prune incidental single co-occurrences, which are mostly list-of-names noise.
    weak = [(u, v) for u, v, d in graph.edges(data=True) if d["weight"] < min_weight]
    graph.remove_edges_from(weak)
    graph.remove_nodes_from([n for n in list(graph.nodes) if graph.degree(n) == 0])
    return graph


def analyse(graph: nx.Graph) -> dict:
    """Centrality and community structure."""
    if graph.number_of_nodes() == 0:
        return {"centrality": {}, "betweenness": {}, "communities": [], "density": 0.0}

    # Weighted degree = raw interaction volume. Betweenness on inverse weight,
    # because a heavy edge is a SHORT social distance, not a long one.
    strength = {n: sum(d["weight"] for _, _, d in graph.edges(n, data=True)) for n in graph.nodes}
    distance = {(u, v): 1.0 / d["weight"] for u, v, d in graph.edges(data=True)}
    nx.set_edge_attributes(graph, {k: {"distance": v} for k, v in distance.items()})

    try:
        betweenness = nx.betweenness_centrality(graph, weight="distance")
    except Exception:  # noqa: BLE001 - tiny/degenerate graphs
        betweenness = {n: 0.0 for n in graph.nodes}

    try:
        communities = [sorted(c) for c in nx.community.greedy_modularity_communities(graph, weight="weight")]
    except Exception:  # noqa: BLE001
        communities = [sorted(graph.nodes)]

    return {
        "centrality": dict(sorted(strength.items(), key=lambda kv: -kv[1])),
        "betweenness": dict(sorted(betweenness.items(), key=lambda kv: -kv[1])),
        "communities": communities,
        "density": nx.density(graph),
    }


def to_json(graph: nx.Graph) -> dict:
    return {
        "nodes": [{"id": n, "mentions": graph.nodes[n].get("mentions", 0)} for n in graph.nodes],
        "edges": [{"source": u, "target": v, "weight": d["weight"]} for u, v, d in graph.edges(data=True)],
    }


def from_json(payload: dict) -> nx.Graph:
    graph = nx.Graph()
    for node in payload.get("nodes", []):
        graph.add_node(node["id"], mentions=node.get("mentions", 0))
    for edge in payload.get("edges", []):
        graph.add_edge(edge["source"], edge["target"], weight=edge["weight"])
    return graph
