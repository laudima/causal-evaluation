"""Failure graph: where a model's errors co-occur across query types.

Nodes are strata (by default rung x query type). The weight of an edge between
strata A and B is the lift P(fail_B | fail_A) / P(fail_B), computed over pairs of
items that share a group (by default the same variant, story and graph). The
lift is symmetric; edges point from lower to higher rung only as a layout
convention. The graph describes errors; it is not a causal graph.
"""

from itertools import combinations

import pandas as pd


def failure_graph(
    df: pd.DataFrame,
    node_cols: tuple[str, ...] = ("rung", "query_type"),
    group_cols: tuple[str, ...] = ("variant", "story_id", "graph_id"),
    min_pairs: int = 30,
    min_lift: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (nodes, edges). `df` needs a boolean `correct` column plus node/group columns."""
    data = df.assign(
        fail=~df["correct"].astype(bool), node=list(zip(*[df[c] for c in node_cols], strict=True))
    )
    nodes = (
        data.groupby("node")["fail"]
        .agg(n="size", errors="sum")
        .assign(error_rate=lambda t: t["errors"] / t["n"])
        .reset_index()
    )
    counts = data.groupby([*group_cols, "node"])["fail"].agg(n="size", f="sum").reset_index()
    per_group = counts.groupby(list(group_cols))
    totals: dict[tuple, list[float]] = {}
    for _, group in per_group:
        rows = list(group[["node", "n", "f"]].itertuples(index=False))
        for a, b in combinations(sorted(rows, key=lambda r: tuple(map(str, r.node))), 2):
            first, second = (a, b) if _order(a.node) <= _order(b.node) else (b, a)
            acc = totals.setdefault((first.node, second.node), [0.0, 0.0, 0.0, 0.0])
            acc[0] += first.n * second.n  # item pairs
            acc[1] += first.f * second.f  # both fail
            acc[2] += first.f * second.n  # first fails
            acc[3] += first.n * second.f  # second fails
    edges = []
    for (source, target), (pairs, both, first_fail, second_fail) in totals.items():
        if pairs < min_pairs or first_fail == 0 or second_fail == 0:
            continue
        lift = (both / first_fail) / (second_fail / pairs)
        if lift > min_lift:
            edges.append({"source": source, "target": target, "lift": lift, "pairs": int(pairs)})
    return nodes, pd.DataFrame(edges, columns=["source", "target", "lift", "pairs"])


def _order(node: tuple) -> tuple:
    return tuple(str(part) for part in node)


SHORT_LABELS = {
    "marginal": "marginal",
    "correlation": "cond. prob.",
    "exp_away": "expl. away",
    "collider_bias": "collider",
    "backadj": "adj. set",
    "ate": "ATE",
    "ett": "ETT",
    "nde": "NDE",
    "nie": "NIE",
    "det-counterfactual": "counterf.",
}


def draw_failure_graph(nodes: pd.DataFrame, edges: pd.DataFrame, ax=None, title: str | None = None):
    """Draw nodes in columns by rung (first node field), shaded by error rate."""
    import matplotlib.pyplot as plt
    import networkx as nx

    graph = nx.DiGraph()
    for row in nodes.itertuples(index=False):
        graph.add_node(row.node, error_rate=row.error_rate)
    for row in edges.itertuples(index=False):
        graph.add_edge(row.source, row.target, lift=row.lift)
    columns: dict = {}
    for node in sorted(graph.nodes, key=_order):
        columns.setdefault(node[0], []).append(node)
    pos = {
        node: (float(col), -float(i))
        for col, members in columns.items()
        for i, node in enumerate(members)
    }
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    colors = [graph.nodes[n]["error_rate"] for n in graph.nodes]
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=colors,
        cmap="Reds",
        vmin=0,
        vmax=1,
        node_size=1500,
        node_shape="s",
        edgecolors="black",
    )
    labels = {
        n: f"{SHORT_LABELS.get(n[1], n[1])}\n{graph.nodes[n]['error_rate']:.2f}"
        for n in graph.nodes
    }
    nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=7)
    if graph.number_of_edges():
        widths = [0.8 + 2.0 * (graph.edges[e]["lift"] - 1) for e in graph.edges]
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            width=widths,
            edge_color="gray",
            arrows=True,
            connectionstyle="arc3,rad=0.1",
            node_size=1500,
            node_shape="s",
        )
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            ax=ax,
            font_size=6,
            edge_labels={e: f"{graph.edges[e]['lift']:.1f}" for e in graph.edges},
        )
    for col in columns:
        ax.text(float(col), 0.6, f"L{col}", ha="center", fontsize=9, fontweight="bold")
    ax.margins(x=0.25, y=0.15)
    ax.set_title(title or "", fontsize=9, pad=18)
    ax.axis("off")
    return ax
