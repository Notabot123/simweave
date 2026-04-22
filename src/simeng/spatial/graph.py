"""Minimal, dependency-free graph representation plus a networkx adapter.

The goal is to let :mod:`simeng.agents.routing` operate on either a pure
Python ``dict``-of-``dict`` adjacency, or a networkx graph, without pulling
networkx into the hard dependency list.

Adjacency format
----------------
``{u: {v: edge_data, ...}, ...}`` where ``edge_data`` is either:

* a ``dict`` with a ``weight`` key (default key name; configurable), or
* a bare number, interpreted as the edge weight.

Both directed and undirected are supported -- the class does not enforce
symmetry, so :meth:`Graph.add_edge` with ``directed=False`` inserts both
``u->v`` and ``v->u``.
"""

from __future__ import annotations

from typing import Any, Hashable, Iterable


Node = Hashable


class Graph:
    """A tiny adjacency-dict graph. Optional subset of the networkx API."""

    def __init__(self, directed: bool = False) -> None:
        self._adj: dict[Node, dict[Node, dict[str, Any]]] = {}
        self.directed = directed

    def __contains__(self, node: Node) -> bool:
        return node in self._adj

    def __getitem__(self, node: Node) -> dict[Node, dict[str, Any]]:
        return self._adj[node]

    def __iter__(self):
        return iter(self._adj)

    def __len__(self) -> int:
        return len(self._adj)

    @property
    def nodes(self) -> Iterable[Node]:
        return tuple(self._adj)

    def edges(self) -> Iterable[tuple[Node, Node, dict[str, Any]]]:
        seen: set[tuple[Node, Node]] = set()
        for u, nbrs in self._adj.items():
            for v, data in nbrs.items():
                if not self.directed and (v, u) in seen:
                    continue
                seen.add((u, v))
                yield u, v, data

    def add_node(self, node: Node) -> None:
        self._adj.setdefault(node, {})

    def add_edge(self, u: Node, v: Node, weight: float = 1.0, **attrs: Any) -> None:
        self.add_node(u)
        self.add_node(v)
        data = {"weight": float(weight), **attrs}
        self._adj[u][v] = data
        if not self.directed:
            self._adj[v][u] = data

    def neighbors(self, node: Node) -> Iterable[Node]:
        return tuple(self._adj[node])

    def degree(self, node: Node) -> int:
        return len(self._adj[node])


def adj_view(graph: Any, node: Node) -> dict[Node, dict[str, Any]]:
    """Return ``{neighbor: edge_data_dict}`` for *either* a simeng Graph or a networkx graph."""
    # simeng Graph or any dict-of-dict
    if isinstance(graph, Graph):
        return graph[node]
    if isinstance(graph, dict):
        nbrs = graph[node]
        return {
            v: (d if isinstance(d, dict) else {"weight": float(d)})
            for v, d in nbrs.items()
        }
    # networkx-style: has .neighbors and __getitem__ returning AtlasView
    if hasattr(graph, "neighbors") and hasattr(graph, "__getitem__"):
        nbrs = graph[node]
        # Ensure each edge dict has a weight key
        result: dict[Node, dict[str, Any]] = {}
        for v in nbrs:
            data = nbrs[v]
            if isinstance(data, dict):
                result[v] = data
            else:
                result[v] = {"weight": float(data)}
        return result
    raise TypeError(f"Unsupported graph type: {type(graph).__name__}")


def edge_weight(
    edge_data: Any, weight_attr: str = "weight", default: float = 1.0
) -> float:
    if isinstance(edge_data, dict):
        return float(edge_data.get(weight_attr, default))
    if isinstance(edge_data, (int, float)):
        return float(edge_data)
    return default


def grid_graph(
    nrows: int,
    ncols: int,
    *,
    diagonal: bool = False,
    weight: float = 1.0,
    diagonal_weight: float | None = None,
) -> Graph:
    """Build a 2D lattice graph with nodes ``(r, c)``.

    Parameters
    ----------
    diagonal:
        If True, add 45-degree edges (cost defaults to ``sqrt(2) * weight``).
    """
    if diagonal and diagonal_weight is None:
        diagonal_weight = weight * (2**0.5)

    g = Graph(directed=False)
    for r in range(nrows):
        for c in range(ncols):
            g.add_node((r, c))

    for r in range(nrows):
        for c in range(ncols):
            if r + 1 < nrows:
                g.add_edge((r, c), (r + 1, c), weight=weight)
            if c + 1 < ncols:
                g.add_edge((r, c), (r, c + 1), weight=weight)
            if diagonal:
                if r + 1 < nrows and c + 1 < ncols:
                    g.add_edge((r, c), (r + 1, c + 1), weight=diagonal_weight)
                if r + 1 < nrows and c - 1 >= 0:
                    g.add_edge((r, c), (r + 1, c - 1), weight=diagonal_weight)
    return g
