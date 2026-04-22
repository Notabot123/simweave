"""A* (and Dijkstra) shortest path over a simeng / networkx / dict graph.

Pure stdlib; networkx is not required. If ``heuristic`` is ``None`` the
algorithm degrades to Dijkstra (heuristic = 0 is admissible).
"""

from __future__ import annotations

import heapq
from math import inf, sqrt
from typing import Any, Callable, Hashable

from simeng.spatial.graph import adj_view, edge_weight


Node = Hashable
Heuristic = Callable[[Node, Node], float]


class NoPathError(ValueError):
    """Raised when A* cannot reach the goal from the start node."""


def a_star(
    graph: Any,
    start: Node,
    goal: Node,
    heuristic: Heuristic | None = None,
    weight_attr: str = "weight",
) -> list[Node]:
    """Compute shortest path from ``start`` to ``goal``.

    Parameters
    ----------
    graph:
        Any object understood by :func:`simeng.spatial.graph.adj_view`.
    heuristic:
        Admissible heuristic ``h(node, goal)``. Defaults to zero (Dijkstra).
    weight_attr:
        Edge attribute to treat as weight. Defaults to ``"weight"``.

    Returns
    -------
    list[Node]
        Path from start to goal inclusive.
    """
    if heuristic is None:

        def heuristic(a: Node, b: Node) -> float:
            return 0.0

    open_heap: list[tuple[float, int, Node]] = [(0.0, 0, start)]
    came_from: dict[Node, Node] = {}
    g_score: dict[Node, float] = {start: 0.0}
    counter = 1
    closed: set[Node] = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            return _reconstruct(came_from, current)
        closed.add(current)

        for neighbor, edge_data in adj_view(graph, current).items():
            if neighbor in closed:
                continue
            w = edge_weight(edge_data, weight_attr)
            tentative = g_score[current] + w
            if tentative < g_score.get(neighbor, inf):
                g_score[neighbor] = tentative
                came_from[neighbor] = current
                f = tentative + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f, counter, neighbor))
                counter += 1

    raise NoPathError(f"No path from {start!r} to {goal!r}.")


def dijkstra(
    graph: Any, start: Node, goal: Node, weight_attr: str = "weight"
) -> list[Node]:
    """Convenience wrapper -- A* with zero heuristic."""
    return a_star(graph, start, goal, heuristic=None, weight_attr=weight_attr)


def _reconstruct(came_from: dict[Node, Node], current: Node) -> list[Node]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Heuristics for 2D grid-like nodes (tuples of (row, col) or (x, y)).
# ---------------------------------------------------------------------------


def manhattan(a: Node, b: Node) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # type: ignore[index]


def euclidean(a: Node, b: Node) -> float:
    dx = a[0] - b[0]  # type: ignore[index]
    dy = a[1] - b[1]  # type: ignore[index]
    return sqrt(dx * dx + dy * dy)


def chebyshev(a: Node, b: Node) -> float:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))  # type: ignore[index]
