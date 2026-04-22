"""Routable agent that inherits from :class:`~simweave.core.entity.Entity`.

An Agent traverses a graph under its own steam. On each tick:

* If it has no plan and has tasks queued, it replans to the next task via A*.
* If it is mid-edge, it consumes distance according to ``speed * dt``.
* On arrival at a node, it pops it from the path, records history, and may
  fire ``on_arrive(node)``.

The Agent holds only a reference to the graph; it does not own it. Several
agents may share a graph.
"""

from __future__ import annotations

from typing import Any, Callable, Hashable, Sequence

from simweave.core.entity import Entity
from simweave.core.logging import get_logger
from simweave.agents.compass import Compass
from simweave.agents.routing import a_star, NoPathError
from simweave.spatial.graph import adj_view, edge_weight

log = get_logger("agents.agent")

Node = Hashable


class Agent(Entity):
    """Graph-routing agent.

    Parameters
    ----------
    graph:
        A simweave Graph, networkx graph, or dict-of-dict adjacency.
    start_node:
        Node at which the agent spawns.
    tasks:
        Ordered list of target nodes to visit.
    speed:
        Distance per unit time.
    heuristic:
        Optional heuristic for A*. If None, A* degenerates to Dijkstra.
    on_arrive:
        Optional callback ``on_arrive(agent, node, env)`` fired when the
        agent arrives at any node (including intermediate path nodes).
    name:
        Optional agent name.
    """

    def __init__(
        self,
        graph: Any,
        start_node: Node,
        tasks: Sequence[Node] | None = None,
        speed: float = 1.0,
        heuristic: Callable[[Node, Node], float] | None = None,
        on_arrive: Callable[["Agent", Node, Any], None] | None = None,
        weight_attr: str = "weight",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.graph = graph
        self.position: Node = start_node
        self.tasks: list[Node] = list(tasks or [])
        self.path: list[Node] = []
        self.speed = float(speed)
        self.heuristic = heuristic
        self.compass = Compass(points=8)
        self.weight_attr = weight_attr
        self.on_arrive = on_arrive

        self._edge_remaining: float = 0.0
        self._edge_weight: float = 0.0
        self.history: list[tuple[float, Node]] = []
        self.completed_tasks: int = 0

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def plan_to(self, target: Node) -> None:
        """Replan to ``target`` from the current position using A*."""
        try:
            path = a_star(
                self.graph,
                self.position,
                target,
                heuristic=self.heuristic,
                weight_attr=self.weight_attr,
            )
        except NoPathError:
            log.warning("%s: no path from %s to %s", self.name, self.position, target)
            self.path = []
            return
        # path[0] is the current position; drop it.
        self.path = path[1:] if path and path[0] == self.position else path

    def _take_next_task(self) -> None:
        while self.tasks and not self.path:
            target = self.tasks.pop(0)
            if target == self.position:
                # Already here; count as arrival.
                self.completed_tasks += 1
                if self.on_arrive is not None:
                    self.on_arrive(self, target, None)
                continue
            self.plan_to(target)

    # ------------------------------------------------------------------
    # Tick loop
    # ------------------------------------------------------------------
    def tick(self, dt: float, env) -> None:
        super().tick(dt, env)
        if not self.path:
            self._take_next_task()
        if not self.path:
            return

        remaining_step = self.speed * dt
        while remaining_step > 0 and self.path:
            if self._edge_remaining <= 0:
                # Begin traversing next edge.
                next_node = self.path[0]
                nbrs = adj_view(self.graph, self.position)
                if next_node not in nbrs:
                    log.warning(
                        "%s: edge %s->%s missing; replanning.",
                        self.name,
                        self.position,
                        next_node,
                    )
                    if self.path:
                        target = self.path[-1]
                        self.path = []
                        self.plan_to(target)
                    return
                self._edge_weight = edge_weight(nbrs[next_node], self.weight_attr)
                self._edge_remaining = self._edge_weight

            step = min(remaining_step, self._edge_remaining)
            self._edge_remaining -= step
            remaining_step -= step

            if self._edge_remaining <= 1e-12:
                self.position = self.path.pop(0)
                self.history.append((env.clock.t + dt, self.position))
                if self.on_arrive is not None:
                    self.on_arrive(self, self.position, env)
                # If we've exhausted the current task plan, count the
                # arrival as a completed task regardless of whether more
                # tasks are queued behind it.
                if not self.path:
                    self.completed_tasks += 1

        # If the path finished and more tasks queued, plan immediately so the
        # next tick doesn't waste a step standing still.
        if not self.path and self.tasks:
            self._take_next_task()

    def has_work(self, env) -> bool:
        return bool(self.path) or bool(self.tasks)
