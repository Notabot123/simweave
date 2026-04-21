"""Agents routing on a 2-D grid using A*.

Two agents with different speeds visit three waypoints each. Their
positions are recorded per-tick and plotted as a simple ASCII trace at
the end. The ``on_arrive`` callback fires whenever an agent reaches any
node along its planned path.

Run:
    python demos/06_agent_astar.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (adds src/ to sys.path when not pip-installed)

from simeng.core.environment import SimEnvironment
from simeng.spatial.graph import grid_graph
from simeng.agents.agent import Agent
from simeng.agents.routing import manhattan


def on_arrive(agent, node, env):
    t = env.clock.t if env is not None else None
    print(f"  t={t}: {agent.name} reached {node}")


def main() -> None:
    g = grid_graph(8, 8)

    fast = Agent(graph=g, start_node=(0, 0),
                 tasks=[(7, 2), (3, 7), (0, 5)],
                 speed=2.0, heuristic=manhattan,
                 on_arrive=on_arrive, name="courier_fast")

    slow = Agent(graph=g, start_node=(7, 7),
                 tasks=[(0, 0), (5, 3)],
                 speed=0.5, heuristic=manhattan,
                 on_arrive=on_arrive, name="courier_slow")

    env = SimEnvironment(dt=1.0, end=60.0)
    env.register(fast); env.register(slow)
    env.run()

    print()
    print(f"fast completed: {fast.completed_tasks}/3  at {fast.position}")
    print(f"slow completed: {slow.completed_tasks}/2  at {slow.position}")

    # ASCII trace of the fast courier.
    print("\nfast courier trace:")
    visited = {pos for _, pos in fast.history}
    for r in range(8):
        row = "".join("*" if (r, c) in visited else "." for c in range(8))
        print(f"  {row}")


if __name__ == "__main__":
    main()
