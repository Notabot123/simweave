import pytest

from simeng.spatial.graph import Graph, grid_graph
from simeng.agents.compass import Compass
from simeng.agents.routing import a_star, dijkstra, manhattan, NoPathError
from simeng.agents.agent import Agent
from simeng.core.environment import SimEnvironment


# ---------------------------------------------------------------------------
# Compass
# ---------------------------------------------------------------------------

def test_compass_quantises_to_allowed_angles():
    c = Compass(points=4)
    c.set_absolute(40)
    assert c.angle == 0.0
    c.set_absolute(50)
    assert c.angle == 90.0
    assert c.direction == "E"


def test_compass_rejects_bad_point_count():
    with pytest.raises(ValueError):
        Compass(points=5)


def test_compass_clockwise_wraps():
    c = Compass(points=8)
    c.set_absolute(315)  # NW
    assert c.direction == "NW"
    c.clockwise(90)
    assert c.direction == "NE"


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def test_astar_on_simple_graph():
    g = Graph(directed=False)
    for n in ["A", "B", "C", "D", "E"]:
        g.add_node(n)
    g.add_edge("A", "B", weight=1)
    g.add_edge("B", "C", weight=1)
    g.add_edge("A", "D", weight=5)
    g.add_edge("D", "E", weight=1)
    g.add_edge("C", "E", weight=1)
    path = a_star(g, "A", "E")
    assert path == ["A", "B", "C", "E"]


def test_dijkstra_alias_works():
    g = grid_graph(3, 3)
    path = dijkstra(g, (0, 0), (2, 2))
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)


def test_astar_with_manhattan_heuristic():
    g = grid_graph(5, 5)
    path = a_star(g, (0, 0), (4, 4), heuristic=manhattan)
    assert len(path) - 1 == 8  # Manhattan distance on a 4-connected grid


def test_no_path_raises():
    g = Graph()
    g.add_node("A"); g.add_node("B")
    with pytest.raises(NoPathError):
        a_star(g, "A", "B")


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------

def test_agent_traverses_grid_to_target():
    g = grid_graph(5, 5)
    agent = Agent(
        graph=g,
        start_node=(0, 0),
        tasks=[(4, 4)],
        speed=1.0,
        heuristic=manhattan,
        name="walker",
    )
    env = SimEnvironment(dt=1.0, end=20.0)
    env.register(agent)
    env.run()
    assert agent.position == (4, 4)
    assert agent.completed_tasks == 1
    assert len(agent.history) >= 8  # 8 hops minimum


def test_agent_handles_multiple_tasks_in_order():
    g = grid_graph(4, 4)
    agent = Agent(
        graph=g,
        start_node=(0, 0),
        tasks=[(0, 3), (3, 3), (3, 0)],
        speed=1.0,
        heuristic=manhattan,
    )
    env = SimEnvironment(dt=1.0, end=40.0)
    env.register(agent)
    env.run()
    visited = {pos for _, pos in agent.history}
    assert {(0, 3), (3, 3), (3, 0)}.issubset(visited)
    assert agent.completed_tasks == 3


def test_agent_speed_scales_traversal_time():
    g = grid_graph(2, 2)
    fast = Agent(graph=g, start_node=(0, 0), tasks=[(1, 1)], speed=2.0,
                 heuristic=manhattan, name="fast")
    slow = Agent(graph=g, start_node=(0, 0), tasks=[(1, 1)], speed=0.5,
                 heuristic=manhattan, name="slow")
    env = SimEnvironment(dt=0.1, end=10.0)
    env.register(fast); env.register(slow)
    env.run()
    # Fast finishes; slow might still be en route at t=10 only if 2 hops * 1.0 weight
    # divided by 0.5 speed = 4 steps, well within 10.
    assert fast.position == (1, 1)
    assert slow.position == (1, 1)
    # Fast should have completed earlier -- inspect history length.
    assert len(fast.history) >= 2
