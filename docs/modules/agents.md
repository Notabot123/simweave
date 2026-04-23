# Agents and spatial

Graphs, shortest-path algorithms, and an `Agent` that traverses them.

## Graphs

```python
import simweave as sw

g = sw.grid_graph(rows=8, cols=12, diagonal=True)
```

`grid_graph` returns a `simweave.spatial.Graph` whose nodes are
`(row, col)` tuples. Bring your own graph by either constructing
`Graph` directly or using a NetworkX graph with the `[graph]` extra.

## Path planners

- `a_star(graph, start, goal, heuristic)`
- `dijkstra(graph, start, goal)`
- Heuristics: `manhattan`, `euclidean`, `chebyshev`
- Errors: `NoPathError` if no route exists

## Agents

```python
agent = sw.Agent(
    graph=g,
    start_node=(0, 0),
    tasks=[(7, 11), (3, 4), (0, 11)],
    speed=2.0,
    heuristic=sw.manhattan,
    name="rover",
)
env = sw.SimEnvironment(dt=0.5, end=50.0)
env.register(agent)
env.run()

sw.plot_agent_path(agent, graph=g).write_html("path.html",
                                              include_plotlyjs="cdn")
```

<iframe src="../../embeds/agent_path.html"
        width="100%" height="560" frameborder="0"
        loading="lazy"
        title="A* agent traversal over an 8x12 grid"></iframe>

`Compass` is a small enum for cardinal-direction logic if you need it
when authoring custom agent behaviours.

## API: `simweave.agents`

::: simweave.agents
    options:
      show_root_heading: false
      show_source: true

## API: `simweave.spatial`

::: simweave.spatial
    options:
      show_root_heading: false
      show_source: true
