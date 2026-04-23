# Install

SimWeave targets Python 3.10 and newer.

## Base install

```bash
pip install simweave
```

This installs only NumPy. The base install is enough to:

- build and run any `SimEnvironment`
- use every queue, service, generator, agent, warehouse and recorder
- run Monte Carlo replications
- work with `Money` and the currency system
- work with the SI units module

## Optional extras

Pick the extras you need. They are additive — install several at once:

```bash
pip install "simweave[viz,optim,graph]"
```

| Extra        | Pulls in              | Enables                                                    |
| ------------ | --------------------- | ---------------------------------------------------------- |
| `viz`        | `plotly>=5.18`        | Every `simweave.viz` plot helper                           |
| `plot`       | `matplotlib>=3.7`     | Legacy matplotlib helpers (kept for backward compat)       |
| `optim`      | `scipy>=1.10`         | Stiff ODE solvers, optimisation utilities                  |
| `graph`      | `networkx>=3.0`       | Larger / richer graphs for `simweave.spatial`              |
| `geo`        | `osmnx`, `networkx`   | Building graphs from real-world map data                   |
| `fast`       | `numba>=0.58`         | JIT-compiled hot loops                                     |
| `intl`       | `babel>=2.14`         | Locale-aware money formatting                              |
| `dev`        | (everything for tests) | Full test suite (`pytest`, `pytest-cov`, plotly, scipy …)  |
| `all`        | (everything runtime)   | All runtime extras at once                                 |

## Development install

For working on SimWeave itself:

```bash
git clone https://github.com/swhipp87/simweave.git
cd simweave
pip install -e ".[dev]"
pytest -q
```

## Building the docs locally

Documentation uses MkDocs Material with mkdocstrings:

```bash
pip install -e ".[docs]"
mkdocs serve
```

Then open `http://127.0.0.1:8000`. Pages auto-reload on edit, including
the API reference (which reads docstrings from the live source tree).
