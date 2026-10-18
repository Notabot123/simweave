"""Regression: ``import simweave`` must work without plotly installed.

plotly lives behind the ``viz`` extra, so the base install (numpy only) has
to import cleanly, and plot helpers must fail with the friendly
``ImportError`` rather than ``ModuleNotFoundError`` at import time.

Each scenario runs in a fresh subprocess with an import-blocking meta path
hook so it is independent of whatever the dev environment has installed.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import simweave

# Subprocesses must import the same simweave as this test run (pytest's
# ``pythonpath = ["src"]`` does not propagate to child processes).
_PKG_ROOT = str(Path(simweave.__file__).resolve().parent.parent)

_BLOCK_PLOTLY = textwrap.dedent(
    """
    import sys
    from importlib.abc import MetaPathFinder

    class _Block(MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "plotly" or name.startswith("plotly."):
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return None

    sys.meta_path.insert(0, _Block())
    for _m in [m for m in sys.modules if m == "plotly" or m.startswith("plotly.")]:
        del sys.modules[_m]
    """
)


def _run(body: str) -> subprocess.CompletedProcess:
    code = _BLOCK_PLOTLY + textwrap.dedent(body)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [_PKG_ROOT, env.get("PYTHONPATH")]))
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


def test_import_simweave_without_plotly():
    proc = _run(
        """
        import simweave
        assert "plotly" not in sys.modules
        assert simweave.have_plotly() is False
        print(simweave.__version__)
        """
    )
    assert proc.returncode == 0, proc.stderr


def test_plot_helpers_raise_friendly_import_error_without_plotly():
    proc = _run(
        """
        import numpy as np
        import simweave
        from simweave.viz import vehicle_dynamics
        from simweave.viz.plots import plot_state_trajectories

        class _R:
            time = np.linspace(0.0, 1.0, 5)
            states = np.zeros((5, 2))

        for fn in (plot_state_trajectories, vehicle_dynamics.plot_vehicle_metrics):
            try:
                fn(_R())
            except ImportError as e:
                assert "pip install simweave[viz]" in str(e), str(e)
            else:
                raise SystemExit(f"{fn.__name__} did not raise ImportError")
        """
    )
    assert proc.returncode == 0, proc.stderr


def test_flight_imports_without_plotly():
    proc = _run(
        """
        import simweave.flight as flight
        assert "plotly" not in sys.modules
        assert "scipy" not in sys.modules
        flight.Quadrotor()
        """
    )
    assert proc.returncode == 0, proc.stderr
