"""Path shim so demos work even when ``simeng`` hasn't been pip-installed.

The ``src/`` layout means ``import simeng`` only resolves after
``pip install -e .`` (or with ``PYTHONPATH=src``). This shim adds
``<repo>/src`` to ``sys.path`` so the demos can be run directly as
scripts without any prior setup.

The recommended workflow for real work is still::

    python -m venv .venv && source .venv/bin/activate
    pip install -e .[dev]
    python demos/01_simple_queue.py

Importing this module is harmless once the package is installed properly.
"""
from __future__ import annotations

import pathlib
import sys


_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
