"""Lightweight logging helpers.

The previous ``print_display`` class has been retired in favour of stdlib
``logging``. Import ``get_logger(name)`` from here for a consistent simeng
logger hierarchy.
"""
from __future__ import annotations

import logging

_ROOT = "simeng"
_configured = False


def configure(level: int | str = logging.INFO,
              fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
              force: bool = False) -> None:
    """Configure the simeng logger hierarchy. Safe to call many times."""
    global _configured
    if _configured and not force:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger(_ROOT)
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    _configured = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the ``simeng`` namespace."""
    if name is None or name == _ROOT:
        return logging.getLogger(_ROOT)
    if name.startswith(_ROOT + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT}.{name}")
