"""Internal helper: lazy import plotly with a friendly error.

simweave's core does not depend on plotly. Users who want viz install via
the optional extra ``simweave[viz]``. Importing :mod:`simweave.viz` at the
top level is cheap; the heavy import only happens when a plot helper is
actually called.
"""

from __future__ import annotations

from typing import Any


_INSTALL_HINT = (
    "simweave.viz requires plotly. Install with `pip install simweave[viz]` "
    "or `pip install plotly>=5.18`."
)


def require_go() -> Any:
    """Return ``plotly.graph_objects`` or raise a clear ImportError."""
    try:
        import plotly.graph_objects as go  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - exercised when plotly absent
        raise ImportError(_INSTALL_HINT) from e
    return go


def have_plotly() -> bool:
    """Cheap probe: does the current environment have plotly available?"""
    try:
        import plotly  # noqa: F401
    except ImportError:
        return False
    return True
