"""Theme registry and helpers for :mod:`simweave.viz`.

A theme is a small bundle of plotly layout settings that every viz helper
applies before returning a figure. The defaults are deliberately minimal so
EdgeWeave (and other downstream consumers of ``fig.to_json()``) can override
freely on the JS side without fighting embedded styling.

Public API
----------
* :func:`set_default_theme(name)` -- choose the theme used when a plot
  helper is called without an explicit ``theme=`` argument.
* :func:`get_default_theme()` -- introspect the current default.
* :func:`register_theme(name, template, palette=None, layout_overrides=None)`
  -- add a custom theme. Once registered, pass ``theme=name`` to any plot
  helper, or set it as the default.
* :func:`apply_theme(fig, theme=None)` -- internal helper, applied
  automatically by every plot helper.
* :func:`available_themes()` -- list registered theme names.

Built-in themes
---------------
* ``"light"`` -- plotly's ``plotly_white`` template, a colour-blind-friendly
  qualitative palette.
* ``"dark"`` -- plotly's ``plotly_dark`` template, same palette tuned for
  dark backgrounds.
* ``"presentation"`` -- plotly's ``presentation`` template (large fonts).
* ``"minimal"`` -- ``simple_white`` template, monochrome grey palette,
  intended for journal-style figures.

Themes are intentionally named with intent (``light``, ``dark``,
``presentation``) rather than brand identity. Users layer brand colours on
top via :func:`register_theme`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Theme:
    """A named plotly styling bundle."""

    name: str
    template: str
    palette: tuple[str, ...] = field(default_factory=tuple)
    layout_overrides: dict[str, Any] = field(default_factory=dict)


# Colour-blind-friendly Okabe & Ito palette plus an extra neutral.
_OKABE_ITO = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#999999",  # grey
)

_GREY_SCALE = (
    "#222222",
    "#555555",
    "#888888",
    "#aaaaaa",
    "#cccccc",
)


_REGISTRY: dict[str, Theme] = {
    "light": Theme(
        name="light",
        template="plotly_white",
        palette=_OKABE_ITO,
    ),
    "dark": Theme(
        name="dark",
        template="plotly_dark",
        palette=_OKABE_ITO,
    ),
    "presentation": Theme(
        name="presentation",
        template="presentation",
        palette=_OKABE_ITO,
        layout_overrides={"margin": {"l": 60, "r": 30, "t": 50, "b": 50}},
    ),
    "minimal": Theme(
        name="minimal",
        template="simple_white",
        palette=_GREY_SCALE,
    ),
}

_DEFAULT_NAME: str = "light"


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def available_themes() -> tuple[str, ...]:
    """Return the names of all currently-registered themes."""
    return tuple(_REGISTRY.keys())


def get_default_theme() -> str:
    """Return the name of the theme used when no ``theme=`` is passed."""
    return _DEFAULT_NAME


def set_default_theme(name: str) -> None:
    """Set the theme applied by every plot helper that omits ``theme=``."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown theme {name!r}. Available: {sorted(_REGISTRY)}. "
            "Use register_theme(...) to add a new one."
        )
    global _DEFAULT_NAME
    _DEFAULT_NAME = name


def register_theme(
    name: str,
    template: str,
    palette: tuple[str, ...] | list[str] | None = None,
    layout_overrides: dict[str, Any] | None = None,
    *,
    overwrite: bool = False,
) -> None:
    """Register a new theme.

    Parameters
    ----------
    name:
        Identifier used by ``set_default_theme(name)`` and ``theme=name``.
    template:
        Name of a plotly built-in template (``"plotly_white"``,
        ``"plotly_dark"``, ``"simple_white"``, ``"ggplot2"``, etc) or any
        template you have already added to ``plotly.io.templates``.
    palette:
        Optional sequence of hex/CSS colour strings used as the
        ``colorway`` for traces.
    layout_overrides:
        Optional mapping merged into ``fig.layout`` after the template is
        applied. Use this for fonts, paper colours, margins, etc.
    overwrite:
        If False (default), raises ``KeyError`` when ``name`` already
        exists. Pass ``overwrite=True`` to replace.
    """
    if not overwrite and name in _REGISTRY:
        raise KeyError(
            f"Theme {name!r} already registered. Pass overwrite=True to replace."
        )
    _REGISTRY[name] = Theme(
        name=name,
        template=template,
        palette=tuple(palette) if palette else (),
        layout_overrides=dict(layout_overrides or {}),
    )


def get_theme(name: str | None = None) -> Theme:
    """Return the :class:`Theme` for ``name`` (or the current default)."""
    key = name if name is not None else _DEFAULT_NAME
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown theme {key!r}. Available: {sorted(_REGISTRY)}."
        )
    return _REGISTRY[key]


def apply_theme(fig: Any, theme: str | None = None) -> Any:
    """Apply ``theme`` to ``fig`` in place and return ``fig``.

    Called by every plot helper. Safe to call again on a returned figure
    if the user wants to re-theme without rebuilding the data.
    """
    t = get_theme(theme)
    layout: dict[str, Any] = {"template": t.template}
    if t.palette:
        layout["colorway"] = list(t.palette)
    layout.update(t.layout_overrides)
    fig.update_layout(**layout)
    return fig
