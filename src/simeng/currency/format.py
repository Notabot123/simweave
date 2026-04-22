"""Formatting for :class:`Money`.

Two paths:

* **ASCII default** (always available): ``"GBP 1,234.50"``. No locale
  assumptions — readable in logs, CSVs, and anywhere a BOM-sensitive
  consumer might see it.
* **Locale-aware via babel** (optional, ``simeng[intl]``):
  ``"£1,234.50"`` with the currency symbol and digit grouping of the
  user's locale.

The format specifier protocol supported by ``f"{m:...}"``:

* ``""``        -> default ASCII (``GBP 1,234.50``)
* ``"r"``       -> round to currency precision first, then default ASCII
* ``"raw"``     -> raw ``Decimal`` text, no grouping, no currency tag
* Anything starting with a locale code containing ``_`` (e.g.
  ``"en_GB"``) -> locale-aware via babel; raises ``ImportError`` if
  ``simeng[intl]`` not installed.
"""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal
from typing import TYPE_CHECKING

from simeng.currency.codes import get_decimals

if TYPE_CHECKING:
    from simeng.currency.money import Money


def _ascii_format(money: "Money", round_first: bool) -> str:
    decimals = get_decimals(money.currency)
    amount = money.amount
    if round_first:
        quant = Decimal(10) ** -decimals
        amount = amount.quantize(quant, rounding=ROUND_HALF_EVEN)
    # Split sign and magnitude so grouping applies to the digits only.
    sign = "-" if amount < 0 else ""
    magnitude = abs(amount)
    if decimals == 0:
        body = f"{magnitude:,.0f}"
    else:
        body = f"{magnitude:,.{decimals}f}"
    return f"{sign}{money.currency} {body}"


def _babel_format(money: "Money", locale: str) -> str:
    try:
        from babel.numbers import format_currency
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "Locale-aware money formatting requires babel. "
            "Install with `pip install simeng[intl]`."
        ) from exc
    return format_currency(money.amount, money.currency, locale=locale)


def format_money(money: "Money", spec: str = "", locale: str | None = None) -> str:
    """Format a :class:`Money` for display.

    Parameters
    ----------
    money:
        The value to render.
    spec:
        Format specifier. See module docstring.
    locale:
        Optional locale identifier (e.g. ``"en_GB"``). Overrides ``spec``
        when provided. Requires ``simeng[intl]`` (babel).

    Returns
    -------
    str
        The formatted string.
    """
    if locale is not None:
        return _babel_format(money, locale)
    if not spec:
        return _ascii_format(money, round_first=True)
    if spec == "r":
        return _ascii_format(money, round_first=True)
    if spec == "raw":
        # Full-precision decimal text, no currency tag, no grouping.
        return str(money.amount)
    if "_" in spec:
        return _babel_format(money, spec)
    raise ValueError(
        f"Unknown Money format spec: {spec!r}. "
        f"Use '' / 'r' / 'raw' or a locale like 'en_GB'."
    )
