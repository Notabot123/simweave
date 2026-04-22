"""Foreign exchange conversion protocol and built-in strategies.

simweave deliberately **does not ship live FX rates.** Network data,
API keys, and rate-limits don't belong in a simulation kernel. Users
supply their own converter — either a static in-memory table for
deterministic testing, or a callable wrapper around whatever service
their application uses (Bloomberg, OANDA, ECB reference rates, etc.).

The :class:`FXConverter` protocol keeps this contract minimal: one
method, ``rate(source, target, at=None) -> Decimal``.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Callable, Mapping, Protocol, runtime_checkable

if TYPE_CHECKING:
    from datetime import datetime


@runtime_checkable
class FXConverter(Protocol):
    """Protocol for anything that can quote an FX rate.

    ``rate(source, target, at=None)`` returns the number of ``target``
    units per one unit of ``source``. A rate of ``1.27`` for
    ``("GBP", "USD")`` means one pound buys 1.27 dollars.
    """

    def rate(
        self,
        source: str,
        target: str,
        at: "datetime | None" = None,
    ) -> Decimal: ...


class StaticFXConverter:
    """Fixed-rate converter backed by an in-memory table.

    Automatically handles inverses: if ``("GBP", "USD") -> 1.27`` is
    registered, ``rate("USD", "GBP")`` returns ``1 / 1.27`` without
    needing an explicit entry.

    Parameters
    ----------
    rates:
        Mapping of ``(source, target)`` tuples to the rate. Rates may
        be ``Decimal``, ``float``, ``int``, or a numeric string; all
        are coerced to ``Decimal`` via ``Decimal(str(value))``.
    """

    __slots__ = ("_rates",)

    def __init__(
        self,
        rates: Mapping[tuple[str, str], Decimal | float | int | str],
    ) -> None:
        self._rates: dict[tuple[str, str], Decimal] = {}
        for (src, tgt), r in rates.items():
            key = (src.upper(), tgt.upper())
            if key[0] == key[1]:
                # Identity rate is always 1 — silently ignore explicit entry.
                continue
            self._rates[key] = Decimal(str(r))

    def rate(
        self,
        source: str,
        target: str,
        at: "datetime | None" = None,  # noqa: ARG002 - signature compatibility
    ) -> Decimal:
        src = source.upper()
        tgt = target.upper()
        if src == tgt:
            return Decimal("1")
        if (src, tgt) in self._rates:
            return self._rates[(src, tgt)]
        if (tgt, src) in self._rates:
            return Decimal("1") / self._rates[(tgt, src)]
        raise KeyError(
            f"No FX rate defined for {src} -> {tgt}. "
            f"Provide it explicitly or register the inverse."
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"StaticFXConverter({len(self._rates)} rates)"


class CallableFXConverter:
    """Wrap an arbitrary callable as an :class:`FXConverter`.

    The wrapped callable must accept ``(source, target, at)`` and return
    something coercible to ``Decimal`` (via ``Decimal(str(...))``).
    Useful when you already have a rate-lookup function.

    Example
    -------
    >>> def my_lookup(src, tgt, at=None):
    ...     return 1.27 if (src, tgt) == ("GBP", "USD") else 1.0
    >>> fx = CallableFXConverter(my_lookup)
    >>> fx.rate("GBP", "USD")
    Decimal('1.27')
    """

    __slots__ = ("_fn",)

    def __init__(
        self,
        fn: Callable[[str, str, "datetime | None"], Decimal | float | int | str],
    ) -> None:
        self._fn = fn

    def rate(
        self,
        source: str,
        target: str,
        at: "datetime | None" = None,
    ) -> Decimal:
        src = source.upper()
        tgt = target.upper()
        if src == tgt:
            return Decimal("1")
        raw = self._fn(src, tgt, at)
        return Decimal(str(raw))
