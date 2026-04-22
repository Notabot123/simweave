"""simeng.currency: Decimal-backed Money, ISO 4217 codes, FX conversion.

Public surface
--------------
* :class:`Money` — frozen, currency-tagged ``Decimal`` value with
  strict same-currency arithmetic and explicit FX conversion.
* :class:`CurrencyMismatchError` — raised on cross-currency ops.
* :class:`FXConverter` — Protocol for rate lookup.
* :class:`StaticFXConverter` — in-memory rate table with automatic
  inverse lookup. Use for deterministic tests and fixed-rate models.
* :class:`CallableFXConverter` — adapts an arbitrary callable to the
  :class:`FXConverter` protocol.
* :func:`format_money` — ASCII default formatter, optional locale-aware
  path via ``simeng[intl]`` (babel).
* :func:`register_custom`, :func:`unregister_custom` — escape hatch for
  crypto, in-game, or test-fixture currency codes.
* :func:`is_valid_currency`, :func:`get_decimals`, :func:`list_codes` —
  registry introspection helpers.

Design contract in short
------------------------
* Decimal everywhere; float inputs routed via ``str()`` to avoid
  binary-float drift.
* Banker's rounding (``ROUND_HALF_EVEN``) at display / quantisation
  time; full precision preserved during computation.
* Cross-currency arithmetic refuses — use ``money.to(target, fx)``.
* Scalar * Money allowed; Money * Money refused.
* Money / Money (same currency) returns ``float``; Money / scalar
  returns Money.
* Negatives are legal (debts, refunds, signed cashflows).
"""

from simeng.currency.codes import (
    get_decimals,
    is_valid_currency,
    list_codes,
    register_custom,
    unregister_custom,
)
from simeng.currency.format import format_money
from simeng.currency.fx import (
    CallableFXConverter,
    FXConverter,
    StaticFXConverter,
)
from simeng.currency.money import CurrencyMismatchError, Money

__all__ = [
    # Core value type
    "Money",
    "CurrencyMismatchError",
    # FX
    "FXConverter",
    "StaticFXConverter",
    "CallableFXConverter",
    # Formatting
    "format_money",
    # Registry
    "register_custom",
    "unregister_custom",
    "is_valid_currency",
    "get_decimals",
    "list_codes",
]
