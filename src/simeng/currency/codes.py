"""ISO 4217 currency registry.

Maps an uppercase three-letter code (e.g. ``"GBP"``) to the number of
minor-unit decimal places defined by ISO 4217 for that currency. A
conservative subset of the ~180 active codes is baked in — enough to
cover >99% of the world's tradable currencies without dragging in
deprecated codes or fund/metal codes (XAU, XDR) that aren't "money"
in any operational sense.

For anything outside this registry (crypto, in-game currencies, test
fixtures), use :func:`register_custom`.

The registry is process-local: custom registrations do not persist
across Python processes. This is deliberate — hard-coding crypto rates
or custom codes across team members would be an anti-pattern.
"""

from __future__ import annotations

# ISO 4217 minor-unit exponents. Listed by decimal places for readability.
_ISO_4217: dict[str, int] = {
    # 0 decimal places (no minor units)
    "BIF": 0,
    "CLP": 0,
    "DJF": 0,
    "GNF": 0,
    "ISK": 0,
    "JPY": 0,
    "KMF": 0,
    "KRW": 0,
    "PYG": 0,
    "RWF": 0,
    "UGX": 0,
    "UYI": 0,
    "VND": 0,
    "VUV": 0,
    "XAF": 0,
    "XOF": 0,
    "XPF": 0,
    # 2 decimal places (the vast majority)
    "AED": 2,
    "AFN": 2,
    "ALL": 2,
    "AMD": 2,
    "ANG": 2,
    "AOA": 2,
    "ARS": 2,
    "AUD": 2,
    "AWG": 2,
    "AZN": 2,
    "BAM": 2,
    "BBD": 2,
    "BDT": 2,
    "BGN": 2,
    "BMD": 2,
    "BND": 2,
    "BOB": 2,
    "BRL": 2,
    "BSD": 2,
    "BWP": 2,
    "BYN": 2,
    "BZD": 2,
    "CAD": 2,
    "CDF": 2,
    "CHF": 2,
    "CNY": 2,
    "COP": 2,
    "CRC": 2,
    "CUP": 2,
    "CVE": 2,
    "CZK": 2,
    "DKK": 2,
    "DOP": 2,
    "DZD": 2,
    "EGP": 2,
    "ERN": 2,
    "ETB": 2,
    "EUR": 2,
    "FJD": 2,
    "FKP": 2,
    "GBP": 2,
    "GEL": 2,
    "GHS": 2,
    "GIP": 2,
    "GMD": 2,
    "GTQ": 2,
    "GYD": 2,
    "HKD": 2,
    "HNL": 2,
    "HRK": 2,
    "HTG": 2,
    "HUF": 2,
    "IDR": 2,
    "ILS": 2,
    "INR": 2,
    "IRR": 2,
    "JMD": 2,
    "KES": 2,
    "KGS": 2,
    "KHR": 2,
    "KPW": 2,
    "KYD": 2,
    "KZT": 2,
    "LAK": 2,
    "LBP": 2,
    "LKR": 2,
    "LRD": 2,
    "LSL": 2,
    "MAD": 2,
    "MDL": 2,
    "MGA": 2,
    "MKD": 2,
    "MMK": 2,
    "MNT": 2,
    "MOP": 2,
    "MRU": 2,
    "MUR": 2,
    "MVR": 2,
    "MWK": 2,
    "MXN": 2,
    "MYR": 2,
    "MZN": 2,
    "NAD": 2,
    "NGN": 2,
    "NIO": 2,
    "NOK": 2,
    "NPR": 2,
    "NZD": 2,
    "PAB": 2,
    "PEN": 2,
    "PGK": 2,
    "PHP": 2,
    "PKR": 2,
    "PLN": 2,
    "QAR": 2,
    "RON": 2,
    "RSD": 2,
    "RUB": 2,
    "SAR": 2,
    "SBD": 2,
    "SCR": 2,
    "SDG": 2,
    "SEK": 2,
    "SGD": 2,
    "SHP": 2,
    "SLE": 2,
    "SOS": 2,
    "SRD": 2,
    "SSP": 2,
    "STN": 2,
    "SVC": 2,
    "SYP": 2,
    "SZL": 2,
    "THB": 2,
    "TJS": 2,
    "TMT": 2,
    "TOP": 2,
    "TRY": 2,
    "TTD": 2,
    "TWD": 2,
    "TZS": 2,
    "UAH": 2,
    "USD": 2,
    "UYU": 2,
    "UZS": 2,
    "VES": 2,
    "WST": 2,
    "XCD": 2,
    "YER": 2,
    "ZAR": 2,
    "ZMW": 2,
    "ZWL": 2,
    # 3 decimal places (Middle East oil currencies)
    "BHD": 3,
    "IQD": 3,
    "JOD": 3,
    "KWD": 3,
    "LYD": 3,
    "OMR": 3,
    "TND": 3,
    # 4 decimal places (rare, mostly historic settlement)
    "CLF": 4,
    "UYW": 4,
}

# User-registered custom codes.
_CUSTOM: dict[str, int] = {}


def is_valid_currency(code: str) -> bool:
    """Return True if ``code`` is a known ISO 4217 or custom-registered code."""
    if not isinstance(code, str):
        return False
    up = code.upper()
    return up in _ISO_4217 or up in _CUSTOM


def get_decimals(code: str) -> int:
    """Return the canonical decimal places for ``code``.

    Raises
    ------
    KeyError
        If the code is not a known ISO 4217 or custom-registered code.
    """
    up = code.upper()
    if up in _CUSTOM:
        return _CUSTOM[up]
    if up in _ISO_4217:
        return _ISO_4217[up]
    raise KeyError(
        f"Unknown currency code: {code!r}. "
        f"Use simeng.currency.register_custom({code!r}, decimals=...) if intentional."
    )


def register_custom(code: str, decimals: int) -> None:
    """Register a non-ISO currency code (crypto, in-game currency, test fixture).

    Parameters
    ----------
    code:
        Uppercase identifier. Overwrites a prior custom registration but
        refuses to shadow a real ISO 4217 code.
    decimals:
        Non-negative number of minor-unit decimal places.

    Examples
    --------
    >>> from simeng.currency import register_custom, Money
    >>> register_custom("BTC", decimals=8)
    >>> Money("0.12345678", "BTC").amount
    Decimal('0.12345678')
    """
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Currency code must be a non-empty string.")
    up = code.upper()
    if up in _ISO_4217:
        raise ValueError(
            f"{up!r} is an ISO 4217 code; refuse to shadow with register_custom."
        )
    if not isinstance(decimals, int) or decimals < 0:
        raise ValueError("decimals must be a non-negative integer.")
    _CUSTOM[up] = decimals


def unregister_custom(code: str) -> None:
    """Remove a previously registered custom code. No-op if absent."""
    _CUSTOM.pop(code.upper(), None)


def list_codes(*, include_custom: bool = True) -> tuple[str, ...]:
    """Return a sorted tuple of all currently known codes."""
    codes = set(_ISO_4217)
    if include_custom:
        codes.update(_CUSTOM)
    return tuple(sorted(codes))
