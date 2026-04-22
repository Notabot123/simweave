"""Monte Carlo cashflow with multi-currency Money.

A toy project with three cost centres (GBP, USD, EUR) and a stochastic
FX rate on settlement day. Each replicate draws one realisation of the
per-currency costs plus the GBP-denominated conversion rates, sums the
total in GBP, and reports the distribution of the base-currency bill.

This is intentionally tiny — the point is to show the Money / FX /
MCResult integration, not to be a realistic accounting model.

Run:
    python demos/13_money_cashflow.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (adds src/ to sys.path when not pip-installed)

from decimal import Decimal
from statistics import mean, stdev

import numpy as np

from simweave.currency import (
    Money,
    StaticFXConverter,
    format_money,
)
from simweave.mc.runner import run_monte_carlo


def sample_project_cost_gbp(seed: int) -> float:
    """One replicate: sample costs in three currencies and convert to GBP.

    Returns the total GBP cost as a float (so ``MCResult.samples`` can be
    numpy-materialised cleanly downstream). Full Decimal precision is
    retained inside the function and only truncated at the final cast.
    """
    rng = np.random.default_rng(seed)

    # --- sample the per-currency cost centres (lognormal-ish) ---------
    gbp_cost = Money(str(round(rng.lognormal(mean=9.0, sigma=0.3), 2)), "GBP")
    usd_cost = Money(str(round(rng.lognormal(mean=8.5, sigma=0.4), 2)), "USD")
    eur_cost = Money(str(round(rng.lognormal(mean=8.8, sigma=0.35), 2)), "EUR")

    # --- sample spot FX rates around sensible central values ----------
    # All rates are "X per 1 GBP" so we can convert into GBP via the
    # inverse, which StaticFXConverter handles automatically.
    gbp_usd = round(rng.normal(loc=1.27, scale=0.02), 4)
    gbp_eur = round(rng.normal(loc=1.17, scale=0.015), 4)

    fx = StaticFXConverter({
        ("GBP", "USD"): Decimal(str(gbp_usd)),
        ("GBP", "EUR"): Decimal(str(gbp_eur)),
    })

    # --- convert to GBP and sum with explicit start= ------------------
    total = sum(
        [gbp_cost, usd_cost.to("GBP", fx), eur_cost.to("GBP", fx)],
        start=Money(0, "GBP"),
    ).round_to_currency()

    return float(total.amount)


def main() -> None:
    n_runs = 500

    result = run_monte_carlo(
        sample_project_cost_gbp,
        n_runs=n_runs,
        executor="serial",
        scenario_name="multi_ccy_cashflow_gbp",
    )
    samples = [float(x) for x in result.samples]

    mu = mean(samples)
    sd = stdev(samples)
    p50 = float(np.quantile(samples, 0.50))
    p90 = float(np.quantile(samples, 0.90))
    p99 = float(np.quantile(samples, 0.99))

    print(f"Monte Carlo project cost over {n_runs} replicates (base = GBP)\n")
    print(f"  mean    : {format_money(Money(str(round(mu, 2)), 'GBP'))}")
    print(f"  stdev   : {format_money(Money(str(round(sd, 2)), 'GBP'))}")
    print(f"  median  : {format_money(Money(str(round(p50, 2)), 'GBP'))}")
    print(f"  p90     : {format_money(Money(str(round(p90, 2)), 'GBP'))}")
    print(f"  p99     : {format_money(Money(str(round(p99, 2)), 'GBP'))}")
    print()
    print("Note: no live FX is shipped with simweave. The rates used here")
    print("were drawn inside each replicate from hard-coded distributions")
    print("purely for illustrative purposes.")


if __name__ == "__main__":
    main()