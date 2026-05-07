"""Demo 26 — Fault injection and predictive-maintenance dataset generation.

Shows how to:

1. Wrap a physics model in a :class:`~simweave.faults.FaultInjector` and inject
   a physically-meaningful parameter fault (thermal insulation degradation).
2. Produce runs for *multiple* degradation profiles (linear vs exponential) and
   a healthy baseline.
3. Stack them into a combined :class:`~simweave.faults.FaultDataset` with RUL
   and failure-mode labels suitable for LSTM / time-series model training.
4. Visualise the degraded sensor signals, health index, and RUL.

System
------
A lumped-capacitance :class:`~simweave.continuous.ThermalRC` model representing
an electric motor winding.  A constant heat input Q_in simulates steady-state
operation.  The fault is *insulation degradation*: the thermal resistance R_th
rises over time, causing the winding temperature to climb toward a dangerous
level.

Two failure profiles are compared:

* ``"linear"`` — constant degradation rate (uniform wear).
* ``"exponential"`` — slow start, rapid end (fatigue / crack propagation).

Run
---
    python demos/26_fault_injection.py
"""

import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

import simweave as sw
from simweave.faults import FaultDataset, FaultInjector, FaultProfile, ParameterFault

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

DT = 0.5          # integration step (seconds)
T_SPAN = (0.0, 1200.0)  # 20-minute run
AMBIENT = 293.15  # 20 °C ambient
Q_IN = 50.0       # constant heat input [W]

RNG = np.random.default_rng(42)

# Fault onset / failure times (seconds into run)
ONSET = 400.0
FAILURE = 1000.0


# ---------------------------------------------------------------------------
# Helper: build a ThermalRC with constant heat input
# ---------------------------------------------------------------------------

def build_system() -> sw.ThermalRC:
    """Motor winding: R_th=2.0 K/W, C_th=800 J/K, starts at ambient."""
    return sw.ThermalRC(
        thermal_resistance=2.0,
        thermal_capacitance=800.0,
        ambient_temperature=AMBIENT,
        initial_temperature=AMBIENT,
    )


def constant_heat(t: float) -> float:
    return Q_IN


# ---------------------------------------------------------------------------
# Run 1 — Healthy baseline (no fault)
# ---------------------------------------------------------------------------

print("Running healthy baseline ...")
healthy_sys = build_system()

healthy_result = sw.simulate(
    healthy_sys,
    t_span=T_SPAN,
    dt=DT,
    inputs=constant_heat,
)

# Build a no-fault FaultInjector just to get a consistent FaultDataset object.
healthy_injector = FaultInjector(system=build_system(), faults=[])
healthy_ds = FaultDataset.from_result(
    healthy_result,
    healthy_injector,
    noise_std=0.3,         # ±0.3 K sensor noise
    rng=np.random.default_rng(10),
)

# ---------------------------------------------------------------------------
# Run 2 — Linear insulation degradation
# ---------------------------------------------------------------------------

print("Running linear-degradation fault ...")

linear_profile = FaultProfile(
    onset_time=ONSET,
    failure_time=FAILURE,
    mode="insulation_loss_linear",
    shape="linear",
)
linear_fault = ParameterFault(
    param="R_th",
    profile=linear_profile,
    max_delta=3.0,     # R_th rises to 4× nominal at full failure
    relative=True,
)
linear_injector = FaultInjector(system=build_system(), faults=[linear_fault])
linear_result = sw.simulate(linear_injector, t_span=T_SPAN, dt=DT, inputs=constant_heat)
linear_ds = FaultDataset.from_result(
    linear_result,
    linear_injector,
    noise_std=0.3,
    rng=np.random.default_rng(20),
)

# ---------------------------------------------------------------------------
# Run 3 — Exponential insulation degradation (slow start, rapid end)
# ---------------------------------------------------------------------------

print("Running exponential-degradation fault ...")

exp_profile = FaultProfile(
    onset_time=ONSET,
    failure_time=FAILURE,
    mode="insulation_loss_exponential",
    shape="exponential",
)
exp_fault = ParameterFault(
    param="R_th",
    profile=exp_profile,
    max_delta=3.0,
    relative=True,
)
exp_injector = FaultInjector(system=build_system(), faults=[exp_fault])
exp_result = sw.simulate(exp_injector, t_span=T_SPAN, dt=DT, inputs=constant_heat)
exp_ds = FaultDataset.from_result(
    exp_result,
    exp_injector,
    noise_std=0.3,
    rng=np.random.default_rng(30),
)

# ---------------------------------------------------------------------------
# Combine into a training corpus
# ---------------------------------------------------------------------------

print("Assembling combined dataset ...")
corpus = FaultDataset.concat([healthy_ds, linear_ds, exp_ds])
train_ds, test_ds = corpus.train_test_split(test_frac=0.2, rng=np.random.default_rng(99))

print(f"\n{corpus!r}")
print(f"  Train samples : {len(train_ds)}")
print(f"  Test  samples : {len(test_ds)}")
print(f"  Feature cols  : {corpus.feature_names}")

# ---------------------------------------------------------------------------
# DataFrame summary
# ---------------------------------------------------------------------------

try:
    import pandas as pd

    df = corpus.to_dataframe()
    print("\nDataFrame head (5 rows):")
    print(df.head(5).to_string(index=False))
    print("\nClass distribution (failure_mode):")
    print(df["failure_mode"].value_counts().to_string())
    print("\nRUL statistics:")
    print(df.groupby("failure_mode")["rul"].describe().round(1).to_string())
except ImportError:
    print("\n(Install pandas for DataFrame output: pip install pandas)")

# ---------------------------------------------------------------------------
# Visualisation (requires simweave[viz])
# ---------------------------------------------------------------------------

if not sw.have_plotly():
    print("\n(Install plotly for plots: pip install simweave[viz])")
    sys.exit(0)

import os

out_dir = Path("demos") / "fault_outputs"
out_dir.mkdir(parents=True, exist_ok=True)


# --- Plot 1: Healthy vs degraded temperature signals (linear fault) ---------
import plotly.graph_objects as go

fig_signals = go.Figure()
fig_signals.add_trace(go.Scatter(
    x=healthy_result.time,
    y=healthy_result.state[:, 0] - AMBIENT,
    mode="lines", name="healthy", line={"color": "royalblue"},
))
fig_signals.add_trace(go.Scatter(
    x=linear_result.time,
    y=linear_result.state[:, 0] - AMBIENT,
    mode="lines", name="linear fault", line={"color": "darkorange"},
))
fig_signals.add_trace(go.Scatter(
    x=exp_result.time,
    y=exp_result.state[:, 0] - AMBIENT,
    mode="lines", name="exponential fault", line={"color": "crimson"},
))
# Shade degradation and failure windows
fig_signals.add_vrect(
    x0=ONSET, x1=FAILURE,
    fillcolor="orange", opacity=0.10, layer="below", line_width=0,
    annotation_text="degrading", annotation_position="top left",
)
fig_signals.add_vrect(
    x0=FAILURE, x1=T_SPAN[1],
    fillcolor="red", opacity=0.10, layer="below", line_width=0,
    annotation_text="failed", annotation_position="top left",
)
fig_signals.update_layout(
    title="Motor winding temperature rise — healthy vs faulted",
    xaxis_title="time (s)",
    yaxis_title="temperature rise above ambient (K)",
    legend={"orientation": "h", "y": -0.2},
)
fig_signals.write_html(str(out_dir / "temperature_signals.html"))
print(f"\nSaved: {out_dir}/temperature_signals.html")

# --- Plot 2: Health index and RUL for both fault profiles -------------------
fig_hi = sw.plot_health_index(linear_ds, title="Health index & RUL — linear fault")
fig_hi.write_html(str(out_dir / "health_index_linear.html"))
print(f"Saved: {out_dir}/health_index_linear.html")

fig_hi_exp = sw.plot_health_index(exp_ds, title="Health index & RUL — exponential fault")
fig_hi_exp.write_html(str(out_dir / "health_index_exponential.html"))
print(f"Saved: {out_dir}/health_index_exponential.html")

# --- Plot 3: Fault signal annotated view ------------------------------------
fig_fault = sw.plot_fault_signals(
    linear_result, linear_injector,
    title="ThermalRC — linear insulation-loss fault",
)
fig_fault.write_html(str(out_dir / "fault_signals_linear.html"))
print(f"Saved: {out_dir}/fault_signals_linear.html")

print("\nDone. Open the HTML files in a browser to explore the plots.")
print("\nTip — training an LSTM with PyTorch / Keras:")
print("  X = corpus.features          # shape (N, n_features)")
print("  y_rul = corpus.rul           # regression target")
print("  y_mode = corpus.failure_mode # multi-class target")
print("  train, test = corpus.train_test_split()")
