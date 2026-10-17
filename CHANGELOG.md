# Changelog

## Unreleased (to ship as 0.8.2 -- patch release required)

### Fixed

- `import simweave` failed on a base install (no extras) with
  `ModuleNotFoundError: No module named 'plotly'`. `simweave.viz.vehicle_dynamics`
  imported `plotly.subplots` at module level, and `simweave.roads` pulls in
  `simweave.viz`. plotly is now imported lazily inside `plot_vehicle_metrics`,
  which raises the usual "pip install simweave[viz]" `ImportError` when plotly
  is missing. **0.8.1 on PyPI is affected; release 0.8.2 with this fix.**

### Added

- Regression test (`tests/test_base_install_imports.py`) that blocks plotly via an
  import hook and checks `import simweave` works and plot helpers raise the
  friendly `ImportError`.
- CI job `base-install` that installs the built wheel with base dependencies
  only in a clean venv and runs an import + smoke test.
