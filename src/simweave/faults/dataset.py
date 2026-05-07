"""FaultDataset: ML-ready arrays assembled from a simulation result and fault labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from simweave.continuous.solver import SimulationResult
    from simweave.faults.injector import FaultInjector
    from simweave.faults.recorder import FaultRecorder


@dataclass
class FaultDataset:
    """ML-ready dataset produced from a faulted simulation run.

    Columns are aligned on a shared time grid so the dataset can be fed
    directly into a time-series model (LSTM, TCN, etc.) or a tabular
    classifier after slicing.

    Attributes
    ----------
    time : np.ndarray, shape (N,)
        Simulation time at each sample.
    features : np.ndarray, shape (N, F)
        Sensor / state observations, optionally with additive Gaussian noise.
    feature_names : list[str]
        Column labels for *features* — state labels first, then input labels.
    health_index : np.ndarray, shape (N,)
        Health index in [0, 1]: 1 = healthy, 0 = fully failed.
    rul : np.ndarray, shape (N,)
        Remaining useful life in simulation time units.
    is_failed : np.ndarray, shape (N,), dtype bool
        ``True`` from the failure time onward.
    failure_mode : np.ndarray, shape (N,), dtype object
        String label for the active failure mode at each sample
        (``"healthy"`` before any fault onset).
    """

    time: np.ndarray
    features: np.ndarray
    feature_names: list[str]
    health_index: np.ndarray
    rul: np.ndarray
    is_failed: np.ndarray
    failure_mode: np.ndarray

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_result(
        cls,
        result: "SimulationResult",
        injector: "FaultInjector",
        noise_std: float | np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> "FaultDataset":
        """Build a dataset from a :class:`~simweave.continuous.solver.SimulationResult`.

        Labels are computed analytically from the injector's fault profiles, so
        this works equally well after a standalone
        :func:`~simweave.continuous.solver.simulate` call or a hybrid
        :class:`~simweave.continuous.solver.ContinuousProcess` run.

        Parameters
        ----------
        result : SimulationResult
            Output of :func:`~simweave.continuous.solver.simulate`.
        injector : FaultInjector
            The injector used for the simulation (provides label computation).
        noise_std : float or array-like, optional
            Gaussian noise standard deviation added to *features* to simulate
            sensor measurement uncertainty.  A scalar applies the same std to
            every feature column; an array of length ``n_features`` gives
            per-column control.
        rng : numpy.random.Generator, optional
            Random generator for reproducible noise.  Defaults to a fresh
            ``np.random.default_rng()`` when *noise_std* is provided.
        """
        t = result.time

        hi = np.array([injector.overall_health(ti) for ti in t])
        rul_arr = np.array([injector.overall_rul(ti) for ti in t])
        modes = np.array([injector.active_mode(ti) for ti in t], dtype=object)

        # Features: state columns (+ input columns when present)
        features = result.state.copy()
        names: list[str] = list(result.state_labels)
        if result.inputs is not None:
            n_in = result.inputs.shape[1]
            features = np.hstack([features, result.inputs])
            names += [f"u{i}" for i in range(n_in)]

        if noise_std is not None:
            _rng = rng if rng is not None else np.random.default_rng()
            std = np.broadcast_to(
                np.asarray(noise_std, dtype=float), (features.shape[1],)
            )
            features = features + _rng.normal(0.0, std, features.shape)

        return cls(
            time=t,
            features=features,
            feature_names=names,
            health_index=hi,
            rul=rul_arr,
            is_failed=hi <= 0.0,
            failure_mode=modes,
        )

    @classmethod
    def from_recorder(
        cls,
        recorder: "FaultRecorder",
        result: "SimulationResult",
        noise_std: float | np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> "FaultDataset":
        """Build a dataset from a :class:`~simweave.faults.recorder.FaultRecorder`.

        Convenience wrapper for the hybrid :class:`~simweave.continuous.solver.ContinuousProcess`
        path where a recorder was already registered with the environment.

        Parameters
        ----------
        recorder : FaultRecorder
            Recorder attached to the running process.
        result : SimulationResult
            History extracted from the process via
            :meth:`~simweave.continuous.solver.ContinuousProcess.result`.
        noise_std : float or array-like, optional
            Sensor noise std (see :meth:`from_result`).
        rng : numpy.random.Generator, optional
            Random generator for reproducible noise.
        """
        return cls.from_result(result, recorder.injector, noise_std=noise_std, rng=rng)

    @staticmethod
    def concat(datasets: list["FaultDataset"]) -> "FaultDataset":
        """Stack multiple :class:`FaultDataset` objects along the sample axis.

        Useful for assembling a training corpus from several independent
        simulation runs (e.g. one healthy run + multiple fault runs with
        different onset times or degradation profiles).

        All datasets must share the same *feature_names*.

        Parameters
        ----------
        datasets : list[FaultDataset]
            Two or more datasets to concatenate.

        Returns
        -------
        FaultDataset
            A single dataset containing all samples in order.
        """
        if not datasets:
            raise ValueError("datasets must be non-empty.")
        names = datasets[0].feature_names
        for i, ds in enumerate(datasets[1:], 1):
            if ds.feature_names != names:
                raise ValueError(
                    f"datasets[{i}].feature_names {ds.feature_names!r} does not "
                    f"match datasets[0].feature_names {names!r}."
                )
        return FaultDataset(
            time=np.concatenate([d.time for d in datasets]),
            features=np.vstack([d.features for d in datasets]),
            feature_names=names,
            health_index=np.concatenate([d.health_index for d in datasets]),
            rul=np.concatenate([d.rul for d in datasets]),
            is_failed=np.concatenate([d.is_failed for d in datasets]),
            failure_mode=np.concatenate([d.failure_mode for d in datasets]),
        )

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def to_dataframe(self):
        """Convert to a :class:`pandas.DataFrame`.

        Requires ``pandas`` (``pip install pandas``).

        Returns
        -------
        pandas.DataFrame
            Columns: ``time``, all feature columns, ``health_index``, ``rul``,
            ``is_failed``, ``failure_mode``.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for FaultDataset.to_dataframe(). "
                "Install it with: pip install pandas"
            ) from exc

        data: dict = {"time": self.time}
        for i, name in enumerate(self.feature_names):
            data[name] = self.features[:, i]
        data["health_index"] = self.health_index
        data["rul"] = self.rul
        data["is_failed"] = self.is_failed
        data["failure_mode"] = self.failure_mode
        return pd.DataFrame(data)

    def train_test_split(
        self,
        test_frac: float = 0.2,
        shuffle: bool = False,
        rng: np.random.Generator | None = None,
    ) -> "tuple[FaultDataset, FaultDataset]":
        """Split into train and test subsets.

        Parameters
        ----------
        test_frac : float
            Fraction of samples to reserve for the test set (default 0.2).
        shuffle : bool
            If ``True``, shuffle indices before splitting.  Set *rng* for
            reproducibility.
        rng : numpy.random.Generator, optional
            Required when *shuffle=True* for reproducibility.

        Returns
        -------
        train, test : tuple[FaultDataset, FaultDataset]
        """
        n = len(self.time)
        idx = np.arange(n)
        if shuffle:
            _rng = rng if rng is not None else np.random.default_rng()
            _rng.shuffle(idx)
        split = int(n * (1.0 - test_frac))
        return self._subset(idx[:split]), self._subset(idx[split:])

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def _subset(self, idx: np.ndarray) -> "FaultDataset":
        return FaultDataset(
            time=self.time[idx],
            features=self.features[idx],
            feature_names=self.feature_names,
            health_index=self.health_index[idx],
            rul=self.rul[idx],
            is_failed=self.is_failed[idx],
            failure_mode=self.failure_mode[idx],
        )

    def __len__(self) -> int:
        return int(self.time.size)

    def __repr__(self) -> str:
        modes = sorted(set(self.failure_mode.tolist()))
        return (
            f"FaultDataset(n={len(self)}, features={self.feature_names}, "
            f"modes={modes}, failed={int(self.is_failed.sum())})"
        )


__all__ = ["FaultDataset"]
