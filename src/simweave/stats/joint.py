"""Joint sampling of named quantities from a copula and marginals."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from simweave.stats.copulas import Copula
from simweave.stats.marginals import Marginal, as_marginal


class JointSampler:
    """Draw dependent values: ``copula`` supplies the dependence, ``marginals`` the shapes.

    Parameters
    ----------
    copula:
        A :class:`~simweave.stats.copulas.Copula` with ``dim == len(marginals)``.
    marginals:
        Mapping ``name -> marginal`` (insertion order = copula axis order).
        Anything :func:`~simweave.stats.marginals.as_marginal` accepts.
    block:
        Scalar :meth:`draw` calls are served from blocks of this many vectorised
        samples. The buffer is tied to the ``rng`` passed in, so a given seed and
        call order is reproducible; a shared ``rng`` advances in blocks.
    """

    def __init__(
        self,
        copula: Copula,
        marginals: Mapping[str, object],
        *,
        block: int = 256,
    ) -> None:
        if not marginals:
            raise ValueError("marginals must be non-empty.")
        self.names: tuple[str, ...] = tuple(marginals)
        self.marginals: dict[str, Marginal] = {k: as_marginal(v) for k, v in marginals.items()}
        if copula.dim != len(self.names):
            raise ValueError(
                f"copula has dim={copula.dim} but {len(self.names)} marginals were given."
            )
        if block < 1:
            raise ValueError("block must be >= 1.")
        self.copula = copula
        self.block = int(block)
        self._buf: np.ndarray | None = None
        self._pos = 0
        self._buf_rng: np.random.Generator | None = None

    def sample_array(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """``(n, dim)`` array of values, columns in ``names`` order."""
        u = self.copula.sample_u(int(n), rng)
        return np.column_stack([self.marginals[k].ppf(u[:, i]) for i, k in enumerate(self.names)])

    def sample(self, rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
        x = self.sample_array(rng, n)
        return {k: x[:, i] for i, k in enumerate(self.names)}

    def draw(self, rng: np.random.Generator) -> dict[str, float]:
        """One joint draw as ``{name: value}``."""
        if self._buf is None or self._buf_rng is not rng or self._pos >= len(self._buf):
            self._buf = self.sample_array(rng, self.block)
            self._buf_rng = rng
            self._pos = 0
        row = self._buf[self._pos]
        self._pos += 1
        return {k: float(row[i]) for i, k in enumerate(self.names)}

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_buf"], d["_buf_rng"], d["_pos"] = None, None, 0
        return d

    @classmethod
    def from_data(
        cls,
        data,
        names: Sequence[str],
        *,
        marginals: Mapping[str, object] | str = "empirical",
        family: str = "auto",
        **fit_kwargs,
    ) -> "JointSampler":
        """Calibrate from observations (rows) with empirical or supplied marginals."""
        from simweave.stats.fit import fit_copula
        from simweave.stats.marginals import Empirical

        x = np.asarray(data, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(names):
            raise ValueError("data must be (n, len(names)).")
        if isinstance(marginals, str):
            if marginals != "empirical":
                raise ValueError("marginals must be 'empirical' or a mapping.")
            m = {k: Empirical(x[:, i]) for i, k in enumerate(names)}
        else:
            m = dict(marginals)
        fit = fit_copula(x, family=family, **fit_kwargs)
        return cls(fit.copula, m)
