"""Control allocation for an X-configuration quadrotor.

Rotor numbering and spin (seen from above, body x forward, y left)::

        front
    2 (CW)   1 (CCW)
         \\ /
         / \\
    3 (CCW)  4 (CW)
        rear

Rotor 1 is front-right, 2 front-left, 3 rear-left, 4 rear-right. A
counter-clockwise rotor pushes the body clockwise, so the yaw torque is
``k * (-f1 + f2 - f3 + f4)`` with ``k`` the torque-to-thrust ratio.
"""

from __future__ import annotations

import math

import numpy as np

from simweave.flight.params import QuadrotorParams


class Mixer:
    """Maps collective thrust and body torques to four rotor thrusts.

    ``wrench = A @ thrusts`` with ``wrench = [T, tau_x, tau_y, tau_z]``.
    """

    def __init__(self, params: QuadrotorParams | None = None) -> None:
        self.params = params if params is not None else QuadrotorParams()
        d = self.params.arm_length / math.sqrt(2.0)
        k = self.params.torque_ratio
        self.allocation = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [-d, d, d, -d],
                [-d, -d, d, d],
                [-k, k, -k, k],
            ]
        )
        self._inverse = np.linalg.inv(self.allocation)

    def wrench(self, thrusts) -> np.ndarray:
        """``[T, tau_x, tau_y, tau_z]`` produced by four rotor thrusts."""
        return self.allocation @ np.asarray(thrusts, dtype=float)

    def thrusts(self, collective: float, torques) -> np.ndarray:
        """Unconstrained rotor thrusts for a requested wrench."""
        w = np.array([collective, *np.asarray(torques, dtype=float)])
        return self._inverse @ w

    def allocate(self, collective: float, torques) -> np.ndarray:
        """Rotor thrusts within ``[0, max_thrust]`` for a requested wrench.

        When the request does not fit, attitude torques take priority over
        collective thrust: the collective is moved first (keeping the torques
        exact), and only if the torques alone cannot be produced are they
        scaled down uniformly.
        """
        f_max = self.params.max_thrust
        base = self._inverse[:, 0] * collective  # collective only: T/4 each
        spread = self._inverse[:, 1:] @ np.asarray(torques, dtype=float)
        # Shift every rotor by the same amount c (collective change) so that
        # base_i + c + s * spread_i stays inside [0, f_max].
        base_level = float(base[0])
        lo = -float(spread.min())
        hi = f_max - float(spread.max())
        scale = 1.0
        if lo <= hi:
            level = min(max(base_level, lo), hi)
        else:
            span = float(spread.max() - spread.min())
            scale = f_max / span
            level = -scale * float(spread.min())
        return np.clip(level + scale * spread, 0.0, f_max)
