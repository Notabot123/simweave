# ---------------------------------------------------------------------------
# PID Controller for Control Systems Engineering
# ---------------------------------------------------------------------------

from __future__ import annotations


class PIDController:
    """Generic PID controller.

    u(t) = Kp * e + Ki * ∫e dt + Kd * de/dt

    With only ``Kp``, ``Ki``, ``Kd`` and ``setpoint`` given it behaves as a
    plain PID. The keyword options below are all off by default.

    Parameters
    ----------
    output_limits:
        ``(low, high)`` clamp on the output; either side may be ``None``.
    anti_windup:
        ``"clamp"`` (default) stops the integral growing while the output is
        saturated and the error is pushing it further into saturation
        (conditional integration). ``"none"`` lets the integral wind up.
        Has no effect without ``output_limits``.
    derivative_on_measurement:
        Differentiate the measurement instead of the error, which avoids a
        derivative kick when the setpoint changes.
    derivative_tau:
        Time constant of a first-order low-pass filter on the derivative
        term; ``None`` or ``0`` disables filtering.
    """

    def __init__(
        self,
        Kp: float,
        Ki: float = 0.0,
        Kd: float = 0.0,
        setpoint: float = 0.0,
        *,
        output_limits: tuple[float | None, float | None] | None = None,
        anti_windup: str = "clamp",
        derivative_on_measurement: bool = False,
        derivative_tau: float | None = None,
    ):
        if anti_windup not in ("clamp", "none"):
            raise ValueError("anti_windup must be 'clamp' or 'none'.")
        if derivative_tau is not None and derivative_tau < 0:
            raise ValueError("derivative_tau must be non-negative.")
        lo, hi = output_limits if output_limits is not None else (None, None)
        if lo is not None and hi is not None and lo > hi:
            raise ValueError("output_limits low must not exceed high.")

        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)
        self.setpoint = float(setpoint)
        self.output_limits = (lo, hi)
        self.anti_windup = anti_windup
        self.derivative_on_measurement = bool(derivative_on_measurement)
        self.derivative_tau = derivative_tau

        self._integral = 0.0
        self._prev_error = None
        self._prev_measurement = None
        self._derivative = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_error = None
        self._prev_measurement = None
        self._derivative = 0.0

    def _clip(self, u: float) -> float:
        lo, hi = self.output_limits
        if lo is not None and u < lo:
            return lo
        if hi is not None and u > hi:
            return hi
        return u

    def __call__(self, measurement: float, dt: float) -> float:
        if dt <= 0:
            raise ValueError("dt must be positive.")
        error = self.setpoint - measurement

        # derivative (zero on the first call)
        if self._prev_error is None:
            raw_derivative = 0.0
        elif self.derivative_on_measurement:
            raw_derivative = -(measurement - self._prev_measurement) / dt
        else:
            raw_derivative = (error - self._prev_error) / dt
        if self.derivative_tau:
            alpha = dt / (self.derivative_tau + dt)
            self._derivative += alpha * (raw_derivative - self._derivative)
        else:
            self._derivative = raw_derivative

        self._prev_error = error
        self._prev_measurement = measurement

        # integral, with optional conditional-integration anti-windup
        new_integral = self._integral + error * dt
        unsaturated = (
            self.Kp * error + self.Ki * new_integral + self.Kd * self._derivative
        )
        output = self._clip(unsaturated)
        winding_up = output != unsaturated and (unsaturated - output) * error > 0
        if not (self.anti_windup == "clamp" and winding_up):
            self._integral = new_integral
        else:
            output = self._clip(
                self.Kp * error
                + self.Ki * self._integral
                + self.Kd * self._derivative
            )
        return output

    def update(self, t: float, state, dt: float) -> float:
        """:class:`~simweave.continuous.solver.Controller` adapter.

        Only valid for single-state plants: the measurement is ``state[0]``.
        For other layouts wrap the controller in a small class.
        """
        return self(float(state[0]), dt)
