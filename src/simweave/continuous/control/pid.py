# ---------------------------------------------------------------------------
# PID Controller for Control Systems Engineering
# ---------------------------------------------------------------------------

class PIDController:
    """Generic PID controller.

    u(t) = Kp * e + Ki * ∫e dt + Kd * de/dt
    """

    def __init__(
        self,
        Kp: float,
        Ki: float = 0.0,
        Kd: float = 0.0,
        setpoint: float = 0.0,
    ):
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)
        self.setpoint = float(setpoint)

        self._integral = 0.0
        self._prev_error = None

    def reset(self):
        self._integral = 0.0
        self._prev_error = None

    def __call__(self, measurement: float, dt: float) -> float:
        error = self.setpoint - measurement

        # integral
        self._integral += error * dt

        # derivative
        if self._prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self._prev_error) / dt

        self._prev_error = error

        return (
            self.Kp * error
            + self.Ki * self._integral
            + self.Kd * derivative
        )