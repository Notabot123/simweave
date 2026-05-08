class SkyhookDamper:
    """Skyhook damping controller.

    Applies force proportional to body velocity.
    """

    def __init__(self, damping: float):
        self.c = float(damping)

    def force(self, body_velocity: float, wheel_velocity: float) -> float:
        # For skyhook uses body velocity only, but signature makes modular/easy to toggle controller
        return -self.c * body_velocity
    
class GroundhookDamper:
    """Groundhook damping controller.
    
    Applies force proportional to wheel velocity."""

    def __init__(self, damping: float):
        self.c = float(damping)

    def force(self, body_velocity: float, wheel_velocity: float) -> float:
        # For groundhook uses wheel velocity only, but signature makes modular/easy to toggle controller
        return self.c * wheel_velocity
    
class HybridActiveDamper:
    "Hybrid Skyhook & Groundhook."
    def __init__(self, c_sky: float, c_ground: float, alpha: float = 0.5):
        self.sky = SkyhookDamper(c_sky)
        self.ground = GroundhookDamper(c_ground)
        self.alpha = float(alpha)

    def force(self, body_velocity: float, wheel_velocity: float) -> float:
        return (
            self.alpha * self.sky.force(body_velocity, 0)
            + (1 - self.alpha) * self.ground.force(0, wheel_velocity)
        )
    
class SemiActiveWrapper:
    """Enforces passive (dissipative) behaviour: F * v_rel <= 0."""

    def __init__(self, controller):
        self.controller = controller

    def force(self, body_velocity: float, wheel_velocity: float) -> float:
        F = self.controller.force(body_velocity, wheel_velocity)

        v_rel = body_velocity - wheel_velocity

        # enforce dissipative constraint
        if F * v_rel > 0:
            return 0.0

        return F