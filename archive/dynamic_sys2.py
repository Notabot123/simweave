import numpy as np
from dataclasses import dataclass, field

SI_LIST = ['m','kg','A','K','mol','cd','s']

# -----------------------------
# UNIT HELPERS
# -----------------------------

def check_known_units(exponents=None):
    if exponents is None:
        exponents = [0]*7

    mapping = {
        (1,0,0,0,0,0,0): Distance,
        (1,0,0,0,0,0,-1): Velocity,
        (1,0,0,0,0,0,-2): Acceleration,
        (0,1,0,0,0,0,0): Mass,
        (2,0,0,0,0,0,0): Area,
        (3,0,0,0,0,0,0): Volume,
        (1,1,0,0,0,0,-2): Force,
    }

    return mapping.get(tuple(exponents), None)


def units_from_exponents(exponents):
    cls = check_known_units(exponents)
    if cls:
        return cls

    parts = []
    for i, e in enumerate(exponents):
        if e != 0:
            parts.append(f"{SI_LIST[i]}^{e}" if e != 1 else SI_LIST[i])
    return "*".join(parts) if parts else "dimensionless"


# -----------------------------
# BASE CLASS
# -----------------------------

@dataclass
class SIUnit:
    value: float
    unit: str
    _exponents: list[int] = field(default_factory=lambda: [0]*7)

    def __post_init__(self):
        if isinstance(self.value, SIUnit):
            self.value = self.value.value

    def __str__(self):
        return f"{self.value} [{self.unit}]"

    # -------- arithmetic --------

    def __add__(self, other):
        if type(self) is type(other):
            return type(self)(self.value + other.value)
        raise TypeError("Cannot add different unit types")

    def __sub__(self, other):
        if type(self) is type(other):
            return type(self)(self.value - other.value)
        raise TypeError("Cannot subtract different unit types")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return type(self)(self.value * other)

        if isinstance(other, SIUnit):
            exp = [a+b for a,b in zip(self._exponents, other._exponents)]
            cls = units_from_exponents(exp)

            if isinstance(cls, str):
                return SIUnit(self.value * other.value, cls, exp)
            return cls(self.value * other.value)

        raise TypeError("Unsupported multiplication")

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return type(self)(self.value / other)

        if isinstance(other, SIUnit):
            exp = [a-b for a,b in zip(self._exponents, other._exponents)]
            cls = units_from_exponents(exp)

            if isinstance(cls, str):
                return SIUnit(self.value / other.value, cls, exp)
            return cls(self.value / other.value)

        raise TypeError("Unsupported division")


# -----------------------------
# CARTESIAN MIXIN
# -----------------------------

class Cartesian:
    def __init__(self, x, y=0, z=0):
        self.vector = np.array([x, y, z], dtype=float)

    @property
    def magnitude(self):
        return np.linalg.norm(self.vector)

    def dot(self, other):
        return np.dot(self.vector, other.vector)

    def cross(self, other):
        return np.cross(self.vector, other.vector)


# -----------------------------
# UNITS
# -----------------------------

class Distance(SIUnit, Cartesian):
    def __init__(self, value):
        SIUnit.__init__(self, value, "m", [1,0,0,0,0,0,0])
        Cartesian.__init__(self, value, 0, 0)


class Velocity(SIUnit, Cartesian):
    def __init__(self, value):
        SIUnit.__init__(self, value, "m/s", [1,0,0,0,0,0,-1])
        Cartesian.__init__(self, value, 0, 0)


class Acceleration(SIUnit, Cartesian):
    def __init__(self, value):
        SIUnit.__init__(self, value, "m/s²", [1,0,0,0,0,0,-2])
        Cartesian.__init__(self, value, 0, 0)


class Mass(SIUnit):
    def __init__(self, value):
        super().__init__(value, "kg", [0,1,0,0,0,0,0])


class Force(SIUnit, Cartesian):
    def __init__(self, value):
        SIUnit.__init__(self, value, "N", [1,1,0,0,0,0,-2])
        Cartesian.__init__(self, value, 0, 0)


class Area(SIUnit):
    def __init__(self, value):
        super().__init__(value, "m²", [2,0,0,0,0,0,0])


class Volume(SIUnit):
    def __init__(self, value):
        super().__init__(value, "m³", [3,0,0,0,0,0,0])


# -----------------------------
# SPRING / DAMPER
# -----------------------------

class Spring:
    def __init__(self, displacement, stiffness):
        self.displacement = Distance(displacement)
        self.stiffness = stiffness

    def force(self):
        return self.displacement * self.stiffness


class Damper:
    def __init__(self, velocity, coefficient):
        self.velocity = Velocity(velocity)
        self.coefficient = coefficient

    def force(self):
        return self.velocity * self.coefficient

import numpy as np

class QuarterCarModel:
    """
    2-DOF Quarter Car Model
    State vector:
        x = [z_s, z_s_dot, z_u, z_u_dot]
    """

    def __init__(self, m_s, m_u, k_s, c_s, k_t):
        self.m_s = m_s  # sprung mass
        self.m_u = m_u  # unsprung mass
        self.k_s = k_s  # suspension stiffness
        self.c_s = c_s  # damping
        self.k_t = k_t  # tyre stiffness

    def derivatives(self, t, x, z_r, z_r_dot=0):
        z_s, z_s_dot, z_u, z_u_dot = x

        # Suspension force
        f_s = self.k_s * (z_s - z_u)
        f_d = self.c_s * (z_s_dot - z_u_dot)

        # Equations of motion
        z_s_ddot = (-f_s - f_d) / self.m_s
        z_u_ddot = (f_s + f_d - self.k_t * (z_u - z_r)) / self.m_u

        return np.array([z_s_dot, z_s_ddot, z_u_dot, z_u_ddot])

    def simulate(self, road_func, t_end=2.0, dt=0.001):
        t = np.arange(0, t_end, dt)
        x = np.zeros((len(t), 4))

        for i in range(1, len(t)):
            z_r = road_func(t[i])

            dx = self.derivatives(t[i], x[i-1], z_r)

            # simple Euler integration
            x[i] = x[i-1] + dx * dt

        return t, x
    
# -----------------------------
# DEMO
# -----------------------------

if __name__ == "__main__":
    d = Distance(10)
    t = SIUnit(2, "s", [0,0,0,0,0,0,1])

    v = d / t
    a = v / t

    print(d)
    print(v)
    print(a)

    f = Mass(5) * a
    print(f)

    # quarter car suspension example
    model = QuarterCarModel(
        m_s=250,   # kg
        m_u=40,    # kg
        k_s=15000, # N/m
        c_s=1500,  # Ns/m
        k_t=200000 # N/m
    )

    # road profile (bump)
    def road(t):
        return 0.05 * np.sin(10*t) if t < 1 else 0

    t, x = model.simulate(road, t_end=2)

    import matplotlib.pyplot as plt
    plt.plot(t, x[:,0], label="Sprung mass")
    plt.plot(t, x[:,2], label="Unsprung mass")
    plt.legend()
    plt.show()