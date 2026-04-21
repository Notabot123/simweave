import numpy as np
import simeng.dynamic_sys2 as ds


# -----------------------------
# BASIC UNIT TESTS
# -----------------------------

def test_predefined_mass():
    m = ds.Mass(5)
    assert isinstance(m, ds.Mass)
    assert m.value == 5


def test_siunit_passthrough():
    m = ds.Mass(5)
    m2 = ds.Mass(m)
    assert m.value == m2.value


# -----------------------------
# DERIVED UNITS
# -----------------------------

def test_newtons_from_mass_acceleration():
    m = ds.Mass(5)
    a = ds.Acceleration(2)

    f = m * a

    assert isinstance(f, ds.Force)
    assert f.unit == "N"


def test_distance_over_time():
    d = ds.Distance(10)
    t = ds.SIUnit(2, "s", [0,0,0,0,0,0,1])

    v = d / t

    assert isinstance(v, ds.Velocity)
    assert v.value == 5


def test_velocity_over_time():
    v = ds.Velocity(10)
    t = ds.SIUnit(2, "s", [0,0,0,0,0,0,1])

    a = v / t

    assert isinstance(a, ds.Acceleration)
    assert a.value == 5


# -----------------------------
# SCALAR OPERATIONS
# -----------------------------

def test_multiply_float_and_int():
    v = ds.Velocity(2)

    v2 = v * 2.5
    v3 = v * 3

    assert v2.value == 5.0
    assert v3.value == 6
    assert isinstance(v3, ds.Velocity)


def test_divide_by_scalar():
    d = ds.Distance(10)

    d2 = d / 2

    assert d2.value == 5


# -----------------------------
# VECTOR BEHAVIOUR
# -----------------------------

def test_vector_initialisation():
    v = ds.Velocity(10)

    assert np.allclose(v.vector, [10, 0, 0])


def test_vector_magnitude():
    v = ds.Velocity(0)
    v.vector = np.array([3, 4, 0])

    assert v.magnitude == 5


def test_dot_product():
    v1 = ds.Velocity(3)
    v2 = ds.Velocity(4)

    result = v1.dot(v2)

    assert result == 12


def test_cross_product():
    v1 = ds.Velocity(0)
    v2 = ds.Velocity(0)

    v1.vector = np.array([1, 0, 0])
    v2.vector = np.array([0, 1, 0])

    result = v1.cross(v2)

    assert np.allclose(result, [0, 0, 1])


# -----------------------------
# SPRING / DAMPER
# -----------------------------

def test_spring_force():
    spring = ds.Spring(0.1, ds.SIUnit(1000, "N/m", [0,1,0,0,0,0,-2]))

    f = spring.force()

    assert f.value == 100


def test_damper_force():
    damper = ds.Damper(2, ds.SIUnit(100, "Ns/m", [0,1,0,0,0,0,-1]))

    f = damper.force()

    assert f.value == 200


# -----------------------------
# QUARTER CAR MODEL
# -----------------------------

def test_quarter_car_static():
    model = ds.QuarterCarModel(
        m_s=250,
        m_u=40,
        k_s=15000,
        c_s=1500,
        k_t=200000
    )

    def flat_road(t):
        return 0

    t, x = model.simulate(flat_road, t_end=0.1)

    # system should remain near zero
    assert np.allclose(x, 0, atol=1e-6)


def test_quarter_car_response():
    model = ds.QuarterCarModel(
        m_s=250,
        m_u=40,
        k_s=15000,
        c_s=1500,
        k_t=200000
    )

    def bump(t):
        return 0.05 if t > 0.01 else 0

    t, x = model.simulate(bump, t_end=0.5)

    # sprung mass should move
    assert np.max(np.abs(x[:,0])) > 0


# -----------------------------
# EDGE CASES
# -----------------------------

def test_invalid_addition():
    d = ds.Distance(10)
    v = ds.Velocity(5)

    try:
        _ = d + v
        assert False
    except TypeError:
        assert True


def test_dimensionless_result():
    d1 = ds.Distance(10)
    d2 = ds.Distance(2)

    result = d1 / d2

    assert result.unit == "dimensionless"