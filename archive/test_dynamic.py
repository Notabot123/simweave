import os
import archive.dynamic_systems as ds
from archive.dynamic_systems import kg, dot, cross
import numpy as np




def test_use_of_predefined_unit():
    """ Use an instantiated unit such as kg, for simple, easy usages """
    m = 5 * kg
    assert isinstance(m,ds.Mass)

def handle_SIUnit_passed_as_value_to__init():
    """ In case user passes a class instead of value to class constructor """
    t = ds.TimeUnit(10)
    t2 = ds.TimeUnit(t)
    assert(t.value == t2.value,"SIunit class handled class as value input improperly")

def test_newtons_from_mXa_():
    """ Newton isn't an SIUnit and would otherwise be kg*m/s^2 """
    m = ds.Mass(5)
    a = ds.Acceleration(5)
    f = m * a
    assert(isinstance(f,ds.Force))
    assert(f.unit == 'N')

def test_multiply_float_and_int():
    """ test normal left multiply with float and int"""
    v = ds.Velocity(1)
    v2 = v * 2.5
    v3 = v * 3
    assert (v2.value,v3.value,type(v3)) == (v.value*2.5,v.value*3,ds.Velocity)

def test_distance_over_time():
    """ test distance/time is a velocity."""
    d = ds.Distance(10)
    t = ds.TimeUnit(10)
    v = d/t
    assert isinstance(v,ds.Velocity)

def test_velocity_over_time():
    """ test velocity/time is a acceleration."""
    v = ds.Velocity(10)
    t = ds.TimeUnit(10)
    a = v/t
    assert isinstance(a,ds.Acceleration)

def test_dynamic_unit_change_nonscalar():
    """ test velocity creation.
    Expected behaviour, is that v.direction_xyz has x==v.value """
    v = ds.Velocity(10)
    assert (v.direction_xyz[0],0,0) == (10,0,0)

def test_hypotenuse_calc_nonscalar():
    """ test diagonal updates correctly.
    Expected behaviour, is that direction_xyz updates hypotenuse as sqrt(x^2+y^2+z^2) """
    v = ds.Velocity(1)
    # knowing sqrt(36) is 6, and 36/3=12
    x = np.sqrt(12)
    v.update_cartesian_xyz([x,x,x])
    assert np.round(v.value) == 6

def test_angles_nonscalar():
    """ test angles update when x,y,z are specified """
    v = ds.Velocity(1)
    v.update_cartesian_xyz([2,2,2])
    rad_45 = np.pi/4
    assert(v.angles_xyz_axis==[rad_45,rad_45,rad_45])

""" consider if we want to keep this feature """
def test_infix_dot():
    """ test infix operator for dot prod """
    # update inputs to x,y,z once mixin issue tackled
    # will bcome result.value if result is a vector
    # interestingly, np.dot(v2,v) works already lol, but only upon value. Perhaps direction_xyz should be value
    v = ds.Velocity(3)
    v2 = ds.Velocity(3)
    result = v |dot| v2
    assert result == np.dot(v2.direction_xyz,v.direction_xyz)

"""def test_infix_cross():
    # test infix operator for cross prod 
    v = ds.Velocity(1)
    v2 = ds.Velocity(1)
    v.update_cartesian_xyz(2,2,-2)
    v2.update_cartesian_xyz(2,-2,2)
    result = v |cross| v2
    assert result == np.cross(v.direction_xyz, v2.direction_xyz)"""

""" Testing alternative units """
def test_alt_unit_time():
    """ Create a timeUnit of 1 hour, 1 min, 1000 ms
    Assert its value, unit and alt unit is as expected """
    t = ds.TimeUnit(1,'hrs')
    v = t.value
    u = t.unit
    alt = t.alt_unit
    v_alt = t.convert_hours()
    assert (v,u,alt,v_alt) == (3600,'s','hrs',1)

    t = ds.TimeUnit(1,'mins')
    v = t.value
    u = t.unit
    alt = t.alt_unit
    v_alt = t.convert_minutes()
    assert (v,u,alt,v_alt) == (60,'s','mins',1)

    t = ds.TimeUnit(1000,'ms')
    v = t.value
    u = t.unit
    alt = t.alt_unit
    v_alt = t.convert_milliseconds()
    assert (v,u,alt,v_alt) == (1,'s','ms',1000)

def test_alt_unit_mass():
    """ Create a Mass of 1 pound, 1 oz, 1 tonne
    Assert its value, unit and alt unit is as expected """
    t = ds.Mass(1,'lb')
    v = t.value
    u = t.unit
    alt = t.alt_unit
    v_alt = t.convert_pounds()
    assert (v,u,alt,v_alt) == (0.45359237,'kg','lb',1)

    t = ds.Mass(1,'T')
    v = t.value
    u = t.unit
    alt = t.alt_unit
    v_alt = t.convert_tonnes()
    assert (v,u,alt,v_alt) == (1000,'kg','metric-tonnes',1)

    t = ds.Mass(1000,'g')
    v = t.value
    u = t.unit
    alt = t.alt_unit
    v_alt = t.convert_grams()
    assert (v,u,alt,v_alt) == (1,'kg','g',1000)

    t = ds.Mass(1,'oz')
    v = t.value
    u = t.unit
    alt = t.alt_unit
    v_alt = t.convert_oz()
    assert (v,u,alt,v_alt) == (1/35.2739619,'kg','oz',1)


if __name__ == '__main__':
    print(os.listdir())