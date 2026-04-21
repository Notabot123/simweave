import numpy as np
from dataclasses import dataclass, field

""" Stu - early development work only.
    Quartercar suspension is incomplete"""
SI_list = ['m','kg','A','K','mol','cd','s']
units_list = ['s','m','kg','A','K','mol','cd','m^2','m²','m^3','m³','N','Nm','N/m','m/s','ms^-1']

def check_known_units(exponents = [0,0,0,0,0,0,0]):
    if np.all(exponents==0):
        return float
    elif exponents == [1,0,0,0,0,0,0]:
        return Distance
    elif exponents == [1,0,0,0,0,0,-1]:
        return Velocity
    elif exponents == [1,0,0,0,0,0,-2]:
        return Acceleration
    elif exponents == [0,1,0,0,0,0,0]:
        return Mass
    elif exponents == [2,0,0,0,0,0,0]:
        return Area
    elif exponents == [3,0,0,0,0,0,0]:
        return Volume
    elif exponents == [1,1,0,0,0,0,-2]:
        return Force
    elif exponents == [2,1,0,0,0,0,-2]:
        return Torque
    elif exponents == [-1,1,0,0,0,0,-2]:
        return Pressure 
    elif exponents == [0,1,0,0,0,0,-2]:
        return SpringConstant
    elif exponents == [0,1,0,0,0,0,-1]:
        return DamperCoefficient
    else:
        return None

def units_from_exponents(exponents = [0,0,0,0,0,0,0]):
    assert len(exponents)==7, f"Should be 7 exponents in input array, found {len(exponents)}"
    d = check_known_units(exponents)
    if not d:
        units = []
        for i,e in enumerate(exponents):
            if e == 0:
                pass
            elif e ==1:
                units.append(SI_list[i])
            else:
                units.append(SI_list[i]+'^'+str(e))
        return ''.join(units)
    else:
        return d


@dataclass
class SI_unit:
    """
    SI unit is a dataclass subclass for physics variables, in specified units. 
    Allows for alternative units, with conversion methods
    Methods:
    -------
    __post__init__ : checks that unit is in allowable list
    __str__ : prints value with unit, and alternative unit if specified
    """
    value: float
    unit: str
    alt_unit: str = ''
    _convert_func: str = ''
    _exponents: list[int] = field(default_factory=lambda: [0,0,0,0,0,0,0])

    def __post_init__(self):
        """ This is called post instantiating dataclass, and checks SIUnit is valid """
        if isinstance(self.value,SI_unit):
            """ If user accidentally passes class instance as value to constructor """
            return SI_unit(self.value.value, unit=self.unit)
        if any([self.unit == u for u in units_list]):
            pass
        else:
            #raise Exception(f"{self.unit} is not a recognised SI unit. See: {units_list}")
            print(f"{self.unit} is not a recognised derived SI unit. See: {units_list}")
    def __str__(self):
        if self.alt_unit:
            return f'Value:{self.value} ({self.unit}) i.e. {self._convert_func()} ({self.alt_unit})'
        else:
            return f'Value:{self.value} ({self.unit})'
    
    ## operator overrides 
    def __add__(self,other):
        """ Adds CartesianSystem subclasses. Allows float and int.
        Allows incremental increases to Velocity via Acceleration etc.
        Otherwise, classes must match.
        """
        t = type(self)
        if isinstance(other, t): 
            return t(self.value + other.value)
        elif (isinstance(self, Acceleration) & isinstance(other, Velocity)) or (
            isinstance(self, Velocity) & isinstance(other, Acceleration)): 
            return Velocity(self.value + other.value)
        elif (isinstance(self, Velocity) & isinstance(other, Distance)) or (
            isinstance(self, Distance) & isinstance(other, Velocity)): 
            return Distance(self.value + other.value)
        elif isinstance(other, float|int): 
            return t(self.value + other.value)
        else:
            raise Exception(f"Only {t} may be added to {t}, not {t} and {type(other)}")
        
    ## in place addition
    def __iadd__(self,other):
        return self.__add__(other)
        
    def __sub__(self,other):
        t = type(self)
        if isinstance(other, t): 
            return t(self.value - other.value)
        else:
            raise Exception(f"Only {t} may be subtracted from {t}")        
    def __rmul__(self,other):
        t = type(self)
        if isinstance(other, float|int): 
            return t(self.value * other, unit=self.unit)
        else:
            raise Exception("Only SIUnits, float or int may be right-multiplied to SIUnits")
        
    def __mul__(self,other):
        t = type(self)
        if isinstance(other, t): 
            return t(self.value * other.value) 
        elif isinstance(other, SI_unit):
            # combining units, add operator not suited for lists, performs concat
            expon = [sum(x) for x in zip(self._exponents, other._exponents)] 
            u = units_from_exponents(expon)
            if isinstance(u, str):
                return SI_unit(self.value * other.value, unit=u)
            else:
                return u(self.value * other.value)
        elif isinstance(other, float|int): 
            return t(self.value * other, unit=self.unit)
        else:
            raise Exception("Only SIUnits, float or int may be multiplied to SIUnits")
    def __truediv__(self,other):
        t = type(self)
        if isinstance(other, t):
            # same class 
            return t(self.value / other.value)
        if isinstance(self, Distance) & isinstance(other, TimeUnit): 
            return Velocity(self.value / other.value)
        if isinstance(self, Velocity) & isinstance(other, TimeUnit): 
            return Acceleration(self.value / other.value)

        if isinstance(other, SI_unit):
            # combining units, subtract operator not suited for lists, performs concat
            expon = [(x-y) for x,y in zip(self._exponents, other._exponents)] 
            u = units_from_exponents(expon)
            if isinstance(u, str):
                return SI_unit(self.value / other.value, unit=u)
            else:
                return u(self.value / other.value)

        if isinstance(other, float|int): 
            return t(self.value / other, unit=self.unit)
        else:
            raise Exception("Only SIUnits may be used for TimeUnit to dividie into")

class TimeUnit(SI_unit):
    """
    SI unit subclass for time variables, in seconds. 
    Allows for hours (hrs) and milliseconds as alternative units, with conversion methods
    Methods:
    -------
    convert_hours
    convert_milliseconds
    """
    value: float
    unit: str = 's'
    def __init__(self, value, unit='s'):        
        """ Store units as seconds but allow hours (hrs), minutes (mins) and milliseconds (s) as alternate units """
        super().__init__(value=value, unit='s', alt_unit=None, _convert_func=None, _exponents=[0,0,0,0,0,0,1])
        if (unit == 'h') | (unit == 'hours') | (unit == 'hrs'):
            self.alt_unit = 'hrs'
            self.value = self._convert_from_hours()
            self._convert_func = self.convert_hours        
        elif unit == 'mins':
            self.alt_unit = 'mins'
            self.value = self._convert_from_minutes()
            self._convert_func = self.convert_minutes
        elif unit == 'ms':
            self.alt_unit = 'ms'
            self.value = self._convert_from_milliseconds()
            self._convert_func = self.convert_milliseconds
        elif unit == 's':
            # default alternate unit can be mins
            self.alt_unit = 'mins'
            self._convert_func = self.convert_minutes
        else:
            raise Exception(f"{unit} is not a recognised alternative unit. Available:['hrs','mins','ms']."+ \
                            "Where possible, use SI units (seconds)")
    
    def convert_hours(self):
        return self.value / 3600
    def convert_minutes(self):
        return self.value / 60
    def convert_milliseconds(self):
        return self.value * 1000
    def _convert_from_hours(self):
        return self.value * 3600
    def _convert_from_minutes(self):
        return self.value * 60
    def _convert_from_milliseconds(self):
        return self.value / 1000
     
class Mass(SI_unit):
    value: float
    unit: str = 'kg'
    def __init__(self,value,unit='kg'):
        """ Store units as kg but allow grams (g), pounds (lbs), metric tonnes (T) and ounces (oz) as alternate units """
        super().__init__(value, unit='kg',alt_unit=None, _convert_func=None,_exponents=[0,1,0,0,0,0,0])      
        if (unit == 'lbs') | (unit == 'lb') :
            self.alt_unit = 'lb'
            self.value = self._convert_from_pounds()
            self._convert_func = self.convert_pounds
        if (unit.lower() == 't') | (unit.lower() == 'mt'):
            self.alt_unit = 'metric-tonnes'
            self.value = self._convert_from_tonnes()
            self._convert_func = self.convert_tonnes
        if (unit == 'g') | (unit.lower() == 'grams'):
            self.alt_unit = 'g'
            self.value = self._convert_from_grams()
            self._convert_func = self.convert_grams
        if (unit.lower() == 'ounces') | (unit.lower() == 'oz'):
            self.alt_unit = 'oz'
            self.value = self._convert_from_oz()
            self._convert_func = self.convert_oz
    def convert_pounds(self):
        return self.value / 0.45359237
    def convert_tonnes(self):
        return self.value / 1000
    def convert_grams(self):
        return self.value * 1000
    def convert_oz(self):
        return self.value * 35.2739619
    def _convert_from_pounds(self):
        return self.value * 0.45359237
    def _convert_from_tonnes(self):
        return self.value * 1000
    def _convert_from_grams(self):
        return self.value / 1000
    def _convert_from_oz(self):
        return self.value / 35.2739619
"""
Cartesian Systems
----------
## Hereafter, mixins which inherit SIUnit and Cartesian
"""
class CartesianSystem:
    """ For use with directional vectors in Euclidean 3D space, with Cartesian co-ordinates """
    _value: float
    direction_xyz: list[float] = [0,0,0]
    angles_xyz_axis = [0,0,0] #planes xy xz yz, ## think on desired orientation
    def __init__(self, x, y=0, z=0):
        """ Class constructor for Cartesian Systems, which are vectors not scalars.
        By default, x axis is the direction and will equal .value if only one argument is used.
        self.update_cartesian_xyz() is called which initialises .value and angles about axes """
        print("this __init__ was called: cartesian")
        self.direction_xyz = [x,y,z]
        self.update_cartesian_xyz(self.direction_xyz)

    @property
    def value(self):
        """ Use property class, so we can control setting behaviour and update x,y,z """
        return self._value

    @value.setter
    def value(self, value):
        """ Naively, if value is set by interaction of other SIUnits and directionality is unknown """
        ## note currently this is called along with SIUnit __init__
        ## surely we want to maintain direction if that was known..?
        #self._value = value
        #self.direction_xyz = [value,0,0]
        #self.angles_xyz_axis = [0,0,0]
        if isinstance(value,SI_unit):
            """ If user accidentally passes class instance as value """
            value = value.value
        if isinstance(value,int|float):
            self.update_cartesian_xyz([value,0,0])
        else:
            self.update_cartesian_xyz(value)

    def update_cartesian_xyz(self, xyz):
        """ update x,y,z and diagonal magnitude (self.value) """
        self.direction_xyz = xyz
        xy_hypot = np.sqrt(self.direction_xyz[0]**2 + self.direction_xyz[1]**2)
        v = np.sqrt(xy_hypot**2 + self.direction_xyz[2]**2)
        self._value = v
        self.angles_xyz_axis[0] = self.planar_angle(xyz[0],xyz[1]) ## think on desired orientation
        self.angles_xyz_axis[1] = self.planar_angle(xyz[0],xyz[2])
        self.angles_xyz_axis[2] = self.planar_angle(xyz[1],xyz[2])

    @staticmethod
    def planar_angle(opposite,adjacent):
        if opposite==0 and adjacent==0:
            angle = 0
        if opposite==0:
            angle = 0
        if  adjacent==0:
            angle = 90
        else:
            angle = np.arctan(opposite/adjacent)
        return angle

    def angle_between_3d_vector(self, other):
        """ Return planar angle from two 3d vectors:
        1. Calculate dot product A•B
        2. Divide by diagonal magnitude of each
        3. Inverse cosine (arcsin) of dot prod/[A][B]"""
        print("not tested implementation of angle between 3d vectors")
        dp = self.dot_prod(other)
        angle = np.arcsin(dp/(self.value * other.value))
        return angle

    def dot_prod(self,other):
        """ Multiply two vectors A•B, but return only scalar magnitude.
        This involves projecting one to the direction of the other i.e. A*Bcos(Ø) in 2D
        This implementation uses x,y,z co-ords, multiplying each, then adding. Alt method:
        #theta = self.angle_between_3d_vector(self,other)        
        #return self.value * (other.value * np.cos(theta))
        """
        #x = self.direction_xyz[0] * other.direction_xyz[0]
        #y = self.direction_xyz[1] * other.direction_xyz[1]
        #z = self.direction_xyz[2] * other.direction_xyz[2]
        #assert((x + y + z)==np.dot(self.direction_xyz,other.direction_xyz))

        return np.dot(self.direction_xyz,other.direction_xyz)
    
    def cross_prod(self,other):
        """ Cross Product (Vector product)."""
        print("not completed cross_prod method, TO DO..")
        t = type(self)
        t2 = type(other)
        if t==t2:            
            if isinstance(self, Distance) & isinstance(other, Distance): 
                return t(np.cross(self.direction_xyz, other.direction_xyz)) # stu, anticipate area however not scalar: x,y,z?
            elif isinstance(self, CartesianSystem) & isinstance(other, CartesianSystem): 
                return t(np.cross(self.direction_xyz, other.direction_xyz))
            else:
                raise(TypeError,"This method is only for CartesianSystem subclasses")
        else:
            raise(TypeError,"This method is only for matching CartesianSystem subclasses")
        
    ## operator overrides, here the cross-product could be accounted
    def __mul__(self,other):
        """ A cross-product of A & B resulting in an area.
        If A & B are perpendicular this will be rectangular, else parrallelogram """
        t = type(self)
        if isinstance(other, t):
            # if directions are equal, just multiply
            if self.angles_xyz_axis == other.angles_xyz_axis:
                return t(self.value * other.value)
            else: 
                return Area(self.value * other.value)
        elif isinstance(other, SI_unit): 
            return SI_unit(self.value * other.value, unit=self.unit+'*'+other.unit)
        else:
            raise Exception("Only SIUnits may be multiplied to TimeUnit")
    ## TO DO - truediv
    def __add__(self,other):
        """ Adds CartesianSystem subclasses. Allows float and int.
        Allows incremental increases to Velocity via Acceleration etc.
        Otherwise, classes must match.
        """
        t = type(self)
        if isinstance(other, t): 
            return t(self.value + other.value)
        elif (isinstance(t, Acceleration) & isinstance(other, Velocity)) or (
            isinstance(t, Velocity) & isinstance(other, Acceleration)): 
            return Velocity(self.value + other.value)
        elif (isinstance(t, Velocity) & isinstance(other, Distance)) or (
            isinstance(t, Distance) & isinstance(other, Velocity)): 
            return Distance(self.value + other.value)
        elif isinstance(other, float|int): 
            return t(self.value + other.value)
        else:
            raise Exception(f"Only {t} may be added to {t}")
        
    ## in place addition
    def __iadd__(self,other):
        self.__add__(self,other)
"""
Infix
-------
##  - for mimicking new operators |dot| and |x|
"""
class Infix:
    def __init__(self, function):
        self.function = function
    def __ror__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))
    def __or__(self, other):
        return self.function(other)

dot=Infix(CartesianSystem.dot_prod)
cross=Infix(CartesianSystem.cross_prod)

"""
Mixin Cartesian & SIUnit
## For vector quantities, with magnitude and direction
"""
class Distance(SI_unit, CartesianSystem):
    value: float
    unit: str = 'm'
    direction_xyz: list[float] = [0,0,0]
    alt_unit: str
    _convert_func: str
    def __init__(self,value,unit='m'):
        super().__init__(value,unit,alt_unit=None,
                          _convert_func=None,_exponents=[1,0,0,0,0,0,0])
    def convert_mm(self):
        return self.value / 1000
    def convert_inches(self):
        return self.value * 25.4 / 1000
    

class Velocity(SI_unit, CartesianSystem):
    value: float
    unit: str = 'm/s'
    direction_xyz: list[float] = [0,0,0]
    def __init__(self,value,unit='m/s'):
        super().__init__(value,unit,alt_unit=None, _convert_func=None,_exponents=[1,0,0,0,0,0,-1])
    def convert_mph(self):
        return self.value * 3600 / 1600

class Acceleration(SI_unit, CartesianSystem):
    value: float
    unit: str = 'm/s²'
    direction_xyz: list[float] = [0,0,0]
    def __init__(self,value,unit='m/s²'):
        super().__init__(value,unit,alt_unit=None, _convert_func=None,_exponents=[1,0,0,0,0,0,-2])

class Force(SI_unit, CartesianSystem):
    value: float
    unit: str = 'N'
    direction_xyz: list[float] = [0,0,0]
    def __init__(self,value,unit='N'):
        super().__init__(value,unit,alt_unit=None, _convert_func=None,_exponents=[1,1,0,0,0,0,-2])

class Torque(SI_unit, CartesianSystem):
    value: float
    unit: str = 'Nm'
    direction_xyz: list[float] = [0,0,0]
    def __init__(self,value,unit='N'):
        super().__init__(value,unit,alt_unit=None, _convert_func=None,_exponents=[2,1,0,0,0,0,-2])

class Pressure(SI_unit):
    value: float
    unit: str = 'Pa'
    def __init__(self,value,unit='Pa'):
        super().__init__(value,unit,alt_unit='N/m²', _convert_func=None,_exponents=[-1,1,0,0,0,0,-2])

class Area(SI_unit):
    value: float
    unit: str = 'm²'
    def __init__(self,value,unit='m²'):
        super().__init__(value,unit,alt_unit=None, _convert_func=None,_exponents=[2,0,0,0,0,0,0])

class Volume(SI_unit):
    value: float
    unit: str = 'm³'
    def __init__(self,value,unit='m³'):
        super().__init__(value,unit,_exponents=[3,0,0,0,0,0,0])

## Vibration and Control Dyanamics
class SpringConstant(SI_unit):
    value: float
    unit: str = 'Nm^-1'
    def __init__(self,value,unit='N/m'):
        super().__init__(value,unit,_exponents=[0,1,0,0,0,0,-2])

class DamperCoefficient(SI_unit):
    value: float
    unit: str = 'N/ms^-1'
    def __init__(self,value,unit='N/m'):
        super().__init__(value,unit,_exponents=[0,1,0,0,0,0,-1])


class Spring:
    displacement: Distance
    stiffness: SpringConstant
    def __init__(self,displacement,stiffness):
        if not isinstance(displacement,Distance):
            displacement = Distance(displacement)
        if not isinstance(stiffness,SpringConstant):
            stiffness = SpringConstant(stiffness)
        self.displacement = displacement
        self.stiffness = stiffness
    def spring_force(self):
        return self.stiffness * self.displacement
    
class Damper:
    velocity: Velocity
    damp_coefficient: DamperCoefficient
    def __init__(self,velocity,damp_coefficient):
        self.velocity = velocity
        self.damp_coefficient = damp_coefficient
    def damping_force(self):
        return self.velocity * self.damp_coefficient
    
class Quarter_Car_SuspensionSystem:
    tyre: Spring
    coil_spring: Spring
    damper: Damper
    sprung_mass: Mass
    unsprung_mass: Mass
    suspension_working_space: Distance
    z0: Distance = 0
    z1: Distance = 0
    z2: Distance = 0
    _delta_z1_x0: Distance = 0
    _delta_z2_z1: Distance = 0
    _prior_delta_z1_x0: Distance = 0 # rqd?
    _prior_delta_z2_z1: Distance = Distance(0) 
    _prior_velocity_z1_x0: Velocity = 0 # rqd?
    _prior_velocity_z2_z1: Velocity = 0 # rqd?
    z1_v = Velocity(0)
    z2_v = Velocity(0)
    z1_a = Acceleration(0)
    z2_a = Acceleration(0)

    def __init__(self, tyre_stiffness, coil_spring_stiffness, damper_coeff, sprung_mass, unsprung_mass, height_wheelhub, height_suspension):
        self.tyre = Spring(0,tyre_stiffness)
        self.coil_spring = Spring(0,SpringConstant(coil_spring_stiffness))
        self.damper = Damper(0,DamperCoefficient(damper_coeff))
        self.sprung_mass = Mass(sprung_mass)
        self.unsprung_mass = Mass(unsprung_mass)
        self.z1 = Distance(height_wheelhub)
        self.z2 = Distance(height_suspension)
        

    def dy_vector(x: np.array):
        """ Where x is a vector, and a vector of differences is returned along axis=0 """
        return np.diff(x, axis=0)
    
    def integral_vector(x: np.array):
        """ Where x is a vector, and a vector of cumulative sum is returned along axis=0 """
        return np.cumsum(x, axis=0)
    
    def _velocity_scalar_delta_z1_x0(self, timestep=1):
        """ Velocity of unsprung mass in Z axis, with time in seconds """
        dy = (self._delta_z1_x0 - self._prior_delta_z1_x0)* (1/timestep)
        return Velocity(dy)
    
    def _velocity_scalar_delta_z2_z1(self, timestep=1):
        """ Velocity of sprung mass in Z axis with respect to unsprung, with time in seconds """
        dy = (self._delta_z2_z1 - self._prior_delta_z2_z1)* (1/timestep)
        return Velocity(dy)
    
    def _acceleration_scalar_delta_z1_x0(self, timestep=1):
        """ Acceleration of unsprung mass in Z axis, with time in seconds """
        dy2 = (self._velocity_scalar_delta_z1_x0(timestep) - self._prior_velocity_z1_x0)* (1/timestep)
        return Acceleration(dy2)
    
    def _acceleration_scalar_delta_z2_z1(self, timestep=1):
        """ Acceleration of sprung mass in Z axis, with time in seconds """
        dy2 = (self._velocity_scalar_delta_z2_z1(timestep) - self._prior_velocity_z2_z1)* (1/timestep)
        return Acceleration(dy2)


    def suspension_kinematics(self, road_displacement: Distance, timestep=1):
        self.z0 = road_displacement
        self._delta_z1_x0 = self.z1 - self.z0
        self._delta_z2_z1 = self.z2 - self.z1

        self.tyre.displacement = self._delta_z1_x0 # stu: what is the natural spring length?
        f_tyre = self.tyre.spring_force()
        self.coil_spring.displacement = self._delta_z2_z1 # stu: what is the natural spring length?
        f_coil = self.coil_spring.spring_force()
        self.damper.velocity = Velocity((self._delta_z2_z1 - self._prior_delta_z2_z1)* (1/timestep))
        f_damper = self.damper.damping_force()

        f_total_sprung = f_coil + f_damper
        f_total_unsprung = f_tyre - f_total_sprung # assume this convention for now
        
        # using absolute velocities rather than relative
        acc_sprung = f_total_sprung / self.sprung_mass
        acc_unsprung = f_total_unsprung / self.unsprung_mass
        self.z2_a = acc_sprung
        self.z1_a = acc_unsprung
        self.z2_v += acc_sprung
        self.z1_v += acc_unsprung
        self.z2 += self.z2_v
        self.z1 += self.z1_v
        # for damper next time step
        self._delta_z2_z1 = self.z2 - self.z1
        self._prior_delta_z2_z1 = self._delta_z2_z1
        

kg = Mass(1)

if __name__ == '__main__':
    v = Velocity(10)
    print(v.direction_xyz)
    print(v.angles_xyz_axis)
    print(v.value)

    v2 = Velocity([2,2,2])
    print(v2.direction_xyz)
    print(v2.angles_xyz_axis)
    print(v2.value)
    # note we shouldn't simply multiply values here. Needs to be xyz based
    print(v2 * v2)
    print(v * v2)

    car = Quarter_Car_SuspensionSystem(1000,1000,1000,250,50,0.2,0.7)
    sprung_movement = []
    unsprung_movement = []
    road = np.cos(np.arange(0,180)) 
    for i in range(10):
        car.suspension_kinematics(Distance(road[i]),0.02)
        #sprung_movement.append(car.z1.value)
        #unsprung_movement.append(car.z2.value)
        sprung_movement.append( road[i] )
    import matplotlib.pyplot as plt
    plt.plot(sprung_movement)
    #plt.plot(unsprung_movement)
    #plt.legend(['sprung','unsprung'])
    plt.show()



