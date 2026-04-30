from .si import Mass, Distance, TimeUnit, Force, Energy, Power, Pressure, Frequency, Temperature

kg = Mass(1)
m = Distance(1)
s = TimeUnit(1)

N = Force(1)
J = Energy(1)
W = Power(1)
Pa = Pressure(1)
Hz = Frequency(1)

# Physical constants

# Acceleration due to gravity
g = 9.80665 * m / s**2

# Speed of light
c = 299_792_458 * m / s

# Planck constant
h = 6.62607015e-34 * J * s

# Boltzmann constant
k_B = 1.380649e-23 * Energy(1) / Temperature(1)