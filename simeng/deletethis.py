A = [1,2,3,4]
B = [2,4,6,8]
expon = [(x-y) for x,y in zip(A, B)] 

print(expon)

import dynamic_systems as ds

m = ds.Mass(5)
a = ds.Acceleration(5)
f = m * a 
print(f)
print(type(f))
