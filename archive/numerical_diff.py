import numpy as np
import matplotlib.pyplot as plt

def F(x):
    
    y = np.cos(x)
    return y


# Array of angles in radians
x = np.arange(-np.pi, np.pi, np.pi/10)
for idx,_ in enumerate(x):        
        if _ == 0:
            print("was 0")
            x[idx] += np.finfo(float).eps
### numeric differentiation
y = F(x)

h     = np.sqrt(np.finfo(float).eps) * x
xph   = x + h
dx    = xph - x
slope = (F(xph) - F(x)) / dx


plt.plot(x,y)
plt.title("cosine")
plt.show()

plt.plot(x,slope)
plt.title("-sin vs F(x)'")
plt.show()

print(x)