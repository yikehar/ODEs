"""
https://opinionator.blogs.nytimes.com/2009/05/26/guest-column-loves-me-loves-me-not-do-the-math/?_r=0
x(t) = cos(t)
y(t) = sin(t)
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# function that returns dz/dx
def model(z, t):
    dxdt = -z[1]
    dydt = z[0]
    dzdt = [dxdt, dydt]
    return dzdt

# initial condition
z0 = [1.0, 0.0]

# time points
t = np.linspace(0,40,401)

# solve ODEs
z = odeint(model, z0, t)

# plot results
plt.plot(t,z[:,0], 'r-', linewidth=2, label='x(t)')
plt.plot(t,z[:,1], 'b--', linewidth=2, label='y(t)')
plt.xlabel('time')
plt.ylabel('x, y')
plt.legend(loc='best')
plt.show()