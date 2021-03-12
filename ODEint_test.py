"""
copied from http://apmonitor.com/pdc/index.php/Main/SolveDifferentialEquations
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
"""
#Sample 01
# function that returns dy/dt
def model(y, t):
    k = 0.3
    dydt = -k * y
    return dydt

# initial condition
y0 = 5

# time points
t = np.linspace(0,20)

# solve ODE
y = odeint(model, y0, t)

# plot results
plt.plot(t,y)
plt.xlabel('time')
plt.ylabel('y(t)')
plt.show()
"""
"""
#Sample 02
# function that returns dy/dt
def model(y, t, k):
    dydt = -k * y
    return dydt

# initial condition
y0 = 5

# time points
t = np.linspace(0,20)

# solve ODEs
k = 0.1
y1 = odeint(model, y0, t, args=(k,))
#print(y1)
k = 0.2
y2 = odeint(model, y0, t, args=(k,))
#print(y2)
k = 0.5
y3 = odeint(model, y0, t, args=(k,))
#print(y3)

# plot results
plt.plot(t,y1, 'r-', linewidth=2, label='k=0.1')
plt.plot(t,y2, 'b--', linewidth=2, label='k=0.2')
plt.plot(t,y3, 'g:', linewidth=2, label='k=0.5')
plt.xlabel('time')
plt.ylabel('y(t)')
plt.legend()
plt.show()
"""
"""
#ex 01
# function that returns dy/dt
def model(y, t):
    dydt = -y + 1.0
    return dydt

# initial condition
y0 = 0

# time points
t = np.linspace(0,5)
#print(t)
# solve ODEs
y = odeint(model, y0, t)

# plot results
plt.plot(t,y, 'r-', linewidth=2, label='k=0.1')
plt.xlabel('time')
plt.ylabel('y(t)')
plt.legend()
plt.show()
"""
"""
# ex 02
# u steps from 0 to 2 at t=10
def func_u(t):
    if t < 10.0:
            u = 0
    else:
            u = 2
    return u

# function that returns dy/dx
def model(y, t):
    u = func_u(t)
    dydt = -0.2 * y + 0.2 * u
    return dydt

# initial condition
y0 = 1

# time points
t = np.linspace(0,20, 500)
#print(t)
# solve ODEs
y = odeint(model, y0, t)
print(y)
# plot results
plt.plot(t,y, 'r-', linewidth=2, label='Output y(t)')
plt.plot([0, 10, 10, 20],[0, 0, 2, 2], 'b--', linewidth=2, label='Input u(t)')
plt.xlabel('time')
plt.ylabel('y(t), u(t)')
plt.legend(loc='best')
plt.show()
"""
"""
#ex 03
# function that returns dy/dx
def model(z, t):
    dxdt = 3 * np.exp(-t)
    dydt = 3 - z[1]
    dzdt = [dxdt, dydt]
    return dzdt

# initial condition
z0 = [0, 0]

# time points
t = np.linspace(0,5)

# solve ODEs
z = odeint(model, z0, t)

# plot results
plt.plot(t,z[:,0], 'r-', linewidth=2, label='x(t)')
plt.plot(t,z[:,1], 'b--', linewidth=2, label='y(t)')
plt.xlabel('time')
plt.ylabel('x, y')
plt.legend(loc='best')
plt.show()
"""
"""
#ex 04
# input function
def func_u(t):
#S is the function that steps from zero to one at t = 5
    if t < 5.0:
        S = 0
    else:
        S = 1
    u = 2 * S
    return u

# function that returns dy/dt, dx/dt
def model(z, t):
    u = func_u(t)
    dxdt = -0.5*z[0] + 0.5*u
    dydt = -0.2*z[1] + 0.2*z[0]
    dzdt = [dxdt, dydt]
    return dzdt

# initial condition
z0 = [0, 0]

# time points
t = np.linspace(0,40)

# solve ODEs
z = odeint(model, z0, t)

# plot results
plt.plot([0, 5, 5, 40],[0, 0, 2, 2], 'g:', linewidth=2, label='u(t)')
plt.plot(t,z[:,0], 'r-', linewidth=2, label='x(t)')
plt.plot(t,z[:,1], 'b--', linewidth=2, label='y(t)')
plt.xlabel('time')
plt.ylabel('x, y')
plt.legend(loc='best')
plt.show()
"""

#ex04 sample answer
# function that returns dz/dt
def model(z, t, u):
    x = z[0]
    y = z[1]
    dxdt = (-x + u)/2.0
    dydt = (-y + x)/5.0
    dzdt = [dxdt, dydt]
    return dzdt

# initial condition
z0 = [0,0]

# number of time points
n=401

# time points
t = np.linspace(0,40,n)

# step input
u = np.zeros(n)
# change to 2.0 at time = 5.0
u[51:] = 2.0

# store solution
x = np.empty_like(t)
y = np.empty_like(t)
# record initial conditions
x[0] = z0[0]
y[0] = z0[1]

#solve ODE
for i in range(1,n):
    #span for next time step
    tspan = [t[i-1],t[i]]
    #solve for next step
    z = odeint(model, z0, tspan, args=(u[i],))
    # store solution for plotting
    x[i] = z[1][0]
    y[i] = z[1][1]
    #next initial condition
    z0 = z[1]
print(t)
# plot results
plt.plot(t,u,'g:',label='u(t)')
plt.plot(t,x,'b-',label='x(t)')
plt.plot(t,y,'r--',label='y(t)')
plt.ylabel('values')
plt.xlabel('time')
plt.legend(loc='best')
plt.show()