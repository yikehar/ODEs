import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

"""
Modified from
https://qiita.com/kotai2003/items/ed28fb723a335a873061

Variables
S   susceptible
E   Exposed or incubation period
I   infectious
R   removed or removed
N   total population

beta    infectious rate
epsilon rate at which ab exposed person becomes infective
gamma   recovery rate
lp      latency period [day], lp = 1/epsilon
ip      infectious period [day], ip = 1/gamma

Assumptions
One who acquired the immunity will never get infected again, nor lose their immunity.
The net population N has no external influx or efflux.
There are no births and no deaths by other causes.

Differential equations
dS/dt = -beta * S * I / N
dE/dt = beta * S * I / N - epsilon * E
dI/dt = epsilon * E - gamma * I
dR/dt = gamma * I

Array that stores the 4 variables
[v[0], v[1], v[2], v[3]] = [S, E, I, R]
"""

# Function that returns the vector dv/dt
def SEIR_EQ(v, t, beta, epsilon, gamma, N):
    return(-beta*v[0]*v[2]/N, beta*v[0]*v[2]/N-epsilon*v[1], epsilon*v[1]-gamma*v[2], gamma*v[2])

# Time points
t_max = 100 #days
dt = 0.01
times = np.arange(0, t_max, dt)
print(times)

# Initial state
S0 = 99
E0 = 1
I0 = 0
R0 = 0
Npop = S0 + E0 + I0 + R0
ini_state = [S0, E0, I0, R0]

# Parameters
beta_const = 1.0
latency_period = 2.0  #days
infectious_period = 7.4 #days

epsilon_const = 1/latency_period
gamma_const = 1/infectious_period

args = (beta_const, epsilon_const, gamma_const, Npop)


# Solve ODE
results = odeint(SEIR_EQ, ini_state, times, args)

plt.plot(times, results)
plt.legend(['Susceptible', 'Exposed', 'Infectious', 'Removed'],loc='best')
plt.title("SEIR model")
plt.xlabel('Time(days)')
plt.ylabel('Population')
plt.grid()
plt.show()