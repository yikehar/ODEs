"""
3.2D

Let's consider the following system in a cell:
dE / dt = p - k*E
E -- protein level
p -- production rate
k*E -- degradation rate

The ODE can be discretely rewritten as:
E(T + 1) = {p - k*E} * dT + E

The analytical solution for the ODE is
E = {1 - exp(-k*T)} * p / k
"""
import matplotlib.pyplot as plt
import numpy as np

#Define params.
Tmax1 = 5000.0
dT = 0.01
p = 1.0
k = 0.1
Tmax2 = Tmax1 * dT

#Init. arrays for numerical sol.
E = np.zeros(int(Tmax1))
time1 = np.linspace(0, int(Tmax1), int(Tmax1))
E1 = np.zeros(int(Tmax2))

#Numerical sol.
for T in range(int(Tmax1) - 1):
    E[T + 1] = (p - k*E[T])*dT + E[T]

#Sampled elements in E are stored in E1
E1 = E[::int(1/dT)]

#Init. arrays for numerical sol.
time2 = np.linspace(0, int(Tmax2), int(Tmax2))
E2 = np.zeros(int(Tmax2))

#Analytical sol.
E2 = (1.0 - np.exp(-k * time2)) * p / k

#Plot results
plt.plot(time2, E1, 'ro', linewidth=2, label='E_numerical')
plt.plot(time2, E2, 'b*', linewidth=2, label='E_analytical')
plt.xlabel('Time')
plt.ylabel('Protein level')
plt.legend(loc='best')
plt.show()