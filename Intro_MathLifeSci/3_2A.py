"""
3.2A

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
Tmax = 5000.0
dT = 0.01
p = 1.0
k = 0.1

#Init. arrays
E = np.zeros(int(Tmax))
time = np.zeros(int(Tmax))

#Calc. numerically
for T in range(int(Tmax) - 1):
    E[T + 1] = (p - k*E[T])*dT + E[T]
    time[T + 1] = T

#Plot results
plt.plot(time, E, 'r-', linewidth=2, label='E_numerical')
plt.xlabel('Time')
plt.ylabel('Protein level')
plt.legend(loc='best')
plt.show()