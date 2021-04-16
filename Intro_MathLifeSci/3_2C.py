"""
3.2C

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
Tmax2 = Tmax * dT

#Init. arrays
E = np.zeros(int(Tmax2))
time = np.linspace(0, int(Tmax2), int(Tmax2) + 1)

#Analytical sol.
E = (1.0 - np.exp(-k * time)) * p / k

#Plot results
plt.plot(time, E, 'r-', linewidth=2, label='E_analytical2')
plt.xlabel('Time')
plt.ylabel('Protein level')
plt.legend(loc='best')
plt.show()