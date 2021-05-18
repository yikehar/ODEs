"""
4.8A
FHN model

ODEs:
d/dt V = c * {-v**3 / 3 + V - W + I}
d/dt W = {V - b * W + a} / c

Params:
V -- membrane potential (non-dimensional)
W -- recovery variables
I -- external input current (non-dimensional)
a, b, c -- constants
"""

import numpy as np
import matplotlib.pyplot as plt

#Initialize params
Tmax, dt = 10000, 0.02
V = np.zeros(Tmax)
W = np.zeros(Tmax)
I, a, b, c = 1., 0.7, 0.8, 10.

#Solve ODEs
for T in range(Tmax-1):
    V[T + 1] = V[T] + dt * (c * (-V[T]**3 / 3. + V[T] - W[T] + I))
    W[T + 1] = W[T] + dt * (V[T] - b * W[T] + a) / c

#Plot results
t = np.linspace(0, Tmax-1, Tmax)
plt.figure()
plt.plot(t, V, 'b-', label="Membrane potential")
plt.plot(t, W, 'r--', label="Recovery variable")
plt.grid()
plt.legend(loc='lower right')
plt.show()
