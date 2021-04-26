import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import kadai3_6eq

#init. params
dt = 0.01
Tmax = 100
X = np.zeros(Tmax)
X [0] = 30
Y = np.zeros(Tmax)
Y[0] = 10
a, b, c, d = 1.0, 0.2, 0.1, 1.0
time = np.arange(0, Tmax, dt)

#Solve ODEs
L = odeint(kadai3_6eq.model, [X[0], Y[0]], time, args=(a, b, c, d,))

#plot results
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(time, L[:, 0], 'g-', linewidth=2, label='Herbivores')
ax1.plot(time, L[:, 1], 'r--', linewidth=2, label='Carnivores')
ax1.legend(loc='best')
ax1.set_xlabel('Time')
ax1.set_ylabel('Number of animals')
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(L[:, 0], L[:, 1], 'k-', linewidth=2)
ax2.set_xlabel('Herbivores (X)')
ax2.set_ylabel('Carnivores (Y)')
plt.show()