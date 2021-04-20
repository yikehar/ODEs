"""
3.6A
Lotka-Volterra Model

Params:
X -- # herbivores
y -- # carnivores

a * X -- spontaneous increase of herbivores
b * X * Y -- decrease of herbivores by prey
c * X * Y -- increase of carnivores by prey
d * Y -- spontaneous decrease of carnivores

ODEs:
dX / dt = a * X - b * X * Y
dY / dt = c * X * Y - d * Y
"""

import numpy as np
import matplotlib.pyplot as plt

# init. params
dt = 0.01   #change dt and Tmax to check the accuracy of Eular's method for this model
Tmax = 10000
X = np.zeros(Tmax)
X[0] = 30   #init. # herbivores
Y = np.zeros(Tmax)
Y[0] = 10   #init. # carnivores
a = 1.0
b = 0.2
c = 0.1
d = 1.0



#calc. S, I, R
for T in range(Tmax - 1):
    X[T + 1] = X[T] + dt * (a * X[T] - b * X[T] * Y[T])
    Y[T + 1] = Y[T] + dt * (c * X[T] * Y[T] - d * Y[T])

#Plot results
time = np.linspace(0, Tmax-1, Tmax)
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(time, X, 'g-', linewidth=2, label='Herbivores')
ax1.plot(time, Y, 'r--', linewidth=2, label='Carnivores')
ax1.legend(loc='best')
ax1.set_xlabel('Time')
ax1.set_ylabel('Number of animals')
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(X, Y, 'k-', linewidth=2)
ax2.set_xlabel('Herbivores (X)')
ax2.set_ylabel('Carnivores (Y)')
plt.show()