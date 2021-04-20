"""
3.5A
SIR Model

Params:
S -- susceptible
I -- infected
R -- recovered
beta -- rate at which an infected person infects a susceptible person
gamma -- rate at which infected people recover from the disease

ODEs:
dS / dt = -beta * S * I
dI / dt = beta * S * I - gamma * I
dR / dt = gamma * I
dS/dt + dI/dt + dR/dt = 0
"""

import numpy as np
import matplotlib.pyplot as plt

# init. params
dt = 0.01
Tmax = 10000
S = np.zeros(Tmax)
S[0] = 99
I = np.zeros(Tmax)
I[0] = 1.0
R = np.zeros(Tmax)
b = 0.01 #beta
g = 0.1 #gamma

#calc. S, I, R
for T in range(Tmax-1):
    S[T+1] = S[T] + dt * (-b * S[T] * I[T])
    I[T+1] = I[T] + dt * (b * S[T] * I[T] - g * I[T])
    R[T+1] = R[T] + dt * (g * I[T])

#Plot results
X = np.linspace(0, Tmax-1, Tmax)
plt.plot(X, S, 'g--', linewidth=2, label='Susceptible')
plt.plot(X, I, 'r-', linewidth=2, label='Infected')
plt.plot(X, R, 'b:', linewidth=2, label='Recovered')
plt.legend(loc='best')
plt.title('$\\beta$ = %.2f / $\gamma$ = %.2f / Total infected = %.3f' %(b, g, S[0]-S[Tmax-1]))
plt.show()