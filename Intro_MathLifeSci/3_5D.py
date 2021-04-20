"""
3.5D
SIR Model

Params:
S -- susceptible
I -- infected
R -- recovered
beta -- infection rate
gamma -- recovery rate
N -- population size
N = S + I + R

ODEs:
dS / dt = -beta * S * I
dI / dt = beta * S * I - gamma * I
dR / dt = gamma * I
dS/dt + dI/dt + dR/dt = 0


Results for different population sizes are compared in this code.
"""

import numpy as np
import matplotlib.pyplot as plt

# init. params
dt = 0.01
Tmax = 10000
S = np.zeros(Tmax)
I = np.zeros(Tmax)
R = np.zeros(Tmax)
X = np.linspace(0, Tmax-1, Tmax)
b = 0.001 #beta
g = 0.1 #gamma
N = [100, 200, 300, 400]

fig = plt.figure()

#calc. S, I, R
for J in range(len(N)):
    I[0] = N[J] * 0.01 #init. I
    S[0] = N[J] - I[0] #init. S
    for T in range(Tmax-1):
        S[T+1] = S[T] + dt * (-b * S[T] * I[T])
        I[T+1] = I[T] + dt * (b * S[T] * I[T] - g * I[T])
        R[T+1] = R[T] + dt * (g * I[T])

#Plot results
    ax = fig.add_subplot(1, len(N), J+1)
    ax.set_ylim(0, N[J])
    ax.plot(X, S, 'g--', linewidth=2, label='Susceptible')
    ax.plot(X, I, 'r-', linewidth=2, label='Infected')
    ax.plot(X, R, 'b:', linewidth=2, label='Recovered')
    ax.legend(loc='best')
    ax.title.set_text('Total infected = %.1f \n Proportion = %.2f' %(S[0]-S[Tmax-1], (S[0]-S[Tmax-1])/N[J]))
plt.show()