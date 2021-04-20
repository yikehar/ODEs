"""
3.5E
SIR Model

Params:
S -- susceptible
I -- infected
R -- recovered
beta -- infection rate
gamma -- recovery rate
R0 -- basic reproductive number
R0 = beta / gamma (at the early stage)

ODEs:
dS / dt = -beta * S * I
dI / dt = beta * S * I - gamma * I
dR / dt = gamma * I
dS/dt + dI/dt + dR/dt = 0

R0 is a constant in this model
"""

import numpy as np
import matplotlib.pyplot as plt

# init. params
dt = 0.01
Tmax = 10000
N = 500
I = np.zeros(Tmax)
I[0] = N * 0.01
S = np.zeros(Tmax)
S[0] = N - I[0]
R = np.zeros(Tmax)
R0 = 2.0
X = np.linspace(0, Tmax-1, Tmax)
b = [0.0001, 0.001, 0.02, 0.1] #beta

fig = plt.figure()

#calc. S, I, R
for J in range(len(b)):
    g = b[J] / R0  #calc. gamma for given beta and R0
    for T in range(Tmax-1):
        S[T+1] = S[T] + dt * (-b[J] * S[T] * I[T])
        I[T+1] = I[T] + dt * (b[J] * S[T] * I[T] - g * I[T])
        R[T+1] = R[T] + dt * (g * I[T])

#Plot results
    ax = fig.add_subplot(1, len(b), J+1)
    ax.plot(X, S, 'g--', linewidth=2, label='Susceptible')
    ax.plot(X, I, 'r-', linewidth=2, label='Infected')
    ax.plot(X, R, 'b:', linewidth=2, label='Recovered')
    ax.legend(loc='best')
    ax.title.set_text('$\\beta$ = {} / $\gamma$ = {} / \n Total infected = {}'.format(b[J], g, S[0]-S[Tmax-1]))
plt.show()