"""
3.5C
SIR Model

Params:
S -- susceptible
I -- infected
R -- recovered
beta -- infection rate
gamma -- recovery rate

ODEs:
dS / dt = -beta * S * I
dI / dt = beta * S * I - gamma * I
dR / dt = gamma * I
dS/dt + dI/dt + dR/dt = 0

Results for different beta values are compared in this code.
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
X = np.linspace(0, Tmax-1, Tmax)
b = [0.01, 0.005, 0.002, 0.001] #beta
g = 0.1 #gamma

fig = plt.figure()



#calc. S, I, R
for J in range(len(b)):
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
    ax.title.set_text('$\\beta$ = %.3f / $\gamma$ = %.2f / \n Total infected = %.3f' %(b[J], g, S[0]-S[Tmax-1]))
plt.show()