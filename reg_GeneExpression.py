"""
Goodwin model (Model 1) that demonstrates regulation of gene expression
dM / dt = V / (D + P ** m) - a * M
dE / dt = b * M - c * E
dP / dt = d * E - e * P

Modified model (Model 2)
dP / dt = d * E - e * P / (k + P)

Equations
Mathematical Biology: I. An Introduction By James D. Murray
https://books.google.co.jp/books?id=4WbpP90Gk1YC&lpg=PP1&dq=murray%20mathematical%20biology%20introduction&pg=PP1#v=onepage&q=murray%20mathematical%20biology%20introduction&f=false

Parameter values
The Goodwin Model: Behind the Hill Function
https://doi.org/10.1371/journal.pone.0069573
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Parameters
V0 = 1.0
D0 = 1.0
ml0 = 6.4
a0 = 0.1
b0 = 1.0
c0 = 0.1
dl0 = 1.0
e1 = 0.1
e2 = 0.1
k0 = 0.1
args1 = (V0, D0, ml0, a0, b0, c0, dl0, e1)
args2 = (V0, D0, ml0, a0, b0, c0, dl0, e2, k0)

# Initial condition
M0 = 0.0    # Initial mRNA level
E0 = 0.2    # Initial level of the translated enzyme
P0 = 2.5    # Initial level of the reaction product
v_ini = [M0, E0, P0]

# Goodwin model
def model1(v, t, V, D, ml, a, b, c, dl, e):
    M = v[0]
    E = v[1]
    P = v[2]
    dMdt = V / (D + P ** ml) - a * M
    dEdt = b * M - c * E
    dPdt = dl * E - e * P
    dvdt = [dMdt, dEdt, dPdt]
    return dvdt

# Modified model
def model2(v, t, V, D, ml, a, b, c, dl, e, k):
    M = v[0]
    E = v[1]
    P = v[2]
    dMdt = V / (D + P ** ml) - a * M
    dEdt = b * M - c * E
    dPdt = dl * E - e * P / (k + P)
    dvdt = [dMdt, dEdt, dPdt]
    return dvdt

# Time points
t = np.linspace(0, 400, 201)

# Solve ODEs
v1 = odeint(model1, v_ini, t, args1)
v2 = odeint(model2, v_ini, t, args2)

# Plot results
plt.plot(t,v1[:,0], 'b-', linewidth=2, label='mRNA (Model 1)')
plt.plot(t,v1[:,1], 'g-', linewidth=2, label='Enzyme (Model 1)')
plt.plot(t,v1[:,2], 'r-', linewidth=2, label='Product (Model 1)')
plt.plot(t,v2[:,0], 'b:', linewidth=2, label='mRNA (Model 2)')
plt.plot(t,v2[:,1], 'g:', linewidth=2, label='Enzyme (Model 2)')
plt.plot(t,v2[:,2], 'r:', linewidth=2, label='Product (Model 2)')
plt.xlabel('Time')
plt.ylabel('Level')
plt.title('m = %.1f' % ml0)
plt.legend(loc='best')
plt.show()