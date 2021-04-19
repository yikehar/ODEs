"""
3.3B
Haematopoiesis

Params:
L -- # long term haematopoietic stem cells (LT-HSC)
S -- # short term haematopoietic stem cells (ST-HSC)
M -- # multipotent progenitor cells (MPP)
Cl --# common lymphoid progenitor cells (CLP)
Cm --# common myeloid progenitor cells (CMP)

ODEs:
dL / dt = p1 * L - d1 * L
dS / dt = d1 * L + p2 * S - d2 * S
dM / dt = d2 * S + p3 * M - d3 * M - d4* M
dCl / dt = d3 * M - d5 * Cl
dCm / dt = d4 * M - d6 * Cm
"""
import numpy as np
import matplotlib.pyplot as plt

dt = 0.01
Tmax = 30000
L = np.zeros(Tmax+1)
L[0] = 1.0
S = np.zeros(Tmax+1)
M = np.zeros(Tmax+1)
Cl = np.zeros(Tmax+1)
Cm = np.zeros(Tmax+1)

p1 = 0.009
p2 = 0.042
p3 = 4.0
d1 = 0.009
d2 = 0.045
d3 = 0.022
d4 = 3.992
d5 = 0.00
d6 = 0.5

for T in range(Tmax):
    L[T+1] = L[T] + dt*(p1 - d1) * L[T]
    S[T+1] = S[T] + dt*(d1*L[T] + (p2 - d2)*S[T])
    M[T+1] = M[T] + dt*(d2*S[T] + (p3 - d3 - d4)*M[T])
    Cl[T+1] = Cl[T] + dt*(d3*M[T] - d5*Cl[T])
    Cm[T+1] = Cl[T] + dt * (d4*M[T] - d6*Cm[T])

X = np.linspace(0, Tmax, Tmax+1)

#Plot results
fig = plt.figure()

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(X, L, 'k--', linewidth=2, label='LT-HSC(L)')
ax1.plot(X, S, 'g:', linewidth=2, label='ST-HSC(S)')
ax1.plot(X, M, 'm-', linewidth=2, label='MPP(M)')
ax1.legend(loc='best')

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(X, L, 'k--', linewidth=2, label='LT-HSC(L)')
ax2.plot(X, Cl, 'b:', linewidth=2, label='CMP(Cl)')
ax2.plot(X, Cm, 'r-', linewidth=2, label='CMP(Cm)')
ax2.legend(loc='best')

plt.tight_layout()
plt.show()