"""
3.3C
Haematopoiesis

Params:
L -- # long term haematopoietic stem cells (LT-HSC)
S -- # short term haematopoietic stem cells (ST-HSC)
M -- # multipotent progenitor cells (MPP)
Cl --# common lymphoid progenitor cells (CLP)
Cm --# common myeloid progenitor cells (CMP)

ODEs, logistic growth
dL / dt = p1 * L * (1 - L / Lmax) - d1 * L
dS / dt = d1 * L + p2 * S * (1 - S / Smax) -d2 * S
dM / dt = d2 * S + p3 * M * (1 - M / Mmax)- d3 * M
(CLPs and CMPs are counted as the same type of cells)
"""
import numpy as np
import matplotlib.pyplot as plt

dt = 0.01
Tmax = 60000
L = np.zeros(Tmax+1)
L[0] = 1.0
S = np.zeros(Tmax+1)
M = np.zeros(Tmax+1)
p1 = 0.02
p2 = 0.06
p3 = 4.01
d1 = 0.01
d2 = 0.05
d3 = 4.0
Lmax = 5.0
Smax = 5.0
Mmax = 5.0

for T in range(Tmax):
    L[T+1] = L[T] + dt * (p1*L[T]*(1.0 - L[T]/Lmax) - d1*L[T])
    S[T+1] = S[T] + dt * (d1 * L[T] + p2*S[T]*(1.0 - S[T]/Smax) - d2*S[T])
    M[T+1] = M[T] + dt * (d2*S[T] + p3*M[T]*(1.0 - M[T]/Mmax) - d3*M[T])

X = np.linspace(0, Tmax, Tmax+1)
#Plot results
plt.plot(X, L, 'k--', linewidth=2, label='LT-HSC(L)')
plt.plot(X, S, 'g:', linewidth=2, label='ST-HSC(S)')
plt.plot(X, M, 'm-', linewidth=2, label='MPP(M)')

plt.xlabel('Time')
plt.ylabel('Cell counts')
plt.legend(loc='best')
plt.show()