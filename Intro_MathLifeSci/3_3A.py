"""
3.3A
Params:
L -- # long term haematopoietic stem cells (LT-HSC)
S -- # short term haematopoietic stem cells (ST-HSC)
M -- # multipotent progenitor cells (MPP)
Cl --# common lymphoid progenitor cells (CLP)
Cm --# common myeloid progenitor cells (CMP)

ODEs
dL / dt = p1 * L - d1 * L
dS / dt = d1 * L + p2 * S -d2 * S
dM / dt = d2 * S + p3 * M - d3 * M
(CLPs and CMPs are counted as the same type of cells)
"""
import numpy as np
import matplotlib.pyplot as plt

dt = 0.01
Tmax = 30000
L = np.zeros(Tmax+1)
L[0] = 1.0
S = np.zeros(Tmax+1)
M = np.zeros(Tmax+1)
p1 = 0.009
p2 = 0.042
p3 = 4.0
d1 = 0.009
d2 = 0.045
d3 = 4.014

for T in range(Tmax):
    L[T+1] = L[T]+dt*(p1 - d1) * L[T]
    S[T+1] = S[T]+dt*(d1 * L[T] + (p2 - d2) * S[T])
    M[T+1] = M[T]+dt*(d2*S[T] + (p3-d3)*M[T])

X = np.linspace(0, Tmax, Tmax+1)
#Plot results
plt.plot(X, L, 'k--', linewidth=2, label='LT-HSC(L)')
plt.plot(X, S, 'g:', linewidth=2, label='ST-HSC(S)')
plt.plot(X, M, 'm-', linewidth=2, label='MPP(M)')

plt.xlabel('Time')
plt.ylabel('Cell counts')
plt.legend(loc='best')
plt.show()