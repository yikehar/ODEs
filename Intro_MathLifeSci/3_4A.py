"""
3.4A
Glucose metabolism

Params:
G -- blood glucose
I -- blood insulin
d -- diet

ODEs:
dG / dt = d - k1 * (1 + a*I) * G
dI / dt = b * G - k2 * I
"""

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

dt = 0.01
Tmax = 10000
dint = 1000    #interval btwn meals

# init. d
d = np.zeros(dint)
d[dint - 101:dint]=1.0 #set 0.5 for diet restriction
d = np.matlib.repmat(d,1,int(Tmax/dint))
d = d.reshape(Tmax,)
p = np.zeros(d.shape)
#p = d  #uncomment to simulate insulin administraion for the treatment of type I Dm

# init. G and I
G = np.zeros(Tmax)
G[0] = 1.0
I = np.zeros(Tmax)
I[0] = 1.0

#constants
a = 0.0 #set a = 0.0 for type II DM
b = 1.0 #set b = 0.0 for type I DM
k1 = 0.1
k2 = 0.1

#calc. G and I numerically
for T in range(Tmax-1):
    G[T+1] = G[T] + dt * (-k1 * G[T] * (1.0 + a * I[T]) + d[T])
    I[T+1] = I[T] + dt * (b * G[T] - k2 * I[T] + p[T])

#calc. mean of G in its stable state
Gave = G[int(Tmax/2) : Tmax].mean()

#Plot results
X = np.linspace(0, Tmax-1, Tmax)
plt.plot(X, G, 'g-', linewidth=2, label='Glucose')
plt.plot(X, I, 'r-', linewidth=2, label='Insulin')
plt.plot(X, d, 'b--', linewidth=2, label='Food intake')
plt.ylim(0, 5)
plt.legend(loc='best')
plt.title('a = %.1f / b= %.1f / G_ave = %.3f' %(a, b, Gave))
plt.show()
