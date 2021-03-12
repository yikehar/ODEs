import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

"""
Damped Linear Oscillator
v" + k * v' + v = 0

(Lienard system)
v_dot = k * (w - v)
w_dot = -v / k
"""
def DLO (vw1, t, k):
    v1 = vw1[0]
    w1 = vw1[1]
    v1_dot = k * (w1 - v1)
    w1_dot = -v1 / k
    return v1_dot, w1_dot

"""
Bonhoeffer - Van der Pol model
v" + c * (v**2 - 1) * v' + v = 0

(Lienard system)
v_dot = c * (w - v**3 / 3 + v)
w_dot = -v / c
"""
def BVP (vw2, t, c):
    v2 = vw2[0]
    w2 = vw2[1]
    v2_dot = c * (w2 - v2**3 / 3 + v2)
    w2_dot = -v2 / c
    return v2_dot, w2_dot

#Constants
k_const = 0.1
c_const = 5.0

#Initial values
vw1_0 = [1.0, 0.0]
vw2_0 = [1.0, 0.0]

# Time points
t = np.linspace(0,40,401)

#Solve ODEs
vw1 = odeint(DLO, vw1_0, t, args=(k_const,))
vw2 = odeint(BVP, vw2_0, t, args=(c_const,))


# plot results
plt.plot(t,vw1[:,0], 'r-', linewidth=2, label='d2v/dt2 + k * dv/dt + v = 0, k = %.1f' % k_const)
plt.plot(t,vw2[:,0], 'b-', linewidth=2, label='d2v/dt2 + c * (v**2 - 1) * dv/dt + v = 0, c = %.1f' % c_const)
plt.xlabel('Time [a. u]')
plt.ylabel('v [a. u.]')
plt.legend(loc='best')
plt.show()