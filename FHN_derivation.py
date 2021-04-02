import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#Constants
k_DLO = 0.1
c_BVP = 5.0
a_oFHN = 0.0
b_oFHN = 0.0
c_oFHN = 2.0
i_oFHN = 0.0
a_mFHN = 0.08
b_mFHN = 1.0
c_mFHN = 0.8
i_mFHN = 0.0

#Initial values
vw1_0 = [0.1, 0.1]
vw2_0 = [0.1, 0.1]
vw3_0 = [0.1, 0.1]
vw4_0 = [0.1, 0.1]

# Time points
t = np.linspace(0,40,401)

"""
Damped Linear Oscillator
v" + k * v' + v = 0

(Lienard system)
v_dot = k * (w - v)
w_dot = -v / k

k > 0
"""
def DLO (vw, t, k):
    v = vw[0]
    w = vw[1]
    v_dot = k * (w - v)
    w_dot = -v / k
    return v_dot, w_dot

"""
Bonhoeffer - Van der Pol model
v" + c * (v**2 - 1) * v' + v = 0

(Lienard system)
v_dot = c * (w - v**3 / 3 + v)
w_dot = -v / c

c > 0
"""
def BVP (vw, t, c):
    v = vw[0]
    w = vw[1]
    v_dot = c * (w + v - pow(v, 3) / 3)
    w_dot = -v / c
    return v_dot, w_dot

"""
Original FitzHugh-Nagumo model
v_dot = c*(w + v - v^3 / 3) + i
w_dot = -(v-a+b*w) / c
1-2b/3 < a < 1, 0 < b < 1, b < c^2
When a = b = i = 0, oFHN = BVP.
"""
def oFHN (vw, t, a, b, c, i):
    v = vw[0]
    w = vw[1]
    v_dot = c * (w + v - pow(v, 3) / 3) + i
    w_dot = -(v - a + b*w) / c
    return v_dot, w_dot



"""
Modified FHN model by Rinzel
v_dot = v*(a-v)*(v-1) - w + i
w_dot = b*v - c*w
0 < a < 1, b > 0, c > 0
"""

"""
def mFHN (vw, t, a, b, c, i):
    v = vw[0]
    w = vw[1]
    v_dot = v * (a - v) * (v - 1) -w + i
    w_dot = -(b*v - c*w)
    return v_dot, w_dot
"""
def mFHN (vw, t, a, b, c, i):
    v = vw[0]
    w = vw[1]
    v_dot = v - pow(v, 3)/3 - w + i
    w_dot = -a*(b*v - c*w)
    return v_dot, w_dot




"""
Jacobian
J = [[dv'/dv,    dv'/dw],
    [dw'/dv,    dw'/dw]]
"""
def J_DLO (k):
    J = np.array([[-k, k],
                  [-1 / k, 0.0]])
    det = np.linalg.det(J)
    tr = np.trace(J)
    return det, tr

def J_BVP (v, c):
    J = np.array([[c*(1 - pow(v, 2)), c],
                  [-1 / c, 0.0]])
    det = np.linalg.det(J)
    tr = np.trace(J)
    return det, tr

def J_oFHN (v, b, c):
    J = np.array([[c*(1 - pow(v, 2)), c],
                  [-1 / c, -b / c]])
    det = np.linalg.det(J)
    tr = np.trace(J)
    return det, tr

def J_mFHN (v, a, b, c):
    J = np.array([[-3.0*pow(v, 2) + 2.0*(a+1.0)*v - a, c],
                  [b, -c]])
    det = np.linalg.det(J)
    tr = np.trace(J)
    return det, tr

#Stability assessment
def stability(detJ, trJ):
    D_J = pow(trJ, 2.0) - 4.0 * detJ
    if D_J < 0.0:  # When eigen values are complex numbers
        if trJ > 0.0:  # When real parts of the eigen values are all positive
            FP_class = 'an unstable spiral'
        elif trJ < 0.0:  # When real parts of the eigen values are all negative
            FP_class = 'a stable spiral'
        else:  # When all eigen values are pure imaginary numbers
            FP_class = 'a center'
    elif D_J == 0.0:
        if trJ > 0.0:
            FP_class = 'an unstable node'
        else:
            FP_class = 'a stable node'
    else:  # When eigen values are real numbers
        if detJ < 0.0:  # When J has both positive and negative eigen values
            FP_class = 'a saddle'
        elif trJ > 0.0:  # When all eigen values are positive or zero
            if detJ == 0.0:  # When an eigen value is zero
                FP_class = 'a saddle'
            else:
                FP_class = 'an unstable node'
        else:           # When all eigen values are negative or zero
            FP_class = 'a stable node'
    print('The fixed point (%.2f, %.2f) is %s.' % (x, y, FP_class))
    return D_J

#Solve ODEs
vw1 = odeint(DLO, vw1_0, t, args=(k_DLO,))
vw2 = odeint(BVP, vw2_0, t, args=(c_BVP,))
vw3 = odeint(oFHN, vw3_0, t, args=(a_oFHN, b_oFHN, c_oFHN, i_oFHN,))
vw4 = odeint(mFHN, vw4_0, t, args=(a_mFHN, b_mFHN, c_mFHN, i_mFHN,))

#Plot v vs time
plt.figure(1)
plt.plot(t,vw1[:,0], 'r-', linewidth=2, label='Damped Linear Ocsillator, k = %.1f' % k_DLO)
plt.plot(t,vw2[:,0], 'b-', linewidth=2, label='Van der Pol model, c = %.1f' % c_BVP)
plt.plot(t,vw3[:,0], 'g-', linewidth=2, label='Original FHN, a = %.1f, b = %.1f, c = %.1f, i = %.1f' % (a_oFHN, b_oFHN, c_oFHN, i_oFHN))
plt.plot(t,vw4[:,0], 'm-', linewidth=2, label='Modified FHN, a = %.1f, b = %.1f, c = %.1f, i = %.1f' % (a_mFHN, b_mFHN, c_mFHN, i_mFHN))
plt.xlabel('Time [a. u.]')
plt.ylabel('v [a. u.]')
plt.legend(loc='best')

#Plot w vs v for

plt.show()