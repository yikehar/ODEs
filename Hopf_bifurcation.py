"""
Working on the exercises in the following PDF:
http://kurodalab.bs.s.u-tokyo.ac.jp/member/Yugi/Textbook/chapter2.pdf
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Constants. a, b > 0
a_const = 2.4
b_const = 1.8

#Initial values
xy0 = [0.1, 0.8]

#Time points
t = np.linspace(0, 40, 401)

def selkov_model(xy, t, a, b):
    #ADP(x), F6P(y)
    x, y = xy

    #ODEs
    x_dot = -x + a*y + pow(x, 2) * y
    y_dot = b - a*y - pow(x, 2) * y
    return x_dot, y_dot

#Stability assessment
def stability(a, x, y):
    """
    Jacobian of the Sel'kov model
    J = [[-1 + 2*x*y, a + x**2],
        [2*x*y,      -a - x**2]]
    """
    J = np.array([[-1 + 2*x*y, a + pow(x, 2)], [2*x*y, -a - pow(x, 2)]])
    detJ = np.linalg.det(J)
    trJ = np.trace(J)
    D_J = pow(trJ, 2) - 4 * detJ
    if D_J < 0:  # When eigen values are complex numbers
        if trJ == 0:  # When all eigen values are pure imaginary numbers
            print('The fixed point (%.1f, %.1f) is a center.' % (x, y))
        elif trJ < 0:  # When real parts of the eigen values are all negative
            print('The fixed point (%.1f, %.1f) is a stable spiral.' % (x, y))
        else:
            print('The fixed point (%.1f, %.1f) is an unstable spiral.' % (x, y))
    else:  # When eigen values are real numbers
        if detJ < 0:  # When J has both positive and negative eigen values
            print('The fixed point (%.1f, %.1f) is a saddle.' % (x, y))
        elif trJ < 0:  # When all eigen values are negative
            print('The fixed point (%.1f, %.1f) is a stable node.' % (x, y))
        else:
            print('The fixed point (%.1f, %.1f) is an unstable node.' % (x, y))
    return 0

"""
At a fixed point the following system of equations hold:
    0 = -x + a*y + x**2*y
    0 = b - a*y - x**2*y
Solution:
    (x, y) = (b, b/(a + b**2))
"""
#Determine a fixed point
def FixedPoint(a, b):
    x = b
    y = b / (a + pow(b, 2))
    return x, y
x_fp, y_fp = FixedPoint(a_const, b_const)
st = stability(a_const, x_fp, y_fp)

#Solve ODEs
xy = odeint(selkov_model, xy0, t, args=(a_const, b_const,))

#Plot x, y vs time
plt.figure(1)
plt.plot(t, xy[:, 0], 'r-', linewidth=2, label='ADP (x)')
plt.plot(t, xy[:, 1], 'b-', linewidth=2, label='F6P (y)')
plt.xlabel('Time')
plt.ylabel('x, y')
plt.legend(loc='upper left')
plt.title("Sel'kov Model: (a, b) = (%.1f, %.1f)" % (a_const, b_const))

#Plot y vs x
plt.figure(2)
plt.plot(xy[:, 0], xy[:, 1], 'g--', linewidth=2, label='Solutions')
meshsize = 61
xmax = max(x_fp,max(xy[:,0]))
ymax = max(y_fp,max(xy[:,1]))
xx, yy = np.meshgrid(np.linspace(0, xmax + 0.2, meshsize), np.linspace(0, ymax + 0.2, meshsize))
xx_dot = -xx + (a_const + pow(xx, 2)) * yy
yy_dot = b_const - (a_const + pow(xx, 2)) * yy
plt.quiver(xx, yy, xx_dot, yy_dot)
plt.contour(xx, yy, xx_dot, levels=[0], colors="b")
plt.contour(xx, yy, yy_dot, levels=[0], colors="r")
plt.xlim([0.0, xmax + 0.2])
plt.ylim([0.0, ymax + 0.2])
plt.plot(x_fp,y_fp, color='magenta', marker='.', markersize=16, linestyle='none', label="Fixed points")
plt.grid()
plt.title("Nullclines: Blue, dx/dt = 0; Red, dy/dt = 0")
plt.xlabel('ADP (x)')
plt.ylabel('F6P (y)')
plt.legend(loc='upper left')
plt.show()
