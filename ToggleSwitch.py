"""
Working on the exercises in the following PDF:
http://kurodalab.bs.s.u-tokyo.ac.jp/member/Yugi/Textbook/chapter2.pdf
"""

import numpy as np
import sympy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Constant. a > 0
a_const = 3.0

#Initial values
xy0 = [1.25, 1.25]

#Time points
t = np.linspace(0, 80, 801)

def toggle_switch(xy, t, a):
    #LacI(x), lambdaCI(y)
    x, y = xy

    #ODEs
    x_dot = a / (1.0 + pow(y, 2)) -x
    y_dot = a / (1.0 + pow(x, 2)) -y
    return x_dot, y_dot

#Stability assessment
def stability(a, x, y):
    """
    Jacobian of the toggle switch
    J = [[-1,                   -2ay / (1 + y**2)**2],
        [-2ax / (1 + x**2)**2,  -1]]
    """
    J = np.array([[-1.0, -2.0*a*y / pow(1.0 + pow(y, 2.0), 2.0)],
                  [-2.0*a*x / pow(1.0 + pow(x, 2.0), 2.0), -1.0]])
    detJ = np.linalg.det(J)
    trJ = np.trace(J)
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
    return detJ, trJ, D_J

"""
At a fixed point the following system of equations hold:
    0 = a / (1 + y**2) - x
    0 = a / (1 + x**2) - y
Eliminating y gives
    (x**3 + x - a)(x**2 - a*x + 1) = 0.
Discriminant D1 of the equation x**3 + x - a = 0:
    D1 = -4 -27*a**2 < 0.
Discriminant D2 of the equation x**2 - a*x + 1 = 0:
    D2 = a**2 - 4
When a = 2, the system has a fixed point at x = 1.
When a < 2, the system has a fixed point where x is a real solution of x**3 + x - a = 0.
When a > 2, the system has three fixed points for a real solution of x**3 + x - a = 0
and two solutions of x**2 - a*x + 1 = 0.
"""
def FixedPoints(a):
    D2 = pow(a, 2) - 4
    u3 = a/2 + np.sqrt(pow(a/2, 2) + 1/27)
    u = pow(u3, 1/3)
    v = -1 / 3 / u
    x1 = u + v  #x**3 + x - a = 0 solved analytically
    if D2 > 0:
        x2 = (a + np.sqrt(D2)) / 2
        x3 = (a - np.sqrt(D2)) / 2
        x = np.array([x1, x2, x3])
    else:
        x = np.array([x1])
    y = a*np.ones(x.shape) / (np.ones(x.shape)+np.power(x,2))
    return x, y

x_fp, y_fp = FixedPoints(a_const)
for k in range(len(x_fp)):
    st = stability(a_const, x_fp[k], y_fp[k])
    print('(det, tr, D) = (%.2f, %.2f, %.2f)' % st)

#Solve ODEs
xy_sol = odeint(toggle_switch, xy0, t, args=(a_const,))

#Plot x, y vs time
plt.figure(1)
plt.plot(t, xy_sol[:, 0], 'r-', linewidth=2, label='LacI (x)')
plt.plot(t, xy_sol[:, 1], 'b-', linewidth=2, label='lambda CI (y)')
plt.xlabel('Time')
plt.ylabel('x, y')
plt.legend(loc='upper left')
plt.title("Toggle switch: a = %.2f" % (a_const))

#Plot y vs x
plt.figure(2)
plt.plot(xy_sol[:, 0], xy_sol[:, 1], 'g--', linewidth=2, label='Solutions')
meshsize = 21
xmax = max(max(x_fp),max(xy_sol[:,0]))
ymax = max(max(y_fp),max(xy_sol[:,1]))
xx, yy = np.meshgrid(np.linspace(0, xmax + 0.2, meshsize), np.linspace(0, ymax + 0.2, meshsize))
xx_dot = a_const*np.ones(xx.shape) / (np.ones(xx.shape) + np.power(yy, 2)) -xx
yy_dot = a_const*np.ones(xx.shape) / (np.ones(xx.shape) + np.power(xx, 2)) -yy
plt.quiver(xx, yy, xx_dot, yy_dot)
plt.contour(xx, yy, xx_dot, levels=[0], colors="b")
plt.contour(xx, yy, yy_dot, levels=[0], colors="r")
plt.xlim([0.0, xmax + 0.2])
plt.ylim([0.0, ymax + 0.2])
plt.plot(x_fp,y_fp, color='magenta', marker='.', markersize=16, linestyle='none', label="Fixed points")
plt.grid()
plt.title("Nullclines: Blue, dx/dt = 0; Red, dy/dt = 0")
plt.xlabel('LacI (x)')
plt.ylabel('lambda CI (y)')
plt.legend(loc='upper left')

plt.show()