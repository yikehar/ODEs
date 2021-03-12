"""
Working on the exercises in the following PDF:
http://kurodalab.bs.s.u-tokyo.ac.jp/member/Yugi/Textbook/chapter2.pdf
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Constants. a, b > 0
a_const = 0.8
b_const = 0.3

#Initial values
xy0 = [0.01, 0.8]

#Time points
t = np.linspace(0, 40, 401)

def griffith_model(xy, t, a, b):
    #x, translated protein lv; t, transcribed mRNA lv
    x, y = xy

    #ODEs
    x_dot = -a * x + y
    y_dot = pow(x,2) / (1 + pow(x, 2)) -b * y
    return x_dot, y_dot

#Stability assessment
def stability(a, b, x, y):
    """
    Jacobian of the Griffith model
    J = [[-a,                1],
        [2*x/(1 + x**2)**2, -b]]
    """
    J = np.array([[-a, 1], [2 * x / pow(1 + pow(x, 2), 2), -b]])
    detJ = np.linalg.det(J)
    trJ = np.trace(J)
    D_J = pow(trJ, 2) - 4 * detJ
    if D_J < 0: #When eigen values are complex numbers
        if trJ == 0:    #When all eigen values are pure imaginary numbers
            print('The fixed point (%.1f, %.1f) is a center.' % (x, y))
        elif trJ < 0:   #When real parts of the eigen values are all negative
            print('The fixed point (%.1f, %.1f) is a stable spiral.' % (x, y))
        else:
            print('The fixed point (%.1f, %.1f) is an unstable spiral.' % (x, y))
    else:   #When eigen values are real numbers
        if detJ < 0:    #When J has both positive and negative eigen values
            print('The fixed point (%.1f, %.1f) is a saddle.' % (x, y))
        elif trJ < 0:   #When all eigen values are negative
            print('The fixed point (%.1f, %.1f) is a stable node.' % (x, y))
        else:
            print('The fixed point (%.1f, %.1f) is an unstable node.' % (x, y))
    return 0

"""
At a fixed point the following system of equations hold:
    0 = -a * x + y
    0 = x**2 / (1 + x**2) -b * y
Eliminating y gives:
    x * (x**2 -x/ab + 1) = 0
"""
#Determine fixed points
def FPs(a, b):
    # Quadratic formula at the fixed points: x = (1 +/- sqrt(1-4*a^2*b^2))/(2*a*b)
    D = 1.0 - 4*pow(a,2)*pow(b,2)   #Discriminant
    if D > 0:
        numFPs = 3
        x1 = 0.0
        x2 = (1 - np.sqrt(D)) / (2 * a * b)
        x3 = (1 + np.sqrt(D)) / (2 * a * b)
        x = np.array([x1, x2, x3])
    elif D < 0:
        numFPs = 1
        x = np.array([0.0])
    else:
        numFPs = 2
        x1 = 0.0
        x2 = 1 / (2*a*b)
        x = np.array([x1, x2])
    y = np.multiply(x, a)
    return x, y, numFPs

#Solve ODEs
xy = odeint(griffith_model, xy0, t, args=(a_const, b_const,))

#Fixed points
x_fp, y_fp, num_fp = FPs(a_const, b_const)
print('Number of fixed points: %d' % num_fp)
for num in range(0, num_fp):
    st = stability(a_const, b_const, x_fp[num], y_fp[num])

#Plot x, y vs time
plt.figure(1)
plt.plot(t, xy[:, 0], 'r-', linewidth=2, label='Protein (x)')
plt.plot(t, xy[:, 1], 'b-', linewidth=2, label='mRNA (y)')
plt.xlabel('Time')
plt.ylabel('x, y')
plt.legend(loc='upper left')
plt.title("Griffith Model: (a, b) = (%.1f, %.1f)" % (a_const, b_const))

#Plot y vs x
plt.figure(2)
plt.plot(xy[:, 0], xy[:, 1], 'g--', linewidth=2, label='Solutions')
meshsize = 101
xmax = max(max(x_fp),max(xy[:,0]))
ymax = max(max(y_fp),max(xy[:,1]))
xx, yy = np.meshgrid(np.linspace(0, xmax + 0.2, meshsize), np.linspace(0, ymax + 0.2, meshsize))
xx_dot = -a_const * xx + yy
yy_dot = pow(xx,2) / (np.ones(xx.shape) + pow(xx, 2)) -b_const * yy
plt.quiver(xx, yy, xx_dot, yy_dot)
plt.contour(xx, yy, xx_dot, levels=[0], colors="b")
plt.contour(xx, yy, yy_dot, levels=[0], colors="r")
plt.xlim([-0.1, xmax + 0.2])
plt.ylim([-0.1, ymax + 0.2])
plt.plot(x_fp,y_fp, color='magenta', marker='.', markersize=16, linestyle='none', label="Fixed points")
plt.grid()
plt.title("Nullclines: Blue, dx/dt = 0; Red, dy/dt = 0")
plt.xlabel('Protein (x)')
plt.ylabel('mRNA (y)')
plt.legend(loc='upper left')
plt.show()
