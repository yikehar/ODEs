"""
Working on the exercises in the following PDF:
http://kurodalab.bs.s.u-tokyo.ac.jp/member/Yugi/Textbook/chapter2.pdf
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Constants. a, b > 0
a_const = 0.01
b_const = 0.3

#Initial values
xy0 = [0.4, 0.1]

#Time points
t = np.linspace(0, 80, 801)

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
        [-2*x*y,      -a - x**2]]
    """
    J = np.array([[-1.0 + 2.0*x*y, a + pow(x, 2.0)],
                  [-2.0*x*y, -a - pow(x, 2.0)]])
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
    0 = -x + a*y + x**2*y
    0 = b - a*y - x**2*y
Solution:
    (x, y) = (b, b/(a + b**2))
"""
#Determine the fixed point
def FixedPoint(a, b):
    #When divided by zero
    if pow(a, 2) + pow(b, 2) == 0:
        x = 0.0
        y = 0.0
    else:
        x = b
        y = b / (a + pow(b, 2))
    return x, y
x_fp, y_fp = FixedPoint(a_const, b_const)
st = stability(a_const, x_fp, y_fp)
print('(det, tr, D) = (%.2f, %.2f, %.2f)' % st)

#Solve ODEs
xy = odeint(selkov_model, xy0, t, args=(a_const, b_const,))

#Plot x, y vs time
plt.figure(1)
plt.plot(t, xy[:, 0], 'r-', linewidth=2, label='ADP (x)')
plt.plot(t, xy[:, 1], 'b-', linewidth=2, label='F6P (y)')
plt.xlabel('Time')
plt.ylabel('x, y')
plt.legend(loc='upper left')
plt.title("Sel'kov Model: (a, b) = (%.2f, %.2f)" % (a_const, b_const))

#Plot y vs x
plt.figure(2)
plt.plot(xy[:, 0], xy[:, 1], 'g--', linewidth=2, label='Solutions')
meshsize = 21
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
plt.plot(x_fp,y_fp, color='magenta', marker='.', markersize=16, linestyle='none', label="Fixed point")
plt.grid()
plt.title("Nullclines: Blue, dx/dt = 0; Red, dy/dt = 0")
plt.xlabel('ADP (x)')
plt.ylabel('F6P (y)')
plt.legend(loc='upper left')

#Plot b vs a
amin = 0.0
amax = 2.5
adivs = 81
bmin = 0.0
bmax = 2.5
bdivs = 81
aa, bb = np.meshgrid(np.linspace(amin, amax, adivs), np.linspace(bmin, bmax, bdivs))
det_grid = np.zeros(aa.shape)
tr_grid = np.zeros(aa.shape)
D_grid = np.zeros(aa.shape)
for m in range(0, aa.shape[0]):
    for n in range(0, aa.shape[1]):
        xfp, yfp = FixedPoint(aa[m,n], bb[m,n])
        det_temp, tr_temp, D_temp = stability(aa[m,n], xfp, yfp)
        det_grid[m,n] = det_temp
        tr_grid[m,n] = tr_temp
        D_grid[m,n] = D_temp
        print(det_grid[m,n],tr_grid[m,n],D_grid[m,n])
plt.figure(3)
contf_D = plt.contourf(aa, bb, D_grid, 4, cmap='hsv')
CB_D = plt.colorbar(contf_D)
CB_D.set_label('D = tr2(J) - 4det(J)')
contf_tr = plt.contourf(aa, bb, tr_grid, 4, hatches=['\\', 'o', '/', '-'],cmap='gray',alpha=0.3)
CB_tr = plt.colorbar(contf_tr)
CB_tr.set_label('tr(J)')
plt.contour(aa, bb, tr_grid, levels=[0], colors="r")
plt.contour(aa, bb, D_grid, levels=[0], colors="b")
plt.xlabel('a')
plt.ylabel('b')
plt.title("Lines: Blue, D = 0; Red, tr(J) = 0")

plt.show()
