import numpy as np

#Constants
a0 = -4.0
b0 = -2.0
c0 = 0.1

w = -0.5 + np.sqrt(3)/(-2j) #One of the complex cube roots of -1
def cubic(a, b, c):         #Cubic formula
    y0 = a / 3.0
    p = -pow(a,2.0)/3.0 + b
    q = 2.0*pow(a,3.0)/27.0 - a*b/3.0 + c
    D_4 = pow(q,2.0)/4.0 + pow(p,3.0)/27.0
    if D_4 < 0.0:
        r = np.sqrt(pow(q/2.0, 2.0) - D_4)
        th = np.arctan(np.sqrt(-D_4)/(-q/2))
        u = pow(r, 1/3)*np.exp(1.0j*th/3.0)
        v = -p / u / 3.0
        y = np.array([u + v, (u + v * w) * w, (u * w + v) * w])
        x = y - y0 * np.ones(y.shape)
        x = x.real
    else:
        u_3 = -q / 2.0 + np.sqrt(D_4)
        if u_3 < 0.0:
            u = -pow(-u_3,1/3)
            v = -p / u / 3.0
        elif u_3 > 0.0:
            u = pow(u_3, 1/3)
            v = -p / u / 3.0
        else:
            u = 0.0
            v = 0.0
        y = np.array([u + v, (u + v*w)*w, (u*w + v)*w])
        x = y - y0*np.ones(y.shape)
    return x
x_cal = cubic(a0, b0, c0)
print(x_cal)

"""
Solve a cubic equation
x^3 + a*x^2 + b*x + c = 0

Steps:
y^3 + p*y + q = 0, x = y - y0
y = u + v

u**3 + v**3 + q = 0
3*u*v + p = 0

u**6 + q*u**3 - p**3/27 = 0
"""