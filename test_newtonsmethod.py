import numpy as np

#x**3 + x -a, solved by Newton's method
def NewtonsMethod(x0, a):
    f = pow(x0, 3) + x0 - a
    dfdx = 3*pow(x0, 2) + 1
    return x0 - f / dfdx

"""
Solution of a cubic equation
x**3 + p*x + q = 0
x = u + v

(u**3 + v**3 + q) + (3*u*v + p)(u + v) = 0

u**3 + v**3 + q = 0
3*u*v + p = 0

u**6 + q*u**3 - p**3/27 =0
u**3 = -q/2 + sqrt(D)
v**3 = -q/2 - sqrt(D)

w**2 + w + 1 = 0
u1 = pow(u**3, 1/3), u2 = u1*w, u3 = u1*w**2
v1 = pow(v**3, 1/3), v2 = v1*w, v3 = v1*w**2

x = u1 + v1, u2 + v3, u3 + v2
  = u1 + v1, u1*w + v1*w**2, u1*w**2 + v1*w
(u1 + v1) is always real.
"""

#x**3 + x - a = 0, solved analytically
def x_analitical(a):
    u3 = a/2 + np.sqrt(pow(a/2, 2) + 1/27)

    u = pow(u3, 1/3)
    v = -1 / 3 / u
    return u + v

#Initial value and a constant
x = 5.0
a_const = 0.1

#Compare results
while True:
    x1 = NewtonsMethod(x, a_const)
    if abs((x1 - x)/x) < 0.0001:
        break
    x = x1

xa = x_analitical(a_const)

print(x, xa)