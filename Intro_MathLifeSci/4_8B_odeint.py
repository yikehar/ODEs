"""
4.8B_odeint
Heartbeat
Diffusion term added to FHN model
Solved with odeint

PDEs:
d/dt V = c * {-v**3 / 3 + V - W + I} + d * (Laplacian) * V
d/dt W = {V - b * W + a} / c

Params:
V -- membrane potential (non-dimensional)
W -- recovery variables
I -- external input current (non-dimensional)
a, b, c -- constants
d -- diffusion coefficient
"""

import numpy as np
import matplotlib.pyplot as plt
import discreteLaplacian
import os
from PIL import Image
from scipy.integrate import odeint

#Model heartbeat
def heartbeat2D(matVW, t):
    # reshape V, W
    n = int(len(matVW) / 2)
    x_max = int(np.sqrt(n))
    v1, w1 = matVW[:n].reshape(x_max, x_max), matVW[n:].reshape(x_max, x_max)

    #Initialize params
    i = np.zeros((x_max, x_max))
    i[49:51, 49:51] = 1.
    d, a, b, c = 1., 0.7, 0.8, 10.
    dx = 1.
    dx2 = dx * dx

    #PDEs
    dvdt = c * (-v1**3 / 3. + v1 - w1 + i) + d * discreteLaplacian.Lap2DMt(v1, dx2)
    dwdt = (v1 - b * w1 + a) / c
    return np.hstack((dvdt.ravel(), dwdt.ravel()))

#Initialize params
Tmax, dt = 5000, 0.02
Xmax = 100
VW_ini = np.zeros(2 * Xmax**2)
V = np.zeros((Xmax, Xmax, Tmax))
W = np.zeros((Xmax, Xmax, Tmax))
t = np.arange(0, Tmax*dt, dt)

#Solve PDEs
VW = odeint(heartbeat2D, VW_ini, t)
V = VW[:,:Xmax**2].T.reshape(Xmax, Xmax, Tmax)
W = VW[:,Xmax**2:].T.reshape(Xmax, Xmax, Tmax)
"""
for i in range(1,num):  #Use for-loop to avoid MemoryError
    #span for next time step
    tspan = [t[i-1],t[i]]
    #solve for next step
    z = odeint(heartbeat2D, VW_ini, tspan, args=(Xmax,))
    # store solution
    V[:, :, i] = z[1][:lenX].reshape(Xmax, Xmax)
    W[:, :, i] = z[1][lenX:].reshape(Xmax, Xmax)
    #next initial condition
    VW_ini = z[1]
    print(i)
"""




#Reduce size of matrices by slicing
STEP = 100
Vr = V[:, :, ::STEP]
Wr = W[:, :, ::STEP]

V_max = V.max().__ceil__()
V_min = V.min().__floor__()
W_max = W.max().__ceil__()
W_min = W.min().__floor__()

#Plot results
len_T = len(str(Vr.shape[2]))  #the number of digits
for time in range(Vr.shape[2]):
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.4))
    heatmapV = axs[0].pcolor(Vr[:, :, time], vmin=V_min, vmax=V_max, cmap='YlOrRd')
    fig.colorbar(heatmapV, ax=axs[0])
    heatmapW = axs[1].pcolor(Wr[:, :, time], vmin=W_min, vmax=W_max, cmap='YlOrRd')
    fig.colorbar(heatmapW, ax=axs[1])
    s = str(time).zfill(len_T)
    fig.tight_layout()
    fig.savefig("GIF\\{}.png".format(s))   #Save image for each loop
    plt.clf()
    plt.cla()
    plt.close()
path = os.getcwd() + "\\GIF\\"
files = [path + str(t).zfill(len_T) + '.png' for t in range(Vr.shape[2])]
images = list(map(lambda file: Image.open(file), files))
images.pop(0).save(path + "4_8B_odeint.gif" ,save_all = True, append_images = images, duration = 10, optimize = False, loop = 0)
