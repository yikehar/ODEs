"""
4.8D
Heartbeat
Diffusion term added to FHN model

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


#Reaction Diffusion Equation
def dvdt_dwdt(v, w, Ie):
    #Constants
    d, a, b, c = 1., 0.7, 0.8, 10.
    dx = 1.
    dx2 = dx * dx
    return c * (-v**3 / 3. + v - w + Ie) + d * discreteLaplacian.Lap2DMt(v, dx2), (v - b * w + a) / c

#Heartbeat
def heartbeat2D(matVW, t):
    # reshape V, W
    n = int(len(matVW) / 2)
    x_max = int(np.sqrt(n))
    v1, w1 = matVW[:n].reshape(x_max, x_max), matVW[n:].reshape(x_max, x_max)

    #External curren generated in the SV node
    i = np.zeros((x_max, x_max))
    i[49:51, 49:51] = 1.

    #PDEs
    dvdt, dwdt = dvdt_dwdt(v1, w1, i)
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



#Reduce size of matrices by slicing
STEP = 80
Vr = V[:, :, ::STEP]
Wr = W[:, :, ::STEP]

#Max and Min values for plotting
V_max = V.max().__ceil__()
V_min = V.min().__floor__()
W_max = W.max().__ceil__()
W_min = W.min().__floor__()

#Calc. vector field
Vgrid, Wgrid = np.meshgrid(np.arange(V_min, V_max + 1, 0.1), np.arange(W_min, W_max + 1, 0.1))
dVgrid, dWgrid = dvdt_dwdt(Vgrid, Wgrid, 0)


#Plot results
len_T = len(str(Vr.shape[2]))  #the number of digits
for time in range(Vr.shape[2]):
    fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.8))
    heatmapV = axs[0].pcolor(Vr[:, :, time], vmin=V_min, vmax=V_max, cmap='YlOrRd')
    axs[0].set_xlabel('$X$')
    axs[0].set_ylabel('$Y$')
    axs[0].set_title('Action Potential $V$')
    axs[0].text(71, 51, "*", size=30, color='magenta', horizontalalignment="center", verticalalignment="center")
    axs[0].text(71, 51, "    (71, 51)", size=10, color='magenta', horizontalalignment="left")
    fig.colorbar(heatmapV, ax=axs[0])
    axs[1].quiver(Vgrid[::2, ::2], Wgrid[::2, ::2], dVgrid[::2, ::2], dWgrid[::2, ::2], scale=800)
    axs[1].contour(Vgrid, Wgrid, dVgrid, levels=[0], colors="g")
    axs[1].contour(Vgrid, Wgrid, dWgrid, levels=[0], colors="r")
    axs[1].plot(10, 10, 'g-', label='$dV/dt = 0$')
    axs[1].plot(10, 10, 'r-', label = '$dW/dt = 0$')
    axs[1].plot(Vr[71, 51, time], Wr[71, 51, time], 'm*', markersize=20, label='$V$ at ($X$, $Y$) = (71, 51)')
    axs[1].set_xlabel('$V$')
    axs[1].set_ylabel('$W$')
    axs[1].set_title('Phase-plane')
    axs[1].set_xlim(V_min, V_max)
    axs[1].set_ylim(W_min, W_max)
    axs[1].legend(loc='upper right')
    s = str(time).zfill(len_T)
    fig.tight_layout()
    fig.savefig("GIF\\{}.png".format(s))   #Save image for each loop
    plt.clf()
    plt.cla()
    plt.close()
path = os.getcwd() + "\\GIF\\"
files = [path + str(t).zfill(len_T) + '.png' for t in range(Vr.shape[2])]
images = list(map(lambda file: Image.open(file), files))
images.pop(0).save(path + "4_8D.gif" ,save_all = True, append_images = images, duration = 10, optimize = False, loop = 0)
