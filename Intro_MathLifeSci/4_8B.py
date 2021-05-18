"""
4.8B
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

#Initialize params
Tmax, dt = 5000, 0.02
Xmax, dx = 100, 1.
dx2 = dx*dx
V = np.zeros((Xmax, Xmax, Tmax))
W = np.zeros((Xmax, Xmax, Tmax))
I = np.zeros((Xmax, Xmax))
I[49:51, 49:51] = 1.
d, a, b, c = 1., 0.7, 0.8, 10.

#Solve ODEs
for T in range(Tmax-1):
    V[:, :, T + 1] = V[:, :, T] + dt * (c * (-V[:, :, T]**3 / 3. + V[:, :, T] - W[:, :, T] + I) + d * discreteLaplacian.Lap2DMt(V[:, :, T], dx2))
    W[:, :, T + 1] = W[:, :, T] + dt * (V[:, :, T] - b * W[:, :, T] + a) / c

#Reduce size of matrices by slicing
STEP = 100
Vr = V[:, :, ::STEP]
Wr = W[:, :, ::STEP]

#Plot results
len_T = len(str(Vr.shape[2]))  #the number of digits
for time in range(Vr.shape[2]):
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.4))
    heatmapV = axs[0].pcolor(Vr[:, :, time], vmin=0.0, vmax=np.ceil(V.max()), cmap='YlOrRd')
    fig.colorbar(heatmapV, ax=axs[0])
    heatmapW = axs[1].pcolor(Wr[:, :, time], vmin=0.0, vmax=np.ceil(W.max()), cmap='YlOrRd')
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
images.pop(0).save(path + "4_8B.gif" ,save_all = True, append_images = images, duration = 10, optimize = False, loop = 0)