"""
4.6A
PDEs:
d/dt B = d * (Laplacian) * B - k * B + cb
d/dt W = d * (Laplacian) * B - k * W + cw
d/dt D = a * (B + R) * W - k * D

B -- morphogen Dpp, Drosophila homolog of vertebrate BMP4
W -- morphogen Wg (Wingless)
D -- Dll gene, regulated by both Dpp and Wg
R -- activated Dpp receptor, artificially introduced
d * (Laplacian) *  -- diffusion
- k * -- breakdown
cb -- production of Dpp
cw -- production of Wg
a * B * W -- production of Dll
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import time
import discreteLaplacian

#init.
Xmax, Tmax = 100, 500
dt, dx = 0.1, 1.
d, k, a = 1., 0.1, 1.
dx2 = dx*dx

B = np.zeros((Xmax, Xmax, Tmax))
W = np.zeros((Xmax, Xmax, Tmax))
D = np.zeros((Xmax, Xmax, Tmax))
cb = np.zeros((Xmax, Xmax))
cw = np.zeros((Xmax, Xmax))
R = np.zeros((Xmax, Xmax))

#Initial conc. distributions
cb[50:99, 40:59] = 0.5
cw[0:49, 40:59] = 0.5

#Mosaic expression of the activated Dpp receptor
R[30:35, 2:7] = np.ones(R[30:35, 2:7].shape)
R[30:35, 22:27] = np.ones(R[30:35, 22:27].shape)
R[30:35, 42:47] = np.ones(R[30:35, 42:47].shape)
R[30:35, 62:67] = np.ones(R[30:35, 62:67].shape)

starttime = time.time() #start timing

#Solve PDEs
for T in range(Tmax - 1):
    B[:, :, T + 1] = dt * (d * discreteLaplacian.Lap2DMt(B[:, :, T], dx2) - k * B[:, :, T] + cb) + B[:, :, T]
    W[:, :, T + 1] = dt * (d * discreteLaplacian.Lap2DMt(W[:, :, T], dx2) - k * W[:, :, T] + cw) + W[:, :, T]
    D[:, :, T + 1] = dt * (a * (B[:, :, T] + R) * W[:, :, T] - k * D[:, :, T]) + D[:, :, T]

elapsedtime = time.time() - starttime
print('Time elapsed (sec): {}'.format(elapsedtime))

#Plot results
len_T = len(str(Tmax))  #the number of digits in Tmax
for time in range(Tmax):
    fig, axs = plt.subplots(2, 2)
    heatmapB = axs[0, 0].pcolor(B[:, :, time], vmin=0.0, vmax=1.0, cmap='YlOrRd')
    fig.colorbar(heatmapB, ax=axs[0, 0])
    heatmapW = axs[0, 1].pcolor(W[:, :, time], vmin=0.0, vmax=1.0, cmap='YlOrRd')
    fig.colorbar(heatmapW, ax=axs[0, 1])
    heatmapD = axs[1, 0].pcolor(D[:, :, time], vmin=0.0, vmax=1.0, cmap='YlOrRd')
    fig.colorbar(heatmapD, ax=axs[1, 0])
    heatmapR = axs[1, 1].pcolor(R, vmin=0.0, vmax=1.0, cmap='YlOrRd')
    fig.colorbar(heatmapR, ax=axs[1, 1])
    s = str(time).zfill(len_T)
    fig.tight_layout()
    fig.savefig("GIF\\{}.png".format(s))   #Save image for each loop
    plt.clf()
    plt.cla()
    plt.close()

path = os.getcwd() + "\\GIF\\"
files = [path + str(t).zfill(len_T) + '.png' for t in range(Tmax)]
images = list(map(lambda file: Image.open(file), files))
images.pop(0).save(path + "4_6B.gif" ,save_all = True, append_images = images, duration = 100, optimize = False, loop = 0)