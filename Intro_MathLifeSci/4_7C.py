"""
4.7C
Turing model

PDEs:
d/dt A = da * (Laplacian) * A + aa * A + ia * I
d/dt I = di * (Laplacian) * I + ai * A + ii * I
(A and I have max and min values)

Params:
A -- activating factor
I -- inhibitory factor
da, di -- diffusion coefficients (> 0)
aa, ai -- activating coefficients (> 0)
ia, ii -- inhibitory coefficients (< 0)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import time
import discreteLaplacian

#init.
Xmax, Tmax = 100, 500
dt, dx = 0.01, 1.
da, di, aa, ia, ai, ii = 20., 0.1, 6., -30., 0.5, -2.
Amax, Imax = 2., 2. #Maximum values for A and I
dx2 = dx*dx

A = np.zeros((Xmax, Xmax, Tmax))
I = np.zeros((Xmax, Xmax, Tmax))

#Initial conc. distributions
A[47:52, 47:52, 0] = 1.

starttime = time.time() #start timing

#Solve PDEs
for T in range(Tmax - 1):
    A[:, :, T + 1] = dt * (da * discreteLaplacian.Lap2DMt(A[:, :, T], dx2) + aa * A[:, :, T] + ia * I[:, :, T]) + A[:, :, T]
    I[:, :, T + 1] = dt * (di * discreteLaplacian.Lap2DMt(I[:, :, T], dx2) + ai * A[:, :, T] + ii * I[:, :, T]) + I[:, :, T]
    #Apply maximum and minimum values
    A[:, :, T + 1] = A[:, :, T + 1] * ((A[:, :, T + 1] > 0.).astype(int) * (A[:, :, T + 1] < Amax).astype(int)) + Amax * (A[:, :, T + 1] >= Amax).astype(int)
    I[:, :, T + 1] = I[:, :, T + 1] * ((I[:, :, T + 1] > 0.).astype(int) * (I[:, :, T + 1] < Imax).astype(int)) + Imax * (I[:, :, T + 1] >= Imax).astype(int)

elapsedtime = time.time() - starttime
print('Time elapsed (sec): {}'.format(elapsedtime))

#Plot results
len_T = len(str(Tmax))  #the number of digits in Tmax
for time in range(Tmax):
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.4))
    heatmapB = axs[0].pcolor(A[:, :, time], vmin=0.0, vmax=2.0, cmap='YlOrRd')
    fig.colorbar(heatmapB, ax=axs[0])
    heatmapW = axs[1].pcolor(I[:, :, time], vmin=0.0, vmax=2.0, cmap='YlOrRd')
    fig.colorbar(heatmapW, ax=axs[1])
    s = str(time).zfill(len_T)
    fig.tight_layout()
    fig.savefig("GIF\\{}.png".format(s))   #Save image for each loop
    plt.clf()
    plt.cla()
    plt.close()

path = os.getcwd() + "\\GIF\\"
files = [path + str(t).zfill(len_T) + '.png' for t in range(Tmax)]
images = list(map(lambda file: Image.open(file), files))
images.pop(0).save(path + "4_7C.gif" ,save_all = True, append_images = images, duration = 100, optimize = False, loop = 0)
