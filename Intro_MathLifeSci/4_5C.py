"""
4.5B-C

PDEs:
d/dt B = d * (Laplacian) * B - k * B + c
d/dt S = a * B - k * S

B -- Dpp, Drosophila homolog of vertebrate BMP4
d * (Laplacian) * B -- diffusion of Dpp
- k * B -- reaction that consumes Dpp
c -- production of Dpp
S -- sal protein
a * B -- production rate of sal protein
- k * S -- degradation of sal protein

Dpp morphogen gradient determines pattering in the Drosophila wings
sal gene is the target of Dpp
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

S = np.zeros((Xmax, Xmax, Tmax))
B = np.zeros((Xmax, Xmax, Tmax))
c = np.zeros((Xmax, Xmax))
#Initial conc. distribution of Dpp
c[:, 47:53] = 0.5

starttime = time.time() #start timing

#Solve. PDEs
for T in range(Tmax - 1):
    B[:, :, T + 1] = dt * (d * discreteLaplacian.Lap2DMt(B[:, :, T], dx2) - k * B[:, :, T] + c) + B[:, :, T]
    S[:, :, T + 1] = dt * (a * B[:, :, T] - k * S[:, :, T]) + S[:, :, T]

elapsedtime = time.time() - starttime
print('Time elapsed (sec): {}'.format(elapsedtime))

#Plot results
len_T = len(str(Tmax))  #the number of digits in Tmax
for time in range(Tmax):
    fig, axs = plt.subplots(1, 2)
    heatmapB = axs[0].pcolor(B[:, :, time], vmin=0.0, vmax=1.0, cmap='YlOrRd')
    fig.colorbar(heatmapB, ax=axs[0])
    heatmapS = axs[1].pcolor(S[:, :, time], vmin=0.0, vmax=1.0, cmap='YlOrRd')
    fig.colorbar(heatmapS, ax=axs[1])
    s = str(time).zfill(len_T)
    fig.savefig("GIF\\{}.png".format(s))   #Save image for each loop
    plt.clf()
    plt.cla()
    plt.close()

path = os.getcwd() + "\\GIF\\"
files = [path + str(t).zfill(len_T) + '.png' for t in range(Tmax)]
images = list(map(lambda file: Image.open(file), files))
images.pop(0).save(path + "4_5C.gif" ,save_all = True, append_images = images, duration = 100, optimize = False, loop = 0)
