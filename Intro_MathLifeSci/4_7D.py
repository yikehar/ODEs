"""
4.7D-E
Turing model

PDEs:
d/dt A = da * (Laplacian) * A - ka * A + Ap,
Ap = c1 * A + c2 * I + c3, 0 <= Ap <= Apmax
d/dt I = di * (Laplacian) * I - ki * I + Ip
Ip = c4 * A + c5 * I + c6, 0 <= Ip <= Ipmax

Params:
A -- activating factor
I -- inhibitory factor
da, di -- diffusion coefficients (> 0)
ka, ki -- activating coefficients (> 0)
Ap, Ip -- production rate of A and I
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import time
import discreteLaplacian

#init.
Xmax, Tmax = 100, 5001
dx, dt = 1., 0.1
da, di, ka, ki, = 0.02, 0.5, 0.03, 0.06
c1, c2, c3 = 0.08, -0.08, 0.01   #c3 = 0.1 (default), 0.01 (spot), 0.2 (mesh)
c4, c5, c6 = 0.11, 0., -0.15
Apmax, Ipmax = 0.2, 0.5 #Maximum values for A and I
dx2 = dx*dx

A = np.zeros((Xmax, Xmax, Tmax))
I = np.zeros((Xmax, Xmax, Tmax))

#Uncomment one of the folloings to set initial conc. distribution of A
#A[45:50, 45:50, 0] = 1.    #uniform
#A[:, :, 0] = np.random.rand(Xmax, Xmax) #random
A[45:55,25:35,0], A[75:80,65:70,0], A[25:30,85:90,0] = 1., 1., 1.

starttime = time.time() #start timing

#Solve PDEs
for T in range(Tmax - 1):
    #Define Ap and Ip
    Ap = c1 * A[:, :, T] + c2 * I[:, :, T] + c3
    Ip = c4 * A[:, :, T] + c5 * I[:, :, T] + c6
    #Apply maximum and minimum values
    Ap = Ap.clip(0., Apmax)
    Ip = Ip.clip(0., Ipmax)
    #Solve PDEs
    A[:, :, T + 1] = dt * (da * discreteLaplacian.Lap2DMt(A[:, :, T], dx2) - ka * A[:, :, T] + Ap) + A[:, :, T]
    I[:, :, T + 1] = dt * (di * discreteLaplacian.Lap2DMt(I[:, :, T], dx2) - ki * I[:, :, T] + Ip) + I[:, :, T]


elapsedtime = time.time() - starttime
print('Time elapsed (sec): {}'.format(elapsedtime))
"""
#Plot results
len_T = len(str(Tmax))  #the number of digits in Tmax
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.4))
heatmapB = axs[0].pcolor(A[:, :, Tmax-1], vmin=0.0, vmax=np.ceil(A.max()), cmap='YlOrRd')
fig.colorbar(heatmapB, ax=axs[0])
heatmapW = axs[1].pcolor(I[:, :, Tmax-1], vmin=0.0, vmax=np.ceil(I.max()), cmap='YlOrRd')
fig.colorbar(heatmapW, ax=axs[1])
fig.tight_layout()
fig.savefig("4_7D_spot_rand.png")
"""


#Reduce size of matrices by slicing
STEP = 100
Ar = I[:, :, ::STEP]
Ir = I[:, :, ::STEP]
len_T = len(str(Ar.shape[2]))  #the number of digits in Tmax
print(Ar.shape[2])
#Plot results as animation
for time in range(Ar.shape[2]):
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.4))
    heatmapB = axs[0].pcolor(Ar[:, :, time], vmin=0.0, vmax=np.ceil(A.max()), cmap='YlOrRd')
    fig.colorbar(heatmapB, ax=axs[0])
    heatmapW = axs[1].pcolor(Ir[:, :, time], vmin=0.0, vmax=np.ceil(I.max()), cmap='YlOrRd')
    fig.colorbar(heatmapW, ax=axs[1])
    s = str(time).zfill(len_T)
    fig.tight_layout()
    fig.savefig("GIF\\{}.png".format(s))   #Save image for each loop
    plt.clf()
    plt.cla()
    plt.close()
path = os.getcwd() + "\\GIF\\"
files = [path + str(t).zfill(len_T) + '.png' for t in range(Ar.shape[2])]
images = list(map(lambda file: Image.open(file), files))
images.pop(0).save(path + "4_7D_spot.gif" ,save_all = True, append_images = images, duration = 10, optimize = False, loop = 0)
