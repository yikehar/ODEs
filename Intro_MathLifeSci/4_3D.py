"""
4.3D
Diffusion across neighboring cells
[    ][E1,2][    ]
[E0,1][E1,1][E2,1]
[    ][E1,0][    ]

PDEs:
d2/dx2 E1,1 = {E0,1 + E2,1 - 2 * E1,1} / dx2
d2/dy2 E1,1 = {E1,0 + E1,2 - 2 * E1,1} / dy2
(diffusion coefficient d = 1)

Under the assumption dx = dy, the system of PDEs for d = 1
can be rewritten as
{d2/dx2 + d2/dy2} E1,1 = {E0,1 + E1,0 + E1,2 + E2,1 - 4 * E1,1} / dx2

When d is any real number which is > 0,
{d2/dx2 + d2/dy2} E1,1 = d * {E0,1 + E1,0 + E1,2 + E2,1 - 4 * E1,1} / dx2
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import time
import discreteLaplacian

#init.
Xmax, Tmax = 100, 100
dt, d, dx = 0.1, 1.0, 1.0
dx2 = dx*dx
E = np.zeros((Xmax, Xmax, Tmax))
E[0:20, 40:60, 0] = np.random.randn(E[0:20, 40:60, 0].shape[0], E[0:20, 40:60, 0].shape[1])    #Initial conc. distribution

starttime = time.time() #start timing

#Solve PDE
for T in range(Tmax - 1):
    E[:, :, T + 1] = dt * d * discreteLaplacian.Lap2DMt(E[:, :, T], dx2) + E[:, :, T]

elapsedtime = time.time() - starttime
print('Time elapsed (sec): {}'.format(elapsedtime))

#Plot results
len_T = len(str(Tmax))  #the number of digits in Tmax
for time in range(Tmax):
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(E[:, :, time], vmin=0.0, vmax=1.0, cmap='YlOrRd')
    fig.colorbar(heatmap, ax=ax)
    s = str(time).zfill(len_T)
    fig.savefig("GIF\\{}.png".format(s))   #Save image for each loop
    plt.clf()
    plt.cla()
    plt.close()

path = os.getcwd() + "\\GIF\\"
files = [path + str(t).zfill(len_T) + '.png' for t in range(Tmax)]
images = list(map(lambda file: Image.open(file), files))
images.pop(0).save(path + "4_3D.gif" ,save_all = True, append_images = images, duration = 100, optimize = False, loop = 0)