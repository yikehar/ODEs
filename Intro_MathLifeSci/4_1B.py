"""
4.1B
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

#init.
Xmax, Tmax = 100, 100
dt, d, dx = 0.1, 1.0, 1.0
E = np.zeros((Xmax, Xmax, Tmax))
E[40:60, 40:60, 0] = 1.0    #Initial conc. distribution

starttime = time.time() #start timing

#Calc. conc. E(x, y, t)
for T in range(Tmax - 1):
    # Do not calc. at the boundaries
    for X in range(1, Xmax - 2):
        for Y in range(1, Xmax - 2):
            E[X, Y, T + 1] = \
            dt * (d * (E[Y, X - 1, T] + E[Y, X + 1, T] + E[Y -1, X, T] + E[Y + 1, X, T] \
            - 4 * E[X, Y, T])) / dx/dx + E[X, Y, T]

    #Calc. at the horizontal boundaries
    for X in range(1, Xmax - 2):
        Y = 0
        E[X, Y, T + 1] = \
        dt * (d * (E[Y, X - 1, T] + E[Y, X + 1, T] + E[Y + 1, X, T] \
        - 3 * E[X, Y, T])) / dx/dx + E[X, Y, T]
        Y = Xmax - 1
        E[X, Y, T + 1] = \
        dt * (d * (E[Y, X - 1, T] + E[Y, X + 1, T] + E[Y - 1, X, T] \
        - 3 * E[X, Y, T])) / dx/dx + E[X, Y, T]

    # Calc. at the vertical boundaries
    for Y in range(1, Xmax - 2):
        X = 0
        E[X, Y, T + 1] = \
        dt * (d * (E[Y, X + 1, T] + E[Y -1, X, T] + E[Y + 1, X, T] \
        - 3 * E[X, Y, T])) / dx/dx + E[X, Y, T]
        X = Xmax - 1
        E[X, Y, T + 1] = \
        dt * (d * (E[Y, X - 1, T] + E[Y -1, X, T] + E[Y + 1, X, T] \
        - 3 * E[X, Y, T])) / dx/dx + E[X, Y, T]

    # Calc. at the corners
    X, Y = 0, 0
    E[X, Y, T + 1] = dt * (d * (E[Y, X + 1, T] + E[Y + 1, X, T] - 2 * E[X, Y, T])) / dx/dx + E[X, Y, T]

    X, Y = 0, Xmax - 1
    E[X, Y, T + 1] = dt * (d * (E[Y, X + 1, T] + E[Y - 1, X, T] - 2 * E[X, Y, T])) / dx/dx + E[X, Y, T]

    X, Y = Xmax - 1, 0
    E[X, Y, T + 1] = dt * (d * (E[Y, X - 1, T] + E[Y + 1, X, T] - 2 * E[X, Y, T])) / dx/dx+ E[X, Y, T]

    X, Y = Xmax - 1, Xmax - 1
    E[X, Y, T + 1] = dt * (d * (E[Y, X - 1, T] + E[Y - 1, X, T] - 2 * E[X, Y, T])) / dx/dx + E[X, Y, T]

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

path = os.getcwd() + "\\GIF\\"
files = [path + str(t).zfill(len_T) + '.png' for t in range(Tmax)]
images = list(map(lambda file: Image.open(file), files))
images.pop(0).save(path + "4_1B.gif" ,save_all = True, append_images = images, duration = 100, optimize = False, loop = 0)