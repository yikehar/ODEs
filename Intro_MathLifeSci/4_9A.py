"""
4.9A
Proneural wave
Two-variable model
A -- AS-C expression level, degree of differentiation
E -- EGF signal

A = 0; undifferentiated
A = 1; completely differentiated

PDEs:
(4.18) dA / dt = ea * (1 - A) * E
(4.19) dE / dt = de * (Laplacian) * E - ke * E + ae * A * (1 - A)

Coefficients:
ea -- production of A
de -- diffusion of E
ke -- degradation of E
ae -- production of E
"""
import numpy as np
import matplotlib.pyplot as plt
import discreteLaplacian
import os
from PIL import Image
from scipy.integrate import odeint

#Function that returns dA/dt, dE/dt
def vec_dt(A, E):
    #Constants
    de, ae, ke, ea = 1., 1., 1., 10.
    dx = 1.
    dx2 = dx * dx
    return ea * (1. - A) * E, de * discreteLaplacian.Lap2DMt(E, dx2) - ke * E + ae * A * (1. - A)

#Model
def twovar_proneural(vec, t):
    # reshape A and E
    n = int(len(vec) / 2)
    x_max = int(np.sqrt(n))
    a1, e1 = vec[:n].reshape(x_max, x_max), vec[n:].reshape(x_max, x_max)

    #PDEs
    dAdt, dEdt = vec_dt(a1, e1)
    return np.hstack((dAdt.ravel(), dEdt.ravel()))

#parameters
Tmax, Xmax = 200, 25
dt = 0.1
t = np.arange(0, Tmax*dt, dt)

#Initial values
E0 = np.zeros(Xmax**2)
A0 = np.zeros((Xmax, Xmax))
A0[:, 0] = 0.5
vec_ini = np.hstack((A0.ravel(), E0))

#Solve PDEs
vec_out = odeint(twovar_proneural,vec_ini, t)

#Reshape A and E
A = vec_out[:,:Xmax**2].T.reshape(Xmax, Xmax, Tmax)
E = vec_out[:,Xmax**2:].T.reshape(Xmax, Xmax, Tmax)

#Reduce size of matrices by slicing
STEP = 4
Ar = A[:, :, ::STEP]
Er = E[:, :, ::STEP]

#Max and Min values for plotting
A_max = A.max().__ceil__()
A_min = A.min().__floor__()
E_max = E.max().__ceil__()
E_min = E.min().__floor__()

#Plot results
len_T = len(str(Ar.shape[2]))  #the number of digits
for time in range(Ar.shape[2]):
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.4))
    heatmapA = axs[0].pcolor(Ar[:, :, time], vmin=A_min, vmax=A_max, cmap='YlOrRd')
    fig.colorbar(heatmapA, ax=axs[0])
    axs[0].set_title('AS-C')
    heatmapE = axs[1].pcolor(Er[:, :, time], vmin=E_min, vmax=E_max, cmap='YlOrRd')
    fig.colorbar(heatmapE, ax=axs[1])
    axs[1].set_title('EGF')
    s = str(time).zfill(len_T)
    fig.tight_layout()
    fig.savefig("GIF\\{}.png".format(s))   #Save image for each loop
    plt.clf()
    plt.cla()
    plt.close()
path = os.getcwd() + "\\GIF\\"
files = [path + str(t).zfill(len_T) + '.png' for t in range(Ar.shape[2])]
images = list(map(lambda file: Image.open(file), files))
images.pop(0).save(path + "4_9A.gif" ,save_all = True, append_images = images, duration = 10, optimize = False, loop = 0)

