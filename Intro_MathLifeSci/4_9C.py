"""
4.9C
Proneural wave
Four-variable model, EGF mutant
A -- AS-C expression level, degree of differentiation
E -- EGF signal
D -- Delta expression level
N -- Notch signal

A = 0; undifferentiated
A = 1; completely differentiated

PDEs:
(4.22) dA / dt = ea * (1 - A) * max{E-N, 0}  <- irreversible process
(4.19) dE / dt = de * (Laplacian) * E - ke * E + ae * A * (1 - A)
(4.20) dD / dt = -kd * D + ad * A * (1 - A)
(4.21) dN / dt = -kn * N + dn * sum(D(l, m)) - dc * D(i, j)

sum(D(l, m)) -- sum of D in neighboring cells

Coefficients:
ea -- production of A by E
de -- diffusion of E
ke -- degradation of E
ae -- production of E by A
kd -- degradation of D
ad -- production of D by A
kn -- degradation of N
dn -- trans-activation of N by D in neighboring cells
dc -- cis-inhibition of N by D in the same cell
"""
import numpy as np
import matplotlib.pyplot as plt
import discreteLaplacian
import os
from PIL import Image
from scipy.integrate import odeint

#Function that returns dA/dt, dE/dt, dD/dt, dN/dt
def vec_dt(A, E, D, N):
    #Constants
    de, ae, ke, ea = 1., 1., 1., 10.
    kn, dn, dc, kd, ad = 1., 0.25, 0.25, 1., 1.

    #sum of D in neighboring cells
    D_upper = np.roll(D, 1, axis=0)
    D_upper[0,:] = 0
    D_lower = np.roll(D, -1, axis=0)
    D_lower[-1, :] = 0
    D_left = np.roll(D, 1, axis=1)
    D_left[0, :] = 0
    D_right = np.roll(D, -1, axis=1)
    D_right[-1, :] = 0
    D_nei = D_upper + D_lower + D_left + D_right

    #EGF mutant
    Emut = np.ones(E.shape)
    Emut[5:20, 5:20] = 0

    dx = 1. #val of dx = dy for calculation of laplacian

    # Calc. max{E - N, 0} by clip(min, max)
    return ea * (1. - A) * (E - N).clip(0, None), \
           de * discreteLaplacian.Lap2DMt(E, dx**2) - ke * E + ae * A * (1. - A) * Emut, \
           -kd * D + ad * A * (1. - A),\
           -kn * N + dn * D_nei - dc * D
            #dA/dt, dE/dt, dD/dt, dN/dt

#Model
def fourvar_proneural(vec, t):
    # reshape A, E, D, N
    n = int(len(vec) / 4)
    x_max = int(np.sqrt(n))
    a1, e1 = vec[:n].reshape(x_max, x_max), vec[n:2*n].reshape(x_max, x_max)
    d1, n1 = vec[2*n:3*n].reshape(x_max, x_max), vec[3*n:].reshape(x_max, x_max)
    #PDEs
    dAdt, dEdt, dDdt, dNdt = vec_dt(a1, e1, d1, n1)
    return np.hstack((dAdt.ravel(), dEdt.ravel(), dDdt.ravel(), dNdt.ravel()))

#parameters
Tmax, Xmax = 200, 25
dt = 0.1
t = np.arange(0, Tmax*dt, dt)

#Initial values
E0 = np.zeros(Xmax**2)
D0 = np.zeros(Xmax ** 2)
N0 = np.zeros(Xmax**2)
A0 = np.zeros((Xmax, Xmax))
A0[:, 0] = 0.5
vec_ini = np.hstack((A0.ravel(), E0, D0, N0))

#Solve PDEs
vec_out = odeint(fourvar_proneural,vec_ini, t)

#Reshape A, E, D, N
A = vec_out[:,:Xmax**2].T.reshape(Xmax, Xmax, Tmax)
E = vec_out[:,Xmax**2:2*Xmax**2].T.reshape(Xmax, Xmax, Tmax)
D = vec_out[:,2*Xmax**2:3*Xmax**2].T.reshape(Xmax, Xmax, Tmax)
N = vec_out[:,3*Xmax**2:].T.reshape(Xmax, Xmax, Tmax)

#Reduce size of matrices by slicing
STEP = 4
Ar = A[:, :, ::STEP]
Er = E[:, :, ::STEP]
Dr = A[:, :, ::STEP]
Nr = E[:, :, ::STEP]

#Max and Min values for plotting
A_max = A.max().__ceil__()
A_min = 0.
E_max = E.max().__ceil__()
E_min = 0.
D_max = D.max().__ceil__()
D_min = 0.
N_max = N.max().__ceil__()
N_min = 0.

#Plot results
len_T = len(str(Ar.shape[2]))  #the number of digits
for time in range(Ar.shape[2]):
    fig, axs = plt.subplots(2, 2, figsize=(6.4, 4.8))
    heatmapA = axs[0][0].pcolor(Ar[:, :, time], vmin=A_min, vmax=A_max, cmap='YlOrRd')
    fig.colorbar(heatmapA, ax=axs[0][0])
    axs[0][0].set_title('AS-C')
    heatmapE = axs[0][1].pcolor(Er[:, :, time], vmin=E_min, vmax=E_max, cmap='YlOrRd')
    fig.colorbar(heatmapE, ax=axs[0][1])
    axs[0][1].set_title('EGF')
    heatmapD = axs[1][0].pcolor(Dr[:, :, time], vmin=D_min, vmax=D_max, cmap='YlOrRd')
    fig.colorbar(heatmapD, ax=axs[1][0])
    axs[1][0].set_title('Delta')
    heatmapN = axs[1][1].pcolor(Nr[:, :, time], vmin=N_min, vmax=N_max, cmap='YlOrRd')
    fig.colorbar(heatmapN, ax=axs[1][1])
    axs[1][1].set_title('Notch')
    s = str(time).zfill(len_T)
    fig.tight_layout()
    fig.savefig("GIF\\{}.png".format(s))   #Save image for each loop
    plt.clf()
    plt.cla()
    plt.close()
path = os.getcwd() + "\\GIF\\"
files = [path + str(t).zfill(len_T) + '.png' for t in range(Ar.shape[2])]
images = list(map(lambda file: Image.open(file), files))
images.pop(0).save(path + "4_9C.gif" ,save_all = True, append_images = images, duration = 10, optimize = False, loop = 0)
