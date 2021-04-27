import numpy as np

#Use this function as a substitute for del2 in MATLAB
def Lap2D(U, dx):
    Xmax, Ymax = U.shape
    L = np.zeros(U.shape)

    # Do not calc. at the boundaries
    for X in range(1, Xmax - 2):
        for Y in range(1, Ymax - 2):
            L[Y, X] = (U[Y, X - 1] + U[Y, X + 1] + U[Y - 1, X] + U[Y + 1, X] - 4 * U[Y, X]) / dx / dx

    # Calc. at the horizontal boundaries
    for X in range(1, Xmax - 2):
        Y = 0
        L[Y, X] = (U[Y, X - 1] + U[Y, X + 1] + U[Y + 1, X] - 3 * U[Y, X]) / dx / dx
        Y = Ymax - 1
        L[Y, X] = (U[Y, X - 1] + U[Y, X + 1] + U[Y - 1, X] - 3 * U[Y, X]) / dx / dx

    # Calc. at the vertical boundaries
    for Y in range(1, Ymax - 2):
        X = 0
        L[Y, X] = (U[Y, X + 1] + U[Y - 1, X] + U[Y + 1, X] - 3 * U[Y, X]) / dx / dx
        X = Xmax - 1
        L[Y, X] = (U[Y, X - 1] + U[Y - 1, X] + U[Y + 1, X] - 3 * U[Y, X]) / dx / dx

    # Calc. at the corners
    X, Y = 0, 0
    L[Y, X] = (U[Y, X + 1] + U[Y + 1, X] - 2 * U[Y, X]) / dx / dx

    X, Y = 0, Ymax - 1
    L[Y, X] = (U[Y, X + 1] + U[Y - 1, X] - 2 * U[Y, X]) / dx / dx

    X, Y = Xmax - 1, 0
    L[Y, X] = (U[Y, X - 1] + U[Y + 1, X] - 2 * U[Y, X]) / dx / dx

    X, Y = Xmax - 1, Ymax - 1
    L[Y, X] = (U[Y, X - 1] + U[Y - 1, X] - 2 * U[Y, X]) / dx / dx

    return L / 4.

#Calc. 2D Laplacian using matrices
def Lap2DMt(Et, dx2):
    #dx2 = dx*dx
    Ymax, Xmax = Et.shape

    Er = np.zeros(Et.shape)
    El = np.zeros(Et.shape)
    Eu = np.zeros(Et.shape)
    Ed = np.zeros(Et.shape)

    Er[:, Xmax - 1] = Et[:, Xmax - 1]
    Er[:, :Xmax - 1] = Et[:, 1:]

    El[:, 0] = Et[:, 0]
    El[:, 1:] = Et[:, :Xmax - 1]

    Ed[Ymax - 1, :] = Et[Ymax - 1, :]
    Ed[:Ymax - 1, :] = Et[1:, :]

    Eu[0, :] = Et[0, :]
    Eu[1:, :] = Et[:Ymax - 1, :]

    return (Er + El + Eu + Ed - 4. * Et) / dx2
