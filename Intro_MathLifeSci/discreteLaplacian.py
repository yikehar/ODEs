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

#Calc. Laplacian using matrices
def Lap2DMt(U, dx):
    Xmax, Ymax = U.shape
    L = np.zeros(U.shape)

    Et = U
    Er = np.zeros(U.shape)
    El = np.zeros(U.shape)
    Eu = np.zeros(U.shape)
    Ed = np.zeros(U.shape)

    Er[:, Ymax] = Et[:, Ymax]
    Er[:, :Ymax - 1] = Et[:, 1:Ymax]
    El[:, 0] = Et[:, 0]
    El[:, 1:]
