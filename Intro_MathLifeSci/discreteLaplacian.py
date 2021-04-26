"""
Use this function as a substitute for del2 in MATLAB
"""
import numpy as np

def Lap2D(U, dx):
    Xmax, Ymax = U.shape
    L = np.zeros(U.shape)

    # Do not calc. at the boundaries
    for X in range(1, Xmax - 2):
        for Y in range(1, Ymax - 2):
            L[X, Y] = (U[Y, X - 1] + U[Y, X + 1] + U[Y - 1, X] + U[Y + 1, X] - 4 * U[X, Y]) / dx / dx

    # Calc. at the horizontal boundaries
    for X in range(1, Xmax - 2):
        Y = 0
        L[X, Y] = (U[Y, X - 1] + U[Y, X + 1] + U[Y + 1, X] - 3 * U[X, Y]) / dx / dx
        Y = Ymax - 1
        L[X, Y] = (U[Y, X - 1] + U[Y, X + 1] + U[Y - 1, X] - 3 * U[X, Y]) / dx / dx

    # Calc. at the vertical boundaries
    for Y in range(1, Ymax - 2):
        X = 0
        L[X, Y] = (U[Y, X + 1] + U[Y - 1, X] + U[Y + 1, X] - 3 * U[X, Y]) / dx / dx
        X = Xmax - 1
        L[X, Y] = (U[Y, X - 1] + U[Y - 1, X] + U[Y + 1, X] - 3 * U[X, Y]) / dx / dx

    # Calc. at the corners
    X, Y = 0, 0
    L[X, Y] = (U[Y, X + 1] + U[Y + 1, X] - 2 * U[X, Y]) / dx / dx

    X, Y = 0, Ymax - 1
    L[X, Y] = (U[Y, X + 1] + U[Y - 1, X] - 2 * U[X, Y]) / dx / dx

    X, Y = Xmax - 1, 0
    L[X, Y] = (U[Y, X - 1] + U[Y + 1, X] - 2 * U[X, Y]) / dx / dx

    X, Y = Xmax - 1, Ymax - 1
    L[X, Y] = (U[Y, X - 1] + U[Y - 1, X] - 2 * U[X, Y]) / dx / dx

    return L / 4.