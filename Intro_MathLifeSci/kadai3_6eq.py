import numpy as np


def model(L, T, a, b, c, d):
    dL = np.zeros(2)
    dL[0] = a * L[0] - b * L[0] * L[1]
    dL[1] = c * L[0] * L[1] - d * L[1]
    return dL