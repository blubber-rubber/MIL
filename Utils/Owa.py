import numpy as np


def OWA(vector, weight):
    if callable(weight):
        weight = weight(len(vector))
    return np.dot(np.flip(np.sort(vector)), weight)
