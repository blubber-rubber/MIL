import numpy as np

"""
Function to determine the OWA_aggregation of a vector using a  weight_vector
"""


def OWA(vector, weight):
    if callable(weight):
        weight = weight(len(vector))
    return np.dot(np.flip(np.sort(vector)), weight)
