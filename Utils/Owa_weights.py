import numpy as np

"""
Collection of OWA_weights
"""


# Determines the orness of a fuction
def orness(weight):
    l = len(weight)
    test = l - np.arange(1, l + 1)
    return np.dot(weight, test) / (l - 1)


def strict(n):
    return np.append(np.ones(1), np.zeros(n - 1))


def additive(n):
    return 2 * np.flip(np.arange(1, n + 1)) / (n * (n + 1))


def exponential(n):
    return np.flip(2 ** np.arange(n) / (2 ** n - 1)) if n < 50 else np.cumprod(np.full(n, 0.5))


def invadd(n):
    temp_weight = 1 / np.arange(1, n + 1)
    return temp_weight / np.sum(temp_weight)


def mean(n):
    return np.ones(n) / n


def lower_weight(weight):
    return lambda n: np.flip(weight(n))


def trimmed(k, weight):
    return lambda n: np.append(weight(k), np.zeros(n - k))


if __name__ == '__main__':
    print(orness(trimmed(5, mean)(20)))
    print(orness(lower_weight(trimmed(5, mean))(20)))
