from Utils import Owa_weights
from Utils.BagEncodings import *
from Utils.Distances import *


def citationKNN(X, test, k=5, dist=None, internal_dist=distance.euclidean, weight=None, encoding=None):
    if k >= len(X):
        return int(sum(bag.iloc[0, -1] for bag in X) / len(X) >= 0.5)
    distances = [math.inf] * k
    neighbour_classes = [None] * k
    for bag in X:
        if weight is not None:  # distances where weight should be passed on
            test_distance = dist(bag.iloc[:, 1:-1], test.iloc[:, 1:-1], internal_dist=internal_dist, weight=weight)
        elif encoding is not None:  # distances where encoding should be passed on
            test_distance = dist(bag.iloc[:, :-1], test.iloc[:, :-1], internal_dist=internal_dist, encoding=encoding)
        else:  # distances that do not use owa or encoding
            test_distance = dist(bag.iloc[:, 1:-1], test.iloc[:, 1:-1], internal_dist=internal_dist)
        index = k - 1
        while test_distance < distances[index] and index >= 0:
            index -= 1
        if index + 1 < k:
            distances.insert(index + 1, test_distance)
            neighbour_classes.insert(index + 1, bag.iloc[0, -1])
            distances = distances[:k]
            neighbour_classes = neighbour_classes[:k]
    return int(sum(neighbour_classes) / k >= 0.5)
