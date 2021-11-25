import math
from Distances import *


def citationKNN(X, test, k=5, dist=hausdorff_distance, internal_dist=distance.euclidean):
    distances = [math.inf] * k
    neighbour_classes = [None] * k
    for bag in X:
        # bag.iloc[:,1:-1] = alle rijen, en gebruik niet de kolommen met label, molecuulnaam en klasse
        test_distance = dist(bag.iloc[:, 1:-1], test.iloc[:, 1:-1], internal_dist=internal_dist)
        index = k - 1
        while test_distance < distances[index] and index >= 0:
            index -= 1
        if index + 1 < k:
            distances.insert(index + 1, test_distance)
            # neigbour.iloc[0,-1]: elke instance heeft zelfde 'class', neem dus gewoon de eerste rij (0) en vraag klasse (-1)
            neighbour_classes.insert(index + 1, bag.iloc[0, -1])
            distances = distances[:k]
            neighbour_classes = neighbour_classes[:k]
    return int(sum(neighbour_classes) / k >= 0.5)
