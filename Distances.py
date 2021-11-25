from scipy.spatial import distance
import math
import numpy as np
from Utils import Owa, Owa_weights


# min Haussdorff afstand
# A en B zijn 2 bags met enkel de waarden van de instances
def hausdorff_distance_min(A, B, internal_dist=distance.euclidean, weight=Owa_weights.strict):
    return Owa.OWA(np.ravel(distance.cdist(A, B, metric=internal_dist)), Owa_weights.lower_weight(weight))


# avg Hausdorff afstand
# A en B zijn 2 bags met enkel de waarden van de instances
def hausdorff_distance_avg(A, B, internal_dist=distance.euclidean, weight=Owa_weights.strict):
    distances = distance.cdist(A, B, metric=internal_dist)
    sum_b = np.sum(
        np.apply_along_axis(lambda row: Owa.OWA(row, Owa_weights.lower_weight(weight)), axis=0, arr=distances))
    sum_a = np.sum(
        np.apply_along_axis(lambda row: Owa.OWA(row, Owa_weights.lower_weight(weight)), axis=1, arr=distances))
    return (sum_a + sum_b) / np.sum(distances.shape)


# max Haussdorf afstand
def hausdorff_distance(A, B, internal_dist=distance.euclidean, weight=Owa_weights.strict):
    distances = distance.cdist(A, B, metric=internal_dist)
    hAB = Owa.OWA(
        np.apply_along_axis(lambda row: Owa.OWA(row, Owa_weights.lower_weight(weight)), axis=0, arr=distances), weight)
    hBA = Owa.OWA(
        np.apply_along_axis(lambda row: Owa.OWA(row, Owa_weights.lower_weight(weight)), axis=1, arr=distances), weight)
    return max(hAB, hBA)


def sum_of_min_distance(A, B, internal_dist=distance.euclidean, weight=Owa_weights.strict):
    distances = distance.cdist(A, B, metric=internal_dist)
    sum_b = np.sum(
        np.apply_along_axis(lambda row: Owa.OWA(row, Owa_weights.lower_weight(weight)), axis=0, arr=distances))
    sum_a = np.sum(
        np.apply_along_axis(lambda row: Owa.OWA(row, Owa_weights.lower_weight(weight)), axis=1, arr=distances))
    return (sum_a + sum_b) / 2


def sum_of_min_distance2(A, B, internal_dist=distance.euclidean):
    distances = distance.cdist(A, B, metric=internal_dist)
    sum_b = np.sum(np.min(distances, axis=0))
    sum_a = np.sum(np.min(distances, axis=1))
    return (sum_a + sum_b) / 2


if __name__ == '__main__':
    from KEEL_DataReader import *

    bags = KEEL_Data('Artificial_data/data1.dat').get_bags()
    for bag1 in bags:
        for bag2 in bags:
            d1 = sum_of_min_distance(bag1.iloc[:, 1:-1], bag2.iloc[:, 1:-1])
            d2 = sum_of_min_distance2(bag1.iloc[:, 1:-1], bag2.iloc[:, 1:-1])
            if d1 != d2:
                print('oh shit')
