from scipy.spatial import distance
import math
import numpy as np
from Utils import Owa, Owa_weights
from scipy.optimize import linear_sum_assignment
import networkx as nx


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


# surjection afstand
# A en B zijn 2 bags met enkel de waarden van de instances
def surjection_distance(A, B, internal_dist=distance.euclidean):
    if len(A) < len(B):
        A, B = B, A
    distances = distance.cdist(A, B, metric=internal_dist)
    cost_matrix = np.concatenate(
        (distances, np.tile(np.min(distances, axis=1), (distances.shape[0] - distances.shape[1], 1)).transpose()),
        axis=1)
    matching = linear_sum_assignment(cost_matrix)
    return cost_matrix[matching[0], matching[1]].sum()


# fair surjection afstand
# A en B zijn 2 bags met enkel de waarden van de instances
def fair_surjection_distance(A, B, internal_dist=distance.euclidean, precision=10):
    if len(A) < len(B):
        A, B = B, A
    distances = distance.cdist(A, B, metric=internal_dist)

    inf = int(round(2 * distances.sum() * 10 ** precision))
    if inf == 0:
        return 0
    d1, d2 = distances.shape
    c0, s, t = math.ceil(d1 / d2), d1 + d2 + 1, d1 + d2 + 2
    G = nx.DiGraph()
    edges = [(s, i + 1, {"capacity": 1, "weight": 0}) for i in range(d1)]
    edges += [(s, i + 1, {"capacity": 1, "weight": inf}) for i in range(d1, d1 + d2)]
    edges += [(i + 1, t, {"capacity": c0, "weight": 0}) for i in range(d1, d1 + d2)]
    for i in range(d1):
        for j in range(d2):
            edges.append((i + 1, d1 + j + 1, {"capacity": 1, "weight": int(round(distances[i, j] * 10 ** precision))}))

    G.add_edges_from(edges)
    mincostFlow = nx.max_flow_min_cost(G, s, t)
    mincost = (nx.cost_of_flow(G, mincostFlow) % inf) / (10 ** precision)
    return mincost


# link afstand
# A en B zijn 2 bags met enkel de waarden van de instances
def link_distance(A, B, internal_dist=distance.euclidean):
    distances = distance.cdist(A, B, metric=internal_dist)
    d1, d2 = distances.shape[0], distances.shape[1]

    inf = np.sum(distances) * 2
    down_left = np.full((d2, d2), inf)
    np.fill_diagonal(down_left, np.min(distances, axis=0))
    upper_right = np.full((d1, d1), inf)
    np.fill_diagonal(upper_right, np.min(distances, axis=1))

    cost_matrix = np.zeros((d1 + d2, d1 + d2))
    cost_matrix[:d1, :d2] = distances
    cost_matrix[d1:, :d2] = down_left
    cost_matrix[:d1, d2:] = upper_right
    matching = linear_sum_assignment(cost_matrix)
    return cost_matrix[matching[0], matching[1]].sum()

def norm_surjection_distance(A, B, internal_dist=distance.euclidean):
    if len(A) < len(B):
        A, B = B, A
    distances = distance.cdist(A, B, metric=internal_dist)
    cost_matrix = np.concatenate(
        (distances, np.tile(np.min(distances, axis=1), (distances.shape[0] - distances.shape[1], 1)).transpose()),
        axis=1)
    matching = linear_sum_assignment(cost_matrix)
    return cost_matrix[matching[0], matching[1]].sum() / sum(distances.shape)


def norm_fair_surjection_distance(A, B, internal_dist=distance.euclidean, precision=10):
    if len(A) < len(B):
        A, B = B, A
    distances = distance.cdist(A, B, metric=internal_dist)

    inf = int(round(2 * distances.sum() * 10 ** precision))
    if inf == 0:
        return 0
    d1, d2 = distances.shape
    c0, s, t = math.ceil(d1 / d2), d1 + d2 + 1, d1 + d2 + 2
    G = nx.DiGraph()
    edges = [(s, i + 1, {"capacity": 1, "weight": 0}) for i in range(d1)]
    edges += [(s, i + 1, {"capacity": 1, "weight": inf}) for i in range(d1, d1 + d2)]
    edges += [(i + 1, t, {"capacity": c0, "weight": 0}) for i in range(d1, d1 + d2)]
    for i in range(d1):
        for j in range(d2):
            edges.append((i + 1, d1 + j + 1, {"capacity": 1, "weight": int(round(distances[i, j] * 10 ** precision))}))

    G.add_edges_from(edges)
    mincostFlow = nx.max_flow_min_cost(G, s, t)
    mincost = (nx.cost_of_flow(G, mincostFlow) % inf) / (10 ** precision)
    return mincost / sum(distances.shape)


def norm_link_distance(A, B, internal_dist=distance.euclidean):
    distances = distance.cdist(A, B, metric=internal_dist)
    d1, d2 = distances.shape[0], distances.shape[1]

    inf = np.sum(distances) * 2
    down_left = np.full((d2, d2), inf)
    np.fill_diagonal(down_left, np.min(distances, axis=0))
    upper_right = np.full((d1, d1), inf)
    np.fill_diagonal(upper_right, np.min(distances, axis=1))

    cost_matrix = np.zeros((d1 + d2, d1 + d2))
    cost_matrix[:d1, :d2] = distances
    cost_matrix[d1:, :d2] = down_left
    cost_matrix[:d1, d2:] = upper_right
    matching = linear_sum_assignment(cost_matrix)
    return cost_matrix[matching[0], matching[1]].sum() / sum(distances.shape)

def encoding_distance(A, B, internal_dist=distance.euclidean, encoding=None):
    return internal_dist(encoding[A.iloc[0, 0]], encoding[B.iloc[0, 0]])

if __name__ == '__main__':
    from KEEL_DataReader import *

    bags = KEEL_Data('../Artificial_data/data1.dat').get_bags()
    for bag1 in bags:
        for bag2 in bags:
            dist1 = surjection_distance(bag1.iloc[:, 1:-1], bag2.iloc[:, 1:-1])
            dist2 = link_distance(bag1.iloc[:, 1:-1], bag2.iloc[:, 1:-1])
            dist3 = fair_surjection_distance(bag1.iloc[:, 1:-1], bag2.iloc[:, 1:-1])
            print(dist1, dist2, dist3)
