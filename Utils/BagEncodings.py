import pandas as pd
from KEEL_DataReader import KEEL_Data
from scipy.spatial import distance
from sklearn import cluster
import numpy as np
from yellowbrick.cluster import KElbowVisualizer


def kmeansEncoding(X, test, seed=8, normalize=True):
    X_OG = pd.concat(X)  # data from before groupby

    # kmeans on positive instances
    # choose best k
    alg = KElbowVisualizer(cluster.KMeans(random_state=seed), k=(2, 21))
    alg.fit(X_OG[X_OG.iloc[:, -1] == 1].iloc[:, 1:-1])
    pos_k = alg.elbow_value_
    if pos_k is None:
        pos_k = 21
    alg_pos = cluster.KMeans(n_clusters=pos_k, random_state=seed)
    alg_pos.fit(X_OG[X_OG.iloc[:, -1] == 1].iloc[:, 1:-1])
    labels_pos = alg_pos.labels_.astype(np.int)

    # kmeans on negative instances
    alg = KElbowVisualizer(cluster.KMeans(random_state=seed), k=(2,21))
    alg.fit(X_OG[X_OG.iloc[:, -1] == 0].iloc[:, 1:-1])
    neg_k = alg.elbow_value_
    if neg_k is None:
        neg_k = 21
    alg_neg = cluster.KMeans(n_clusters=neg_k, random_state=seed)
    alg_neg.fit(X_OG[X_OG.iloc[:, -1] == 0].iloc[:, 1:-1])
    labels_neg = alg_neg.labels_.astype(np.int) + 3

    # construction of array clusterslabels for whole trainingset
    pos = 0
    neg = 0
    labels = [0] * X_OG.shape[0]
    for rijnr, waarde in enumerate(X_OG.iloc[:, -1]):
        if waarde == 1:
            labels[rijnr] = labels_pos[pos]
            pos = pos + 1
        else:
            labels[rijnr] = labels_neg[neg]
            neg = neg + 1

    # k-means-encoding
    index = 0
    repr_encoding = []
    for bag in X:
        toep = labels[index:index + len(bag)]
        repr_encoding.append((bag.iloc[0, 0], np.array(
            [toep.count(i) / (normalize * (len(bag) - 1) + 1) for i in range(pos_k + neg_k)])))
        index = index + len(bag)

    clusters = np.concatenate((alg_pos.cluster_centers_, alg_neg.cluster_centers_))
    for test_bag in test:
        label_bag = [np.argmin(
            [distance.euclidean(instance, center) for center in clusters]
        ) for index, instance in test_bag.iloc[:, 1:-1].iterrows()]
        repr_encoding.append((test_bag.iloc[0, 0],
                              np.array([label_bag.count(i) / (normalize * (len(test_bag) - 1) + 1) for i in
                                        range(pos_k + neg_k)])))

    return dict(repr_encoding)


# average = average(x), average(y), average(z), ...
def averageEncoding(X, test):
    repr_mean = dict(
        [(bag.iloc[0, 0], bag.iloc[:, 1:-1].mean()) for bag in X] + [(bag.iloc[0, 0], bag.iloc[:, 1:-1].mean()) for bag
                                                                     in test])
    return repr_mean


# center = median(x), median(y), median(z), ...
def centerEncoding(X, test):
    repr_mean = dict(
        [(bag.iloc[0, 0], bag.iloc[:, 1:-1].median()) for bag in X] + [(bag.iloc[0, 0], bag.iloc[:, 1:-1].median()) for
                                                                       bag
                                                                       in test])
    return repr_mean
