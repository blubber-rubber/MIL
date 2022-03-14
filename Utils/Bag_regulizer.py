import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy.stats import iqr
from random import sample
from sklearn.cluster import KMeans
from scipy.spatial import distance


def bag_regulizer(training, test, picker, undersampler, oversampler):
    size = picker(training)
    test_labels = [bag.iloc[0, 0] for bag in test]
    bags = training + test
    bags = undersampler(bags, size)
    bags = oversampler(bags, size)
    training = [bag for bag in bags if bag.iloc[0, 0] not in test_labels]
    test = [bag for bag in bags if bag.iloc[0, 0] in test_labels]

    return training, test


def nothing(bags, size):
    return bags


def random_oversampling(bags, size):
    for i, bag in enumerate(bags):
        new_bag = bag
        while new_bag.shape[0] < size:
            new_bag = new_bag.append(bag.iloc[np.random.randint(bag.shape[0]), :], ignore_index=True)
        bags[i] = new_bag
    return bags


def random_undersampling(bags, size):
    for i, bag in enumerate(bags):
        while bag.shape[0] > size:
            bag = bag.drop(bag.index[sample(range(bag.shape[0]), bag.shape[0] - size)], axis=0)
        bags[i] = bag
    return bags


def smote_oversampling(bags, size):
    bags = random_oversampling(bags, min(size, 6))
    if size > 6:
        sm = SMOTE(random_state=42, sampling_strategy='auto', k_neighbors=5)
        data = pd.concat(bags)
        X, y = data.iloc[:, 1:], data.iloc[:, 0]
        X_oversampled, y_oversampled = sm.fit_resample(X, y)
        bagdata = pd.DataFrame(X_oversampled, columns=X.columns)
        bagdata.insert(0, data.columns[0], y_oversampled)

        bagdata = bagdata.groupby(data.columns[0])
        bag_labels = bagdata.groups.keys()
        bags = [bagdata.get_group(label) for label in bag_labels]
    return bags


def median_picker(bags):
    bag_sizes = np.array([bag.shape[0] for bag in bags])
    return int(np.ceil(np.median(bag_sizes)))


def no_outlier_picker(bags):
    bag_sizes = np.array([bag.shape[0] for bag in bags])
    median = np.median(bag_sizes)
    i = iqr(bag_sizes)
    return int(np.max(bag_sizes[bag_sizes < median + 1.5 * i]))


def k_means_undersampling(bags, size):
    for i, bag in enumerate(bags):
        if bag.shape[0] > size:
            kmeans = KMeans(n_clusters=size, random_state=0).fit(bag.iloc[:, 1: -1])
            centers = kmeans.cluster_centers_
            new_bag = pd.DataFrame(columns=bag.columns)
            for center in centers:
                new_bag = new_bag.append(
                    bag.iloc[min(range(bag.shape[0]), key=lambda x: distance.euclidean(center, bag.iloc[x, 1:-1])), :],
                    ignore_index=True)
            bag = new_bag

        bags[i] = bag
    return bags


if __name__ == '__main__':
    from KEEL_DataReader import *

    i = 1
    root = 'musk2'
    training_bags = KEEL_Data(f'../multiInstance/{root}/{root}-10-{i}tra.dat').get_bags()
    test_bags = KEEL_Data(f'../multiInstance/{root}/{root}-10-{i}tst.dat').get_bags()
    print(bag_regulizer(training_bags, test_bags, no_outlier_picker, k_means_undersampling, smote_oversampling))
