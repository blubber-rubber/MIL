import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
import math


class KEEL_Data:

    def __init__(self, filename=None):
        self.name = ""
        self.inputs = []
        self.attributes = []
        self.outputs = []
        self.data = None
        self.bag_labels = None
        if filename is not None:
            self.read(filename)

    def read(self, filename):

        file = open(filename)
        line = file.readline()
        self.name = line.lstrip("@relation ")
        line = file.readline()
        while not str.startswith(line, '@data'):
            line = line.rstrip('\n')
            if str.startswith(line, '@attribute'):
                line = line.lstrip('@attribute ').split(' ')[0]
                self.attributes.append(line)

            elif str.startswith(line, '@inputs'):
                line = line.lstrip('@inputs ')
                self.inputs.extend(line.rstrip(',').split(', '))

            elif str.startswith(line, '@outputs'):
                line = line.lstrip('@outputs ')
                self.outputs.extend(line.rstrip(',').split(', '))
            line = file.readline()
        temp_data = pd.DataFrame([line.rstrip('\n').split(", ") for line in file], columns=self.attributes)
        convert_dict = dict(
            [(output, int) for output in self.outputs] + [(inp, float) for inp in self.attributes[1:-1]])
        temp_data = temp_data.astype(convert_dict)
        self.data = temp_data

    """
    get_bags function returns list of all bags in the dataset
    """

    def get_bags(self):
        bagdata = self.data.groupby(self.attributes[0])
        self.bag_labels = bagdata.groups.keys()
        bags = [bagdata.get_group(label) for label in self.bag_labels]
        return bags

    """
    k_fold returns a generator object that can be used to stratisfied k-fold validation.
    Stratisfied means that the distribution of the class_labels is kept in the folds.


    Usage for k-fold cross-validation:
        data = KEEL_Data(f'multiInstance/eastWest/eastWest.dat')
        for training_bags, test_bags in data.k_fold(k):
            ...
    """

    def k_fold(self, k=10):
        bags = self.get_bags()
        X = [bag.iloc[:, :] for bag in bags]
        y = [bag.iloc[1, -1] for bag in bags]
        kf = StratifiedKFold(n_splits=k)
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
            yield (X_train, X_test)
