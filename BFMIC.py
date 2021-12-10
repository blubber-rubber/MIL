import numpy as np
from Utils import Owa, Owa_weights, Distances

def BFMIC(X, y, classes, bag_relation, weight):
    best_class = None
    best_score = 0
    for cl in classes:
        test = filter(lambda x: x.iloc[0, -1] == cl, X)
        vector = np.array([bag_relation(x.iloc[:, 1:-1], y.iloc[:, 1:-1]) for x in test])
        score = Owa.OWA(vector, weight)
        if score >= best_score:
            best_score = score
            best_class = cl
    return best_class


if __name__ == '__main__':
    import os
    from KEEL_DataReader import *
    from CitationKNN import *
    from Utils.Distances import *
    import time

    for root in os.listdir('multiInstance'):
        average = []
        for i in range(1, 11):
            solutions = []
            training_bags = KEEL_Data(f'multiInstance/{root}/{root}-10-{i}tra.dat').get_bags()
            test_bags = KEEL_Data(f'multiInstance/{root}/{root}-10-{i}tst.dat').get_bags()
            dt = time.time()
            bag_dist = hausdorff_distance
            bag_relation = lambda x, y: 1 / (1 + (1 + bag_dist(x, y)))
            for test_bag in test_bags:
                solutions.append(
                    BFMIC(training_bags, test_bag, {0, 1}, bag_relation, Owa_weights.strict) == test_bag.iloc[0, -1])

            a = sum(solutions)
            b = len(solutions)
            print(f'{root} {i}: {a}/{b}={100 * a / b}%')
            average.append(a / b)

        print(f'{root}: average_correct={100 * sum(average) / len(average)}%')
        print('---------------------------')
