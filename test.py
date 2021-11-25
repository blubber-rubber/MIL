from KEEL_DataReader import KEEL_Data
from CitationKNN import citationKNN
from Distances import *
import os
import time

for root in os.listdir('multiInstance'):
    average = []
    for i in range(1, 11):
        solutions = []
        training_bags = KEEL_Data(f'multiInstance/{root}/{root}-10-{i}tra.dat').get_bags()
        test_bags = KEEL_Data(f'multiInstance/{root}/{root}-10-{i}tst.dat').get_bags()
        dt = time.time()
        for test_bag in test_bags:
            solutions.append(citationKNN(training_bags, test_bag, dist=hausdorff_distance_avg2) == test_bag.iloc[0, -1])
        print(time.time() - dt)
        a = sum(solutions)
        b = len(solutions)
        print(f'{root} {i}: {a}/{b}={100*a / b}%')
        average.append(a / b)

    print(f'{root}: average_correct={100*sum(average) / len(average)}%')
    print('---------------------------')
