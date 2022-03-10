from CitationKNN import citationKNN
from tqdm import tqdm
from Utils.Distances import *
import os
import time
from sklearn import metrics
from BFMIC import BFMIC
from KEEL_DataReader import *
import json


# function to add to JSON
def write_json(new_data, filename='result.json'):
    with open(filename, 'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data["results"].append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent=4)


def is_in_file(data, filename='result.json'):
    with open(filename, 'r') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        for result in file_data['results']:
            if all([result[key] == value for key, value in data.items()]):
                return True
        return False


file = 'results.json'
for root in os.listdir('multiInstance'):
    bag_distances = [(hausdorff_distance, 'H'),
                     (lambda A, B: hausdorff_distance(A, B, weight=Owa_weights.exponential), 'HExp'),
                     (lambda A, B: hausdorff_distance(A, B, weight=Owa_weights.invadd), 'HInvadd'),
                     (lambda A, B: hausdorff_distance(A, B, weight=Owa_weights.additive), 'HAdd'),

                     (hausdorff_distance_avg, 'AvgH'),
                     (lambda A, B: hausdorff_distance_avg(A, B, weight=Owa_weights.exponential), 'AvgHExp'),
                     (lambda A, B: hausdorff_distance_avg(A, B, weight=Owa_weights.invadd), 'AvgHInvadd'),
                     (lambda A, B: hausdorff_distance_avg(A, B, weight=Owa_weights.additive), 'AvgHAdd'),

                     (hausdorff_distance_min, 'MinH'),
                     (lambda A, B: hausdorff_distance_min(A, B, weight=Owa_weights.exponential), 'MinHExp'),
                     (lambda A, B: hausdorff_distance_min(A, B, weight=Owa_weights.invadd), 'MinHInvadd'),
                     (lambda A, B: hausdorff_distance_min(A, B, weight=Owa_weights.additive), 'MinHAdd'),

                     (sum_of_min_distance, 'SumMin'),
                     (lambda A, B: sum_of_min_distance(A, B, weight=Owa_weights.exponential), 'SumMinExp'),
                     (lambda A, B: sum_of_min_distance(A, B, weight=Owa_weights.invadd), 'SumMinInvadd'),
                     (lambda A, B: sum_of_min_distance(A, B, weight=Owa_weights.additive), 'SumMinAdd'),

                     (link_distance, 'link'),
                     (surjection_distance, 'surj'),
                     (fair_surjection_distance, 'fair_surj')
                     ]
    OWA_weight = [(Owa_weights.strict, 'strict'), (Owa_weights.additive, 'additive'),
                  (Owa_weights.invadd, 'inverse_additive'),
                  (Owa_weights.exponential, 'exp'), (Owa_weights.mean, 'average')]
    for OWA_w, owa_name in OWA_weight:
        for bag_dist, dist_name in bag_distances:
            print(root, dist_name, owa_name)
            if not is_in_file({'data': root, 'dist': dist_name, 'OWA': owa_name}, 'result.json'):
                predictions = []
                values = []
                bag_relation = lambda x, y: 1 / (1 + (1 + bag_dist(x, y)))
                for i in tqdm(range(1, 11)):
                    training_bags = KEEL_Data(f'multiInstance/{root}/{root}-10-{i}tra.dat').get_bags()
                    test_bags = KEEL_Data(f'multiInstance/{root}/{root}-10-{i}tst.dat').get_bags()
                    for test_bag in test_bags:
                        predictions.append(BFMIC(training_bags, test_bag, {0, 1}, bag_relation, OWA_w))
                        values.append(test_bag.iloc[0, -1])
                conf_matrix = metrics.confusion_matrix(predictions, values)
                result = {'data': root, 'dist': dist_name, 'OWA': owa_name, 'TP': int(conf_matrix[0, 0]),
                          'TN': int(conf_matrix[1, 1]),
                          'FP': int(conf_matrix[0, 1]), 'FN': int(conf_matrix[1, 0])}
                write_json(result)
