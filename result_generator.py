from CitationKNN import citationKNN
from tqdm import tqdm

from Utils import Owa_weights
import os
import time
from sklearn import metrics
from KEEL_DataReader import *
import json
from Utils.Distances import *
from scipy.spatial import distance
from itertools import product
from BFMIC import BFMIC


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


raw_bag_distances = [(hausdorff_distance, 'H', True), (hausdorff_distance_avg, 'AvgH', True),
                     (hausdorff_distance_min, 'MinH', True), (sum_of_min_distance, 'SumMin', True),
                     (link_distance, 'link', False), (surjection_distance, 'surj', False),
                     (fair_surjection_distance, 'fair_surj', False)]
internal_distances = [(distance.euclidean, 'euclidean'), (distance.cityblock, 'manhattan')]
internal_OWA = [(Owa_weights.strict, 'strict'), (Owa_weights.additive, 'additive'),
                (Owa_weights.invadd, 'inverse_additive'), (Owa_weights.exponential, 'exp'),
                (Owa_weights.mean, 'average')]
external_OWA = [(Owa_weights.strict, 'strict'), (Owa_weights.additive, 'additive'),
                (Owa_weights.invadd, 'inverse_additive'), (Owa_weights.exponential, 'exp'),
                (Owa_weights.mean, 'average')]

filename = 'results2.json'

filesize = os.path.getsize(filename)

if filesize == 0:
    file = open(filename, 'w')
    file.write('{"results": []}')
    file.close()

roots = os.listdir('multiInstance')
######


#####
total = len(roots) * len(raw_bag_distances) * len(internal_distances) * len(internal_OWA) * len(
    external_OWA)
progress = 0

for int_dists, int_owas, ext_owas, root, raw_b_dists in product(internal_distances, internal_OWA, external_OWA,
                                                                roots, raw_bag_distances):
    raw_b_dist, raw_b_dist_name, is_owa = raw_b_dists
    int_dist, int_dist_name = int_dists
    int_owa, int_owa_name = int_owas
    ext_owa, ext_owa_name = ext_owas
    test = {'data': root, 'b_dist': raw_b_dist_name, 'int_dist': int_dist_name, 'ext_owa': ext_owa_name}
    if is_owa:
        test['int_owa'] = int_owa_name


        def bag_dist(A, B, internal_dist=int_dist):
            return raw_b_dist(A, B, internal_dist=int_dist, weight=int_owa)

    else:
        test['int_owa'] = None


        def bag_dist(A, B, internal_dist=int_dist):
            return raw_b_dist(A, B, internal_dist=int_dist)

    print(test, round(100 * progress / total, 2))
    progress += 1
    if not is_in_file(test, filename):
        bag_relation = lambda x, y: 1 / (1 + (1 + bag_dist(x, y)))
        predictions = []
        values = []
        for i in tqdm(range(1, 11)):
            training_bags = KEEL_Data(f'multiInstance/{root}/{root}-10-{i}tra.dat').get_bags()
            test_bags = KEEL_Data(f'multiInstance/{root}/{root}-10-{i}tst.dat').get_bags()
            for test_bag in test_bags:
                predictions.append(BFMIC(training_bags, test_bag, {0, 1}, bag_relation, ext_owa))
                values.append(test_bag.iloc[0, -1])
        conf_matrix = metrics.confusion_matrix(predictions, values)
        test['TP'] = int(conf_matrix[0, 0])
        test['TN'] = int(conf_matrix[1, 1])
        test['FP'] = int(conf_matrix[0, 1])
        test['FN'] = int(conf_matrix[1, 0])
        write_json(test, filename)
