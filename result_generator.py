from CitationKNN import citationKNN
from tqdm import tqdm
from sklearn import preprocessing
from Utils import Owa_weights
import os
import time
from sklearn import metrics
from KEEL_DataReader import *
import json
from Utils.Distances import *
from scipy.spatial import distance
from itertools import product
from Utils import BagEncodings


#################################################SET-UP#################################################################


owa_distances = ['H', 'AvgH', 'MinH', 'SumMin']  # These distances should be considered as OWA-distances
encoding_distances = ['enc']  # This distance should be considered as an encoding distance

neighbours = [1, 5, 10, 15]  # Number of neighbours to consider
normalizers = [(None, None)]  # Whether or not to normalize/scale the data
raw_bag_distances = [(hausdorff_distance, 'H'), (hausdorff_distance_avg, 'AvgH'),
                     (hausdorff_distance_min, 'MinH'), (sum_of_min_distance, 'SumMin'),
                     (link_distance, 'link'), (surjection_distance, 'surj'),
                     (fair_surjection_distance, 'fair_surj'), (norm_link_distance, 'norm_link'),
                     (norm_surjection_distance, 'norm_surj'),
                     (norm_fair_surjection_distance, 'norm_fair_surj'),
                     (encoding_distance, 'enc')]  # distances to consider
internal_distances = [(distance.euclidean, 'euclidean'),
                      (distance.cityblock, 'manhattan')]  # Distances used inside the bag-distances
internal_OWA = [(Owa_weights.strict, 'strict'), (Owa_weights.additive, 'additive'),
                (Owa_weights.invadd, 'inverse_additive'), (Owa_weights.exponential, 'exp'),
                (Owa_weights.mean, 'average')]  # Owa-weights to be used
encodings = [(BagEncodings.averageEncoding, 'avg_enc'), (BagEncodings.centerEncoding, 'center_enc')]  # Encodings to be used

filename = 'default_results.json'  # Where to save the results

roots = os.listdir('multiInstance')  # Where to get the data from

kfold = 'default'  # Make this a number if you want custom k-fold, or string 'default' to use the KEEL 10-fold


#################################################CODE###################################################################
# Ideally you shouldn't have to change anything here

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


# fucntion to check if the test has been done before
def is_in_file(data, filename='result.json'):
    with open(filename, 'r') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        for result in file_data['results']:
            if all([result[key] == value for key, value in data.items()]):
                return True
        return False


filesize = os.path.getsize(filename)

if filesize == 0:
    file = open(filename, 'w')
    file.write('{"results": []}')
    file.close()

total = len(roots) * len(neighbours) * len(raw_bag_distances) * len(internal_distances) * len(internal_OWA) * len(
    encodings) * len(normalizers)
progress = 0

for root, k, raw_b_dists, int_dists, int_owas, encods, normals in product(roots, neighbours, raw_bag_distances,
                                                                          internal_distances, internal_OWA, encodings,
                                                                          normalizers):
    raw_b_dist, raw_b_dist_name = raw_b_dists
    int_dist, int_dist_name = int_dists
    int_owa, int_owa_name = int_owas
    enc, enc_name = encods
    normalizer, normalize_name = normals

    # Info for Json-file
    test = {'data': root, 'b_dist': raw_b_dist_name, 'int_dist': int_dist_name, 'int_owa': None, 'k': k,
            'norm': normalize_name}
    if raw_b_dist_name in owa_distances:
        test['int_owa'] = int_owa_name

    elif raw_b_dist_name in encoding_distances:
        test['b_dist'] = enc_name

    print(test, round(100 * progress / total, 2))
    progress += 1

    if not is_in_file(test, filename):
        if kfold == 'default':  # Default or custom KFold
            generator = [(KEEL_Data(f'multiInstance/{root}/{root}-10-{i}tra.dat').get_bags(),
                          KEEL_Data(f'multiInstance/{root}/{root}-10-{i}tst.dat').get_bags()) for i in range(1, 11)]
        else:
            generator = KEEL_Data(f'multiInstance/{root}/{root}.dat').k_fold(kfold)
        y_predictions = []
        y_true = []
        for training_bags, test_bags in tqdm(generator):
            if normalizer is not None:
                normer = normalizer()
                X_OG = pd.concat(training_bags)  # recovered data voor de groupby
                normer.fit(X_OG.iloc[:, 1:-1])
                for bag in training_bags + test_bags:
                    bag.iloc[:, 1:-1] = normer.transform(bag.iloc[:, 1:-1])

            if raw_b_dist_name in owa_distances:  # Which version of citationKNN should be used
                def KNN(trainingb, testb):
                    return citationKNN(training_bags, test_bag, k=k, dist=raw_b_dist, internal_dist=int_dist,
                                       weight=int_owa, encoding=None)

            elif raw_b_dist_name in encoding_distances:
                encoding = enc(training_bags, test_bags)


                def KNN(trainingb, testb):
                    return citationKNN(trainingb, testb, k=k, dist=raw_b_dist, internal_dist=int_dist,
                                       weight=None, encoding=encoding)

            else:

                def KNN(trainingb, testb):
                    return citationKNN(trainingb, testb, k=k, dist=raw_b_dist, internal_dist=int_dist,
                                       weight=None, encoding=None)

            fold_predictions = []
            fold_true = []
            for test_bag in test_bags:
                fold_predictions.append(KNN(training_bags, test_bag))
                fold_true.append(int(test_bag.iloc[0, -1]))
            y_predictions.append(''.join(str(number) for number in fold_predictions))
            y_true.append(''.join(str(number) for number in fold_true))

        test['predictions'] = y_predictions
        test['true'] = y_true
        write_json(test, filename)
