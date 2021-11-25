import numpy as np
import random, math
from sklearn.datasets import make_blobs

# colors to be used in plots
colors = np.array(['r', 'g', 'b', 'y', 'c'])

n_attributes = 2
n_positive_bags = 30
n_negative_bags = 70
p = 4 / 5

file = open('Artificial_data/data1.dat', 'w')
bag_ids = '{' + ', '.join(str(i) for i in range(n_positive_bags + n_negative_bags)) + '}'
lines = ['@relation artificial\n', f'@attribute Bag_id {bag_ids}\n']

inputs = '@inputs Bag_id'
for i in range(n_attributes):
    inputs += f', F{i}'
    lines.append(f'@attribute F{i} real')
lines.append('@attribute Class {0, 1}\n')
inputs += '\n'
lines.append(inputs)
lines.append('@outputs Class\n')
lines.append('@data\n')

min_attr = [math.inf] * n_attributes
max_attr = [-math.inf] * n_attributes

positive_center = (-5, -5)
negative_center = (5, 5)
pbags = 0
for bag in range(n_positive_bags + n_negative_bags):
    label = random.random() < 1 / 2 and pbags < n_positive_bags
    pbags += label
    center = positive_center if label else negative_center
    center = (center[0] + random.normalvariate(0, 1), center[1] + random.normalvariate(0, 1))
    n_samples = 1
    while random.random() < p:
        n_samples += 1

    X, y = make_blobs(n_samples=n_samples, n_features=n_attributes, cluster_std=1.0,
                      centers=[center, center], shuffle=False)
    for i in range(len(X)):
        data = [bag]
        for j, data_point in enumerate(X[i]):
            if max_attr[j] < data_point:
                max_attr[j] = data_point
            if min_attr[j] > data_point:
                min_attr[j] = data_point
        data.extend(X[i])
        data.append(int(label))
        line = ', '.join(str(d) for d in data) + '\n'
        lines.append(line)

for attr in range(n_attributes):
    lines[attr + 2] += f' [{min_attr[attr]}, {max_attr[attr]}]\n'
file.writelines(lines)
file.close()

