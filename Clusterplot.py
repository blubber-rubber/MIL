import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import random
from sklearn import datasets, cluster, neighbors, metrics, tree, svm, mixture
from itertools import cycle, islice, chain

# colors to be used in plots
colors = np.array(['y', 'b', 'g', 'r', 'c'])

# visualize the data

from KEEL_DataReader import *
from Bamic import *

data = KEEL_Data('Artificial_data/data1.dat')
X = data.data
plt.scatter(X.iloc[:, 1], X.iloc[:, 2], color=colors[X.iloc[:, 3]])
plt.show()
training_bags = data.get_bags()
tr_bags = [bag.iloc[:, 1:-1] for bag in training_bags]

clusters, medioids = bamic(tr_bags)
cl1 = sum([list(bag.index) for bag in clusters[0]], [])
kleuren = [1 if i in cl1 else 0 for i in range(len(X))]
m1 = list(medioids[0].index)
m2 = list(medioids[1].index)

for i in m1:
    kleuren[i] = 2
for i in m2:
    kleuren[i] = 3

print(m1)
print(m2)

plt.scatter(X.iloc[:, 1], X.iloc[:, 2], color=colors[kleuren])
plt.show()
