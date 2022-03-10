from collections import defaultdict

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from KEEL_DataReader import KEEL_Data

from matplotlib.colors import ListedColormap
import random
import numpy as np

root = 'musk2'
X = KEEL_Data(f'multiInstance/{root}/{root}.dat').get_bags()

bag_sizes = {bag.iloc[0, 0]: bag.shape[0] for bag in X}
print(bag_sizes)

pos_bag_sizes = defaultdict(int)
neg_bag_sizes = defaultdict(int)
for bag in X:
    if bag.iloc[0, -1] == 1:
        pos_bag_sizes[bag.shape[0]] += 1
    else:
        neg_bag_sizes[bag.shape[0]] += 1

print(pos_bag_sizes)
print(neg_bag_sizes)

plt.bar([key - 0.1 for key in pos_bag_sizes.keys()], pos_bag_sizes.values(), color='g', width=0.5)
plt.bar([key + 0.1 for key in neg_bag_sizes.keys()], neg_bag_sizes.values(), color='r', width=0.5)
plt.show()

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from KEEL_DataReader import KEEL_Data
from sklearn import cluster
from yellowbrick.cluster import KElbowVisualizer
from matplotlib.colors import ListedColormap
import random
import numpy as np

pca = PCA()

data = KEEL_Data(f'multiInstance/{root}/{root}.dat')
X = data.data

# Colours
bagnames = set(X.iloc[:, 0])
colors = np.array([(random.random(), random.random(), random.random()) for i in range(len(bagnames))])
bagnames = {name: index for (index, name) in enumerate(bagnames)}
kleuren = [bagnames[naam] for naam in X.iloc[:, 0]]

# PCA on data
pca_dims = pca.fit_transform(X.iloc[:, 1:-1])

pos_inst = pca_dims[X.iloc[:, 1] == 'NON-MUSK-252']
neg_inst = pca_dims[X.iloc[:, 1] == 'NON-MUSK-j146']



kleuren_pos = np.array(kleuren)[X.iloc[:, -1] == 1]
kleuren_neg = np.array(kleuren)[X.iloc[:, -1] == 0]

cmap = ListedColormap([colors[0], colors[1]])
# PLOTTING

# Each class different colour
scatter = plt.scatter(pca_dims[:, 0], pca_dims[:, 1], c=X.iloc[:, -1], cmap=cmap)
plt.title("each class diffirent colour")
plt.legend(handles=scatter.legend_elements()[0], labels=["Negative", "Positive"])
plt.show()
