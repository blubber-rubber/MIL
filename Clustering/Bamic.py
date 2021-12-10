from Utils.Distances import *
import random


def dynamic_distance_calc(bags, index1, index2, distances, bag_dist):
    if distances[index1, index2] is None:
        distances[index1, index2] = bag_dist(bags[index1], bags[index2])
        distances[index2, index1] = distances[index1, index2]
    return distances[index1, index2]


def bamic(bags, n_clusters=2, bag_dist=hausdorff_distance_min):
    clusters_indices = []
    medioid_indices = random.sample(range(len(bags)), n_clusters)
    dyn_distances = np.array([[None for i in bags] for _ in bags])
    changed = True
    score = math.inf
    while changed:
        changed = False
        clusters_indices = [[medioid_indices[j]] for j in range(n_clusters)]

        for i, _ in enumerate(bags):
            index = np.argmin([
                dynamic_distance_calc(bags, i, medioid_indices[j], dyn_distances, bag_dist) for j in range(n_clusters)])
            if i not in clusters_indices[index]:
                clusters_indices[index].append(i)

                
        new_medioids = []
        new_score = 0
        print(f'mediods:  {medioid_indices}')
        print(dynamic_distance_calc(bags, medioid_indices[0], medioid_indices[1], dyn_distances, bag_dist))
        for j in range(n_clusters):
            distances = np.sum(np.array(
                [[dynamic_distance_calc(bags, i, k, dyn_distances, bag_dist) for i in clusters_indices[j]] for k in
                 clusters_indices[j]]), axis=0)
            new_index = np.argmin(distances)
            new_score += distances[new_index]
            new_medioids.append(clusters_indices[j][new_index])
        if new_score < score:
            changed = True
            score = new_score
            medioid_indices = new_medioids
        print(new_score)

    clusters = [[bags[i] for i in clusters_indices[k]] for k in range(n_clusters)]
    medioids = [bags[j] for j in medioid_indices]
    return clusters, medioids


if __name__ == "__main__":
    from KEEL_DataReader import *

    training_bags = KEEL_Data('Artificial_data/data1.dat').get_bags()
    tr_bags = [bag.iloc[:, 1:-1] for bag in training_bags]
    a, b = bamic(tr_bags)
    print(b)
