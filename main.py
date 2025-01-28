# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import pylab as p
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd

from sklearn.manifold._utils import _binary_search_perplexity

import networkx as nx
import matplotlib.pyplot as plt
import networkx.algorithms.community as nxcom
from sklearn.cluster import AgglomerativeClustering

import math

from sklearn import preprocessing

import sklearn.datasets as datasets

MACHINE_EPSILON = np.finfo(np.double).eps


def _joint_probabilities(distances, desired_perplexity):
    """Compute joint probabilities p_ij from distances.

    Parameters
    ----------
    distances : ndarray of shape (n_samples * (n_samples-1) / 2,)
        Distances of samples are stored as condensed matrices, i.e.
        we omit the diagonal and duplicate entries and store everything
        in a one-dimensional array.

    desired_perplexity : float
        Desired perplexity of the joint probability distributions.

    verbose : int
        Verbosity level.

    Returns
    -------
    P : ndarray of shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.
    """
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    distances = distances.astype(np.float32, copy=False)
    conditional_P = _binary_search_perplexity(
        distances, desired_perplexity, 0
    )
    P = (conditional_P + conditional_P.T) / (2 * len(distances))
    return squareform(P)

    # sum_P = np.maximum(np.sum(P), MACHINE_EPSILON)  # is this correct? the sum of the entire matrix?
    # P = np.maximum(squareform(P) / sum_P, MACHINE_EPSILON)
    # return P


def test():
    raw = datasets.load_iris(as_frame=True)
    X = raw.data.to_numpy()
    X = preprocessing.MinMaxScaler().fit_transform(X)

    label = raw.target.to_numpy()
    size = len(X)

    perplexity = 5
    metric = 'euclidean'

    distances = pairwise_distances(X, metric=metric, squared=True)
    P = _joint_probabilities(distances, perplexity)

    # remove edges with very low probability
    overall_sum = 0.95  # overall sum of probabilities
    sorted_vect = -np.sort(-P.copy())

    # find the minimum value which the commulative sum reaches overall_sum
    min_val = -1
    cum_sum = 0
    for i in range(len(sorted_vect)):
        cum_sum = cum_sum + (2 * sorted_vect[i])
        if cum_sum >= overall_sum:
            min_val = sorted_vect[i]
            break

    # remove entries for which the value is lower than min_val
    for i in range(len(P)):
        if P[i] < min_val:
            P[i] = 0

    # calculating precision
    same_class_nodes = np.zeros(size)
    different_class_nodes = np.zeros(size)
    k = 0
    for i in range(size-1):
        for j in range(i+1, size):
            if label[i] != label[j]:
                different_class_nodes[i] = different_class_nodes[i] + P[k]
                different_class_nodes[j] = different_class_nodes[j] + P[k]
            else:
                same_class_nodes[i] = same_class_nodes[i] + P[k]
                same_class_nodes[j] = same_class_nodes[j] + P[k]
            k = k + 1

    precision_per_point = np.zeros(size)
    for i in range(size):
        if same_class_nodes[i] > 0:
            precision_per_point[i] = same_class_nodes[i] / (same_class_nodes[i] + different_class_nodes[i])

    print('average precision: ', np.average(precision_per_point))

    # remove edges connecting nodes of different classes (for recall calculation)
    k = 0
    for i in range(size-1):
        for j in range(i+1, size):
            if label[i] != label[j]:
                P[k] = 0
            k = k + 1

    # creating the graph
    g = nx.Graph()

    for i in range(size):
        g.add_node(i)

    k = 0
    for i in range(size-1):
        for j in range(i+1, size):
            if P[k] > 0:
                g.add_edge(i, j, length=(1 - P[k]), weigth=P[k])
            k = k + 1

    # calculating recall
    labels_count = {}
    for i in range(size):
        item = labels_count.get(label[i])

        if item is None:
            labels_count.update({label[i]: 1})
        else:
            labels_count.update({label[i]: item + 1})

    recall_per_point = np.zeros(size)
    clust_coefficients = nx.clustering(g)
    components = [c for c in sorted(nx.connected_components(g), key=len, reverse=True)]

    for i in range(len(components)):
        component = list(components[i])
        component_label = label[component[0]]
        component_size = len(component)
        component_recall = (component_size / labels_count.get(component_label))

        for j in range(component_size):
            recall_per_point[component[j]] = (component_recall * clust_coefficients[component[j]])

    print('average recall: ', np.average(recall_per_point))

    # drawing the graph
    nx.draw(g, pos=nx.fruchterman_reingold_layout(g),
            with_labels=False,
            node_color=label,
            cmap=plt.cm.Set1,
            node_size=50,
            edge_color='gray',
            width=0.5)
    ax = plt.gca()  # to get the current axis
    ax.collections[0].set_edgecolor("#000000")
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test()