# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np

from sklearn.manifold._t_sne import _joint_probabilities_nn
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
import sklearn.datasets as datasets

import math

import networkx as nx

import matplotlib.pyplot as plt

MACHINE_EPSILON = np.finfo(np.double).eps
n_jobs = 5


def calculate_joint_probabilities(X, perplexity, metric):
    n_samples = len(X)

    # Compute the number of nearest neighbors to find.
    # LvdM uses 3 * perplexity as the number of neighbors.
    # In the event that we have very small # of points
    # set the neighbors to n - 1.
    n_neighbors = min(n_samples - 1, int(3.0 * perplexity + 1))

    # Find the nearest neighbors for every point
    knn = NearestNeighbors(
        algorithm="auto",
        n_jobs=n_jobs,
        n_neighbors=n_neighbors,
        metric=metric,
        metric_params=None,
    )

    knn.fit(X)

    distances_nn = knn.kneighbors_graph(mode="distance")

    # Free the memory used by the ball_tree
    del knn

    # knn return the euclidean distance but we need it squared
    # to be consistent with the 'exact' method. Note that the
    # method was derived using the euclidean method as in the
    # input space. Not sure of the implication of using a different
    # metric.
    distances_nn.data **= 2

    # compute the joint probability distribution for the input space
    P = _joint_probabilities_nn(distances_nn, perplexity, 0)

    return P


def precision_recall(X, labels, perplexity, metric):
    size = len(X)

    P = calculate_joint_probabilities(X, perplexity, metric).tocoo()

    data = P.data
    row_idx = P.row
    col_idx = P.col

    # remove edges with very low probability
    overall_sum = 0.95  # overall sum of probabilities
    sorted_vect = -np.sort(-P.data.copy())

    # find the minimum value which the commulative sum reaches overall_sum
    min_val = -1
    cum_sum = 0
    for i in range(len(sorted_vect)):
        cum_sum = cum_sum + sorted_vect[i]
        if cum_sum >= overall_sum:
            min_val = sorted_vect[i]
            break

    # creating the graph
    g = nx.Graph()

    for i in range(size):
        g.add_node(i)

    # set labels as node attribute
    nx.set_node_attributes(g, dict(enumerate(map(str, labels))), name='label')

    for i in range(len(data)):
        if data[i] >= min_val:
            g.add_edge(row_idx[i], col_idx[i], length=(1 - data[i]))

    # calculate precision
    same_class_nodes = np.zeros(size)
    different_class_nodes = np.zeros(size)
    for i in range(size):
        for j in g.neighbors(i):
            if g.nodes[i]['label'] == g.nodes[j]['label']:
                same_class_nodes[j] = same_class_nodes[j] + (1 - g.edges[i, j]['length'])
            else:
                different_class_nodes[j] = different_class_nodes[j] + (1 - g.edges[i, j]['length'])

    precision_per_point = np.zeros(size)
    for i in range(size):
        if different_class_nodes[i] > 0:
            precision_per_point[i] = same_class_nodes[i] / (same_class_nodes[i] + different_class_nodes[i])
        else:
            precision_per_point[i] = 1

    # remove edges connecting nodes of different classes (for recall calculation)
    for i, j in g.edges():
        if g.nodes[i]['label'] != g.nodes[j]['label']:
            g.remove_edge(i, j)

    # calculate recall
    labels_count = {}
    for i in range(size):
        item = labels_count.get(labels[i])
        if item is None:
            labels_count.update({labels[i]: 1})
        else:
            labels_count.update({labels[i]: item + 1})

    recall_per_point = np.zeros(size)
    clust_coefficients = nx.clustering(g, weight='length')
    components = [c for c in sorted(nx.connected_components(g), key=len, reverse=True)]

    for i in range(len(components)):
        component = list(components[i])
        component_label = labels[component[0]]
        component_size = len(component)
        component_recall = (component_size / labels_count.get(component_label))

        for j in range(component_size):
            if component_size == 1:
                recall_per_point[component[j]] = component_recall  # components with one element has precision==1,
                                                                   # so recall should be the component recall.
            else:
                recall_per_point[component[j]] = (component_recall * clust_coefficients[component[j]])

    del g

    return precision_per_point, recall_per_point


def f_score(precision, recall, beta=1):
    return (1 + beta*beta) * np.divide(np.multiply(precision, recall), np.add((beta*beta)*precision, recall))


def test_multiple_perplexity():
    raw = datasets.load_digits(as_frame=True)
    X = raw.data.to_numpy()
    X = preprocessing.MinMaxScaler().fit_transform(X)
    labels = raw.target.to_numpy()

    max_perplexity = 150
    min_perplexity = 1
    increment = 1

    avg_pre = np.zeros(math.ceil((max_perplexity - min_perplexity)/increment))
    avg_rec = np.zeros(math.ceil((max_perplexity - min_perplexity)/increment))
    avg_f_s = np.zeros(math.ceil((max_perplexity - min_perplexity)/increment))
    perp_vals = np.zeros(math.ceil((max_perplexity - min_perplexity)/increment))

    k = 0
    for perplexity in range(min_perplexity, max_perplexity, increment):
        print('perplexity: ', perplexity)
        perp_vals[k] = perplexity

        pre, rec = precision_recall(X, labels, perplexity, 'euclidean')
        f_s = f_score(pre, rec, 1.0)

        for i in range(len(pre)):
            if (pre[i] + rec[i]) == 0:
                print('>>> ', pre[i], rec[i])

        avg_pre[k] = np.average(pre)
        avg_rec[k] = np.average(rec)
        avg_f_s[k] = np.average(f_s)

        k = k + 1

    plt.plot(perp_vals, avg_pre, label='precision')
    plt.plot(perp_vals, avg_rec, label='recall')
    plt.plot(perp_vals, avg_f_s, label='f-score')
    plt.legend()
    plt.show()


def test():
    raw = datasets.load_iris(as_frame=True)
    X = raw.data.to_numpy()
    X = preprocessing.MinMaxScaler().fit_transform(X)

    labels = raw.target.to_numpy()
    size = len(X)

    perplexity = 15
    metric = 'euclidean'

    P = calculate_joint_probabilities(X, perplexity, metric).tocoo()

    data = P.data
    row_idx = P.row
    col_idx = P.col

    # remove edges with very low probability
    overall_sum = 0.95  # overall sum of probabilities
    sorted_vect = -np.sort(-P.data.copy())

    # find the minimum value which the commulative sum reaches overall_sum
    min_val = -1
    cum_sum = 0
    for i in range(len(sorted_vect)):
        cum_sum = cum_sum + sorted_vect[i]
        if cum_sum >= overall_sum:
            min_val = sorted_vect[i]
            break

    # creating the graph
    g = nx.Graph()

    for i in range(size):
        g.add_node(i)

    # set labels as node attribute
    nx.set_node_attributes(g, dict(enumerate(map(str, labels))), name='label')

    for i in range(len(data)):
        if data[i] > min_val:
            g.add_edge(row_idx[i], col_idx[i], length=(1 - data[i]))

    original_g = g.copy()

    # calculate precision
    same_class_nodes = np.zeros(size)
    different_class_nodes = np.zeros(size)
    for i in range(size):
        for j in g.neighbors(i):
            if g.nodes[i]['label'] == g.nodes[j]['label']:
                same_class_nodes[j] = same_class_nodes[j] + (1 - g.edges[i, j]['length'])
            else:
                different_class_nodes[j] = different_class_nodes[j] + (1 - g.edges[i, j]['length'])

    precision_per_point = np.zeros(size)
    for i in range(size):
        if different_class_nodes[i] > 0:
            precision_per_point[i] = same_class_nodes[i] / (same_class_nodes[i] + different_class_nodes[i])
        else:
            precision_per_point[i] = 1

    print('average precision: ', np.average(precision_per_point))

    # # remove edges connecting nodes of different classes (for recall calculation)
    # for i, j in g.edges():
    #     if g.nodes[i]['label'] != g.nodes[j]['label']:
    #         g.remove_edge(i, j)

    # calculate recall
    labels_count = {}
    for i in range(size):
        item = labels_count.get(labels[i])

        if item is None:
            labels_count.update({labels[i]: 1})
        else:
            labels_count.update({labels[i]: item + 1})

    recall_per_point = np.zeros(size)
    clust_coefficients = nx.clustering(g, weight='length')
    components = [c for c in sorted(nx.connected_components(g), key=len, reverse=True)]

    for i in range(len(components)):
        component = list(components[i])
        component_label = labels[component[0]]
        component_size = len(component)
        component_recall = (component_size / labels_count.get(component_label))

        for j in range(component_size):
            recall_per_point[component[j]] = (component_recall * clust_coefficients[component[j]])

    print('average recall: ', np.average(recall_per_point))

    # drawing the graph
    nx.draw(original_g, pos=nx.fruchterman_reingold_layout(original_g),
            with_labels=False,
            node_color=labels,
            cmap=plt.cm.tab10,
            node_size=50,
            edge_color='gray',
            width=0.5)
    ax = plt.gca()  # to get the current axis
    ax.collections[0].set_edgecolor("#000000")
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test()
