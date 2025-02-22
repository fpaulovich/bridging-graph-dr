# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright (c) 2024 Fernando V. Paulovich
# License: MIT

import numpy as np
from sklearn.manifold._t_sne import _joint_probabilities_nn
from sklearn.manifold._t_sne import _joint_probabilities
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from util import draw_graph, write_graphml
import networkx as nx
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

MACHINE_EPSILON = np.finfo(np.double).eps
n_jobs = 5
metric_params = None


def calculate_joint_probabilities(X, perplexity, metric):
    if metric == "euclidean":
        # Euclidean is squared here, rather than using **= 2,
        # because euclidean_distances already calculates
        # squared distances, and returns np.sqrt(dist) for
        # squared=False.
        # Also, Euclidean is slower for n_jobs>1, so don't set here
        distances = pairwise_distances(X, metric=metric, squared=True)
    else:
        metric_params_ = metric_params or {}
        distances = pairwise_distances(
            X, metric=metric, n_jobs=n_jobs, **metric_params_
        )

    P = _joint_probabilities(distances, perplexity, 0)

    return P


def calculate_joint_probabilities_bh(X, perplexity, metric):
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


def tsne_prob_graph(X, perplexity, metric, labels=None, epsilon=0.95):
    size = len(X)

    P = 2 * calculate_joint_probabilities(X, perplexity, metric)

    # remove edges with very low probability
    overall_sum = epsilon  # overall sum of probabilities
    sorted_vect = -np.sort(-P.copy())

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
    if labels is not None:
        nx.set_node_attributes(g, dict(enumerate(map(str, labels))), name='label')

    k = 0
    for i in range(size):
        for j in range(i + 1, size):
            if P[k] >= min_val:
                g.add_edge(i, j, weight=P[k] / cum_sum)
            k = k + 1

    return g


def tsne_bh_prob_graph(X, perplexity, metric, labels=None, epsilon=0.95):
    size = len(X)

    P = calculate_joint_probabilities_bh(X, perplexity, metric).tocoo()

    data = P.data
    row_idx = P.row
    col_idx = P.col

    # remove edges with very low probability
    overall_sum = epsilon  # overall sum of probabilities
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
    if labels is not None:
        nx.set_node_attributes(g, dict(enumerate(map(str, labels))), name='label')

    for i in range(len(data)):
        if data[i] >= min_val:
            g.add_edge(row_idx[i], col_idx[i], weight=data[i] / cum_sum)

    return g


def gd_tsne(X, labels, filename_fig, filename_graph, perplexity=10):
    metric = 'euclidean'

    # g = tsne_prob_graph(X, perplexity, metric, labels, epsilon=0.9) # original probability t-SNE graph
    g = tsne_bh_prob_graph(X, perplexity, metric, labels, epsilon=0.9)  # Barnes-Hut probability t-SNE graph

    pos = draw_graph(X, g, labels, filename_fig)
    write_graphml(g, pos, filename_graph)

    return g


def tsne(X, labels, filename_fig, filename_graph, perplexity=10):
    size = len(X)
    metric = 'euclidean'

    y = TSNE(n_components=2, perplexity=perplexity, metric=metric, random_state=0).fit_transform(X)

    # creating the graph
    g = nx.Graph()
    pos = {}

    for i in range(size):
        g.add_node(i)
        pos.update({i: [y[i][0], y[i][1]]})  # set node position as t-SNE coordinates

    write_graphml(g, pos, filename_graph)

    # set labels as node attribute
    nx.set_node_attributes(g, dict(enumerate(map(str, labels))), name='label')

    # drawing the graph
    plt.figure(figsize=(6, 5))

    nx.draw(g, pos=pos,
            with_labels=False,
            node_color=labels,
            cmap=plt.cm.tab10,
            node_size=25,
            edge_color='white',
            width=0.5)
    ax = plt.gca()  # to get the current axis
    ax.collections[0].set_edgecolor("#000000")

    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)

    if filename_fig is not None:
        plt.savefig(filename_fig, dpi=400, bbox_inches='tight')
    plt.close()
