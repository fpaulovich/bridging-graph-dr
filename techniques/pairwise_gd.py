# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright (c) 2024 Fernando V. Paulovich
# License: MIT

import math
import numpy as np
from sklearn.metrics import DistanceMetric
from sklearn.metrics import euclidean_distances
from util import draw_graph_forceatlas2, write_graphml
import networkx as nx
import matplotlib.pyplot as plt


def pairwise_graph(X, metric, labels=None):
    size = len(X)
    dist_mat = DistanceMetric.get_metric(metric).pairwise(X)
    dist_mat = dist_mat / np.amax(dist_mat)

    # creating the graph
    g = nx.Graph()

    for i in range(size):
        g.add_node(i)

    # set labels as node attribute
    if labels is not None:
        nx.set_node_attributes(g, dict(enumerate(map(str, labels))), name='label')

    for i in range(size):
        for j in range(size):
            if i != j:
                weight = math.pow((1 - dist_mat[i][j]), 2)

                g.add_edge(i, j, weight=weight)
                g.add_edge(j, i, weight=weight)

    return g


def gd_pairwise(X, labels, filename_fig, filename_graph):
    metric = 'euclidean'

    g = pairwise_graph(X,
                       metric=metric,
                       labels=labels)

    pos = draw_graph_forceatlas2(X, g, labels, filename_fig)
    write_graphml(g, pos, filename_graph)


def mds(X, labels, filename_fig):
    size = len(X)

    dissimilarities = euclidean_distances(X)

    # init = PCA(n_components=2).fit_transform(X)
    # y, stress = smacof(dissimilarities,
    #                    n_components=2,
    #                    init=init,
    #                    max_iter=300,
    #                    eps=0.001,
    #                    random_state=0,
    #                    return_n_iter=False)

    y, evals = cmds(dissimilarities)

    # creating the graph
    g = nx.Graph()
    pos = {}

    for i in range(size):
        g.add_node(i)
        pos.update({i: [y[i][0], y[i][1]]})  # set node position as c-MDS coordinates

    # set labels as node attribute
    nx.set_node_attributes(g, dict(enumerate(map(str, labels))), name='label')

    plt.figure(figsize=(6, 5))

    # drawing the graph
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

    plt.savefig(filename_fig, dpi=400, bbox_inches='tight')
    plt.close()


def cmds(D, n_components=2):
    """Classical multidimensional scaling (MDS)

    Args:
        D (numpy.array)
            Symmetric distance matrix (n, n)
        n_components
            number of reduced dimensions

    Returns:
        Y (numpy.array)
            Configuration matrix (n, p). Each column represents a dimension. Only the
            p dimensions corresponding to positive eigenvalues of B are returned.
            Note that each dimension is only determined up to an overall sign,
            corresponding to a reflection.
        e (numpy.array)
            Eigenvalues of B (n, 1)
    """
    # Number of points
    n = len(D)

    # Centering matrix
    H = np.eye(n) - np.ones((n, n)) / n

    # YY^T
    B = -H.dot(D ** 2).dot(H) / 2

    # Diagonalize
    evals, evecs = np.linalg.eigh(B)

    # Sort by eigenvalue in descending order
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Compute the coordinates using positive-eigenvalued components only
    w, = np.where(evals > 0)
    L = np.diag(np.sqrt(evals[w]))
    V = evecs[:, w]
    Y = V.dot(L)

    return Y[:, :n_components], evals[evals > 0]
