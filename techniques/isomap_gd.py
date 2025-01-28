# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright (c) 2024 Fernando V. Paulovich
# License: MIT

import math
from sklearn.manifold import Isomap
from util import draw_graph, write_graphml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import DistanceMetric
import networkx as nx
import heapq


def knn_graph(X, nr_neighbors, metric):
    size = len(X)
    dist_mat = DistanceMetric.get_metric(metric).pairwise(X)
    dist_mat = dist_mat / np.amax(dist_mat)

    # adjusting the number of neighbors in case it is larger than the dataset
    nr_neighbors = min(nr_neighbors, size - 1)

    # creating the graph
    g = nx.Graph()

    for i in range(size):
        g.add_node(i)

    for i in range(size):
        heap = []

        for j in range(size):
            dist = dist_mat[i][j]

            if i != j:
                heapq.heappush(heap, (dist, j))

        for k in range(nr_neighbors):
            item = heapq.heappop(heap)
            weight = item[0]
            g.add_edge(i, item[1], weight=weight)
            g.add_edge(item[1], i, weight=weight)

    return g


def isomap_graph(X, nr_neighbors, metric, labels=None):
    size = len(X)

    knn_g = knn_graph(X, nr_neighbors, metric)
    shortest_paths_length = nx.shortest_path_length(knn_g, weight='weight')
    geodesic_matrix = dict(shortest_paths_length)

    # creating the graph
    g = nx.Graph()

    for i in range(size):
        g.add_node(i)

    # set labels as node attribute
    if labels is not None:
        nx.set_node_attributes(g, dict(enumerate(map(str, labels))), name='label')

    max_val = -1
    for i in geodesic_matrix.keys():
        for j in geodesic_matrix[i].keys():
            if i != j:
                max_val = max(max_val, geodesic_matrix[i][j])

    for i in geodesic_matrix.keys():
        for j in geodesic_matrix[i].keys():
            if i != j:
                weight = math.pow(1 - (geodesic_matrix[i][j] / max_val), 2)
                g.add_edge(i, j, weight=weight)
                g.add_edge(j, i, weight=weight)

    return g


def gd_isomap(X, labels, filename_fig, filename_graph, nr_neighbors=10):
    metric = 'euclidean'

    g = isomap_graph(X,
                     nr_neighbors=nr_neighbors,
                     metric=metric,
                     labels=labels)

    pos = draw_graph(X, g, labels, filename_fig)
    write_graphml(g, pos, filename_graph)


def isomap(X, labels, filename_fig, nr_neighbors=10):
    size = len(X)
    metric = 'euclidean'

    y = Isomap(n_neighbors=nr_neighbors,
               metric=metric,
               n_components=2).fit_transform(X)

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
