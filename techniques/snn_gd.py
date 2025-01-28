# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright (c) 2024 Fernando V. Paulovich
# License: MIT

import numpy as np
from sklearn.metrics import DistanceMetric
from util import draw_graph, write_graphml
import networkx as nx
import heapq


def snn_graph(X, nr_neighbors, metric, labels=None):
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
            g.add_edge(i, item[1], weight=(1 - item[0]))

    # creating the graph
    snn_g = nx.Graph()

    for i in range(size):
        snn_g.add_node(i)

    # set labels as node attribute
    if labels is not None:
        nx.set_node_attributes(snn_g, dict(enumerate(map(str, labels))), name='label')

    for i in range(size):
        for j in range(size):
            if i != j:
                snn_size = len(nx.common_neighbors(g, i, j))

                if snn_size > 0:
                    snn_g.add_edge(i, j, weight=snn_size)

    return snn_g


def gd_snn(X, labels, filename_fig, filename_graph, nr_neighbors=10):
    metric = 'euclidean'

    g = snn_graph(X,
                  nr_neighbors=nr_neighbors,
                  metric=metric,
                  labels=labels)

    pos = draw_graph(X, g, labels, filename_fig)
    write_graphml(g, pos, filename_graph)
