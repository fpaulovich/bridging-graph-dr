# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright (c) 2024 Fernando V. Paulovich
# License: MIT

from umap import UMAP
from util import draw_graph_forceatlas2, write_graphml
import networkx as nx
import matplotlib.pyplot as plt


def umap_graph(X, nr_neighbors, metric, labels=None):
    size = len(X)

    mapper = UMAP(n_neighbors=nr_neighbors,
                  metric=metric,
                  random_state=42,
                  transform_mode='graph')

    y = mapper.fit_transform(X)

    # creating the graph
    g = nx.Graph()

    for i in range(size):
        g.add_node(i)

    # set labels as node attribute
    if labels is not None:
        nx.set_node_attributes(g, dict(enumerate(map(str, labels))), name='label')

    P = mapper.graph_.tocoo()

    for i in range(len(P.data)):
        g.add_edge(P.row[i], P.col[i], weight=P.data[i])

    return g


def umap(X, labels, filename_fig, filename_graph, nr_neighbors=10):
    size = len(X)
    metric = 'euclidean'

    y = UMAP(n_neighbors=nr_neighbors,
             metric=metric,
             random_state=42).fit_transform(X)

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

    plt.savefig(filename_fig, dpi=400, bbox_inches='tight')
    plt.close()


def gd_umap(X, labels, filename_fig, filename_graph, nr_neighbors=10):

    metric = 'euclidean'

    g = umap_graph(X, nr_neighbors, metric, labels=labels)

    pos = draw_graph_forceatlas2(X, g, labels, filename_fig)
    write_graphml(g, pos, filename_graph)
