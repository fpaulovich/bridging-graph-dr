# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright (c) 2024 Fernando V. Paulovich
# License: MIT

from sklearn.decomposition import PCA
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math


def load_data(dataset):
    X = np.load(dataset + "/X.npy", allow_pickle=True).astype(np.float64)
    y = np.load(dataset + "/y.npy", allow_pickle=True).astype(np.int64)
    return X, y


def write_graphml(g, pos, filename):
    if filename is not None:
        for node, (x, y) in pos.items():
            g.nodes[node]['x'] = float(x)
            g.nodes[node]['y'] = float(y)
        nx.write_graphml(g, filename, named_key_ids=True)


def init(X):
    size = len(X)
    pca = PCA(n_components=2).fit_transform(X)

    pos = {}
    for i in range(size):
        pos[i] = pca[i]

    return pos


def draw_graph_forceatlas2(X, g, labels, filename=None):
    init_pos = init(X)
    size = len(X)

    pos = nx.forceatlas2_layout(g,
                                pos=init_pos,
                                weight='weight',
                                seed=42,
                                max_iter=1000,
                                scaling_ratio=2,
                                dissuade_hubs=False,
                                linlog=False)

    # pos = nx.fruchterman_reingold_layout(g,
    #                                      pos=init_pos,
    #                                      weight='weight',
    #                                      seed=1,
    #                                      iterations=1000,
    #                                      k=1 / math.sqrt(size))

    plt.figure(figsize=(6, 6))

    # drawing the graph
    nodes = nx.draw_networkx_nodes(g,
                                   pos=pos,
                                   node_color=labels,
                                   cmap=plt.cm.Set1,
                                   node_size=15,
                                   linewidths=0.25)

    edges = nx.draw_networkx_edges(g,
                                   pos=pos,
                                   edge_color='silver',
                                   width=0.5)

    ax = plt.gca()  # to get the current axis
    ax.collections[0].set_edgecolor("#000000")

    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    return pos


def draw_graph_with_positions(g, labels=None, cmap=plt.cm.Set1, color_bar_title=None, vmin=None, vmax=None, filename=None):
    pos = {}
    for n, data in g.nodes.items():
        pos[n] = (data['x'], data['y'])

    if labels is None:
        labels = list(map(int, nx.get_node_attributes(g, 'label').values()))

    plt.figure(figsize=(7, 6))

    nodes = nx.draw_networkx_nodes(g,
                                   pos=pos,
                                   node_color=labels,
                                   cmap=cmap,
                                   node_size=15,
                                   linewidths=0.25,
                                   vmin=vmin,
                                   vmax=vmax)

    edges = nx.draw_networkx_edges(g,
                                   pos=pos,
                                   edge_color='silver',
                                   width=0.5)

    if color_bar_title is not None:
        cb = plt.colorbar(nodes,
                          orientation='vertical',
                          pad=0.025,
                          label=color_bar_title)

    ax = plt.gca()  # to get the current axis
    ax.collections[0].set_edgecolor("#000000")

    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def draw_projection(y, label, cmap=plt.cm.Set1, color_bar_title=None, filename=None):
    # create nodes position based on the DR coordinates
    pos = {}
    for n in range(len(y)):
        pos[n] = (y[n][0], y[n][1])

    # creating the graph
    g = nx.Graph()

    for i in range(len(y)):
        g.add_node(i)

    plt.figure(figsize=(6, 6))

    nodes = nx.draw_networkx_nodes(g,
                                   pos=pos,
                                   node_color=label,
                                   cmap=cmap,
                                   node_size=15,
                                   linewidths=0.25)

    if color_bar_title is not None:
        cb = plt.colorbar(nodes,
                          orientation='horizontal',
                          pad=0.025,
                          label=color_bar_title)

    ax = plt.gca()  # to get the current axis
    ax.collections[0].set_edgecolor("#000000")

    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def draw_graph_no_positions(g, y, labels, filename=None):
    pos = {}
    for n in range(len(y)):
        pos[str(n)] = (y[n][0], y[n][1])

    plt.figure(figsize=(6, 6))

    nodes = nx.draw_networkx_nodes(g,
                                   pos=pos,
                                   node_color=labels,
                                   cmap=plt.cm.Set1,
                                   node_size=15,
                                   linewidths=0.25)

    edges = nx.draw_networkx_edges(g,
                                   pos=pos,
                                   edge_color='silver',
                                   width=0.5)

    # cb = plt.colorbar(nodes,
    #                   orientation='horizontal',
    #                   pad=0.025,
    #                   label='Digits')

    ax = plt.gca()  # to get the current axis
    ax.collections[0].set_edgecolor("#000000")

    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
