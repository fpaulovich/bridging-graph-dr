# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright (c) 2024 Fernando V. Paulovich
# License: MIT

from sklearn.decomposition import PCA
import networkx as nx
import matplotlib.pyplot as plt

import math


def write_graphml(g, pos, filename):
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


def draw_graph(X, g, labels, filename):
    init_pos = init(X)
    size = len(X)

    # pos = nx.forceatlas2_layout(g,
    #                             pos=init_pos,
    #                             weight='weight',
    #                             seed=1,
    #                             max_iter=1000,
    #                             scaling_ratio=2,
    #                             dissuade_hubs=False,
    #                             linlog=False)

    pos = nx.fruchterman_reingold_layout(g,
                                         pos=init_pos,
                                         weight='weight',
                                         seed=1,
                                         iterations=1000,
                                         k=1 / math.sqrt(size))

    plt.figure(figsize=(6, 5))

    # drawing the graph
    nx.draw(g, pos=pos,
            with_labels=False,
            node_color=labels,
            cmap=plt.cm.tab10,
            node_size=25,
            edge_color='silver',
            width=0.5)
    ax = plt.gca()  # to get the current axis
    ax.collections[0].set_edgecolor("#000000")

    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)

    plt.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close()

    return pos
