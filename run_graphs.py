# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright (c) 2024 Fernando V. Paulovich
# License: MIT

from techniques.pairwise_gd import mds, gd_pairwise
from techniques.snn_gd import gd_snn
from techniques.knn_gd import gd_knn
from techniques.umap_gd import umap, gd_umap
from techniques.tsne_gd import tsne, gd_tsne

from sklearn import preprocessing
import sklearn.datasets as datasets

import networkx as nx

import matplotlib.pyplot as plt


def run_wine():
    raw = datasets.load_wine(as_frame=True)
    X = raw.data.to_numpy()
    X = preprocessing.MinMaxScaler().fit_transform(X)
    labels = raw.target.to_numpy()

    nr_neighbors = 10
    perplexity = 10

    dir_name = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/'

    print('MDS')
    mds(X=X,
        labels=labels,
        filename_fig=dir_name + 'wine_cmds.png')

    print('GD Pairwise')
    gd_pairwise(X=X,
                labels=labels,
                filename_fig=dir_name + 'wine_gd_pairwise.png',
                filename_graph=dir_name + 'wine_gd_pairwise.graphml')

    print('GD SNN')
    gd_snn(X=X,
           labels=labels,
           nr_neighbors=nr_neighbors,
           filename_fig=dir_name + 'wine_gd_snn.png',
           filename_graph=dir_name + 'wine_gd_snn.graphml')

    print('GD KNN')
    gd_knn(X=X,
           labels=labels,
           nr_neighbors=nr_neighbors,
           filename_fig=dir_name + 'wine_gd_knn.png',
           filename_graph=dir_name + 'wine_gd_knn.graphml')

    print('UMAP')
    umap(X=X,
         labels=labels,
         nr_neighbors=nr_neighbors,
         filename_fig=dir_name + 'wine_umap.png')

    print('GD UMAP')
    gd_umap(X=X,
            labels=labels,
            nr_neighbors=nr_neighbors,
            filename_fig=dir_name + 'wine_gd_umap.png',
            filename_graph=dir_name + 'wine_gd_umap.graphml')

    print('t-SNE')
    tsne(X=X,
         labels=labels,
         perplexity=perplexity,
         filename_fig=dir_name + 'wine_tsne.png'
         )

    print('GD t-SNE')
    gd_tsne(X=X,
            labels=labels,
            perplexity=perplexity,
            filename_fig=dir_name + 'wine_gd_tsne.png',
            filename_graph=dir_name + 'wine_gd_tsne.graphml'
            )


def run_digits():
    raw = datasets.load_digits(as_frame=True)
    X = raw.data.to_numpy()
    X = preprocessing.MinMaxScaler().fit_transform(X)
    labels = raw.target.to_numpy()

    nr_neighbors = 15
    perplexity = 15

    dir_name = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/'

    print('MDS')
    mds(X=X,
        labels=labels,
        filename_fig=dir_name + 'digits_cmds.png')

    print('GD Pairwise')
    gd_pairwise(X=X,
                labels=labels,
                filename_fig=dir_name + 'digits_gd_pairwise.png',
                filename_graph=dir_name + 'digits_gd_pairwise.graphml')

    print('GD SNN')
    gd_snn(X=X,
           labels=labels,
           nr_neighbors=nr_neighbors,
           filename_fig=dir_name + 'digits_gd_snn.png',
           filename_graph=dir_name + 'digits_gd_snn.graphml')

    print('GD KNN')
    gd_knn(X=X,
           labels=labels,
           nr_neighbors=nr_neighbors,
           filename_fig=dir_name + 'digits_gd_knn.png',
           filename_graph=dir_name + 'digits_gd_knn.graphml')

    print('UMAP')
    umap(X=X,
         labels=labels,
         nr_neighbors=nr_neighbors,
         filename_fig=dir_name + 'digits_umap.png',
         filename_graph=dir_name + 'digits_umap.graphml')

    print('GD UMAP')
    gd_umap(X=X,
            labels=labels,
            nr_neighbors=nr_neighbors,
            filename_fig=dir_name + 'digits_gd_umap.png',
            filename_graph=dir_name + 'digits_gd_umap.graphml')

    print('t-SNE')
    tsne(X=X,
         labels=labels,
         perplexity=perplexity,
         filename_fig=dir_name + 'digits_tsne.png',
         filename_graph=dir_name + 'digits_tsne.graphml'
         )

    print('GD t-SNE')
    gd_tsne(X=X,
            labels=labels,
            perplexity=perplexity,
            filename_fig=dir_name + 'digits_gd_tsne.png',
            filename_graph=dir_name + 'digits_gd_tsne.graphml'
            )


def draw_graphs():
    dir_name = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/'
    filename_fig = dir_name + 'digits_gd_tsne.png'
    filename_graph = dir_name + 'digits_gd_tsne.graphml'

    g = nx.read_graphml(filename_graph)

    pos = {}
    for n, data in g.nodes.items():
        pos[n] = (data['x'], data['y'])

    plt.figure(figsize=(6, 6))

    nodes = nx.draw_networkx_nodes(g,
                                   pos=pos,
                                   node_color=list(map(int, nx.get_node_attributes(g, 'label').values())),
                                   cmap=plt.cm.tab10,
                                   node_size=25)

    edges = nx.draw_networkx_edges(g,
                                   pos=pos,
                                   edge_color='silver',
                                   width=0.5)

    cb = plt.colorbar(nodes,
                      orientation='horizontal',
                      pad=0.025,
                      label='Digits')

    ax = plt.gca()  # to get the current axis
    ax.collections[0].set_edgecolor("#000000")

    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)

    plt.show()

    # plt.savefig(filename, dpi=400, bbox_inches='tight')
    # plt.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # run_wine()
    # run_digits()
    draw_graphs()
