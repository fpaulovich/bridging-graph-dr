# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright (c) 2024 Fernando V. Paulovich
# License: MIT

import networkx as nx
import numpy as np

import matplotlib.pyplot as plt
from techniques.knn_gd import knn_graph


def graph_color_clustering(filename_graph_topology, filename_graph_position, filename_fig):
    g_top = nx.read_graphml(filename_graph_topology)
    g_pos = nx.read_graphml(filename_graph_position)

    pos = {}
    for n, data in g_pos.nodes.items():
        pos[n] = (data['x'], data['y'])

    plt.figure(figsize=(6, 7))

    nodes = nx.draw_networkx_nodes(g_pos,
                                   pos=pos,
                                   node_color=list(map(float, nx.clustering(g_top).values())),
                                   cmap=plt.cm.cividis_r,
                                   node_size=25)

    edges = nx.draw_networkx_edges(g_pos,
                                   pos=pos,
                                   edge_color='silver',
                                   width=0.5)

    plt.colorbar(nodes,
                 orientation='horizontal',
                 label='Clustering coefficient')

    ax = plt.gca()  # to get the current axis
    ax.collections[0].set_edgecolor("#000000")

    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)

    plt.savefig(filename_fig, dpi=400, bbox_inches='tight')
    plt.close()


def graph_color_closeness_centrality(filename_graph_topology, filename_graph_position, filename_fig):
    g_top = nx.read_graphml(filename_graph_topology)
    g_pos = nx.read_graphml(filename_graph_position)

    pos = {}
    for n, data in g_pos.nodes.items():
        pos[n] = (data['x'], data['y'])

    plt.figure(figsize=(6, 7))

    nodes = nx.draw_networkx_nodes(g_pos,
                                   pos=pos,
                                   node_color=list(map(float, nx.closeness_centrality(g_top).values())),
                                   cmap=plt.cm.cividis_r,
                                   node_size=25)

    edges = nx.draw_networkx_edges(g_pos,
                                   pos=pos,
                                   edge_color='silver',
                                   width=0.5)

    plt.colorbar(nodes,
                 orientation='horizontal',
                 label='Closeness centrality')

    ax = plt.gca()  # to get the current axis
    ax.collections[0].set_edgecolor("#000000")

    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)

    plt.savefig(filename_fig, dpi=400, bbox_inches='tight')
    plt.close()


def faithfulness_graph_topologies():
    filename_g1 = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_gd_umap.graphml'
    filename_g2 = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_gd_tsne.graphml'
    filename_g3 = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_gd_knn.graphml'

    g_umap = nx.read_graphml(filename_g1)
    g_tsne = nx.read_graphml(filename_g2)
    g_knn = nx.read_graphml(filename_g3)

    print("jaccard index (tsne, umap): ",
          len(nx.intersection(g_tsne, g_umap).edges) / len(nx.compose(g_tsne, g_umap).edges))
    print("jaccard index (tsne, knn): ",
          len(nx.intersection(g_tsne, g_knn).edges) / len(nx.compose(g_tsne, g_knn).edges))
    print("jaccard index (umap, knn): ",
          len(nx.intersection(g_umap, g_knn).edges) / len(nx.compose(g_umap, g_knn).edges))


def run_1():
    print('SNN')
    filename_graph_topology = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_gd_snn.graphml'
    filename_fig_clustering = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_gd_snn_clustering.png'
    filename_fig_centrality = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_gd_snn_closeness_centrality.png'
    graph_color_clustering(filename_graph_topology=filename_graph_topology,
                           filename_graph_position=filename_graph_topology,
                           filename_fig=filename_fig_clustering)
    graph_color_closeness_centrality(filename_graph_topology=filename_graph_topology,
                                     filename_graph_position=filename_graph_topology,
                                     filename_fig=filename_fig_centrality)

    print('KNN')
    filename_graph_topology = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_gd_knn.graphml'
    filename_fig_clustering = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_gd_knn_clustering.png'
    filename_fig_centrality = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_gd_knn_closeness_centrality.png'
    graph_color_clustering(filename_graph_topology=filename_graph_topology,
                           filename_graph_position=filename_graph_topology,
                           filename_fig=filename_fig_clustering)
    graph_color_closeness_centrality(filename_graph_topology=filename_graph_topology,
                                     filename_graph_position=filename_graph_topology,
                                     filename_fig=filename_fig_centrality)

    print('t-SNE')
    filename_graph_topology = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_gd_tsne.graphml'
    filename_fig_clustering = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_gd_tsne_clustering.png'
    filename_fig_centrality = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_gd_tsne_closeness_centrality.png'
    graph_color_clustering(filename_graph_topology=filename_graph_topology,
                           filename_graph_position=filename_graph_topology,
                           filename_fig=filename_fig_clustering)
    graph_color_closeness_centrality(filename_graph_topology=filename_graph_topology,
                                     filename_graph_position=filename_graph_topology,
                                     filename_fig=filename_fig_centrality)

    print('UMAP')
    filename_graph_topology = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_gd_umap.graphml'
    filename_fig_clustering = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_gd_umap_clustering.png'
    filename_fig_centrality = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_gd_umap_closeness_centrality.png'
    graph_color_clustering(filename_graph_topology=filename_graph_topology,
                           filename_graph_position=filename_graph_topology,
                           filename_fig=filename_fig_clustering)
    graph_color_closeness_centrality(filename_graph_topology=filename_graph_topology,
                                     filename_graph_position=filename_graph_topology,
                                     filename_fig=filename_fig_centrality)


def run_2():
    print('t-SNE')
    filename_graph_topology = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_gd_tsne.graphml'
    filename_graph_position = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_tsne.graphml'

    filename_fig_clustering = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_tsne_clustering.png'
    filename_fig_centrality = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_tsne_closeness_centrality.png'
    graph_color_clustering(filename_graph_topology=filename_graph_topology,
                           filename_graph_position=filename_graph_position,
                           filename_fig=filename_fig_clustering)
    graph_color_closeness_centrality(filename_graph_topology=filename_graph_topology,
                                     filename_graph_position=filename_graph_position,
                                     filename_fig=filename_fig_centrality)

    print('UMAP')
    filename_graph_topology = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_gd_umap.graphml'
    filename_graph_position = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_umap.graphml'

    filename_fig_clustering = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_umap_clustering.png'
    filename_fig_centrality = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_umap_closeness_centrality.png'
    graph_color_clustering(filename_graph_topology=filename_graph_topology,
                           filename_graph_position=filename_graph_position,
                           filename_fig=filename_fig_clustering)
    graph_color_closeness_centrality(filename_graph_topology=filename_graph_topology,
                                     filename_graph_position=filename_graph_position,
                                     filename_fig=filename_fig_centrality)


def faithfulness(filename_graph_topology, filename_graph_position, nr_neighbors):
    g_top = nx.read_graphml(filename_graph_topology, node_type=int)
    g_pos = nx.read_graphml(filename_graph_position, node_type=int)

    # getting positions in 2D
    pos_2d = np.zeros([len(g_pos.nodes), 2])
    for n, data in g_pos.nodes.items():
        pos_2d[int(n)][0] = data['x']
        pos_2d[int(n)][1] = data['y']

    # creating knn graph from 2D data
    g_knn_2d = knn_graph(pos_2d,
                         nr_neighbors=nr_neighbors,
                         metric='euclidean')

    return len(nx.intersection(g_top, g_knn_2d).edges) / len(nx.compose(g_top, g_knn_2d).edges)


def faithfulness_layouts(filename_graph_position1, filename_graph_position2, nr_neighbors):
    g_pos1 = nx.read_graphml(filename_graph_position1, node_type=int)
    g_pos2 = nx.read_graphml(filename_graph_position2, node_type=int)

    # getting positions in 2D
    pos_2d_1 = np.zeros([len(g_pos1.nodes), 2])
    for n, data in g_pos1.nodes.items():
        pos_2d_1[int(n)][0] = data['x']
        pos_2d_1[int(n)][1] = data['y']

    pos_2d_2 = np.zeros([len(g_pos2.nodes), 2])
    for n, data in g_pos2.nodes.items():
        pos_2d_2[int(n)][0] = data['x']
        pos_2d_2[int(n)][1] = data['y']

    # creating knn graph from 2D data
    g_knn_2d_1 = knn_graph(pos_2d_1,
                           nr_neighbors=nr_neighbors,
                           metric='euclidean')

    g_knn_2d_2 = knn_graph(pos_2d_2,
                           nr_neighbors=nr_neighbors,
                           metric='euclidean')

    return len(nx.intersection(g_knn_2d_1, g_knn_2d_2).edges) / len(nx.compose(g_knn_2d_1, g_knn_2d_2).edges)


def run_faithfulness():
    print('faithfulness t-sne:',
          faithfulness(
              filename_graph_topology='/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_gd_tsne.graphml',
              filename_graph_position='/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_tsne.graphml',
              nr_neighbors=10))

    print('faithfulness umap:',
          faithfulness(
              filename_graph_topology='/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_gd_umap.graphml',
              filename_graph_position='/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_umap.graphml',
              nr_neighbors=10))

    print('faithfulness gd t-sne:',
          faithfulness(
              filename_graph_topology='/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_gd_tsne.graphml',
              filename_graph_position='/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_gd_tsne.graphml',
              nr_neighbors=10))

    print('faithfulness gd umap:',
          faithfulness(
              filename_graph_topology='/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_gd_umap.graphml',
              filename_graph_position='/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_gd_umap.graphml',
              nr_neighbors=10))

    print('faithfulness umap vs t-sne:',
          faithfulness_layouts(
              filename_graph_position1='/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_umap.graphml',
              filename_graph_position2='/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/digits_tsne.graphml',
              nr_neighbors=10))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_faithfulness()
    # graph_difference()
    # run_1()
    # run_2()
