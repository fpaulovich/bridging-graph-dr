import math

import numpy as np

from techniques.t_sne import TSNE

import matplotlib.pyplot as plt

from timeit import default_timer as timer
from datetime import timedelta

import networkx as nx
from scipy.sparse import csr_matrix

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.manifold import trustworthiness

from techniques.tsne_gd import gd_tsne

from metrics.local import sortedness
from scipy.stats import weightedtau

import sklearn.datasets as datasets
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA

from util import draw_graph, write_graphml

MACHINE_EPSILON = np.finfo(np.double).eps

from scipy.spatial.distance import pdist
import scipy.stats


def stress(X, y, metric='euclidean'):
    D_high = pdist(X, metric=metric)
    D_low = pdist(y, metric=metric)
    return math.sqrt(np.average(((D_high - D_low) ** 2) / np.sum(D_high ** 2)))


def neighborhood_preservation(X, y, nr_neighbors=10, metric='euclidean'):
    dists_high, indexes_high = KDTree(X, leaf_size=2, metric=metric).query(X, k=nr_neighbors)
    dists_low, indexes_low = KDTree(y, leaf_size=2, metric=metric).query(y, k=nr_neighbors)

    neigh_pres = np.zeros(len(X))
    for i in range(len(X)):
        for p in range(nr_neighbors):
            for q in range(nr_neighbors):
                if indexes_high[i][p] == indexes_low[i][q]:
                    neigh_pres[i] = neigh_pres[i] + 1
        neigh_pres[i] = neigh_pres[i] / nr_neighbors

    return np.average(neigh_pres)


def neighborhood_hit(y, label, nr_neighbors=10, metric='euclidean'):
    dists_low, indexes_low = KDTree(y, leaf_size=2, metric=metric).query(y, k=nr_neighbors)

    neigh_hit = np.zeros(len(y))
    for i in range(len(y)):
        for j in range(nr_neighbors):
            if label[i] == label[indexes_low[i][j]]:
                neigh_hit[i] = neigh_hit[i] + 1
        neigh_hit[i] = neigh_hit[i] / nr_neighbors

    return np.average(neigh_hit)


def load_data(dataset):
    X = np.load(dataset + "/X.npy", allow_pickle=True).astype(np.float64)
    y = np.load(dataset + "/y.npy", allow_pickle=True).astype(np.int64)
    return X, y


def generate_all_tsne_graphs():
    dir_base = '/Users/fpaulovich/Documents/data/'
    datasets = ['bank', 'cifar10', 'cnae9', 'coil20', 'epileptic', 'fashion_mnist', 'fmd', 'har', 'hatespeech',
                'hiva', 'imdb', 'orl', 'secom', 'seismic', 'sentiment', 'sms', 'spambase', 'svhn']

    perplexity = {'bank': 30, 'cifar10': 15, 'cnae9': 5, 'coil20': 50, 'epileptic': 50, 'fashion_mnist': 50,
                  'fmd': 50, 'har': 30, 'hatespeech': 30, 'hiva': 50, 'imdb': 50, 'orl': 15, 'secom': 30, 'seismic': 50,
                  'sentiment': 15, 'sms': 50, 'spambase': 5, 'svhn': 15}

    dir_name_output = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/survey_dr/tsne/'

    for dataset in datasets:
        print('>>>processing:', dataset)
        X, y = load_data(dir_base + dataset)
        X = MinMaxScaler().fit_transform(X)
        y = LabelEncoder().fit_transform(y)

        gd_tsne(X=X,
                labels=y,
                perplexity=int(perplexity[dataset]),
                filename_fig=dir_name_output + dataset + '-gd_tsne.png',
                filename_graph=dir_name_output + dataset + '-tsne.graphml'
                )
    return


def draw_graph_by_tsne(X, g):
    label = list(map(int, nx.get_node_attributes(g, 'label').values()))
    weight = nx.get_edge_attributes(g, name='weight')

    row = np.zeros(g.number_of_edges())
    col = np.zeros(g.number_of_edges())
    data = np.zeros(g.number_of_edges())

    k = 0
    for i, j in g.edges():
        row[k] = i
        col[k] = j
        data[k] = weight[(i, j)]
        k = k + 1

    P = csr_matrix((data, (row, col)), shape=(g.number_of_nodes(), g.number_of_nodes()))
    P = P + P.T
    P /= np.maximum(P.sum(), MACHINE_EPSILON)

    start = timer()
    y = TSNE(n_components=2,
             perplexity=2,
             metric='euclidean',
             random_state=42,
             method='barnes_hut',
             # init='random',
             init=PCA(n_components=2).fit_transform(X),
             probabilities=P).fit_transform(np.zeros((g.number_of_nodes(), 2)))
    end = timer()
    print('t-SNE took {0} to execute'.format(timedelta(seconds=end - start)))

    return y, label


# def draw_all_graphs_by_tsne():
#     dir_base = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/survey_dr/tsne/'
#     datasets = ['bank', 'cifar10', 'cnae9', 'coil20', 'epileptic', 'fashion_mnist', 'fmd', 'har', 'hatespeech',
#                 'hiva', 'imdb', 'orl', 'secom', 'seismic', 'sentiment', 'sms', 'spambase', 'svhn']
#
#     for dataset in datasets:
#         filename_graph = dir_base + dataset + '-tsne.graphml'
#         filename_fig = dir_base + dataset + '-tsne.png'
#         g = nx.read_graphml(filename_graph)
#         y, label = draw_graph_by_tsne(g)
#
#         plt.figure()
#         plt.scatter(y[:, 0], y[:, 1], c=label,
#                     cmap='Set1', edgecolors='face', linewidths=0.5, s=4)
#
#         if filename_fig is not None:
#             plt.savefig(filename_fig, dpi=400, bbox_inches='tight')
#         else:
#             plt.show()
#         plt.close()
#
#         return


def calculate_metrics_original_techniques():
    dir_base = '/Users/fpaulovich/Documents/data/'
    datasets = ['bank', 'cifar10', 'cnae9', 'coil20', 'epileptic', 'fashion_mnist', 'fmd', 'har', 'hatespeech',
                'hiva', 'imdb', 'orl', 'secom', 'seismic', 'sentiment', 'sms', 'spambase', 'svhn']

    perplexity = {'bank': 30, 'cifar10': 15, 'cnae9': 5, 'coil20': 50, 'epileptic': 50, 'fashion_mnist': 50,
                  'fmd': 50, 'har': 30, 'hatespeech': 30, 'hiva': 50, 'imdb': 50, 'orl': 15, 'secom': 30, 'seismic': 50,
                  'sentiment': 15, 'sms': 50, 'spambase': 5, 'svhn': 15}

    for dataset in datasets:
        print('>>>processing:', dataset)
        X, y = load_data(dir_base + dataset)
        X = MinMaxScaler().fit_transform(X)
        # y = LabelEncoder().fit_transform(y)

        y_tsne = TSNE(n_components=2,
                      perplexity=int(perplexity[dataset]),
                      metric='euclidean',
                      random_state=42,
                      method='barnes_hut',
                      init='random').fit_transform(X)

        trust = trustworthiness(X, y_tsne, n_neighbors=int(perplexity[dataset]))
        print('trustworthiness: ', trust)

        sort = sortedness(X, y_tsne, f=weightedtau)
        print('sortedness:', np.average(sort))

    return


# def remove_edges_by_centrality(g, percentage_edges_to_keep=0.9):
#     # # create an edge attribute to inverse weight
#     # weight = nx.get_edge_attributes(g, 'weight')
#     # for key in weight.keys():
#     #     weight[key] = 1 - weight[key]
#     # nx.set_edge_attributes(g, weight, 'length')
#
#     centrality = nx.edge_betweenness_centrality(g, weight='weight')
#
#     centrality = {k: v for k, v in sorted(centrality.items(), key=lambda item: item[1])}
#     nr_original_edges = len(centrality)
#     nr_edges_to_keep = int(nr_original_edges * percentage_edges_to_keep)
#     keys = list(centrality.keys())
#
#     edges_to_remove = []
#     for i in range(nr_edges_to_keep, nr_original_edges):
#         edges_to_remove.append(keys[i])
#
#     for i, j in edges_to_remove:
#         if g.has_edge(i, j):
#             g.remove_edge(i, j)
#
#         if g.has_edge(j, i):
#             g.remove_edge(i, j)
#
#     print('number original edges:', nr_original_edges)
#     print('removed: ', len(edges_to_remove))
#     print('percentage: ', g.number_of_edges() / nr_original_edges)
#
#     y, label = draw_graph_by_tsne(g)
#
#     return y, label, g


# def run_pipeline():
#     dir_base_dataset = '/Users/fpaulovich/Documents/data/'
#     dir_base_graph = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/survey_dr/tsne/'
#
#     # datasets = ['bank', 'cifar10', 'cnae9', 'coil20', 'epileptic', 'fashion_mnist', 'fmd', 'har', 'hatespeech',
#     #             'hiva', 'imdb', 'orl', 'secom', 'seismic', 'sentiment', 'sms', 'spambase', 'svhn']
#     datasets = ['bank', 'cifar10', 'cnae9', 'coil20', 'fashion_mnist', 'fmd', 'har', 'hatespeech',
#                 'hiva', 'imdb', 'orl', 'secom', 'seismic', 'sentiment', 'sms', 'spambase', 'svhn']
#
#     perplexity = {'bank': 30, 'cifar10': 15, 'cnae9': 5, 'coil20': 50, 'epileptic': 50, 'fashion_mnist': 50,
#                   'fmd': 50, 'har': 30, 'hatespeech': 30, 'hiva': 50, 'imdb': 50, 'orl': 15, 'secom': 30, 'seismic': 50,
#                   'sentiment': 15, 'sms': 50, 'spambase': 5, 'svhn': 15}
#
#     for dataset in datasets:
#         print('--')
#         print('>>>processing:', dataset)
#         X, _ = load_data(dir_base_dataset + dataset)
#         X = MinMaxScaler().fit_transform(X)
#
#         # reduce edges by centrality
#         filename_graph = dir_base_graph + dataset + '-tsne.graphml'
#         g = nx.read_graphml(filename_graph)
#         y, label, g_reduced = remove_edges_by_centrality(g, percentage_edges_to_keep=0.8)
#
#         # save reduced graph
#         filename_graph_reduced = dir_base_graph + 'reduced/' + dataset + '-reduced_tsne.graphml'
#         for node in g_reduced.nodes:
#             g_reduced.nodes[node]['x'] = float(y[int(node)][0])
#             g_reduced.nodes[node]['y'] = float(y[int(node)][1])
#         nx.write_graphml(g_reduced, filename_graph_reduced, named_key_ids=True)
#
#         # save reduced DR figure
#         filename_fig_reduced = dir_base_graph + 'reduced/' + dataset + '-reduced_tsne.png'
#         plt.figure()
#         plt.scatter(y[:, 0], y[:, 1], c=label,
#                     cmap='Set1', edgecolors='face', linewidths=0.5, s=4)
#         plt.savefig(filename_fig_reduced, dpi=400, bbox_inches='tight')
#         plt.close()
#
#         # calculate metrics
#         trust = trustworthiness(X, y, n_neighbors=int(perplexity[dataset]))
#         print('trustworthiness: ', trust)
#
#         sort = sortedness(X, y, f=weightedtau)
#         print('sortedness:', np.average(sort))


# def run_one_dataset_pipeline():
#     dir_base_graph = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/'
#     filename_graph = dir_base_graph + 'digits_gd_tsne.graphml'
#
#     percentages = [1.00, 0.95, 0.90]
#
#     raw = datasets.load_digits(as_frame=True)
#     X = raw.data.to_numpy()
#     X = preprocessing.MinMaxScaler().fit_transform(X)
#
#     for percentage in percentages:
#         print('---')
#         print('percentage: ', percentage)
#         g = nx.read_graphml(filename_graph)
#         y, label, _ = remove_edges_by_centrality(g, percentage_edges_to_keep=percentage)
#
#         # save reduced DR figure
#         filename_fig_reduced = dir_base_graph + 'survey_dr/tsne/reduced/digits-[' + str(
#             percentage) + ']-reduced_tsne.png'
#         plt.figure()
#         plt.scatter(y[:, 0], y[:, 1], c=label,
#                     cmap='Set1', edgecolors='face', linewidths=0.5, s=4)
#         plt.savefig(filename_fig_reduced, dpi=400, bbox_inches='tight')
#         plt.close()
#
#         # calculate metrics
#         dr_metric = sortedness(X, y, f=weightedtau)
#         print('sortedness after:', np.average(dr_metric))
#         dr_metric = trustworthiness(X, y, n_neighbors=7)
#         print('trustworthiness after:', dr_metric)
#         dr_metric = stress(X, y, metric='euclidean')
#         print('stress after:', dr_metric)
#         dr_metric = silhouette_score(X, label)
#         print('silhouette_score after:', dr_metric)
#         dr_metric = neighborhood_preservation(X, y, nr_neighbors=7)
#         print('neighborhood_preservation after:', dr_metric)
#         dr_metric = neighborhood_hit(y, label, nr_neighbors=7)
#         print('neighborhood_hit after:', dr_metric)


# def calculate_metrics_graphs():
#     dir_base_graph = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/'
#     filename_graph = dir_base_graph + 'digits_gd_tsne.graphml'
#     g = nx.read_graphml(filename_graph)
#
#     # draw graph using tsne
#     raw = datasets.load_digits(as_frame=True)
#     X = raw.data.to_numpy()
#     X = preprocessing.MinMaxScaler().fit_transform(X)
#     y, label = draw_graph_by_tsne(g)
#
#     # calculate DR metrics
#     dr_metric = sortedness(X, y, f=weightedtau)
#     # dr_metric = stress(X, y)
#     print('dr_metric:', np.average(dr_metric))
#
#     # calculate graph metric
#     graph_metric = nx.degree_centrality(g)
#     graph_metric = list(map(float, graph_metric.values()))
#
#     print('correlation: ', scipy.stats.pearsonr(graph_metric, dr_metric))
#
#     return


# def removing_nodes():
#     dir_base_graph = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/'
#     filename_graph = dir_base_graph + 'iris_gd_tsne.graphml'
#     g = nx.read_graphml(filename_graph)
#     y, label = draw_graph_by_tsne(g)
#
#     raw = datasets.load_iris(as_frame=True)
#     X = raw.data.to_numpy()
#     X = preprocessing.MinMaxScaler().fit_transform(X)
#
#     # dir_base_graph = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/survey_dr/tsne/'
#     # filename_graph = dir_base_graph + 'fashion_mnist-tsne.graphml'
#     # g = nx.read_graphml(filename_graph)
#     # y, label = draw_graph_by_tsne(g)
#     #
#     # dir_base = '/Users/fpaulovich/Documents/data/'
#     # dataset = 'fashion_mnist'
#     # X, _ = load_data(dir_base + dataset)
#     # X = MinMaxScaler().fit_transform(X)
#
#     ##########################################################
#
#     dr_metric = sortedness(X, y, f=weightedtau)
#     print('sortedness before:', np.average(dr_metric))
#     dr_metric = trustworthiness(X, y, n_neighbors=10)
#     print('trustworthiness before:', dr_metric)
#     dr_metric = stress(X, y, metric='euclidean')
#     print('stress before:', dr_metric)
#     dr_metric = silhouette_score(y, label)
#     print('silhouette_score before:', dr_metric)
#     dr_metric = neighborhood_preservation(X, y, nr_neighbors=10)
#     print('neighborhood_preservation before:', dr_metric)
#     dr_metric = neighborhood_hit(y, label, nr_neighbors=10)
#     print('neighborhood_hit before:', dr_metric)
#
#     graph_metric = nx.closeness_centrality(g)
#     graph_metric = {k: v for k, v in sorted(graph_metric.items(), key=lambda item: item[1])}
#
#     print('----')
#     print('number of nodes before: ', g.number_of_nodes())
#     print('number of edges before: ', g.number_of_edges())
#
#     number_nodes = g.number_of_nodes()
#     percentage_to_keep = 0.85
#     to_remove = []
#     keys = list(graph_metric.keys())
#     for removed in range(int(number_nodes * percentage_to_keep), number_nodes):
#         g.remove_node(keys[removed])
#         to_remove.append(int(keys[removed]))
#     g = nx.convert_node_labels_to_integers(g, first_label=0)
#
#     print('number of nodes after: ', g.number_of_nodes())
#     print('number of edges after: ', g.number_of_edges())
#     print('----')
#
#     y_removed, label_removed = draw_graph_by_tsne(g)
#
#     # remove instances from the dataset
#     X_removed = np.delete(X, to_remove, axis=0)
#
#     ######
#     # put back the removed nodes
#     #
#     tree = KDTree(X_removed, leaf_size=2, metric='euclidean')
#     nr_neighbors = 1
#
#     X_recovered = np.copy(X_removed)
#     labels_recovered = np.copy(label_removed)
#     y_recovered = np.copy(y_removed)
#
#     for removed in to_remove:
#         # look for nearest neighbors inside the dataset with removed instances
#         dists, indexes = tree.query(X[removed].reshape(1, -1), k=nr_neighbors)
#         weights = 1.0 / (dists[0] + 0.0001)  # np.full(nr_neighbors, (1.0 / nr_neighbors)) #
#         weights = weights / np.sum(weights)
#
#         p = np.dot(weights.T, y_removed[indexes[0]])
#
#         X_recovered = np.append(X_recovered, [X[removed]], axis=0)
#         labels_recovered = np.append(labels_recovered, [label[removed]], axis=0)
#         y_recovered = np.append(y_recovered, [p], axis=0)
#     #
#     #####
#
#     dr_metric = sortedness(X_recovered, y_recovered, f=weightedtau)
#     print('sortedness after:', np.average(dr_metric))
#     dr_metric = trustworthiness(X_recovered, y_recovered, n_neighbors=10)
#     print('trustworthiness after:', dr_metric)
#     dr_metric = stress(X_recovered, y_recovered, metric='euclidean')
#     print('stress after:', dr_metric)
#     dr_metric = silhouette_score(y_recovered, labels_recovered)
#     print('silhouette_score after:', dr_metric)
#     dr_metric = neighborhood_preservation(X_recovered, y_recovered, nr_neighbors=10)
#     print('neighborhood_preservation after:', dr_metric)
#     dr_metric = neighborhood_hit(y_recovered, labels_recovered, nr_neighbors=10)
#     print('neighborhood_hit after:', dr_metric)
#
#     plt.figure()
#     plt.scatter(y_recovered[:, 0], y_recovered[:, 1], c=labels_recovered, cmap='Set1', edgecolors='face',
#                 linewidths=0.5, s=4)
#     # plt.savefig(filename_fig_reduced, dpi=400, bbox_inches='tight')
#     plt.show()
#     plt.close()
#
#     return


def remove_nodes_centrality(X, g, nodes_to_keep=0.8):
    # draw the graph using t-SNE
    y, label = draw_graph_by_tsne(X, g)

    ###################################
    # calculate metrics original
    #
    dr_metric = sortedness(X, y, f=weightedtau)
    print('sortedness before:', np.average(dr_metric))
    dr_metric = trustworthiness(X, y, n_neighbors=7)
    print('trustworthiness before:', dr_metric)
    dr_metric = stress(X, y, metric='euclidean')
    print('stress before:', dr_metric)
    dr_metric = silhouette_score(y, label)
    print('silhouette_score before:', dr_metric)
    dr_metric = neighborhood_preservation(X, y, nr_neighbors=7)
    print('neighborhood_preservation before:', dr_metric)
    dr_metric = neighborhood_hit(y, label, nr_neighbors=7)
    print('neighborhood_hit before:', dr_metric)
    print('---')

    ###################################
    # creating the indexes to remove
    #
    graph_metric = nx.closeness_centrality(g)
    graph_metric = {k: v for k, v in sorted(graph_metric.items(), key=lambda item: item[1])}

    number_nodes = g.number_of_nodes()
    number_nodes_to_keep = int(number_nodes * nodes_to_keep)
    to_remove = []
    keys = list(graph_metric.keys())
    for removed in range(number_nodes_to_keep, number_nodes):
        g.remove_node(keys[removed])
        to_remove.append(int(keys[removed]))
    g = nx.convert_node_labels_to_integers(g, first_label=0)
    ###################################

    ###################################
    # remove instances
    #
    X_removed = np.delete(X, to_remove, axis=0)
    label_removed = np.delete(label, to_remove, axis=0)
    ###################################

    ###################################
    # project reduced
    #
    # y_removed, _ = draw_graph_by_tsne(g)
    y_removed = TSNE(n_components=2,
                     perplexity=50,
                     metric='euclidean',
                     random_state=42,
                     method='barnes_hut',
                     # init='random'
                     init=PCA(n_components=2).fit_transform(X_removed)
                     ).fit_transform(X_removed)
    ###################################

    ###################################
    # calculate metrics reduced
    #
    dr_metric = sortedness(X_removed, y_removed, f=weightedtau)
    print('sortedness after:', np.average(dr_metric))
    dr_metric = trustworthiness(X_removed, y_removed, n_neighbors=7)
    print('trustworthiness after:', dr_metric)
    dr_metric = stress(X_removed, y_removed, metric='euclidean')
    print('stress after:', dr_metric)
    dr_metric = silhouette_score(y_removed, label_removed)
    print('silhouette_score after:', dr_metric)
    dr_metric = neighborhood_preservation(X_removed, y_removed, nr_neighbors=7)
    print('neighborhood_preservation after:', dr_metric)
    dr_metric = neighborhood_hit(y_removed, label_removed, nr_neighbors=7)
    print('neighborhood_hit after:', dr_metric)

    return y_removed, label_removed


def run_remove_nodes_centrality():
    # raw = datasets.load_digits(as_frame=True)
    # X = raw.data.to_numpy()
    # X = preprocessing.MinMaxScaler().fit_transform(X)
    #
    # dir_base_graph = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/'
    # filename_graph = dir_base_graph + 'digits_gd_tsne.graphml'
    # g = nx.read_graphml(filename_graph)

    dir_base_graph = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/survey_dr/tsne/'
    filename_graph = dir_base_graph + 'fashion_mnist-tsne.graphml'
    g = nx.read_graphml(filename_graph)

    dir_base = '/Users/fpaulovich/Documents/data/'
    dataset = 'fashion_mnist'
    X, _ = load_data(dir_base + dataset)
    X = MinMaxScaler().fit_transform(X)

    # remove nodes
    y_removed, label_removed = remove_nodes_centrality(X, g, nodes_to_keep=0.8)

    plt.figure()
    plt.scatter(y_removed[:, 0], y_removed[:, 1], c=label_removed, cmap='Set1', edgecolors='face',
                linewidths=0.5, s=4)
    # plt.savefig(filename_fig_reduced, dpi=400, bbox_inches='tight')
    plt.show()
    plt.close()


def run_remove_nodes_centrality_batch():
    dir_base_dataset = '/Users/fpaulovich/Documents/data/'
    dir_base_graph = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/survey_dr/tsne/'

    datasets = ['bank', 'cifar10', 'cnae9', 'coil20', 'fashion_mnist', 'fmd', 'har', 'hatespeech',
                'hiva', 'imdb', 'orl', 'secom', 'seismic', 'sentiment', 'sms', 'spambase', 'svhn']

    percentages = [0.95, 0.90, 0.85, 0.80, 0.75, 0.7]

    for percentage in percentages:
        print('--')
        print('--')
        print('--')
        print('>> percentage: ', percentage)

        for dataset in datasets:
            print('--')
            print('--')
            print('>>>processing:', dataset)

            # read the dataset
            X, _ = load_data(dir_base_dataset + dataset)
            X = MinMaxScaler().fit_transform(X)

            # read the graph
            filename_graph = dir_base_graph + dataset + '-tsne.graphml'
            g = nx.read_graphml(filename_graph)

            # remove nodes by centrality
            y_removed, label_removed = remove_nodes_centrality(X, g, nodes_to_keep=percentage)

            filename_fig_reduced = dir_base_graph + 'reduced/' + dataset + '[' + str(percentage) + ']-reduced_tsne.png'

            plt.figure()
            plt.scatter(y_removed[:, 0], y_removed[:, 1], c=label_removed, cmap='Set1', edgecolors='face',
                        linewidths=0.5, s=4)
            plt.savefig(filename_fig_reduced, dpi=400, bbox_inches='tight')
            plt.close()


if __name__ == '__main__':
    # generate_all_tsne_graphs()
    # draw_all_graphs_by_tsne()
    # calculate_metrics_original_techniques()
    # run_pipeline()
    # run_one_dataset_pipeline()
    # calculate_metrics_graphs()
    # run_remove_nodes_centrality()
    run_remove_nodes_centrality_batch()