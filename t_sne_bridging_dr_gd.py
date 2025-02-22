import math

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from techniques.t_sne import TSNE
from techniques.tsne_gd import gd_tsne

from timeit import default_timer as timer
from datetime import timedelta

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA

from metrics.local import sortedness

from scipy.stats import weightedtau
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist

import pandas as pd
import random

MACHINE_EPSILON = np.finfo(np.double).eps


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


def draw_graph_by_tsne(X, g):
    label = list(map(int, nx.get_node_attributes(g, 'label').values()))
    weight = nx.get_edge_attributes(g, name='weight')

    ###################################
    # get all edges from the graph
    #
    row = np.zeros(g.number_of_edges())
    col = np.zeros(g.number_of_edges())
    data = np.zeros(g.number_of_edges())

    k = 0
    for i, j in g.edges():
        row[k] = i
        col[k] = j
        data[k] = weight[(i, j)]
        k = k + 1

    ###################################
    # from the edges, create the probability matrix and endure sum=1
    #
    P = csr_matrix((data, (row, col)), shape=(g.number_of_nodes(), g.number_of_nodes()))
    P = P + P.T
    P /= np.maximum(P.sum(), MACHINE_EPSILON)

    ###################################
    # draw the probability matrix
    #
    start = timer()
    y = TSNE(n_components=2,
             perplexity=2,  # this is ignored
             metric='euclidean',
             random_state=42,
             method='barnes_hut',
             init=PCA(n_components=2).fit_transform(X),
             probabilities=P).fit_transform(np.zeros((g.number_of_nodes(), 2)))
    end = timer()
    print('t-SNE took {0} to execute'.format(timedelta(seconds=end - start)))

    return y, label


def remove_nodes_centrality(X, label, g, perplexity, nodes_to_keep=0.8):
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
                     perplexity=perplexity,
                     metric='euclidean',
                     random_state=42,
                     method='barnes_hut',
                     init=PCA(n_components=2).fit_transform(X_removed)
                     ).fit_transform(X_removed)
    ###################################

    ###################################
    # calculate metrics reduced
    #
    metrics = {}

    dr_metric = sortedness(X_removed, y_removed)
    print('sortedness after:', np.average(dr_metric))
    metrics.update({'sortedness': np.average(dr_metric)})

    dr_metric = sortedness(X_removed, y_removed, f=weightedtau)
    print('sortedness after (weightedtau):', np.average(dr_metric))
    metrics.update({'sortedness_weightedtau': np.average(dr_metric)})

    dr_metric = trustworthiness(X_removed, y_removed, n_neighbors=7)
    print('trustworthiness after:', dr_metric)
    metrics.update({'trustworthiness': np.average(dr_metric)})
    dr_metric = stress(X_removed, y_removed, metric='euclidean')

    print('stress after:', dr_metric)
    metrics.update({'stress': np.average(dr_metric)})
    dr_metric = silhouette_score(y_removed, label_removed)

    print('silhouette_score after:', dr_metric)
    metrics.update({'silhouette_score': np.average(dr_metric)})
    dr_metric = neighborhood_preservation(X_removed, y_removed, nr_neighbors=7)

    print('neighborhood_preservation after:', dr_metric)
    metrics.update({'neighborhood_preservation': np.average(dr_metric)})

    dr_metric = neighborhood_hit(y_removed, label_removed, nr_neighbors=7)
    print('neighborhood_hit after:', dr_metric)
    metrics.update({'neighborhood_hit': np.average(dr_metric)})

    return y_removed, label_removed, metrics


def run_generate_all_tsne_graphs():
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


def run_draw_all_graphs_by_tsne():
    dir_base = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/survey_dr/tsne/'
    datasets = ['bank', 'cifar10', 'cnae9', 'coil20', 'epileptic', 'fashion_mnist', 'fmd', 'har', 'hatespeech',
                'hiva', 'imdb', 'orl', 'secom', 'seismic', 'sentiment', 'sms', 'spambase', 'svhn']

    for dataset in datasets:
        filename_graph = dir_base + dataset + '-tsne.graphml'
        filename_fig = dir_base + dataset + '-tsne.png'
        g = nx.read_graphml(filename_graph)
        y, label = draw_graph_by_tsne(g)

        plt.figure()
        plt.scatter(y[:, 0], y[:, 1], c=label, cmap='Set1', edgecolors='face', linewidths=0.5, s=4)
        plt.savefig(filename_fig, dpi=400, bbox_inches='tight')
        plt.close()

    return


def run_calculate_metrics_original_techniques():
    dir_base = '/Users/fpaulovich/Documents/data/'
    datasets = ['bank', 'cifar10', 'cnae9', 'coil20', 'epileptic', 'fashion_mnist', 'fmd', 'har', 'hatespeech',
                'hiva', 'imdb', 'orl', 'secom', 'seismic', 'sentiment', 'sms', 'spambase', 'svhn']

    perplexity = {'bank': 30, 'cifar10': 15, 'cnae9': 5, 'coil20': 50, 'epileptic': 50, 'fashion_mnist': 50,
                  'fmd': 50, 'har': 30, 'hatespeech': 30, 'hiva': 50, 'imdb': 50, 'orl': 15, 'secom': 30, 'seismic': 50,
                  'sentiment': 15, 'sms': 50, 'spambase': 5, 'svhn': 15}

    for dataset in datasets:
        print('>>>processing:', dataset)
        X, label = load_data(dir_base + dataset)
        X = MinMaxScaler().fit_transform(X)
        label = LabelEncoder().fit_transform(label)

        y = TSNE(n_components=2,
                 perplexity=int(perplexity[dataset]),
                 metric='euclidean',
                 random_state=42,
                 method='barnes_hut',
                 init=PCA(n_components=2).fit_transform(X)
                 ).fit_transform(X)

        dr_metric = sortedness(X, y)
        print('sortedness:', np.average(dr_metric))
        dr_metric = sortedness(X, y, f=weightedtau)
        print('sortedness (weightedtau):', np.average(dr_metric))
        dr_metric = trustworthiness(X, y, n_neighbors=7)
        print('trustworthiness:', dr_metric)
        dr_metric = stress(X, y, metric='euclidean')
        print('stress:', dr_metric)
        dr_metric = silhouette_score(y, label)
        print('silhouette_score:', dr_metric)
        dr_metric = neighborhood_preservation(X, y, nr_neighbors=7)
        print('neighborhood_preservation:', dr_metric)
        dr_metric = neighborhood_hit(y, label, nr_neighbors=7)
        print('neighborhood_hit:', dr_metric)

    return


def run_remove_nodes_centrality():
    dir_base_graph = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/survey_dr/tsne/'
    filename_graph = dir_base_graph + 'fashion_mnist-tsne.graphml'

    dir_base = '/Users/fpaulovich/Documents/data/'
    dataset = 'fashion_mnist'

    perplexity = 50

    percentages = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]

    metrics_df = pd.DataFrame(columns=['percentage',
                                       'sortedness',
                                       'sortedness_weightedtau',
                                       'trustworthiness',
                                       'stress',
                                       'silhouette_score',
                                       'neighborhood_preservation',
                                       'neighborhood_hit'])

    # read the dataset
    X, _ = load_data(dir_base + dataset)
    X = MinMaxScaler().fit_transform(X)

    # read the graph
    g = nx.read_graphml(filename_graph)

    # draw the graph using t-SNE
    y, label = draw_graph_by_tsne(X, g)

    # calculate metrics for the original
    print('---')
    print('>>>original')
    metrics = {}

    dr_metric = sortedness(X, y)
    print('sortedness after:', np.average(dr_metric))
    metrics.update({'sortedness': np.average(dr_metric)})

    dr_metric = sortedness(X, y, f=weightedtau)
    print('sortedness after (weightedtau):', np.average(dr_metric))
    metrics.update({'sortedness_weightedtau': np.average(dr_metric)})

    dr_metric = trustworthiness(X, y, n_neighbors=7)
    print('trustworthiness after:', dr_metric)
    metrics.update({'trustworthiness': np.average(dr_metric)})
    dr_metric = stress(X, y, metric='euclidean')

    print('stress after:', dr_metric)
    metrics.update({'stress': np.average(dr_metric)})
    dr_metric = silhouette_score(y, label)

    print('silhouette_score after:', dr_metric)
    metrics.update({'silhouette_score': np.average(dr_metric)})
    dr_metric = neighborhood_preservation(X, y, nr_neighbors=7)

    print('neighborhood_preservation after:', dr_metric)
    metrics.update({'neighborhood_preservation': np.average(dr_metric)})

    dr_metric = neighborhood_hit(y, label, nr_neighbors=7)
    print('neighborhood_hit after:', dr_metric)
    metrics.update({'neighborhood_hit': np.average(dr_metric)})

    metrics_df.loc[len(metrics_df)] = [1.00,
                                       metrics['sortedness'],
                                       metrics['sortedness_weightedtau'],
                                       metrics['trustworthiness'],
                                       metrics['stress'],
                                       metrics['silhouette_score'],
                                       metrics['neighborhood_preservation'],
                                       metrics['neighborhood_hit']]

    for percentage in percentages:
        print('--')
        print('>> percentage: ', percentage)

        # read the graph
        g = nx.read_graphml(filename_graph)

        # remove nodes by centrality
        y_removed, label_removed, metrics = remove_nodes_centrality(X, label, g, perplexity, nodes_to_keep=percentage)

        metrics_df.loc[len(metrics_df)] = [percentage,
                                           metrics['sortedness'],
                                           metrics['sortedness_weightedtau'],
                                           metrics['trustworthiness'],
                                           metrics['stress'],
                                           metrics['silhouette_score'],
                                           metrics['neighborhood_preservation'],
                                           metrics['neighborhood_hit']]

        # save image
        filename_fig_reduced = dir_base_graph + 'reduced/' + dataset + '[' + str(percentage) + ']-reduced_tsne.png'

        plt.figure()
        plt.scatter(y_removed[:, 0], y_removed[:, 1], c=label_removed, cmap='Set1', edgecolors='face',
                    linewidths=0.5, s=4)
        plt.savefig(filename_fig_reduced, dpi=400, bbox_inches='tight')
        plt.close()

    # save CSV
    metrics_df.to_csv(dir_base_graph + dataset + '-metrics.csv', sep=',')

    return


def run_remove_nodes_centrality_batch():
    dir_base_dataset = '/Users/fpaulovich/Documents/data/'
    dir_base_graph = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/survey_dr/tsne/'

    datasets = ['bank', 'cifar10', 'cnae9', 'coil20', 'epileptic', 'fashion_mnist', 'fmd', 'har', 'hatespeech',
                'hiva', 'imdb', 'secom', 'seismic', 'sentiment', 'sms', 'spambase', 'svhn']

    perplexity = {'bank': 30, 'cifar10': 15, 'cnae9': 5, 'coil20': 50, 'epileptic': 50, 'fashion_mnist': 50,
                  'fmd': 50, 'har': 30, 'hatespeech': 30, 'hiva': 50, 'imdb': 50, 'orl': 15, 'secom': 30, 'seismic': 50,
                  'sentiment': 15, 'sms': 50, 'spambase': 5, 'svhn': 15}

    percentages = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]

    for dataset in datasets:
        print('--')
        print('--')
        print('>>>processing:', dataset)
        print('--')

        metrics_df = pd.DataFrame(columns=['percentage',
                                           'sortedness',
                                           'sortedness_weightedtau',
                                           'trustworthiness',
                                           'stress',
                                           'silhouette_score',
                                           'neighborhood_preservation',
                                           'neighborhood_hit'])

        # read the dataset
        X, _ = load_data(dir_base_dataset + dataset)
        X = MinMaxScaler().fit_transform(X)

        # read the graph
        filename_graph = dir_base_graph + dataset + '-tsne.graphml'
        g = nx.read_graphml(filename_graph)

        # draw the graph using t-SNE
        y, label = draw_graph_by_tsne(X, g)

        # calculate metrics for the original
        print('---')
        print('>>>original')
        metrics = {}

        dr_metric = sortedness(X, y)
        print('sortedness after:', np.average(dr_metric))
        metrics.update({'sortedness': np.average(dr_metric)})

        dr_metric = sortedness(X, y, f=weightedtau)
        print('sortedness after (weightedtau):', np.average(dr_metric))
        metrics.update({'sortedness_weightedtau': np.average(dr_metric)})

        dr_metric = trustworthiness(X, y, n_neighbors=7)
        print('trustworthiness after:', dr_metric)
        metrics.update({'trustworthiness': np.average(dr_metric)})
        dr_metric = stress(X, y, metric='euclidean')

        print('stress after:', dr_metric)
        metrics.update({'stress': np.average(dr_metric)})
        dr_metric = silhouette_score(y, label)

        print('silhouette_score after:', dr_metric)
        metrics.update({'silhouette_score': np.average(dr_metric)})
        dr_metric = neighborhood_preservation(X, y, nr_neighbors=7)

        print('neighborhood_preservation after:', dr_metric)
        metrics.update({'neighborhood_preservation': np.average(dr_metric)})

        dr_metric = neighborhood_hit(y, label, nr_neighbors=7)
        print('neighborhood_hit after:', dr_metric)
        metrics.update({'neighborhood_hit': np.average(dr_metric)})

        metrics_df.loc[len(metrics_df)] = [1.00,
                                           metrics['sortedness'],
                                           metrics['sortedness_weightedtau'],
                                           metrics['trustworthiness'],
                                           metrics['stress'],
                                           metrics['silhouette_score'],
                                           metrics['neighborhood_preservation'],
                                           metrics['neighborhood_hit']]

        for percentage in percentages:
            print('>> percentage: ', percentage)

            # read the graph
            filename_graph = dir_base_graph + dataset + '-tsne.graphml'
            g = nx.read_graphml(filename_graph)

            # remove nodes by centrality
            y_removed, label_removed, metrics = remove_nodes_centrality(X, label, g, int(perplexity[dataset]),
                                                                        nodes_to_keep=percentage)

            metrics_df.loc[len(metrics_df)] = [percentage,
                                               metrics['sortedness'],
                                               metrics['sortedness_weightedtau'],
                                               metrics['trustworthiness'],
                                               metrics['stress'],
                                               metrics['silhouette_score'],
                                               metrics['neighborhood_preservation'],
                                               metrics['neighborhood_hit']]

            # save image
            filename_fig_reduced = dir_base_graph + 'reduced/' + dataset + '[' + str(percentage) + ']-reduced_tsne.png'

            plt.figure()
            plt.scatter(y_removed[:, 0], y_removed[:, 1], c=label_removed, cmap='Set1', edgecolors='face',
                        linewidths=0.5, s=4)
            plt.savefig(filename_fig_reduced, dpi=400, bbox_inches='tight')
            plt.close()

        filename_metrics = dir_base_graph + 'reduced/' + dataset + '-metrics.csv'
        metrics_df.to_csv(filename_metrics, sep=',')

    return


def remove_nodes_random(X, label, g, perplexity, nodes_to_keep=0.8):
    ###################################
    # creating the indexes to remove
    #
    number_nodes = g.number_of_nodes()
    number_nodes_to_remove = int(number_nodes * (1-nodes_to_keep))
    to_remove = random.sample(range(len(X)), number_nodes_to_remove)
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
                     perplexity=perplexity,
                     metric='euclidean',
                     random_state=42,
                     method='barnes_hut',
                     init=PCA(n_components=2).fit_transform(X_removed)
                     ).fit_transform(X_removed)
    ###################################

    ###################################
    # calculate metrics reduced
    #
    metrics = {}

    dr_metric = sortedness(X_removed, y_removed)
    print('sortedness after:', np.average(dr_metric))
    metrics.update({'sortedness': np.average(dr_metric)})

    dr_metric = sortedness(X_removed, y_removed, f=weightedtau)
    print('sortedness after (weightedtau):', np.average(dr_metric))
    metrics.update({'sortedness_weightedtau': np.average(dr_metric)})

    dr_metric = trustworthiness(X_removed, y_removed, n_neighbors=7)
    print('trustworthiness after:', dr_metric)
    metrics.update({'trustworthiness': np.average(dr_metric)})
    dr_metric = stress(X_removed, y_removed, metric='euclidean')

    print('stress after:', dr_metric)
    metrics.update({'stress': np.average(dr_metric)})
    dr_metric = silhouette_score(y_removed, label_removed)

    print('silhouette_score after:', dr_metric)
    metrics.update({'silhouette_score': np.average(dr_metric)})
    dr_metric = neighborhood_preservation(X_removed, y_removed, nr_neighbors=7)

    print('neighborhood_preservation after:', dr_metric)
    metrics.update({'neighborhood_preservation': np.average(dr_metric)})

    dr_metric = neighborhood_hit(y_removed, label_removed, nr_neighbors=7)
    print('neighborhood_hit after:', dr_metric)
    metrics.update({'neighborhood_hit': np.average(dr_metric)})

    return y_removed, label_removed, metrics


def run_remove_nodes_random_batch():
    dir_base_dataset = '/Users/fpaulovich/Documents/data/'
    dir_base_graph = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/survey_dr/tsne/'

    datasets = ['cnae9', 'coil20', 'fashion_mnist', 'har', 'spambase']

    perplexity = {'bank': 30, 'cifar10': 15, 'cnae9': 5, 'coil20': 50, 'epileptic': 50, 'fashion_mnist': 50,
                  'fmd': 50, 'har': 30, 'hatespeech': 30, 'hiva': 50, 'imdb': 50, 'orl': 15, 'secom': 30, 'seismic': 50,
                  'sentiment': 15, 'sms': 50, 'spambase': 5, 'svhn': 15}

    percentages = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]

    for dataset in datasets:
        print('--')
        print('--')
        print('>>>processing:', dataset)
        print('--')

        metrics_df = pd.DataFrame(columns=['percentage',
                                           'sortedness',
                                           'sortedness_weightedtau',
                                           'trustworthiness',
                                           'stress',
                                           'silhouette_score',
                                           'neighborhood_preservation',
                                           'neighborhood_hit'])

        # read the dataset
        X, _ = load_data(dir_base_dataset + dataset)
        X = MinMaxScaler().fit_transform(X)

        # read the graph
        filename_graph = dir_base_graph + dataset + '-tsne.graphml'
        g = nx.read_graphml(filename_graph)

        # draw the graph using t-SNE
        y, label = draw_graph_by_tsne(X, g)

        # calculate metrics for the original
        print('---')
        print('>>>original')
        metrics = {}

        dr_metric = sortedness(X, y)
        print('sortedness after:', np.average(dr_metric))
        metrics.update({'sortedness': np.average(dr_metric)})

        dr_metric = sortedness(X, y, f=weightedtau)
        print('sortedness after (weightedtau):', np.average(dr_metric))
        metrics.update({'sortedness_weightedtau': np.average(dr_metric)})

        dr_metric = trustworthiness(X, y, n_neighbors=7)
        print('trustworthiness after:', dr_metric)
        metrics.update({'trustworthiness': np.average(dr_metric)})
        dr_metric = stress(X, y, metric='euclidean')

        print('stress after:', dr_metric)
        metrics.update({'stress': np.average(dr_metric)})
        dr_metric = silhouette_score(y, label)

        print('silhouette_score after:', dr_metric)
        metrics.update({'silhouette_score': np.average(dr_metric)})
        dr_metric = neighborhood_preservation(X, y, nr_neighbors=7)

        print('neighborhood_preservation after:', dr_metric)
        metrics.update({'neighborhood_preservation': np.average(dr_metric)})

        dr_metric = neighborhood_hit(y, label, nr_neighbors=7)
        print('neighborhood_hit after:', dr_metric)
        metrics.update({'neighborhood_hit': np.average(dr_metric)})

        metrics_df.loc[len(metrics_df)] = [1.00,
                                           metrics['sortedness'],
                                           metrics['sortedness_weightedtau'],
                                           metrics['trustworthiness'],
                                           metrics['stress'],
                                           metrics['silhouette_score'],
                                           metrics['neighborhood_preservation'],
                                           metrics['neighborhood_hit']]

        for percentage in percentages:
            print('>> percentage: ', percentage)

            # read the graph
            filename_graph = dir_base_graph + dataset + '-tsne.graphml'
            g = nx.read_graphml(filename_graph)

            # remove nodes randomly
            y_removed, label_removed, metrics = remove_nodes_random(X, label, g, int(perplexity[dataset]),
                                                                    nodes_to_keep=percentage)

            metrics_df.loc[len(metrics_df)] = [percentage,
                                               metrics['sortedness'],
                                               metrics['sortedness_weightedtau'],
                                               metrics['trustworthiness'],
                                               metrics['stress'],
                                               metrics['silhouette_score'],
                                               metrics['neighborhood_preservation'],
                                               metrics['neighborhood_hit']]

            # save image
            filename_fig_reduced = dir_base_graph + 'reduced_random/' + dataset + '[' + str(percentage) + ']-reduced_tsne.png'

            plt.figure()
            plt.scatter(y_removed[:, 0], y_removed[:, 1], c=label_removed, cmap='Set1', edgecolors='face',
                        linewidths=0.5, s=4)
            plt.savefig(filename_fig_reduced, dpi=400, bbox_inches='tight')
            plt.close()

        filename_metrics = dir_base_graph + 'reduced_random/' + dataset + '-metrics.csv'
        metrics_df.to_csv(filename_metrics, sep=',')

    return


def draw_line_graph():
    # datasets = ['bank', 'cifar10', 'cnae9', 'coil20', 'epileptic', 'fashion_mnist', 'fmd', 'har', 'hatespeech',
    #             'hiva', 'imdb', 'secom', 'seismic', 'sentiment', 'sms', 'spambase', 'svhn']

    metrics = ['sortedness', 'sortedness_weightedtau', 'trustworthiness', 'stress',
               'silhouette_score', 'neighborhood_preservation', 'neighborhood_hit']

    datasets = ['fashion_mnist']  # ['cnae9', 'coil20', 'fashion_mnist', 'har', 'spambase']

    dir_base_graph = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/survey_dr/tsne/'

    plt.figure()
    for metric in metrics:
        for dataset in datasets:
            filename_metrics = dir_base_graph + 'reduced_random/' + dataset + '-metrics.csv'
            df_metrics = pd.read_csv(filename_metrics, sep=',')
            plt.plot(df_metrics['percentage'], df_metrics[metric], label=dataset)

        plt.xlabel('percentage')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.ylabel(metric)
        plt.gca().invert_xaxis()

        filename_fig = dir_base_graph + 'reduced_random/metric_' + metric + '.png'
        plt.savefig(filename_fig, dpi=400, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    # run_remove_nodes_centrality()
    # run_remove_nodes_centrality_batch()
    # run_remove_nodes_random_batch()
    draw_line_graph()
