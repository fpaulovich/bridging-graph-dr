import math

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from techniques.t_sne import TSNE
from techniques.metrics import stress, neighborhood_preservation, neighborhood_hit

from util import load_data
from sklearn import preprocessing
import sklearn.datasets as datasets

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
             metric='euclidean',
             random_state=42,
             method='barnes_hut',
             init='pca',
             probabilities=P).fit_transform(X)
    end = timer()
    print('t-SNE took {0} to execute'.format(timedelta(seconds=end - start)))

    return y, label


def tsne_prob_graph(X, perplexity, metric='euclidean', labels=None, epsilon=0.9):
    size = len(X)

    ####################################
    # execute t-SNE to calculate the probabilities
    start = timer()
    P = TSNE(n_components=2,
             perplexity=perplexity,
             metric=metric,
             random_state=42,
             method='barnes_hut',
             init='pca').fit(X).get_probabilities().tocoo()
    end = timer()
    print('t-SNE took {0} to execute'.format(timedelta(seconds=end - start)))
    ####################################

    data = P.data
    row_idx = P.row
    col_idx = P.col

    # remove edges with very low probability
    sorted_vect = -np.sort(-P.data.copy())

    # find the minimum value which the commulative sum reaches overall_sum
    min_val = -1
    cum_sum = 0
    for i in range(len(sorted_vect)):
        cum_sum = cum_sum + sorted_vect[i]
        if cum_sum >= epsilon:
            min_val = sorted_vect[i]
            break

    # creating the graph
    g = nx.Graph()

    for i in range(size):
        g.add_node(i)

    # set labels as node attribute
    if labels is not None:
        nx.set_node_attributes(g, dict(enumerate(map(str, labels))), name='label')

    for i in range(len(data)):
        if data[i] >= min_val:
            g.add_edge(row_idx[i], col_idx[i], weight=data[i] / cum_sum)

    return g


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
                     init='pca').fit_transform(X_removed)
    ###################################

    ###################################
    # update graph coordinates
    #
    for node in g.nodes:
        g.nodes[node]['x'] = float(y_removed[node][0])
        g.nodes[node]['y'] = float(y_removed[node][1])

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

    return y_removed, label_removed, g, metrics


def remove_nodes_random(X, label, g, perplexity, nodes_to_keep=0.8):
    ###################################
    # creating the indexes to remove
    #
    number_nodes = g.number_of_nodes()
    number_nodes_to_remove = int(number_nodes * (1-nodes_to_keep))
    to_remove = random.sample(range(number_nodes), number_nodes_to_remove)

    for remove in to_remove:
        g.remove_node(str(remove))
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
    # update graph coordinates
    #
    for node in g.nodes:
        g.nodes[node]['x'] = float(y_removed[node][0])
        g.nodes[node]['y'] = float(y_removed[node][1])

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

    return y_removed, label_removed, g, metrics


if __name__ == '__main__':
    raw = datasets.load_digits(as_frame=True)
    X = raw.data.to_numpy()
    X = preprocessing.MinMaxScaler().fit_transform(X)
    labels = raw.target.to_numpy()

    perplexity = 15

    g = tsne_prob_graph(X, perplexity, metric='euclidean', labels=labels, epsilon=0.9)
    y, label = draw_graph_by_tsne(X, g)

    plt.figure()
    plt.scatter(y[:, 0], y[:, 1], c=label, cmap='Set1', edgecolors='face', linewidths=0.5, s=4)
    plt.show()
    plt.close()
