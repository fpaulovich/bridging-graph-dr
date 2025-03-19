import math

from sklearn.manifold._t_sne import _joint_probabilities_nn
from sklearn.neighbors import NearestNeighbors

import numpy as np
import networkx as nx

import pandas as pd

from sklearn import preprocessing
import sklearn.datasets as datasets

from bayes_opt import BayesianOptimization

from timeit import default_timer as timer
from datetime import timedelta

MACHINE_EPSILON = np.finfo(np.double).eps
n_jobs = 5
metric_params = None


def calculate_neighborhood_distances(X, perplexity, metric):
    n_samples = len(X)

    # Compute the number of nearest neighbors to find.
    # LvdM uses 3 * perplexity as the number of neighbors.
    # In the event that we have very small # of points
    # set the neighbors to n - 1.
    n_neighbors = min(n_samples - 1, int(3.0 * perplexity + 1))

    # Find the nearest neighbors for every point
    knn = NearestNeighbors(
        algorithm="auto",
        n_jobs=n_jobs,
        n_neighbors=n_neighbors,
        metric=metric,
        metric_params=None,
    )

    knn.fit(X)

    distances_nn = knn.kneighbors_graph(mode="distance")

    # Free the memory used by the ball_tree
    del knn

    # knn return the euclidean distance but we need it squared
    # to be consistent with the 'exact' method. Note that the
    # method was derived using the euclidean method as in the
    # input space. Not sure of the implication of using a different
    # metric.
    distances_nn.data **= 2

    return distances_nn


def tsne_bh_prob_graph(distances_nn, size, labels, perplexity, epsilon=0.9):
    P = _joint_probabilities_nn(distances_nn, perplexity, 0).tocoo()

    data = P.data
    row_idx = P.row
    col_idx = P.col

    # remove edges with very low probability
    overall_sum = epsilon  # overall sum of probabilities
    sorted_vect = -np.sort(-P.data.copy())

    # find the minimum value which the commulative sum reaches overall_sum
    min_val = -1
    cum_sum = 0
    for i in range(len(sorted_vect)):
        cum_sum = cum_sum + sorted_vect[i]
        if cum_sum >= overall_sum:
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


def precision(g):
    size = g.number_of_nodes()

    # calculate precision
    true_positive = np.zeros(size)
    false_positive = np.zeros(size)
    for i in range(size):
        for j in g.neighbors(i):
            if g.nodes[i]['label'] == g.nodes[j]['label']:
                true_positive[j] = true_positive[j] + g.edges[i, j]['weight']
            else:
                false_positive[j] = false_positive[j] + g.edges[i, j]['weight']

    precision_per_point = np.zeros(size)
    for i in range(size):
        if false_positive[i] > 0:
            precision_per_point[i] = true_positive[i] / (true_positive[i] + false_positive[i])
        else:
            precision_per_point[i] = 1

    return precision_per_point


def recall(g):
    size = g.number_of_nodes()

    true_positive = np.zeros(size)
    for i in range(size):
        for j in g.neighbors(i):
            if g.nodes[i]['label'] == g.nodes[j]['label']:
                true_positive[j] = true_positive[j] + 1

    labels_count = {}
    for i in range(size):
        item = labels_count.get(g.nodes[i]['label'])
        if item is None:
            labels_count.update({g.nodes[i]['label']: 1})
        else:
            labels_count.update({g.nodes[i]['label']: item + 1})

    recall_per_point = np.zeros(size)
    for i in range(size):
        recall_per_point[i] = true_positive[i] / labels_count.get(g.nodes[i]['label'])

    return recall_per_point


def recall_component(g):
    size = g.number_of_nodes()

    # calculate true positive
    true_positive = np.zeros(size)
    for i in range(size):
        for j in g.neighbors(i):
            if g.nodes[i]['label'] == g.nodes[j]['label']:
                true_positive[j] = true_positive[j] + 1

    # remove edges depending on the difference in class
    edges_to_remove = []
    for node in g.nodes():
        sum_intra_class_weight = 0
        sum_inter_class_weight = 0

        intra_class_edges = []
        inter_class_edges = []

        for edge in g.edges(node):
            # if the another node of the edge has the same label of node
            if g.nodes[edge[0]]['label'] == g.nodes[edge[1]]['label']:
                sum_intra_class_weight = sum_intra_class_weight + g.get_edge_data(*edge)['weight']
                intra_class_edges.append(edge)
            else:
                sum_inter_class_weight = sum_inter_class_weight + g.get_edge_data(*edge)['weight']
                inter_class_edges.append(edge)

        # if there is more inter class weight, remove the edge that connects to the same class
        if sum_inter_class_weight > sum_intra_class_weight:
            edges_to_remove = edges_to_remove + intra_class_edges

        # always remove inter class edges
        edges_to_remove = edges_to_remove + inter_class_edges

    for i, j in edges_to_remove:
        if g.has_edge(i, j):
            g.remove_edge(i, j)

        if g.has_edge(j, i):
            g.remove_edge(i, j)

    # remove edges connecting nodes of different classes (for recall calculation)
    for i, j in g.edges():
        if g.nodes[i]['label'] != g.nodes[j]['label']:
            g.remove_edge(i, j)

    # calculate recall
    labels_count = {}
    for i in range(size):
        item = labels_count.get(g.nodes[i]['label'])
        if item is None:
            labels_count.update({g.nodes[i]['label']: 1})
        else:
            labels_count.update({g.nodes[i]['label']: item + 1})

    recall_per_point = np.zeros(size)
    components = [c for c in sorted(nx.connected_components(g), key=len, reverse=True)]

    for i in range(len(components)):
        component = list(components[i])
        component_label = g.nodes[component[0]]['label']
        component_size = len(component)
        label_size = labels_count.get(component_label)

        for j in range(component_size):
            recall_per_point[component[j]] = true_positive[component[j]] / \
                                             (true_positive[component[j]] + (label_size - component_size))

    return recall_per_point


def f_score(prec, rec, beta=1):
    prec = prec + 0.000001
    rec = rec + 0.000001
    return (1 + beta * beta) * np.divide(np.multiply(prec, rec), np.add((beta * beta) * prec, rec))


def optimization_precision_recall_bayesian(X, labels, metric, min_perplexity, max_perplexity):
    # https://bayesian-optimization.github.io/BayesianOptimization/2.0.3/
    start = timer()

    distances_nn = calculate_neighborhood_distances(X, max_perplexity, metric)

    pbounds = {'perplexity': (min_perplexity, max_perplexity)}

    def cost_p_r(perplexity):
        g = tsne_bh_prob_graph(distances_nn, len(X), labels, perplexity, epsilon=0.9)
        pre = precision(g)
        rec = recall_component(g)
        return np.average(f_score(pre, rec, 1.0))

    optimizer = BayesianOptimization(
        f=cost_p_r,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )

    optimizer.maximize(
        init_points=5,
        n_iter=10,
    )

    print(optimizer.max)

    end = timer()
    print('Bayesian optimization took {0} to execute'.format(timedelta(seconds=end - start)))

    return


def optimization_precision_recall_bisection_search(X, labels, metric, min_perplexity, max_perplexity):
    start = timer()

    distances_nn = calculate_neighborhood_distances(X, max_perplexity, metric)

    def cost_p_r(perplexity):
        g = tsne_bh_prob_graph(distances_nn, len(X), labels, perplexity, epsilon=0.9)
        pre = precision(g)
        rec = recall_component(g)
        return np.average(f_score(pre, rec, 1.0))

    def bisect_search(lower, cost_lower, larger, cost_larger):
        while lower < (larger - 1):
            mid = int(lower + (larger - lower) / 2)
            cost_mid = cost_p_r(mid)

            if cost_lower < cost_larger:
                lower = mid
                cost_lower = cost_mid
            else:
                larger = mid
                cost_larger = cost_mid

        if cost_lower > cost_larger:
            return lower, cost_lower
        else:
            return larger, cost_larger

    n_partitions = 5
    best_perplexity = -1
    best_f_measure = -1
    for i in range(n_partitions):
        interval_min = int(((max_perplexity - min_perplexity) / n_partitions) * i + min_perplexity)
        interval_max = int(((max_perplexity - min_perplexity) / n_partitions) * (i + 1) + min_perplexity)

        perp, f_measure = bisect_search(interval_min,
                                        cost_p_r(interval_min),
                                        interval_max,
                                        cost_p_r(interval_max))

        if f_measure > best_f_measure:
            best_perplexity = perp
            best_f_measure = f_measure

        print('--batch--')
        print('best perplexity:', best_perplexity)
        print('best f-score:', best_f_measure)
        print('--batch--')

    print('global best perplexity:', best_perplexity)
    print('global best f-score:', best_f_measure)

    end = timer()
    print('Bisection search optimization took {0} to execute'.format(timedelta(seconds=end - start)))

    return


def test_1():
    raw = datasets.load_digits(as_frame=True)

    X = raw.data.to_numpy()
    X = preprocessing.MinMaxScaler().fit_transform(X)
    labels = raw.target.to_numpy()

    optimization_precision_recall_bisection_search(X, labels,
                                                   metric='euclidean',
                                                   min_perplexity=5,
                                                   max_perplexity=300)

    print('---')

    optimization_precision_recall_bayesian(X, labels,
                                           metric='euclidean',
                                           min_perplexity=5,
                                           max_perplexity=300)

    return


def test_2():
    data_file = "/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/datasets/csv/19k_fourier30_spatial10_normalized.csv"
    df = pd.read_csv(data_file, header=0, engine='python')

    labels = np.array(df['label'].values)  # .reshape(-1, 1)

    df = df.drop(['id', 'label'], axis=1)
    X = df.values
    X = preprocessing.MinMaxScaler().fit_transform(X)

    # optimization_precision_recall_bisection_search(X, labels,
    #                                                metric='euclidean',
    #                                                min_perplexity=5,
    #                                                max_perplexity=1000)

    print('---')

    optimization_precision_recall_bayesian(X, labels,
                                           metric='euclidean',
                                           min_perplexity=5,
                                           max_perplexity=1000)
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_2()
