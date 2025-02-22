# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import gmean

import networkx as nx

from sklearn import preprocessing
import sklearn.datasets as datasets

from techniques.tsne_gd import tsne, tsne_bh_prob_graph
from util import draw_graph


def precision(g):
    size = g.number_of_nodes()

    # calculate precision
    true_positive = np.zeros(size)
    false_positive = np.zeros(size)
    for i in range(size):
        for j in g.neighbors(i):
            if g.nodes[i]['label'] == g.nodes[j]['label']:
                true_positive[j] = true_positive[j] + 1  # g.edges[i, j]['weight']
            else:
                false_positive[j] = false_positive[j] + 1  # g.edges[i, j]['weight']

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


# def recall_paths(g):
#     size = g.number_of_nodes()
#
#     # create an edge attribute to inverse weight
#     length = nx.get_edge_attributes(g, 'weight')
#     for key in length.keys():
#         length[key] = 1 - length[key]
#     nx.set_edge_attributes(g, length, 'length')
#
#     # remove edges connecting nodes of different classes (for recall calculation)
#     for i, j in g.edges():
#         if g.nodes[i]['label'] != g.nodes[j]['label']:
#             g.remove_edge(i, j)
#
#     # get the nodes with same label
#     groups = {}
#     for i in range(size):
#         item = groups.get(g.nodes[i]['label'])
#         if item is None:
#             groups.update({g.nodes[i]['label']: np.array([i])})
#         else:
#             groups.update({g.nodes[i]['label']: np.append(item, i)})
#
#     # calculate the average path lengths between elements of the same label
#     all_lengths = np.zeros((size, size))
#
#     for label in groups.keys():
#         g_elements = groups[label]
#         g_size = len(g_elements)
#
#         for i in range(g_size - 1):
#             for j in range(i + 1, g_size):
#                 if nx.has_path(g, source=g_elements[i], target=g_elements[j]):
#                     all_lengths[g_elements[i]][g_elements[j]] = nx.shortest_path_length(g,
#                                                                                         weight='length',
#                                                                                         source=g_elements[i],
#                                                                                         target=g_elements[j])
#                     all_lengths[g_elements[j]][g_elements[i]] = all_lengths[g_elements[i]][g_elements[j]]
#                 else:
#                     all_lengths[g_elements[i]][g_elements[j]] = -1
#                     all_lengths[g_elements[j]][g_elements[i]] = -1
#
#     # if the elements of the same label are disconnected, penalize
#     max_val = np.max(all_lengths)
#     for i in range(size-1):
#         for j in range(i+1, size):
#             if all_lengths[i][j] == -1:
#                 all_lengths[i][j] = max_val
#                 all_lengths[j][i] = max_val
#
#     # normalizing to 1
#     all_lengths = all_lengths / np.max(all_lengths)
#     average_lengths = np.zeros(size)
#     for i in range(size):
#         average_lengths[i] = 1 - np.average(all_lengths[i])
#
#     return average_lengths


# def recall(g):
#     size = g.number_of_nodes()
#
#     # create an edge attribute to inverse weight
#     length = nx.get_edge_attributes(g, 'weight')
#     for key in length.keys():
#         length[key] = 1 - length[key]
#     nx.set_edge_attributes(g, length, 'length')
#
#     # remove weak connections
#     edges_to_remove = []
#     for edge in g.edges():
#         neighbors_i = [n for n in g[edge[0]]]
#         neighbors_j = [n for n in g[edge[1]]]
#         intersection = np.intersect1d(neighbors_i, neighbors_j)
#         union = np.union1d(neighbors_i, neighbors_j)
#
#         if len(intersection)/len(union) < 0.1:
#             edges_to_remove.append(edge)
#
#     for i, j in edges_to_remove:
#         if g.has_edge(i, j):
#             g.remove_edge(i, j)
#
#         if g.has_edge(j, i):
#             g.remove_edge(i, j)
#
#     # remove edges depending on the difference in class
#     edges_to_remove = []
#     for node in g.nodes():
#         sum_intra_class_weight = 0
#         sum_inter_class_weight = 0
#
#         intra_class_edges = []
#         inter_class_edges = []
#
#         for edge in g.edges(node):
#             # if the another node of the edge has the same label of node
#             if g.nodes[edge[0]]['label'] == g.nodes[edge[1]]['label']:
#                 sum_intra_class_weight = sum_intra_class_weight + g.get_edge_data(*edge)['weight']
#                 intra_class_edges.append(edge)
#             else:
#                 sum_inter_class_weight = sum_inter_class_weight + g.get_edge_data(*edge)['weight']
#                 inter_class_edges.append(edge)
#
#         # if there is more inter class weight, remove the edge that connects to the same class
#         if sum_inter_class_weight > sum_intra_class_weight:
#             edges_to_remove = edges_to_remove + intra_class_edges
#
#         # always remove inter class edges
#         edges_to_remove = edges_to_remove + inter_class_edges
#
#     for i, j in edges_to_remove:
#         if g.has_edge(i, j):
#             g.remove_edge(i, j)
#
#         if g.has_edge(j, i):
#             g.remove_edge(i, j)
#
#     # calculate recall
#     labels_count = {}
#     for i in range(size):
#         item = labels_count.get(g.nodes[i]['label'])
#         if item is None:
#             labels_count.update({g.nodes[i]['label']: 1})
#         else:
#             labels_count.update({g.nodes[i]['label']: item + 1})
#
#     recall_per_point = np.zeros(size)
#     clust_coefficients = nx.clustering(g, weight='length')
#     components = [c for c in sorted(nx.connected_components(g), key=len, reverse=True)]
#
#     for i in range(len(components)):
#         component = list(components[i])
#         component_label = g.nodes[component[0]]['label']
#         component_size = len(component)
#         component_recall = (component_size / labels_count.get(component_label))
#
#         for j in range(component_size):
#             if component_size == 1:
#                 recall_per_point[component[j]] = component_recall  # components with one element has precision==1,
#                 # so recall should be the component recall.
#             else:
#                 recall_per_point[component[j]] = (component_recall * clust_coefficients[component[j]])
#
#     return recall_per_point


def f_score(prec, rec, beta=1):
    prec = prec + 0.000001
    rec = rec + 0.000001
    return (1 + beta * beta) * np.divide(np.multiply(prec, rec), np.add((beta * beta) * prec, rec))


def test_multiple_perplexity(metric='euclidean'):
    raw = datasets.load_digits(as_frame=True)

    X = raw.data.to_numpy()
    X = preprocessing.MinMaxScaler().fit_transform(X)
    labels = raw.target.to_numpy()

    max_perplexity = 200
    min_perplexity = 2
    increment = 5

    avg_pre = np.zeros(math.ceil((max_perplexity - min_perplexity) / increment))
    avg_rec = np.zeros(math.ceil((max_perplexity - min_perplexity) / increment))
    avg_f_s = np.zeros(math.ceil((max_perplexity - min_perplexity) / increment))
    perp_vals = np.zeros(math.ceil((max_perplexity - min_perplexity) / increment))

    k = 0
    max_f_score = 0
    best_perplexity = 0
    for perplexity in range(min_perplexity, max_perplexity, increment):
        print('perplexity: ', perplexity)
        perp_vals[k] = perplexity

        g = tsne_bh_prob_graph(X, perplexity, metric, labels, epsilon=0.9)

        pre = precision(g)
        rec = recall_component(g)
        f_s = f_score(pre, rec, 0.5)

        for i in range(len(pre)):
            if (pre[i] + rec[i]) == 0:
                print('>>> ', perplexity, pre[i], rec[i])

        avg_pre[k] = np.average(pre)
        avg_rec[k] = np.average(rec)
        avg_f_s[k] = np.average(f_s)

        if max_f_score < avg_f_s[k]:
            max_f_score = avg_f_s[k]
            best_perplexity = perplexity

        k = k + 1

    print('best perplexity:', best_perplexity, max_f_score)

    plt.plot(perp_vals, avg_pre, label='precision')
    plt.plot(perp_vals, avg_rec, label='recall')
    plt.plot(perp_vals, avg_f_s, label='f-score')
    plt.legend()
    plt.show()


def test():
    raw = datasets.load_digits(as_frame=True)
    X = raw.data.to_numpy()
    X = preprocessing.MinMaxScaler().fit_transform(X)
    labels = raw.target.to_numpy()

    perplexity = 22
    g = tsne(X, labels, filename_fig='../tsne.png', filename_graph=None, perplexity=perplexity)

    # recall(g)

    # recall_labels = recall_paths(g)
    # draw_graph(X, g, labels, filename=None)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test()
    # test_multiple_perplexity()
