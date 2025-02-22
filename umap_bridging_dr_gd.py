import numpy as np

from umap import UMAP

import matplotlib.pyplot as plt

from timeit import default_timer as timer
from datetime import timedelta

import networkx as nx
from scipy.sparse import csr_matrix

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from techniques.umap_gd import gd_umap

MACHINE_EPSILON = np.finfo(np.double).eps


def load_data(dataset):
    X = np.load(dataset + "/X.npy", allow_pickle=True).astype(np.float64)
    y = np.load(dataset + "/y.npy", allow_pickle=True).astype(np.int64)
    return X, y


def generate_all_umap_graphs():
    dir_base = '/Users/fpaulovich/Documents/data/'
    datasets = ['bank', 'cifar10', 'cnae9', 'coil20', 'epileptic', 'fashion_mnist', 'fmd', 'har', 'hatespeech',
                'hiva', 'imdb', 'orl', 'secom', 'seismic', 'sentiment', 'sms', 'spambase', 'svhn']

    nr_neighbors = {'bank': 5, 'cifar10': 15, 'cnae9': 10, 'coil20': 5, 'epileptic': 5, 'fashion_mnist': 5,
                  'fmd': 15, 'har': 5, 'hatespeech': 15, 'hiva': 15, 'imdb': 15, 'orl': 15, 'secom': 5, 'seismic': 5,
                  'sentiment': 5, 'sms': 5, 'spambase': 10, 'svhn': 10}

    dir_name_output = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/survey_dr/umap/'

    for dataset in datasets:
        print('>>>processing:', dataset)
        X, y = load_data(dir_base + dataset)
        X = MinMaxScaler().fit_transform(X)
        y = LabelEncoder().fit_transform(y)

        gd_umap(X=X,
                labels=y,
                nr_neighbors=int(nr_neighbors[dataset]),
                filename_fig=dir_name_output + dataset + '-gd_umap.png',
                filename_graph=dir_name_output + dataset + '-umap.graphml'
                )
    return


def draw_graph_by_umap(filename_graph, filename_fig=None):
    # g = nx.read_graphml(filename_graph)
    # label = list(map(int, nx.get_node_attributes(g, 'label').values()))
    # weight = nx.get_edge_attributes(g, name='weight')
    #
    # row = np.zeros(g.number_of_edges())
    # col = np.zeros(g.number_of_edges())
    # data = np.zeros(g.number_of_edges())
    #
    # k = 0
    # for i, j in g.edges():
    #     row[k] = i
    #     col[k] = j
    #     data[k] = weight[(i, j)]
    #     k = k + 1
    #
    # P = csr_matrix((data, (row, col)), shape=(g.number_of_nodes(), g.number_of_nodes()))
    # P = P + P.T
    # P /= np.maximum(P.sum(), MACHINE_EPSILON)
    #
    # start = timer()
    # mapper = UMAP(n_neighbors=2,
    #               metric='euclidean',
    #               random_state=42,
    #               transform_mode='graph').fit(np.zeros((g.number_of_nodes(), 2)))
    # end = timer()
    #
    # print('t-SNE took {0} to execute'.format(timedelta(seconds=end - start)))
    #
    # plt.figure()
    # plt.scatter(y[:, 0], y[:, 1], c=label,
    #             cmap='Set1', edgecolors='face', linewidths=0.5, s=4)
    # # plt.grid(linestyle='dotted')
    #
    # if filename_fig is not None:
    #     plt.savefig(filename_fig, dpi=400, bbox_inches='tight')
    # else:
    #     plt.show()
    # plt.close()

    return


def draw_all_graphs_by_umap():
    dir_base = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/survey_dr/umap/'
    datasets = ['bank', 'cifar10', 'cnae9', 'coil20', 'epileptic', 'fashion_mnist', 'fmd', 'har', 'hatespeech',
                'hiva', 'imdb', 'orl', 'secom', 'seismic', 'sentiment', 'sms', 'spambase', 'svhn']

    for dataset in datasets:
        filename_graph = dir_base + dataset + '-umap.graphml'
        filename_fig = dir_base + dataset + '-umap.png'
        draw_graph_by_umap(filename_graph, filename_fig)


if __name__ == '__main__':
    generate_all_umap_graphs()
    # draw_all_graphs_by_umap()
