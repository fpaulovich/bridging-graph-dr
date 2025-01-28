import math

import numpy as np
import random
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from scipy.spatial import distance

import numpy as np
import scipy as sp

import math
from sklearn import preprocessing


def landmark_MDS(D, lands, dim):
    Dl = D[:, lands]
    n = len(Dl)

    # Centering matrix
    H = - np.ones((n, n)) / n
    np.fill_diagonal(H, 1 - 1 / n)
    # YY^T
    H = -H.dot(Dl ** 2).dot(H) / 2

    # Diagonalize
    evals, evecs = np.linalg.eigh(H)

    # Sort by eigenvalue in descending order
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Compute the coordinates using positive-eigenvalued components only
    w, = np.where(evals > 0)
    if dim:
        arr = evals
        w = arr.argsort()[-dim:][::-1]
        if np.any(evals[w] < 0):
            print('Error: Not enough positive eigenvalues for the selected dim.')
            return []
    if w.size == 0:
        print('Error: matrix is negative definite.')
        return []

    V = evecs[:, w]
    L = V.dot(np.diag(np.sqrt(evals[w]))).T
    N = D.shape[1]
    Lh = V.dot(np.diag(1. / np.sqrt(evals[w]))).T
    Dm = D - np.tile(np.mean(Dl, axis=1), (N, 1)).T
    dim = w.size
    X = -Lh.dot(Dm) / 2.
    X -= np.tile(np.mean(X, axis=1), (N, 1)).T

    _, evecs = sp.linalg.eigh(X.dot(X.T))

    return (evecs[:, ::-1].T.dot(X)).T


def test():
    iris = load_digits()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    X = preprocessing.MinMaxScaler().fit_transform(X)

    lands = random.sample(range(0, X.shape[0], 1), int(math.sqrt(len(X))))
    lands = np.array(lands, dtype=int)
    Dl2 = distance.cdist(X[lands, :], X, 'euclidean')

    xl_2 = landmark_MDS(Dl2, lands, 2)

    plt.figure()
    plt.scatter(xl_2[:, 0], xl_2[:, 1], c=y,
                cmap='tab10', edgecolors='face', linewidths=0.5, s=10, label=target_names)
    plt.grid(linestyle='dotted')
    plt.show()


    # plt.figure()
    # colors = ['navy', 'turquoise', 'darkorange']
    # for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    #     plt.scatter(xl_2[y == i, 0], xl_2[y == i, 1], alpha=.8, color=color, label=target_name)
    # plt.legend(loc='best', shadow=False, scatterpoints=1)
    # plt.title('Landmark MDS of IRIS dataset')
    # plt.show()

    print(len(Dl2))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test()
