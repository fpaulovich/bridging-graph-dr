#  Copyright (c) 2023. Davi Pereira dos Santos
#  This file is part of the sortedness project.
#  Please respect the license - more about this in the section (*) below.
#
#  sortedness is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  sortedness is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with sortedness.  If not, see <http://www.gnu.org/licenses/>.
#
#  (*) Removing authorship by any means, e.g. by distribution of derived
#  works or verbatim, obfuscated, compiled or rewritten versions of any
#  part of this work is illegal and it is unethical regarding the effort and
#  time spent here.
#

import gc
import math
from functools import partial
from math import exp
from math import pi

import numpy as np
import pathos.multiprocessing as mp
from numpy import eye, mean, sqrt, ndarray, nan
from numpy.random import permutation
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import rankdata
from scipy.stats import weightedtau, kendalltau

# from parallel import rank_alongrow, rank_alongcol

from metrics.parallel import rank_alongrow, rank_alongcol


def common(S, S_, i, symmetric, f, isweightedtau, return_pvalues, pmap, kwargs):
    def thread(a, b):
        return f(a, b, **kwargs)

    def oneway(scores_a, scores_b):
        jobs = pmap((thread if kwargs else f), scores_a, scores_b)
        result, pvalues = [], []
        for tup in jobs:
            corr, pvalue = tup if isinstance(tup, tuple) else (tup, nan)
            result.append(corr)
            pvalues.append(pvalue)
        return np.array(result, dtype=float), np.array(pvalues, dtype=float)

    handle_f = lambda tup: tup if isinstance(tup, tuple) else (tup, nan)
    result, pvalues = oneway(S, S_) if i is None else handle_f(f(S, S_, **kwargs))
    if not isweightedtau and symmetric:
        result_, pvalues_ = oneway(S_, S) if i is None else handle_f(f(S_, S, **kwargs))
        result = (result + result_) / 2
        pvalues = (pvalues + pvalues_) / 2

    if return_pvalues:
        return np.array(list(zip(result, pvalues)))
    return result


# todo: see if speed can benefit from:
# gen = pairwise_distances_chunked(X, method='cosine', n_jobs=-1)
# Z = np.concatenate(list(gen), axis=0)
# Z_cond = Z[np.triu_indices(Z.shape[0], k=1)
# https://stackoverflow.com/a/55940399/9681577
#
# import dask.dataframe as dd
# from dask.multiprocessing import get
# # o - is pandas DataFrame
# o['dist_center_from'] = dd.from_pandas(o, npartitions=8).map_partitions(lambda df: df.apply(lambda x: vincenty((x.fromlatitude, x.fromlongitude), center).km, axis=1)).compute(get=get)

def remove_diagonal(X):
    n_points = len(X)
    nI = ~eye(n_points, dtype=bool)  # Mask to remove diagonal.
    return X[nI].reshape(n_points, -1)


weightedtau.isweightedtau = True


def geomean_np(lo, gl, beta=0.5):
    """
    >>> round(geomean_np(0.6, 0.64), 4)
    0.6199
    """
    l = (lo + 1) / 2
    g = (gl + 1) / 2
    return math.exp((1 - beta) * math.log(l + 0.000000000001) + beta * math.log(g + 0.000000000001)) * 2 - 1


def balanced_kendalltau(unordered_values, unordered_values_, beta=0.5, gamma=4):
    """
    >>> round(balanced_kendalltau(np.array([2,1,3,4,5]), np.array([2,1,3,4,5]), beta=1), 5)
    1.0
    >>> round(balanced_kendalltau(np.array([1,2,3,4,5]), np.array([5,4,3,2,1]), beta=1), 5)
    -1.0
    >>> round(balanced_kendalltau(np.array([1,2,3,4,5]), np.array([1,2,3,4,5]), beta=0), 5)
    1.0
    >>> round(balanced_kendalltau(np.array([1,2,3,4,5]), np.array([5,4,3,2,1]), beta=0), 5)
    -1.0
    >>> # strong break of trustworthiness = the last distance value (the one with lower weight) becomes the nearest neighbor.
    >>> round(balanced_kendalltau(np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14, 0]), beta=0), 5) # strong break of trustworthiness
    0.83258
    >>> # order of importance is defined by internally sorting the first sequence.
    >>> round(balanced_kendalltau(np.array([15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]), np.array([0,14,13,12,11,10,9,8,7,6,5,4,3,2,1]), beta=0), 5) # strong break of trustworthiness
    0.83258
    >>> # weaker break of trustworthiness = an intermediate median distance value (one with intermediate weight) becomes the nearest neighbor.
    >>> round(balanced_kendalltau(np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), np.array([1,2,3, 0,5,6,7,8,9,10,11,12,13,14,15]), beta=0), 5) # weaker break of trustworthiness
    0.88332
    >>> round(balanced_kendalltau(np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), np.array([17, 2,3,4,5,6,7,8,9,10,11,12,13,14,15]), beta=0), 5) # strong break of continuity
    0.53172
    >>> round(balanced_kendalltau(np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), np.array([1,2,3,17,5,6,7,8,9,10,11,12,13,14,15]), beta=0), 5) # weaker break of continuity
    0.76555
    """
    if beta == 1:
        tau_local = 1
    else:
        idx = np.argsort(unordered_values, kind="stable")
        tau_local = weightedtau(unordered_values, unordered_values_, weigher=lambda r: 1 / pi * gamma / (gamma ** 2 + r ** 2), rank=idx)[0]
    tau_global = 1 if beta == 0 else kendalltau(unordered_values, unordered_values_)[0]
    return geomean_np(tau_local, tau_global, beta)


def sortedness(X, X_, i=None, symmetric=True, f=balanced_kendalltau, return_pvalues=False, parallel=True,
               parallel_n_trigger=500, parallel_kwargs=None, **kwargs):
    """
     Calculate the sortedness (correlation-based measure that focuses on ordering of points) value for each point
     Functions available as scipy correlation coefficients:
         ρ-sortedness (Spearman): f=spearmanr
         𝜏-sortedness (Kendall's 𝜏): f=kendalltau
         w𝜏-sortedness (Sebastiano Vigna weighted Kendall's 𝜏): f=weightedtau
         balanced sortedness  ← default

    Note:
        Depending on the chosen correlation coefficient:
            Categorical, or pathological data might present values lower than one due to the presence of ties even with a perfect projection.
            Ties might be penalized, as they do not contribute to establishing any order.

    Hints:
        Swap X and X_ to focus trustworthiness instead of continuity.

        Swap two points A and B at X_ to be able to calculate sortedness between A and B in the same space (i.e., originally, `X = X_`):
            `X = [A, B, C, ..., Z]`
            `X_ = [B, A, C, ..., Z]`
            `sortedness(X, X_, i=0)`

    Parameters
    ----------
    X
        matrix with an instance by row in a given space (often the original one)
    X_
        matrix with an instance by row in another given space (often the projected one)
    i
        None:   calculate sortedness for all instances
        `int`:  index of the instance of interest
    symmetric
        True: Take the mean between extrusion and intrusion emphasis
            Equivalent to `(sortedness(a, b, symmetric=False) + sortedness(b, a, symmetric=False)) / 2` at a slightly lower cost.
            Might increase memory usage.
        False: Weight by original distances (extrusion emphasis), not the projected distances.
    f
        Agreement function:
        callable    =   scipy correlation function:
            weightedtau (weighted Kendall’s τ is the default), kendalltau, spearmanr
            Meaning of resulting values for correlation-based functions:
                1.0:    perfect projection          (regarding order of examples)
                0.0:    random projection           (enough distortion to have no information left when considering the overall ordering)
               -1.0:    worst possible projection   (mostly theoretical; it represents the "opposite" of the original ordering)
    return_pvalues
        For scipy correlation functions, return a 2-column matrix 'corr, pvalue' instead of just 'corr'
        This makes more sense for Kendall's tau. [the weighted version might not have yet a established pvalue calculation method at this moment]
        The null hypothesis is that the projection is random, i.e., sortedness = 0.0.
    parallel
        None: Avoid high-memory parallelization
        True: Full parallelism
        False: No parallelism
    parallel_kwargs
        Any extra argument to be provided to pathos parallelization
    parallel_n_trigger
        Threshold to disable parallelization for small n values
    kwargs
        Arguments to be passed to the correlation measure

     Returns
     -------
         ndarray containing a sortedness value per row, or a single float (include pvalues as a second value if requested)


    >>> ll = [[i] for i in range(17)]
    >>> a, b = np.array(ll), np.array(ll[0:1] + list(reversed(ll[1:])))
    >>> b.ravel()
    array([ 0, 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1])
    >>> r = sortedness(a, b, f=weightedtau)
    >>> from statistics import median
    >>> round(min(r), 12), round(max(r), 12), round(median(r),12)
    (-1.0, 0.998638259786, 0.937548981983)

    >>> rnd = np.random.default_rng(0)
    >>> rnd.shuffle(ll)
    >>> b = np.array(ll)
    >>> b.ravel()
    array([ 2, 10,  3, 11,  0,  4,  7,  5, 16, 12, 13,  6,  9, 14,  8,  1, 15])
    >>> r = sortedness(a, b, f=weightedtau)
    >>> r
    array([ 0.24691868, -0.17456491,  0.19184376, -0.18193532,  0.07175694,
            0.27992254,  0.04121859,  0.16249574, -0.03506842,  0.27856259,
            0.40866965, -0.07617887,  0.12184064,  0.24762942, -0.05049511,
           -0.46277399,  0.12193493])
    >>> round(min(r), 12), round(max(r), 12)
    (-0.462773990559, 0.408669653064)
    >>> round(mean(r), 12)
    0.070104521222

    >>> import numpy as np
    >>> from functools import partial
    >>> from scipy.stats import spearmanr, weightedtau
    >>> me = (1, 2)
    >>> cov = eye(2)
    >>> rng = np.random.default_rng(seed=0)
    >>> original = rng.multivariate_normal(me, cov, size=12)
    >>> from sklearn.decomposition import PCA
    >>> projected2 = PCA(n_components=2).fit_transform(original)
    >>> projected1 = PCA(n_components=1).fit_transform(original)
    >>> np.random.seed(0)
    >>> projectedrnd = permutation(original)

    >>> s = sortedness(original, original, f=weightedtau)
    >>> round(min(s), 12), round(max(s), 12), s
    (1.0, 1.0, array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))

    # Measure sortedness between two points in the same space.
    >>> M = original.copy()
    >>> M[0], M[1] = original[1], original[0]
    >>> round(sortedness(M, original, 0, f=weightedtau), 12)
    0.547929184934

    >>> s = sortedness(original, projected2, f=weightedtau)
    >>> round(min(s), 12), round(max(s), 12), s
    (1.0, 1.0, array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
    >>> s = sortedness(original, projected1, f=weightedtau)
    >>> round(min(s), 12), round(max(s), 12)
    (0.393463224666, 0.944810120534)
    >>> s = sortedness(original, projectedrnd, f=weightedtau)
    >>> round(min(s), 12), round(max(s), 12)
    (-0.648305479567, 0.397019507592)

    >>> np.round(sortedness(original, original, f=kendalltau, return_pvalues=True), 12)
    array([[1.0000e+00, 5.0104e-08],
           [1.0000e+00, 5.0104e-08],
           [1.0000e+00, 5.0104e-08],
           [1.0000e+00, 5.0104e-08],
           [1.0000e+00, 5.0104e-08],
           [1.0000e+00, 5.0104e-08],
           [1.0000e+00, 5.0104e-08],
           [1.0000e+00, 5.0104e-08],
           [1.0000e+00, 5.0104e-08],
           [1.0000e+00, 5.0104e-08],
           [1.0000e+00, 5.0104e-08],
           [1.0000e+00, 5.0104e-08]])
    >>> sortedness(original, projected2, f=kendalltau)
    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    >>> sortedness(original, projected1, f=kendalltau)
    array([0.56363636, 0.52727273, 0.81818182, 0.96363636, 0.70909091,
           0.85454545, 0.74545455, 0.92727273, 0.85454545, 0.89090909,
           0.6       , 0.74545455])
    >>> sortedness(original, projectedrnd, f=kendalltau)
    array([ 0.2       , -0.38181818,  0.23636364, -0.09090909, -0.05454545,
            0.23636364, -0.09090909,  0.23636364, -0.63636364, -0.01818182,
           -0.2       , -0.01818182])

    >>> wf = partial(weightedtau, weigher=lambda x: 1 / (x**2 + 1))
    >>> sortedness(original, original, f=wf, return_pvalues=True)
    array([[ 1., nan],
           [ 1., nan],
           [ 1., nan],
           [ 1., nan],
           [ 1., nan],
           [ 1., nan],
           [ 1., nan],
           [ 1., nan],
           [ 1., nan],
           [ 1., nan],
           [ 1., nan],
           [ 1., nan]])
    >>> sortedness(original, projected2, f=wf)
    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    >>> sortedness(original, projected1, f=wf)
    array([0.89469168, 0.89269637, 0.92922928, 0.99721669, 0.86529591,
           0.97806422, 0.94330979, 0.99357377, 0.87959707, 0.92182767,
           0.87256459, 0.87747329])
    >>> sortedness(original, projectedrnd, f=wf)
    array([ 0.23771513, -0.2790059 ,  0.3718005 , -0.16623167,  0.06179047,
            0.40434396, -0.00130294,  0.46569739, -0.67581876, -0.23852189,
           -0.39125007,  0.12131153])
    >>> np.random.seed(14980)
    >>> projectedrnd = permutation(original)
    >>> sortedness(original, projectedrnd, f=weightedtau)
    array([ 0.24432153, -0.19634576, -0.00238081, -0.4999116 , -0.01625951,
            0.22478766,  0.07176118, -0.48092843,  0.19345964, -0.44895295,
           -0.42044773,  0.06942218])
    >>> sortedness(original, np.flipud(original), f=weightedtau)
    array([-0.28741742,  0.36769361,  0.06926091,  0.02550202,  0.21424544,
           -0.3244699 , -0.3244699 ,  0.21424544,  0.02550202,  0.06926091,
            0.36769361, -0.28741742])
    >>> original = np.array([[0],[1],[2],[3],[4],[5],[6]])
    >>> projected = np.array([[6],[5],[4],[3],[2],[1],[0]])
    >>> sortedness(original, projected, f=weightedtau)
    array([1., 1., 1., 1., 1., 1., 1.])
    >>> projected = np.array([[0],[6],[5],[4],[3],[2],[1]])
    >>> sortedness(original, projected, f=weightedtau)
    array([-1.        ,  0.51956213,  0.81695345,  0.98180162,  0.98180162,
            0.81695345,  0.51956213])
    >>> round(sortedness(original, projected, 1, f=weightedtau), 12)
    0.519562134793
    >>> round(sortedness(original, projected, 1, symmetric=False, f=weightedtau), 12)
    0.422638894922
    >>> round(sortedness(projected, original, 1, symmetric=False, f=weightedtau), 12)
    0.616485374665
    >>> round(sortedness(original, projected, f=weightedtau, rank=True)[1], 12)
    0.519562134793
    >>> round(sortedness(original, projected, f=weightedtau, rank=False)[1], 12)  # warning: will consider indexes as ranks!
    0.074070734162
    >>> round(sortedness([[1,2,3,3],[1,2,7,3],[3,4,7,8],[5,2,6,3],[3,5,4,8],[2,7,7,5]], [[7,1,2,3],[3,7,7,3],[5,4,5,6],[9,7,6,3],[2,3,5,1],[1,2,6,3]], 1, f=weightedtau), 12)
    -1.0
    >>> from scipy.stats import weightedtau
    >>> weightedtau.isweightedtau = False  # warning: will deactivate wau's auto-negativation of scores!
    >>> round(sortedness(original, projected, 1, f=weightedtau, rank=None), 12)
    0.275652884819
    >>> weightedtau.isweightedtau = True
    """
    isweightedtau = False
    if hasattr(f, "isweightedtau") and f.isweightedtau:
        isweightedtau = True
        if not symmetric:
            if "rank" in kwargs:  # pragma: no cover
                raise Exception(f"Cannot set `symmetric=False` and provide `rank` at the same time.")
            kwargs["rank"] = None
    if parallel_kwargs is None:
        parallel_kwargs = {}
    npoints = len(X)

    if i is None:
        tmap = mp.ThreadingPool(**parallel_kwargs).imap if parallel and npoints > parallel_n_trigger else map
        pmap = mp.ProcessingPool(**parallel_kwargs).imap if parallel and npoints > parallel_n_trigger else map
        sqdist_X, sqdist_X_ = tmap(lambda M: cdist(M, M, metric='sqeuclidean'), [X, X_])
        D = remove_diagonal(sqdist_X)
        D_ = remove_diagonal(sqdist_X_)
        scores_X, scores_X_ = (-D, -D_) if isweightedtau else (D, D_)
    else:
        pmap = None
        if not isinstance(X, ndarray):
            X, X_ = np.array(X), np.array(X_)
        x, x_ = X[i], X_[i]
        X = np.delete(X, i, axis=0)
        X_ = np.delete(X_, i, axis=0)
        d_ = np.sum((X_ - x_) ** 2, axis=1)
        d = np.sum((X - x) ** 2, axis=1)
        scores_X, scores_X_ = (-d, -d_) if isweightedtau else (d, d_)

    return common(scores_X, scores_X_, i, symmetric, f, isweightedtau, return_pvalues, pmap, kwargs)

