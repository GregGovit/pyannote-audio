#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Herve BREDIN - http://herve.niderb.fr

from __future__ import print_function
from __future__ import unicode_literals

from pyannote.algorithms.clustering.hac import \
    HierarchicalAgglomerativeClustering
from pyannote.algorithms.clustering.hac.model import HACModel
from pyannote.algorithms.clustering.hac.stop import DistanceThreshold
from scipy.spatial.distance import euclidean, pdist, cdist, squareform
import numpy as np
from xarray import DataArray


class _Model(HACModel):
    """Euclidean distance between (weighted) average descriptor"""

    def __init__(self):
        super(_Model, self).__init__(is_symmetric=True)

    def compute_model(self, cluster, parent=None):

        coverage = parent.current_state.label_coverage(cluster)
        X = parent.features.crop(coverage, mode='strict')

        n = len(X)
        x = X.mean(axis=0)

        return (x, n)

    def compute_merged_model(self, clusters, parent=None):
        X, N = zip(*[self[cluster] for cluster in clusters])
        x = np.average(X, axis=0, weights=N)
        n = np.sum(N)
        return (x, n)

    def compute_distance_matrix(self, parent=None):
        clusters = list(self._models)
        X = np.vstack([self[cluster][0] for cluster in clusters])
        return DataArray(
            squareform(pdist(X, 'euclidean')),
            [('i', clusters), ('j', clusters)])

    def compute_distances(self, cluster, clusters, dim='i', parent=None):
        x = self[cluster][0].reshape((1, -1))
        X = np.vstack([self[c][0] for c in clusters])
        return DataArray(
            cdist(x, X, metric='euclidean').reshape((-1, )),
            [(dim, clusters)])

    def compute_distance(self, cluster1, cluster2, parent=None):
        x1, _ = self[cluster1]
        x2, _ = self[cluster2]
        return euclidean(x1, x2)


class SpeechTurnClustering(HierarchicalAgglomerativeClustering):
    """Face clustering

    Parameters
    ----------
    threshold : float, optional
        Defaults to 1.0.

    Usage
    -----
    >>> clustering = SpeechTurnClustering()
    >>> # embedding = SlidingWindowFeature
    >>> features = clustering.model.preprocess(embedding)
    >>> result = clustering(starting_point, features=features)
    """

    def __init__(self, threshold=1.0, force=False, logger=None):
        model = _Model()
        stopping_criterion = DistanceThreshold(threshold=threshold,
                                               force=force)
        super(SpeechTurnClustering, self).__init__(
            model,
            stopping_criterion=stopping_criterion,
            constraint=None,
            logger=logger)
