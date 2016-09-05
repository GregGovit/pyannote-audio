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
# Hervé BREDIN - http://herve.niderb.fr

from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from xarray import DataArray

from pyannote.core import SlidingWindow, SlidingWindowFeature
from pyannote.generators.batch import FileBasedBatchGenerator
from pyannote.generators.fragment import SlidingSegments

from ..generators.yaafe import YaafeMixin

from pyannote.algorithms.clustering.hac import \
    HierarchicalAgglomerativeClustering
from pyannote.algorithms.clustering.hac.model import HACModel
from pyannote.algorithms.clustering.hac.stop import DistanceThreshold
from scipy.spatial.distance import euclidean, pdist, cdist, squareform


class _FasterAverageDistanceModel(HACModel):
    """Euclidean distance between (weighted) average descriptor"""

    def __init__(self):
        super(_FasterAverageDistanceModel, self).__init__(is_symmetric=True)

    def compute_model(self, cluster, parent=None):

        coverage = parent.current_state.label_coverage(cluster)

        indices = parent.features.sliding_window.crop(coverage,
                                                      mode='strict')

        print(indices)
        if len(indices) < 1:
            indices = parent.features.sliding_window.crop(coverage,
                                                          mode='center')

        n = parent.features.getNumber()
        indices = indices[np.where((indices > -1) * (indices < n))]

        return indices

    def compute_merged_model(self, clusters, parent=None):
        return np.vstack([self[cluster] for cluster in clusters])

    def compute_distance(self, cluster1, cluster2, parent=None):

        if not hasattr(self, '_precomputed'):
            print('Precomputing only once')
            # compute overall distance matrix once and for all
            X = parent.features.data
            self._precomputed = squareform(pdist(X, 'euclidean'))

        X1 = self[cluster1]
        X2 = self[cluster2]
        return np.nanmean(self._precomputed[X1][:, X2])


class _AverageDistanceModel(HACModel):
    """Euclidean distance between (weighted) average descriptor"""

    def __init__(self):
        super(_AverageDistanceModel, self).__init__(is_symmetric=True)

    def compute_model(self, cluster, parent=None):

        coverage = parent.current_state.label_coverage(cluster)

        X = parent.features.crop(coverage, mode='strict')

        if len(X) < 1:
            X = parent.features.crop(coverage, mode='center')

        return X

    def compute_merged_model(self, clusters, parent=None):
        return np.vstack([self[cluster] for cluster in clusters])

    def compute_distance(self, cluster1, cluster2, parent=None):
        X1 = self[cluster1]
        X2 = self[cluster2]
        return np.mean(cdist(X1, X2, metric='euclidean'))

class _Model(HACModel):
    """Euclidean distance between (weighted) average descriptor"""

    def __init__(self):
        super(_Model, self).__init__(is_symmetric=True)

    def compute_model(self, cluster, parent=None):

        coverage = parent.current_state.label_coverage(cluster)
        X = parent.features.crop(coverage, mode='strict')

        if len(X) < 1:
            X = parent.features.crop(coverage, mode='center')

        n = len(X)
        x = X.mean(axis=0)

        x = x / np.linalg.norm(x, 2)

        return (x, n)

    def compute_merged_model(self, clusters, parent=None):
        X, N = zip(*[self[cluster] for cluster in clusters])
        x = np.average(X, axis=0, weights=N)
        x = x / np.linalg.norm(x, 2)
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


class _Clustering(HierarchicalAgglomerativeClustering):
    """Speech turn clustering

    Parameters
    ----------
    threshold : float, optional
        Defaults to 1.0.

    Usage
    -----
    >>> clustering = _Clustering()
    >>> # embedding = SlidingWindowFeature
    >>> features = clustering.model.preprocess(embedding)
    >>> result = clustering(starting_point, features=features)
    """

    def __init__(self, threshold=1.0, force=False, logger=None):
        model = _FasterAverageDistanceModel()
        stopping_criterion = DistanceThreshold(threshold=threshold,
                                               force=force)
        super(_Clustering, self).__init__(
            model,
            stopping_criterion=stopping_criterion,
            constraint=None,
            logger=logger)


class SequenceEmbeddingClustering(YaafeMixin, FileBasedBatchGenerator):
    """Clustering based on sequence embedding

    Parameters
    ----------
    sequence_embedding : SequenceEmbedding
        Pre-trained sequence embedding.
    feature_extractor : YaafeFeatureExtractor
        Yaafe feature extractor
    normalize : boolean, optional
        Set to True to z-score normalize
    duration : float, optional
    overlap : float, optional
        Sliding window duration (in seconds) and step (ratio).
        Defaults to 3 seconds windows with 0.5 (50 percent) overlap.

    Usage
    -----
    >>> sequence_embedding = SequenceEmbedding.from_disk('architecture_yml', 'weights.h5')
    >>> feature_extractor = YaafeFeatureExtractor(...)
    >>> clustering = SequenceEmbeddingClustering(sequence_embedding, feature_extractor)
    >>> iterations = clustering.apply('audio.wav')

    See also
    --------
    pyannote.audio.embedding.models.SequenceEmbedding

    """
    def __init__(self, sequence_embedding, feature_extractor, normalize=False,
                 duration=3.0, overlap=0.5):

        # feature sequence
        self.feature_extractor = feature_extractor
        self.normalize = normalize

        # sequence embedding
        self.sequence_embedding = sequence_embedding

        # sliding windows
        self.duration = duration
        self.overlap = overlap

        self.step_ = duration * (1. - overlap)
        generator = SlidingSegments(duration=self.duration, step=self.step_,
                                    source='wav')

        super(SequenceEmbeddingClustering, self).__init__(generator,
                                                          batch_size=-1)

    def signature(self):
        shape = self.get_shape()
        return {'type': 'sequence', 'shape': shape}

    def postprocess_sequence(self, mono_batch):
        return self.sequence_embedding.transform(mono_batch)

    def apply(self, wav):
        """

        Parameter
        ---------
        wav : str
            Path to wav audio file

        Returns
        -------
        predictions : SlidingWindowFeature
        """

        current_file = wav, None, None

        window = SlidingWindow(start=0.,
                               duration=self.duration,
                               step=self.step_)
        data = next(self.from_file(current_file))
        features = SlidingWindowFeature(data, window)

        return features

        # clustering = _Clustering(threshold=1., force=False)
        #
        # return clustering
