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

import scipy.signal
import numpy as np
from pyannote.core import Segment, Timeline, Annotation
from pyannote.core.util import pairwise


class Peak(object):
    """Peak detection

    Peaks are detected on min_duration windows.
    Only peaks greater than p(5) + alpha * (p(95) - p(5)) are kept
    where p(x) is xth percentile.

    Parameters
    ----------
    alpha : float, optional
        Adaptative threshold coefficient. Defaults to 0.5
    min_duration : float, optional
        Defaults to 1 second.

    """
    def __init__(self, alpha=0.5, min_duration=1.0):
        super(Peak, self).__init__()
        self.alpha = alpha
        self.min_duration = min_duration

    def apply(self, predictions):
        """Peak detection

        Parameter
        ---------
        predictions : SlidingWindowFeature
            Predictions returned by segmentation approaches.

        Returns
        -------
        segmentation : Timeline
            Partition.
        """
        y = predictions.data
        sw = predictions.sliding_window

        precision = sw.step
        order = int(np.rint(self.min_duration / precision))
        indices = scipy.signal.argrelmax(y, order=order)[0]

        mini = np.nanpercentile(y, 5)
        maxi = np.nanpercentile(y, 95)
        threshold = mini + self.alpha * (maxi - mini)

        peak_time = np.array([sw[i].middle for i in indices if y[i] > threshold])

        n_windows = len(y)
        start_time = sw[0].start
        end_time = sw[n_windows].end

        boundaries = np.hstack([[start_time], peak_time, [end_time]])
        segmentation = Timeline()
        for i, (start, end) in enumerate(pairwise(boundaries)):
            segment = Segment(start, end)
            segmentation.add(segment)

        return segmentation


class Binarize(object):
    """Binarize predictions using onset/offset thresholding"""

    def __init__(self, onset=0.7, offset=0.7):

        super(Binarize, self).__init__()
        self.onset = onset
        self.offset = offset

        self.min_duration = [0., 0.]
        self.pad_onset = 0.
        self.pad_offset = 0.

    def apply(self, predictions, dimension=0):
        """
        Parameters
        ----------
        predictions : SlidingWindowFeature
            Must be mono-dimensional
        dimension : int, optional
            Which dimension to process
        """

        if len(predictions.data.shape) == 1:
            data = predictions.data
        elif predictions.data.shape[1] == 1:
            data = predictions.data[:, 0]
        else:
            data = predictions.data[:, dimension]

        n_samples = predictions.getNumber()
        window = predictions.sliding_window
        timestamps = [window[i].middle for i in range(n_samples)]

        # initial state
        start = timestamps[0]
        label = data[0] > self.onset

        # timeline meant to store 'active' segments
        active = Timeline()

        for t, y in zip(timestamps[1:], data[1:]):

            # currently active
            if label:
                # switching from active to inactive
                if y < self.offset:
                    segment = Segment(start - self.pad_onset,
                                      t + self.pad_offset)
                    active.add(segment)
                    start = t
                    label = False

            # currently inactive
            else:
                # switching from inactive to active
                if y > self.onset:
                    start = t
                    label = True

        # if active at the end, add final segment
        if label:
            segment = Segment(start - self.pad_onset, t + self.pad_offset)
            active.add(segment)

        # because of padding, some 'active' segments might be overlapping
        # therefore, we merge those overlapping segments
        active = active.coverage()

        # remove short 'active' segments
        active = Timeline(
            [s for s in active if s.duration > self.min_duration[1]])

        # fill short 'inactive' segments
        inactive = active.gaps()
        for s in inactive:
            if s.duration < self.min_duration[0]:
                active.add(s)
        active = active.coverage()

        return active
