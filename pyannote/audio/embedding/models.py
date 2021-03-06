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

import keras.backend as K
from keras.models import Model

from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import merge
from keras.layers.pooling import GlobalAveragePooling1D


class TristouNet(object):
    """TristouNet sequence embedding

    Reference
    ---------
    Hervé Bredin, "TristouNet: Triplet Loss for Speaker Turn Embedding"
    Submitted to ICASSP 2017.
    https://arxiv.org/abs/1609.04301

    Parameters
    ----------
    lstm: list, optional
        List of output dimension of stacked LSTMs.
        Defaults to [16, ] (i.e. one LSTM with output dimension 16)
    bidirectional: boolean, optional
        When True, use bi-directional LSTMs
    pooling: {'last', 'average'}
        By default ('last'), only the last output of the last LSTM layer is
        returned. Use 'average' pooling if you want the last LSTM layer to
        return the whole sequence and take the average.
    dense: list, optional
        Number of units of additionnal stacked dense layers.
        Defaults to [16, ] (i.e. add one dense layer with 16 units)
    output_dim: int, optional
        Embedding dimension. Defaults to 16
    space: {'sphere', 'quadrant'}, optional
        When 'sphere' (resp. 'quadrant'), use 'tanh' (resp. 'sigmoid') as
        final activation. Defaults to 'sphere'.
    """

    def __init__(self, lstm=[16,], bidirectional=True, pooling='average',
                 dense=[16,], output_dim=16, space='sphere'):

        self.lstm = lstm
        self.bidirectional = bidirectional
        self.pooling = pooling
        self.dense = dense
        self.output_dim = output_dim
        self.space = space

    def __call__(self, input_shape):
        """Design embedding

        Parameters
        ----------
        input_shape : (n_frames, n_features) tuple
            Shape of input sequence.

        Returns
        -------
        model : Keras model
        """

        inputs = Input(shape=input_shape,
                       name="embedding_input")
        x = inputs

        # stack LSTM layers
        n_lstm = len(self.lstm)
        for i, output_dim in enumerate(self.lstm):

            if self.pooling == 'last':
                # only last LSTM should not return a sequence
                return_sequences = i+1 < n_lstm
            elif self.pooling == 'average':
                return_sequences = True
            else:
                raise NotImplementedError(
                    'unknown "{pooling}" pooling'.format(pooling=self.pooling))

            if i:
                # all but first LSTM
                forward = LSTM(name='forward_{i:d}'.format(i=i),
                               output_dim=output_dim,
                               return_sequences=return_sequences,
                               activation='tanh',
                               dropout_W=0.0,
                               dropout_U=0.0)(forward)
                if self.bidirectional:
                    backward = LSTM(name='backward_{i:d}'.format(i=i),
                                    output_dim=output_dim,
                                    return_sequences=return_sequences,
                                    activation='tanh',
                                    dropout_W=0.0,
                                    dropout_U=0.0)(backward)
            else:
                # first forward LSTM needs to be given the input shape
                forward = LSTM(name='forward_{i:d}'.format(i=i),
                               input_shape=input_shape,
                               output_dim=output_dim,
                               return_sequences=return_sequences,
                               activation='tanh',
                               dropout_W=0.0,
                               dropout_U=0.0)(x)
                if self.bidirectional:
                    # first backward LSTM needs to be given the input shape
                    # AND to be told to process the sequence backward
                    backward = LSTM(name='backward_{i:d}'.format(i=i),
                                    go_backwards=True,
                                    input_shape=input_shape,
                                    output_dim=output_dim,
                                    return_sequences=return_sequences,
                                    activation='tanh',
                                    dropout_W=0.0,
                                    dropout_U=0.0)(x)

        if self.pooling == 'average':
            forward = GlobalAveragePooling1D()(forward)
            if self.bidirectional:
                backward = GlobalAveragePooling1D()(backward)

        # concatenate forward and backward
        if self.bidirectional:
            x = merge([forward, backward], mode='concat', concat_axis=1)
        else:
            x = forward

        # stack dense layers
        for i, output_dim in enumerate(self.dense):
            x = Dense(output_dim,
                      activation='tanh',
                      name='dense_{i:d}'.format(i=i))(x)

        # stack final dense layer
        if self.space == 'sphere':
            activation = 'tanh'
        elif self.space == 'quadrant':
            activation = 'sigmoid'
        x = Dense(self.output_dim, activation=activation)(x)

        # stack L2 normalization layer
        embeddings = Lambda(lambda x: K.l2_normalize(x, axis=-1),
                            name="embedding_output")(x)

        return Model(input=inputs, output=embeddings)
