from __future__ import division
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from lasagne.nonlinearities import softmax
from sklearn.preprocessing import StandardScaler
import numpy as np

__author__ = 'Vladimir Iglovikov'


import pandas as pd

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/train.csv')

target = train['label']
training = train.drop('label', 1)

scaler = StandardScaler()
X = scaler.fit_transform(training).astype(np.float32)


net1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 28 * 28),  # 28x28 input pixels per batch
    hidden_num_units=100,  # number of units in hidden layer
    output_nonlinearity=softmax,  # output layer uses identity function
    output_num_units=1,  # 1 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=False,  # flag to indicate we're dealing with regression problem
    max_epochs=400,  # we want to train this many epochs
    verbose=1,
    )


net1.fit(X, target.astype(np.int32))