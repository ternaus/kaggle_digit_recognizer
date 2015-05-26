from __future__ import division
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from lasagne.nonlinearities import softmax
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import LabelEncoder
__author__ = 'Vladimir Iglovikov'


import pandas as pd

train = pd.read_csv('../data/train.csv')
# test = pd.read_csv('../data/test.csv')

target = train['label']
training = train.drop('label', 1)

scaler = StandardScaler(with_mean=False)

X = scaler.fit_transform(training.values.astype(np.float64)).astype(np.float32)
# X = (training.values / 255.0).astype(np.float32)

# X_test = (test.values / 255.0)
np.random.shuffle(X)

encoder = LabelEncoder()

y = encoder.fit_transform(target.values).astype(np.int32)

net1 = NeuralNet(
     layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),
        ('conv3', layers.Conv2DLayer),
       # ('pool3', layers.MaxPool2DLayer),
        ('dropout3', layers.DropoutLayer),
        ('hidden4', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],

    # layer parameters:
    input_shape=(None, 1, 28, 28),  # 28x28 input pixels per batch

    conv1_num_filters=16, conv1_filter_size=(3, 3), conv1_stride=(2,2), pool1_pool_size=(2, 2),
    conv2_num_filters=16, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=16, conv3_filter_size=(2, 2),# pool3_pool_size=(2, 2),
    dropout3_p = 0.5,
    hidden4_num_units=32, dropout4_p=0.5,
    hidden5_num_units=32,
    output_num_units=10,
    output_nonlinearity=softmax,

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.3,
    update_momentum=0.9,
    use_label_encoder=True, #What is this?
    regression=False,  # flag to indicate we're dealing with regression problem
    max_epochs=10,  # we want to train this many epochs
    verbose=1,
    )


X_reshaped = X.reshape(-1, 1, 28, 28)

net1.fit(X_reshaped, y)