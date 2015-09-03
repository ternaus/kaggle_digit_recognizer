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
import os
from pylab import *
import seaborn as sns

from pylab import *

from sklearn.utils import shuffle
import cPickle as pickle

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

target = train['label']
training = train.drop('label', 1)

scaler = StandardScaler(with_mean=False)

X = scaler.fit_transform(training.values.astype(np.float64)).astype(np.float32)
# X = (training.values / 255.0).astype(np.float32)

# X_test = (test.values / 255.0).astype(np.float32)

X_test = scaler.transform(test.values.astype(np.float64)).astype(np.float32)
# np.random.shuffle(X)

encoder = LabelEncoder()

y = encoder.fit_transform(target.values).astype(np.int32)

random_state = 42

params = {
  'update_learning_rate': 0.01,
  'update_momentum': 0.9,
  'max_epochs': 10
}

method = 'convolutional_update_learning_rate{ulr}_update_momentum{um}_max_epochs{me}'.format(ulr=params['update_learning_rate'],
                                                                               um=params['update_momentum'],
                                                                               me=params['max_epochs'])
net1 = NeuralNet(
  input_shape=(None, 1, 28, 28),  # 28x28 input pixels per batch
  # layer parameters:
  layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    conv1_num_filters=32, 
    conv1_filter_size=(3, 3), 
    pool1_pool_size=(2, 2),
    conv2_num_filters=64, 
    conv2_filter_size=(2, 2), 
    pool2_pool_size=(2, 2),
    conv3_num_filters=128, 
    conv3_filter_size=(2, 2), 
    pool3_pool_size=(2, 2),
    hidden4_num_units=500,
    hidden5_num_units=500,

    output_nonlinearity=softmax,  # output layer uses identity function
    output_num_units=10,  # 1 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=params['update_learning_rate'],
    update_momentum=params['update_momentum'],
    use_label_encoder=True,
    regression=False,  # flag to indicate we're dealing with regression problem
    max_epochs=params['max_epochs'],  # we want to train this many epochs
    verbose=1,
    )


X, y = shuffle(X, y, random_state=random_state)

X_reshaped = X.reshape(-1, 1, 28, 28)

net1.fit(X_reshaped, y)

X_test_reshaped = X_test.reshape(-1, 1, 28, 28)

#save model to file

try:
  os.mkdir('models')
except:
  pass

with open('models/conv.pickle', 'wb') as f:
  pickle.dump(net1, f, -1)


#save plot to file

try:
  os.mkdir('plots')
except:
  pass

train_loss = np.array([i["train_loss"] for i in net1.train_history_])
valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])

plot(train_loss, linewidth=3, label='train')
plot(valid_loss, linewidth=3, label='valid')
yscale("log")
legend()
savefig('plots/{method}.png'.format(method=method))

#predicting
predictions = net1.predict(X_test_reshaped)

try:
  os.mkdir('predictions')
except:
  pass

pd.DataFrame({"ImageId": range(1, len(predictions) + 1), "Label": predictions}).to_csv('predictions/' + method +'.csv',
                                                                                       index=False,
                                                                                       header=True)