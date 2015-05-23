__author__ = 'Vladimir Iglovikov'

import pandas as pd
from lasagne import layers
from lasagne.nonlinearities import  softmax, rectify
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np
from pylab import *
import seaborn as sns
import time

def fit_convolutional_model(reshaped_train_x, y, image_width, image_height, reshaped_test_x):
    """Convolutional neural network for kaggle digit recognizer competition.
    Convolutional nets run slower than conventional NN's.  This network is limited to speed up execution
    The most obvious way to improve performance at the expense of execution speed would be to increase
    the number of filters in the convolutional layers.

    """
    print("\n\nRunning Convolutional Net.  Optimization progress below\n\n")
    net2 = NeuralNet(
        layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),      #Convolutional layer.  Params defined below
        ('pool1', layers.MaxPool2DLayer),   #Like downsampling, for execution speed
        ('conv2', layers.Conv2DLayer),
        ('hidden3', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],

    input_shape=(None, 1, image_width, image_height),
    conv1_num_filters=7, conv1_filter_size=(3, 3), conv1_nonlinearity=rectify,
    pool1_ds=(3, 3),
    conv2_num_filters=12, conv2_filter_size=(2, 2), conv2_nonlinearity=rectify,
    hidden3_num_units=50,
    output_num_units=10, output_nonlinearity=softmax,

    update_learning_rate=0.05,
    update_momentum=0.7,

    regression=False,
    max_epochs=5,
    verbose=1,
    )
    net2.fit(reshaped_train_x, y)
    predictions = net2.predict(reshaped_test_x)

    #save plot
    train_loss = np.array([i['train_loss'] for i in net2.train_history])
    valid_loss = np.array([i['valid_loss'] for i in net2.train_history])
    plot(train_loss, label='train')
    plot(valid_loss, label='valid')
    savefig('plots/conv_{timestamp}.eps'.format(timestamp=time.time()))

    return(predictions)

def write_preds(preds, fname):
    pd.DataFrame({"ImageId": range(1, len(preds)+1), "Label": preds}).to_csv(fname, index=False, header=True)


# Read data
train = pd.read_csv('../data/train.csv')
train_y = train['label'].values.astype('int32')
# Divide pixel brightnesses by max brightness so they are between 0 and 1
# This helps the network optimizer make changes on the right order of magnitude
pixel_brightness_scaling_factor = train.max().max()


train_x = (train.drop('label', 1).values/pixel_brightness_scaling_factor).astype('float32')

test_x = (pd.read_csv('../data/test.csv').values / pixel_brightness_scaling_factor).astype('float32')
# Fit a (non-convolutional) neural network

# Reshape the array of pixels so a convolutional neural network knows where
# each pixel belongs in the image
image_width = image_height = int(train_x.shape[1] ** 0.5)
train_x_reshaped = train_x.reshape(train_x.shape[0], 1, image_height, image_width)
test_x_reshaped = test_x.reshape(test_x.shape[0], 1, image_height, image_width)
# Run the convolutional neural network
convolution_based_preds = fit_convolutional_model(train_x_reshaped, train_y, image_width, image_height, test_x_reshaped)
write_preds(convolution_based_preds, "convolutional_nn.csv")

print("For the non-convolutional net, check out https://www.kaggle.com/users/9028/danb/digit-recognizer/big-ish-neural-network-in-python")