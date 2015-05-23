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
import theano
import cPickle as pickle

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()


def float32(k):
    return np.cast['float32'](k)



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
    pool1_pool_size=(3, 3),
    conv2_num_filters=12, conv2_filter_size=(2, 2), conv2_nonlinearity=rectify,
    hidden3_num_units=50,
    output_num_units=10, output_nonlinearity=softmax,


    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    regression=False,
    max_epochs=1000,
    verbose=1,
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        EarlyStopping(patience=200),
        ],
    )
    net2.fit(reshaped_train_x, y)
    predictions = net2.predict(reshaped_test_x)

    #save plot
    train_loss = np.array([i['train_loss'] for i in net2.train_history_])
    valid_loss = np.array([i['valid_loss'] for i in net2.train_history_])
    plot(train_loss, label='train')
    plot(valid_loss, label='valid')
    grid()
    legend()
    savefig('plots/conv_{timestamp}.png'.format(timestamp=time.time()))

    #save model
    try:
      os.mkdir('models')
    except:
      pass

    with open('net2.pickle', 'wb') as f:
      pickle.dump(net2, f, -1)

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