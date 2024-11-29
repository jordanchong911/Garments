import numpy as np
import pandas as pd

class AnalyticalMethod(object):

    def __init__(self):
        """Class constructor for AnalyticalMethod
        """
        self.W = None

    def feature_transform(self, X):
        """Appends a vector of ones for the bias term.

        Arguments:
            X {np.ndarray} -- A numpy array of shape (N, D) consisting of N
            samples each of dimension D.

        Returns:
            np.ndarray -- A numpy array of shape (N, D + 1)
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        f_transform = np.hstack([np.ones((X.shape[0], 1)), X])

        return f_transform

    def compute_weights(self, X, y):
        """Compute the weights based on the analytical solution.

        Arguments:
            X {np.ndarray} -- A numpy array of shape (N, D) containing the
            training data; there are N training samples each of dimension D.
            y {np.ndarray} -- A numpy array of shape (N, 1) containing the
            ground truth values.

        Returns:
            np.ndarray -- weight vector; has shape (D, 1) for dimension D
        """
        
        X = self.feature_transform(X)

        self.W = np.linalg.pinv(X.T @ X) @ X.T @ y

        return self.W

    def predict(self, X):
        """Predict values for test data using analytical solution.

        Arguments:
            X {np.ndarray} -- A numpy array of shape (num_test, D) containing
            test data consisting of num_test samples each of dimension D.

        Returns:
            np.ndarray -- A numpy array of shape (num_test, 1) containing
            predicted values for the test data, where y[i] is the predicted
            value for the test point X[i].
        """

        X = self.feature_transform(X)
        prediction = X @ self.W

        return prediction

import torch.nn as nn
import torch.nn.init

class DataLoader(object):
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size

        self.indices = np.array([i for i in range(self.X.shape[0])])
        np.random.seed(1)

    def shuffle(self):
        np.random.shuffle(self.indices)

    def get_batch(self, mode='train'):
        """Returns self.X and self.y divided into different batches of size
        self.batch_size according to the shuffled self.indices."""

        X_batch = []
        y_batch = []

        if mode == 'train':
            self.shuffle()
        elif mode == 'test':
            self.indices = np.array([i for i in range(self.X.shape[0])])

        for i in range(0, len(self.indices), self.batch_size):
            if i + self.batch_size <= len(self.indices):
                indices = self.indices[i:i + self.batch_size]
            else:
                indices = self.indices[i:]

            X_batch.append(self.X[indices])
            y_batch.append(self.y[indices])

        return X_batch, y_batch

class NeuralNetwork(nn.Module):

    def __init__(self,
                 input_size,
                 num_classes,
                 list_hidden):
        super(NeuralNetwork, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.list_hidden = list_hidden

    def create_network(self):
        layers = []

        layers.append(torch.nn.Linear(in_features=self.input_size, out_features=self.list_hidden[0]))
        layers.append(nn.ReLU())

        for i in range(len(self.list_hidden) - 1):
            layers.append(torch.nn.Linear(in_features=self.list_hidden[i], out_features=self.list_hidden[i+1]))
            layers.append(nn.ReLU())

        layers.append(torch.nn.Linear(in_features=self.list_hidden[-1], out_features=1)) # made output neuron always one
        
        self.layers = nn.Sequential(*layers) # removed softmax layer

    def init_weights(self):
        torch.manual_seed(2)

        for module in self.modules():

            if isinstance(module, nn.Linear):

                nn.init.normal_(module.weight, mean=0, std=0.1)

                nn.init.constant_(module.bias, 0)

    def forward(self,
                x,
                verbose=False):

        for i, layer in enumerate(self.layers):
            x = layer(x)

        if verbose:
            print(f'Output of layer {i}:', x, '\n')

        return x  # final output for regression