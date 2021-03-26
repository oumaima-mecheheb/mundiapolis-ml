#!/usr/bin/env python3

import numpy as np


class NeuralNetwork():
    """ Class neural network"""

    def __init__(self, nx, nodes):

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nx, nodes).reshape(nodes, nx)
        self.__b1 = np.zeros(nodes).reshape(nodes, 1)
        self.__A1 = 0
        self.__W2 = np.random.randn(nodes).reshape(1, nodes)
        self.__b2 = 0
        self.__A2 = 0


    @property
    def W1(self):
        """
            self: Private attribute
            Returns: Weight vector 1 hidden layer
        """
        return self.__W1

    @property
    def b1(self):
        """
            self: Private attribute
            Returns: Bias1
        """
        return self.__b1

    @property
    def A1(self):
        """
            self: Private attribute
            Returns: Activated1
        """
        return self.__A1

    @property
    def W2(self):
        """
            self: Private attribute
            Returns: Weight vector 2
        """
        return self.__W2

    @property
    def b2(self):
        """
            self: Private attribute
            Returns: Bias2
        """
        return self.__b2

    @property
    def A2(self):
        """
            self: Private attribute
            Returns: Activated output 2 prediction
        """
        return self.__A2



NN = __import__('9-neural_network').NeuralNetwork

lib_train = np.load('C:/Users/lenovo/Desktop/OumaimaI/ml/0x01-classification/data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
nn = NN(X.shape[0], 3)
print(nn.W1)
print(nn.b1)
print(nn.W2)
print(nn.b2)
print(nn.A1)
print(nn.A2)
nn.A1 = 10
print(nn.A1)