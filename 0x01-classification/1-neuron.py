#!/usr/bin/env python3

import numpy as np


class Neuron():
    """ Class Neuron """

    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(nx).reshape(1, nx)  # Weight
        self.__b = 0  # Bias
        self.__A = 0  # output

    @property
    def W(self):
        """
        Returns: private instance weight

        """
        return self.__W

    @property
    def b(self):
        """
        Returns: private instance bias

        """
        return self.__b

    @property
    def A(self):
        """
        Returns: private instance output

        """
        return self.__A

Neuron = __import__('1-neuron').Neuron

lib_train = np.load('C:/Users/lenovo/Desktop/OumaimaI/ml/0x01-classification/data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T


np.random.seed(0)
neuron = Neuron(X.shape[0])
print(neuron.W)
print(neuron.b)
print(neuron.A)
neuron.A = 10
print(neuron.A)