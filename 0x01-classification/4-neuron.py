#!/usr/bin/env python3

import numpy as np

class Neuron():
    """ Class Neuron """

    def __init__(self, nx):
        """
        Args:
            nx: Type int the number of n inputs into the ANN
        """
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

    def forward_prop(self, X):
        """
        Function of forward propagation
        activated by a sigmoid function
        """
        x = np.matmul(self.__W, X) + self.__b  # z (sum of weight and X's)
        sigmoid = 1 / (1 + np.exp(-x))  # (σ): g(z) = 1 / (1 + e^{-z})
        self.__A = sigmoid
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        """

        m = Y.shape[1]
        C = - (1 / m) * np.sum(
            np.multiply(
                Y, np.log(A)) + np.multiply(
                1 - Y, np.log(1.0000001 - A)))
        return C

        def evaluate(self, X, Y):
        """
        Returns: The neuron prediction and the cost
                of the network
        """

        self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        prediction = np.where(self.__A >= 0.5, 1, 0)  # broadcasting
        return prediction, cost

Neuron = __import__('4-neuron').Neuron

lib_train = np.load('C:/Users/lenovo/Desktop/OumaimaI/ml/0x01-classification/data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
A, cost = neuron.evaluate(X, Y)
print(A)
print(cost)