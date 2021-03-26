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

    def forward_prop(self, X):
        """
        Function of forward propagation
        activated by a sigmoid function
        """
        x = np.matmul(self.__W, X) + self.__b  
        sigmoid = 1 / (1 + np.exp(-x))  # g(x) = 1 / (1 + e^{-x})
        self.__A = sigmoid
        return self.__A
