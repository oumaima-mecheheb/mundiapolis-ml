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
