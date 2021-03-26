#!/usr/bin/env python3

import numpy as np


class Neuron:
   """ Class Neuron """
   
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.W = np.random.normal(0, 1, (1, nx)) # Weight
        self.b = 0 # Bias
        self.A = 0 # Output
