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

    def forward_prop(self, X):
        """
        Le calcule de la propagation directe du réseau de neurones
         Le paramètre X: tableau np avec les données d'entrée de forme (nx, m)
         elle retourne : les attributs privés __A1 et __A2
        """
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        calcule le coût du modèle à l'aide de la régression logistique
         elle retourne : le coût
        """
        cost = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = np.sum(cost)
        cost = - cost / A.shape[1]
        return cost

    def evaluate(self, X, Y):
        """
        évalue la prédiction du réseau neuronal
        """
        self.forward_prop(X)
        prediction = np.where(self.__A2 >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A2)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Le calcule une passe de descente de gradient sur le neurone
        """
        # gradient descent for hidden layer
        dz2 = A2 - Y
        dw2 = np.matmul(A1, dz2.T) / A1.shape[1]
        db2 = np.sum(dz2, axis=1, keepdims=True) / A2.shape[1]

        # derivative of the sigmoid function
        da1 = A1 * (1 - A1)
        # gradient descent for output layer
        dz1 = np.matmul(self.__W2.T, dz2)
        dz1 = dz1 * da1
        dw1 = np.matmul(X, dz1.T) / A1.shape[1]
        db1 = np.sum(dz1, axis=1, keepdims=True) / A1.shape[1]
        # updated value for weights and bias
        self.__W2 = self.__W2 - alpha * dw2.T
        self.__b2 = self.__b2 - alpha * db2
        self.__W1 = self.__W1 - alpha * dw1.T
        self.__b1 = self.__b1 - alpha * db1

NN = __import__('13-neural_network').NeuralNetwork

lib_train = np.load('C:/Users/lenovo/Desktop/OumaimaI/ml/0x01-classification/data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
nn = NN(X.shape[0], 3)
A1, A2 = nn.forward_prop(X)
nn.gradient_descent(X, Y, A1, A2, 0.5)
print(nn.W1)
print(nn.b1)
print(nn.W2)
print(nn.b2)