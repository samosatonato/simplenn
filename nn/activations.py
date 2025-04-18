import numpy as np

import nn.nn as nn



class Linear(nn.Activation):

    def _activation(self, net):
        return net

    def _activation_derivative(self, x):
        return np.ones_like(x)

class Tanh(nn.Activation):

    def _activation(self, net):
        return np.tanh(net)

    def _activation_derivative(self, x):
        return 1 - np.tanh(x) ** 2

class Heaviside(nn.Activation):

    def _activation(self, net):
        return np.heaviside(net, 0)

    def _activation_derivative(self, x):
        return np.zeros_like(x)

class ReLU(nn.Activation):

    def _activation(self, net):
        return np.maximum(0, net) 

    def _activation_derivative(self, x):
        return np.where(x > 0, 1, 0)

class Sigmoid(nn.Activation):

    def _activation(self, net):
        return 1 / (1 + np.exp(-net)) 

    def _activation_derivative(self, x):
        return x * (1 - x)

class Softmax(nn.Activation):

    def _activation(self, net):
        exps = np.exp(net - np.max(net, axis=0, keepdims=True))
        return exps / np.sum(exps, axis=0, keepdims=True)

    def _activation_derivative(self, x):
        return x * (1 - x)
    
