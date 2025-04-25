import numpy as np

from . import model as nn


class Optimizer:

    def __init__(self):
        
        self.model = None


    def __call__(self):
        pass

    def add_model(self, model):
        self.model = model
    
    def step(self):

        raise NotImplementedError
    

    def update_weights(self):
        pass

    def update_biases(self):
        pass


class SGD(Optimizer):

    """
    Stochastic Gradient Descent optimizer.
    """

    def __init__(self, learning_rate=None, momentum=None):
        super().__init__()

        self.learning_rate = learning_rate
        self.momentum = momentum

        self.gradients = None


    def step(self, gradients):
        self.dw, self.db = map(list, zip(*gradients))

        delta_w = self.update_weights(self.dw)
        delta_b = self.update_biases(self.db)

        self.model.sub_weights(delta_w)
        self.model.sub_biases(delta_b)


    def update_weights(self, dw):
        return [w*self.learning_rate for w in dw]
        
    def update_biases(self, db):
        return [b*self.learning_rate for b in db]


class RMSProp(SGD):

    """
    RMSProp optimizer.
    """

    def __init__(self, learning_rate):
        super().__init__(learning_rate)


    def run(self):
        raise NotImplementedError


class Adam(SGD):

    """
    Adam optimizer.
    """

    def __init__(self, learning_rate):
        super().__init__(learning_rate)


    def run(self):
        raise NotImplementedError
    
