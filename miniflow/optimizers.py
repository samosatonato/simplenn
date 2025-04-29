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

    def __init__(self, learning_rate=0.01, momentum=0.0):
        super().__init__()

        self.learning_rate = learning_rate
        
        self.momentum = momentum
        self.vel_w = None
        self.vel_b = None

        self.gradients = None

    def add_model(self, model):
        super().add_model(model)
    
        if self.momentum > 0:
            self.velocities_w = [np.zeros_like(l.get_weights()) for l in self.model.layers]
            self.velocities_b = [np.zeros_like(l.get_biases()) for l in self.model.layers]

    def step(self, gradients):
        dw_list, db_list = map(list, zip(*gradients))

        for i in range(len(self.model.layers)):
            dw = dw_list[-(i+1)]
            db = db_list[-(i+1)]

            if self.momentum > 0:
                self.velocities_w[i] = self.momentum * self.velocities_w[i] + self.learning_rate * dw
                self.velocities_b[i] = self.momentum * self.velocities_b[i] + self.learning_rate * db
                delta_w = self.velocities_w[i]
                delta_b = self.velocities_b[i]
            else:
                delta_w = self.learning_rate * dw
                delta_b = self.learning_rate * db

            self.model.layers[i].sub_weights(delta_w)
            self.model.layers[i].sub_biases(delta_b)


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
    
