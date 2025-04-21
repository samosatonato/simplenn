import numpy as np

from . import nnmodel as nn


class Optimizer:

    """
    Base class for all optimizers.
    
    
    - Loads a model to be optimized.
    """

    def __init__(self, parameters_interface=None):
        self.parameters_interface = parameters_interface


    def __call__(self, parameters_interface):
        self.load_parameters_interface(parameters_interface)


    def load_parameters_interface(self, parameters_interface):
        if isinstance(parameters_interface, nn.ParametersInterface):
            self.parameters_interface = parameters_interface
        else:
            raise ValueError('Parameter interface must be ParametersInterface object.')
    
    def step(self, x, y):

        """
        Step function to be called after each iteration.
        - Updates model parameters using the optimizer.

        # Step
        Step is defined as one full update of all parameters.
        """
        

        raise NotImplementedError
    


class SGD(Optimizer):

    """
    Stochastic Gradient Descent optimizer.
    """

    def __init__(self, learning_rate):
        super().__init__(learning_rate)

        self.momentum = None

    def run(self):
        raise NotImplementedError
    


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
    
