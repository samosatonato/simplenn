import numpy as np

import nn.nn as nn
import nn.data_processing as data_proc


class ParameterOptimizer:

    """
    Base class for all optimizers.
    """

    def __init__(self, data_processor=None, model=None):
        default_data_processor = data_proc.DataProcessor()
        self.data_processor = default_data_processor or data_processor

        self.model = None  # Model to be optimized


    def run():
        raise NotImplementedError
    
    def load_model(self, model):
        if self.model is not None:
            raise ValueError('Model already loaded.')
        self.model = model
        return self.model
    
    def step(self, x, y):

        """
        Step function to be called after each iteration.
        - Updates model parameters using the optimizer.
        """
        
        raise NotImplementedError
    

class GradientDescent(ParameterOptimizer):

    """
    Base class for all gradient descent optimizers.
    """

    def __init__(self, learning_rate):
        super().__init__(learning_rate)


    def run(self):
        raise NotImplementedError
    


class SGD(GradientDescent):

    """
    Stochastic Gradient Descent optimizer.
    """

    def __init__(self, learning_rate):
        super().__init__(learning_rate)


    def run(self):
        raise NotImplementedError
    

class BGD(GradientDescent):

    """
    Batch Gradient Descent optimizer.
    """

    def __init__(self, learning_rate):
        super().__init__(learning_rate)


    def run(self):
        raise NotImplementedError


class MBGD(GradientDescent):

    """
    Mini-Batch Gradient Descent optimizer.
    """

    def __init__(self, learning_rate):
        super().__init__(learning_rate)


    def run(self):
        raise NotImplementedError


class Adam(GradientDescent):

    """
    Adam optimizer.
    """

    def __init__(self, learning_rate):
        super().__init__(learning_rate)


    def run(self):
        raise NotImplementedError
    
