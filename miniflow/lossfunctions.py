import numpy as np
from . import activations



class LossFunction:

    """
    Base class for all loss functions.

    - This class computes loss using specific loss function.
    - Also implements unified forward and backward method.

    **Notation**:
    - g: predicted output
    - y: true/target output
    """

    eps = 1e-9

    def __init__(self):
        self.model = None

        self.g = None
        self.y = None

        self.lossval = None  # Loss value
        self.lossgrad = None # Loss gradient
        self.delta_net = None


    def __call__(self, g, y):
        
        """
        Call the loss function.
        """
        self.loss_d(g, y)

        activation = self.model.layers[-1].activation
        if activation is not None:
            if not isinstance(activation, activations.Activation):
                raise ValueError
            self._delta_net(g, y, activation)

        return self.loss(g, y)

    def add_model(self, model):
        self.model = model

    def loss(self, g, y):

        """
        Forward pass of the loss function.
        """
        
        self.lossval = self._loss(g, y)

        return self.lossval

    def _loss(self, g, y):

        """
        Loss function.
        """

        pass

    def loss_d(self, g, y):

        """
        Gradient of the loss function.
        """

        self.lossgrad = self._loss_gradient(g, y)

        return self.lossgrad

    def _loss_gradient(self, g, y):
        
        """
        Derivative of the loss function.
        """

        pass
    
    def _delta_net(self, y, g, activation):
        pass



class MeanSquaredErrorLoss(LossFunction):

    def _loss(self, g, y):
        return np.sum((g - y) ** 2) / g.shape[0]
        
    def _loss_gradient(self, g, y):
        return 2 * (g - y) / g.shape[0]

class CategoricalCrossEntropyLoss(LossFunction):

    def _loss(self, g, y):
        return -1 * np.sum(y * np.log(g + self.eps), axis=1)

    def _loss_gradient(self, g, y):
        return -1 * (y / (g + self.eps))
    
    def _delta_net(self, g, y, activation):
        if isinstance(activation, activations.Softmax):
            self.delta_net = g - y 
        else:
            self.delta_net = self.lossgrad * activation.d(g)

    
class BinaryCrossEntropyLoss(LossFunction):

    def _loss(self, g, y):
        return -1 * np.sum(y * np.log(g + self.eps) + (1 - y) * np.log(1 - g + self.eps))

    def _loss_gradient(self, g, y):
        return -1 * (y / g) + (1 - y) / (1 - g)

class SparseCategoricalCrossEntropyLoss(LossFunction):

    pass

