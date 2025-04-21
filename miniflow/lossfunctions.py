import numpy as np



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

    def __init__(self, g, y):

        if y.shape[0] != g.shape[0]:
            raise ValueError('x and y must have the same number of samples.')
        
        self.g = g
        self.y = y

        self.lossval = None  # Loss value
        self.lossgrad = None # Loss gradient


    def __call__(self, g, y):
        
        """
        Call the loss function.
        """

        return self.get_loss(g, y)


    def get_loss(self, g, y):

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

    def get_loss_gradient(self, g, y):

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



class MeanSquaredErrorLoss(LossFunction):

    def _loss(self, g, y):
        return np.sum((g - y) ** 2) / g.shape[0]
        
    def _loss_gradient(self, g, y):
        return 2 * (g - y) / g.shape[0]

class CategoricalCrossEntropyLoss(LossFunction):

    def _loss(self, g, y):
        return -1 * np.sum(y * np.log(g+self.eps))

    def _loss_gradient(self, g, y):
        return -1 * (y / (g+self.eps))
    
class BinaryCrossEntropyLoss(LossFunction):

    def _loss(self, g, y):
        return -1 * np.sum(y * np.log(g + self.eps) + (1 - y) * np.log(1 - g + self.eps))

    def _loss_gradient(self, g, y):
        return -1 * (y / g) + (1 - y) / (1 - g)

class SparseCategoricalCrossEntropyLoss(LossFunction):

    raise NotImplementedError('SparseCategoricalCrossEntropyLoss not implemented yet!')

