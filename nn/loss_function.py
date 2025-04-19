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

    def __init__(self, input_size=None, output_size=1):
        self.input_size = input_size
        self.output_size = output_size

        self.loss = None  # Loss value
        self.keep_loss_history = True
        self.loss_history = []  # Loss history

    def __call__(self, x, y):
        """
        Call the loss function.
        """

        return self.forward(x, y)


    def forward(self, x, y):

        """
        Forward pass of the loss function.
        """

        if x.shape[0] != y.shape[0]:
            raise ValueError('x and y must have the same number of samples.')
        
        self.loss = self.loss(x, y)

        if self.keep_loss_history:
            self.loss_history.append(self.loss)    

        return self.loss

    def _loss(self, x, y):

        """
        Loss function.
        """

        pass


    def backward(self, x, y):

        """
        Backward pass of the loss function.
        """

        return self._loss_derivative(x, y)

    def _loss_derivative(self, x, y):
        
        """
        Derivative of the loss function.
        """

        pass



class MeanSquaredErrorLoss(LossFunction):

    def _loss(self, predictions, labels):
        return np.sum((predictions - labels) ** 2) / predictions.shape[1]
        
    def _loss_derivative(self, predictions, labels):
        return 2 * (predictions - labels) / predictions.shape[1]

class CrossEntropyLoss(LossFunction):

    def _loss(self, predictions, labels):
        return -1 * np.sum(labels * np.log(predictions))

    def _loss_derivative(self, predictions, labels):
        return -1 * (labels / predictions) / labels.shape[1]