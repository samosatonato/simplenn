"""
Machine module
"""

from . import dataprocessor as dp
from . import optimizers as opt

class Machine:

    """
    A class handling the neural network model, training and validation.
    """

    def __init__(self,
            model=None,
            lossfunction=None,
            optimizer=None,
            evaluator=None,
            hypertuner=None,):
        

        self.model = model
        self.lossfunction = lossfunction
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.hypertuner = hypertuner



    def learn(self, x, y):
        self.train(x, y)

    def train(self, x, y, batch_size=1, epochs=1):
        
        # TODO: add loss tracking (hooks)
        if self.hypertuner is None:
            
            # Load parameter interface to allow the optimizer to adjust model weights and biases
            self.optimizer(self.model.parameters_interface())
            for epoch in range(epochs):

                batches = self.dataprocessor.batchify(x, y, batch_size=batch_size)
                for x_batch, y_batch in batches:
                    g_batch = self.model.predict(x_batch)
                    loss = self.lossfunction(g_batch, y_batch)
                    self.model.gradient(loss)
                    self.optimizer.step()



    def evaluate(self, x, y, evaluator):
        self.test(x, y, evaluator)

    def test(self, x, y, evaluator):
        pass


    def predict(self, x):
        return self.model.predict(x)
