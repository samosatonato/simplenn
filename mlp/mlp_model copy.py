import numpy as np
import mlp.training as training
import nn.evaluation as evaluation
from nn.nn import Module


class Module():

    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)

    def forward(self, x):
        return self.activation(self.net(x))
    
    def net(self, x):
        return self.weights @ x + self.biases

    def backward(self, x):
        pass

    def activation(self, net):
        pass



class Linear(Module):

    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)

    def forward(self, x):
        return super().forward(x)

    def backward(self, x):
        return super().backward(x)

    def activation(self, net):
        return net



class Tanh(Module):

    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)

    def forward(self, x):
        return super().forward(x)

    def activation(self, net):
        return np.tanh(net)



class Heaviside(Module):

    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)

    def forward(self, x):
        return super().forward(x)

    def activation(self, net):
        return np.heaviside(net, 0)



class ReLU(Module):

    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)

    def forward(self, x):
        return super().forward(x)

    def activation(self, net):
        return np.maximum(0, net) 



class Sigmoid(Module):

    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)

    def forward(self, x):
        return super().forward(x)
    
    def activation(self, net):
        return 1 / (1 + np.exp(-net)) 



class Softmax(Module):

    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)

    def forward(self, x):
        return super().forward(x)

    def activation(self, net):
        exps = np.exp(net - np.max(net, axis=0, keepdims=True))
        return exps / np.sum(exps, axis=0, keepdims=True)



class MLP():

    def __init__(self):
        self.modules = []
    
    def _forwardpass(self, x):
        activations = []
        for module in self.modules:
            a = module.forward(x)
            activations.append(a)
        return activations

    def add(self, module):
        self.modules.append(module)

    def train(self, data):
        training.train(self, data)

    def evaluate(self, data):
        evaluation.evaluate(self, data)

    def predict(self, x):
        return self._forwardpass(x)[-1]

    def get_input_dim(self):
        if len(self.modules) == 0:
            raise ValueError('No modules in the model.')
        return self.modules[0].weights.shape[1]
    
    def get_output_dim(self):
        if len(self.modules) == 0:
            raise ValueError('No modules in the model.')
        return self.modules[-1].weights.shape[0]

