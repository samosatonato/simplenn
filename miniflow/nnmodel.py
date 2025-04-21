"""
# Core module for neural network construction.

- This module contains the base classes for the construction of neural networks.
- The goal is to strike a balance between programmatic and theoretic modularity.

**Notation**:

- x - input tensor
- w - weight tensor
- b - bias tensor
- net - linear transformation of the input tensor (WX + B)
- a - activation tensor (activation(net))
- g - output prediction tensor (the last activation in the neural network)
- y - true/target output tensor
- c - loss scalar (vector in case of batches) (loss(G, Y))
- d - number of classes
- n - number of samples
- input_dim - dimension of input features
- output_dim - dimension of labels
"""

import numpy as np

from . import modules
from . import layers
from . import activations
from . import hypertuner
from . import lossfunctions as lf
from . import optimizers


class ParametersInterface:

    def __init__(self, model):

        pass


class NNModel:

    """
    Base class for neural network skeleton.
    - Accepts modules as input (layers, activations).
    - Stores modules in order in a list.
    - TODO: is python linked list + custom linked list necessary ???
    """

    def __init__(self, parameters_obj=None):

        if parameters_obj is None:
            self.paremeters_interface = ParametersInterface(self)

        self.modules = []  # List of modules in the neural network
        self.head_module = None
        self.tail_module = None
    

        self.is_built = False  # Flag to check if the model is built

        self.update_algorithm = 'backprop'

        self.gradients = []

    def __call__(self, *args, **kwds):
        """
        Call the predict method of the model.
        """
        return self.predict(*args, **kwds)

    def __str__(self):
        return f'Neural Network with {len(self.modules)} modules.'

    def parameters_interface(self) -> ParametersInterface:
        return self.parameters_interface


    def add(self, modules):

        """
        - Modules.
        """

        if isinstance(modules, (list, tuple)):
            for module in modules:
                self.add(module)
            return
        else:
            module = modules
        

        if isinstance(module, modules.Module):
            self.modules.append(module)

        else:
            raise ValueError('Invalid model building block.')


    def predict(self, x):

        """
        Predict the output of the model.
        - This method is to be called after building the model.
        - It returns the output of the model for the given input x.
        """
        
        if not self.is_built:
            raise ValueError('Model not built.')

        return self._forwardpass(x)

    def _forwardpass(self, x):

        """
        A full forward pass through the model.
        - Takes an input x
        - Produces an output g
        """

        if not self.is_built:
            raise ValueError('Model not built.')

        if self.head_module is None:
            raise ValueError('No modules in the model.')
        
        module = self.head_module
        while module is not None:
            x = module.forward(x)
            module = module.next
        return x


    def _backpropagate(self, lossgrad):
        if isinstance(lossgrad, np.ndarray):
            if len(np.shape(lossgrad)) != 1:
                raise ValueError('Expecting scalar (stochastic) or vector (batch) loss gradient.')

        tape = []
        grad = lossgrad

        module = self.tail_module
        while module is not None:
            grad = module.backward(grad)
            self.gradients.append(grad)
            module = module.next
        return grad


    def train(self, x, y):

        """
        Train the model.
        - This method is to be called after building the model.
        - It trains the model using the specified parameter optimizer and loss function.
        """

        if not self.is_built:
            raise ValueError('Model not built.')
        
        raise NotImplementedError('Train method not implemented.')

    def evaluate(self, x):
        pass

    def get_input_dim(self):
        if self.head_module is None:
            raise ValueError('No modules in the model.')
        return self.head_module.input_dim

    def get_output_dim(self):
        if self.tail_module is None:
            raise ValueError('No modules in the model.')
        return self.tail_module.output_dim

    def build(self, modules=None):

        """
        # Builds the model. 
        - This method is to be called after adding all the modules to the model or if supplied with modules here.
        - Connects the modules in the model and checks if the model is valid.
        - Fixes loss function and/or parameter optimizer - if not set can be fed to the optimizer object.
        """

        if self.is_built:
            raise ValueError('Neural network already built.')
        if len(self.modules) == 0 and modules is None:
            raise ValueError('No modules in the neural network.')
        if len(self.modules) > 0 and modules is not None:
            raise ValueError('Modules already added to the neural network.')
        
        
        
        if len(self.modules) == 0 and modules is not None:
            # Add modules to the neural network
            self.modules = modules


        if len(self.modules) == 1:
            if not isinstance(self.modules[0], modules.Layer):
                    raise ValueError('Only one module in the neural network, but it is not a layer.')
            self.head_module = self.modules[0]
            self.tail_module = self.modules[0]
        else:
            self.head_module = self.modules[0]
            self.tail_module = self.modules[-1]
            for i in range(len(self.modules) - 1):
                self.modules[i].build()
                self.modules[i].next = self.modules[i + 1]
                self.modules[i + 1].prev = self.modules[i]

            print('Builder: Neural network built.')
        self.is_built = True


    def configurate_architecture(self):
        
        """
        Configurate the architecture of the model.
        - This method is to be called after building the model.
        - This method should be called by hyperparameter optimizer.
        """
        
        raise NotImplementedError('Configurate architecture method not implemented.')

    
    def conf_param(self):

        """
        API for parameter optimizer.
        """


    def get_param(self):

        pass


    def gradient(self, loss: lf.LossFunction = None):
        tape = []

