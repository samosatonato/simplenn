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

import nn.utils as utils
import nn.loss_function as lf
import nn.parameter_optimizer as po



class Module:

    """
    Base class for all modules.
    - Contains the next and prev pointers for the linked list of modules.
    - Contains the input and output sizes for the module.
        - input_dim: size of the input tensor
        - output_dim: size of the output tensor
        - in case of activation function, they are equal to the output size of the previous layer.
    - Contains (interface) the forward and backward methods for the module.
    """

    def __init__(self, input_dim=None, output_dim=None):
        self.next = None
        self.prev = None

        self.input_dim = input_dim
        self.output_dim = output_dim

    def __call__(self, x):

        """
        Call the forward method of the module.
        """

        return self.forward(x)


    def forward(self, x):
        raise NotImplementedError('Forward method not implemented.')

    def backward(self, x):
        raise NotImplementedError('Backward method not implemented.')



class Layer(Module):

    """
    Implements base module class.

    Base class for all layers.
    """

    def __init__(self, input_dim, output_dim, weights_distribution='Normal', has_bias=True):
        super().__init__(input_dim, output_dim)

        self.weights_distribution = weights_distribution
        self.has_bias = has_bias

        self.w = None  # Weights
        self.b = None  # Biases

        self.is_initialized = False  # Flag to check if the layer is initialized


    def initialize(self):

        """
        Initializes the layer.

        TODO: other initializations than weights and biases
        """

        if self.is_initialized:
            raise ValueError('Layer already initialized.')
        if self.w is None and self.b is None:
            self._initialize_weights(self.input_dim, self.output_dim)
            self.is_initialized = True
        else:
            # TODO: maybe raise an error here?
            raise ValueError('WeirdError: Weights and biases already initialized.')

    def _initialize_weights(self, input_dim, output_dim):

        """
        Initializes the weights and biases for the layer.

        - It initializes the weights and biases for the layer using the specified distribution.

        """

        try:
            if self.weights_distribution == 'Normal':
                self.w, self.b = utils.initialize_normal_weights(input_dim, output_dim)
            elif self.weights_distribution == 'Uniform':
                self.w, self.b = utils.initialize_uniform_weights(input_dim, output_dim)
            elif self.weights_distribution == 'Zero':
                self.w, self.b = utils.initialize_zero_weights(input_dim, output_dim)
            else:
                raise ValueError('Invalid weights distribution.')
        except ValueError:
            print('Warning: Invalid weights distribution, using n
            ormal distribution.')
            self.weights_distribution = 'Normal'
            self.w, self.b = utils.initialize_normal_weights(input_dim, output_dim)


    def forward(self, x):
        return self._net(x)

    def _net(self, x):
        return x @ self.w + self.b
    

    def backward(self):
        pass

    def 
    

    # TODO: through pointers
    def activation(self, net):
        if isinstance(self.next, Activation):
            return self.next.activation(net)
        else:
            print(f'No activation function found for {self}, returning net...')
            return net



class Activation(Module):

    """
    Implements base module class.

    Base class for all activation functions.
    """

    def __init__(self, input_dim=None, output_dim=None):
        super().__init__(input_dim, output_dim)


    def forward(self, x):
        return self._activation(x)

    def _activation(self, net):
        pass
    

    def backward(self, x):
        return self._activation_derivative(x)

    def _activation_derivative(self, x):
        pass



class NN:

    """
    Base class for neural network skeleton.
    - Accepts building blocks of neural networks.
        - Modules
        - Loss function
        - Parameter optimizer
        - TODO: hyperparameter optimizer
    - Stores modules in order in a list.
    - TODO: is python linked list + custom linked list necessary ???
    """

    def __init__(self):
        self.modules = []  # List of modules in the neural network
        self.head_module = None
        self.tail_module = None
        
        self.loss_function = None  # Loss function
        self.parameter_optimizer = None  # Optimizer

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


    def add(self, building_block):

        """
        - Modules.
        - Loss functions.
        - Parameter optimizer.
        TODO: 
        - Hyperparameter optimizer.
        - Evaluation engine.
        """

        if isinstance(building_block, Module):
            self.modules.append(building_block)
            
        elif isinstance(building_block, lf.LossFunction):
            if self.loss_function is not None:
                raise ValueError('Loss function already set.')
            self.loss_function = building_block

        elif isinstance(building_block, po.ParameterOptimizer):
            if self.parameter_optimizer is not None:
                raise ValueError('Parameter optimizer already set.')
            self.parameter_optimizer = building_block

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

        x = lossgrad
        module = self.tail_module
        while module is not None:
            x = module.backward(x)
            self.gradients.append(x)
            module = module.next
        return x


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

    def build(self):

        """
        # Builds the model. 
        - This method is to be called after adding all the modules to the model.
        - Connects the modules in the model and checks if the model is valid.
        - Fixes loss function and/or parameter optimizer - if not set can be fed to the optimizer object.
        """

        if self.is_built:
            raise ValueError('Model already built.')
        if len(self.modules) == 0:
            raise ValueError('No modules in the model.')
        if self.loss_function is None:
            print('Builder: No loss function in the model.')
        if self.parameter_optimizer is None:
            print('Builder: No parameter optimizer in the model.')

        if len(self.modules) == 1:
            self.head_module = self.modules[0]
            self.tail_module = self.modules[0]
        else:
            self.head_module = self.modules[0]
            self.tail_module = self.modules[-1]
            for i in range(len(self.modules) - 1):
                self.modules[i].next = self.modules[i + 1]
                self.modules[i + 1].prev = self.modules[i]

            print('Builder: Model built.')
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


