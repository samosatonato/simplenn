from . import utils

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

    def build(self):
        pass


class Layer(Module):

    """
    Implements base module class.

    Base class for all layers.
    """

    def __init__(self, input_dim, output_dim, activation=None, initial_weights_distribution='Normal', has_bias=True):
        super().__init__(input_dim, output_dim)

        self.initial_weights_distribution = initial_weights_distribution
        self.has_bias = has_bias

        self.w = None  # Weights
        self.b = None  # Biases

        self.is_initialized = False  # Flag to check if the layer is initialized

    def __call__(self, x):

        """
        Call the forward method of the layer.
        """

        return self.forward(x)


    def build(self):

        """
        Initializes the layer.

        TODO: other initializations than weights and biases ?
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
            if self.initial_weights_distribution == 'Normal':
                self.w, self.b = utils.initialize_normal_weights(input_dim, output_dim)
            elif self.initial_weights_distribution == 'Uniform':
                self.w, self.b = utils.initialize_uniform_weights(input_dim, output_dim)
            elif self.initial_weights_distribution == 'Zero':
                self.w, self.b = utils.initialize_zero_weights(input_dim, output_dim)
            else:
                raise ValueError('Invalid weights distribution.')
            
        except ValueError:
            print('Warning: Invalid weights distribution, using normal distribution.')
            self.initial_weights_distribution = 'Normal'
            self.w, self.b = utils.initialize_normal_weights(input_dim, output_dim)


    def forward(self, x):
        self.cached_x = x
        net = self.net(x)
        return net

    def _net(self, x):
        return x @ self.w + self.b
    

    def backward(self):
        pass
    

    """
    # TODO: through pointers
    def activation(self, net):
        if isinstance(self.next, Activation):
            return self.next.activation(net)
        else:
            print(f'No activation function found for {self}, returning net...')
            return net
    """


class Activation(Module):

    """
    Implements base module class.

    Base class for all activation functions.
    """

    def __init__(self, input_dim=None, output_dim=None):
        super().__init__(input_dim, output_dim)


    def forward(self, x):
        self.cached_x = x
        a = self._activation(x)
        self.cached_a = a
        return a

    def _activation(self, net):
        pass
    

    def backward(self, x):
        return self._activation_derivative(x)

    def _activation_derivative(self, x):
        pass