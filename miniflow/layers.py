from . import modules



class Dense(modules.Layer):

    """
    Dense layer with a linear activation function.
    """

    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)

    def forward(self, x):
        return self.net(x)

    def backward(self, x):
        return super().backward(x)


    def activation(self, net):
        if isinstance(self.next, modules.Activation):
            return self.next.activation(net)
        else:
            return net


# TODO: more layers, sparse, dropout?
    
