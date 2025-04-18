import nn.nn as nn



class Dense(nn.Layer):

    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)

    def forward(self, x):
        return self.net(x)

    def backward(self, x):
        return super().backward(x)

    def activation(self, net):
        return net
    
    