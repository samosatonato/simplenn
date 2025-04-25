import numpy as np

from . import layers
from . import activations
from . import hypertuner
from . import lossfunctions as lf


class Model:

    def __init__(self, is_eager=False):
        
        self.is_eager = is_eager

        self.is_training = False

    def train(self):
        self.is_training = True

    def untrain(self):
        self.is_training = False

 
class Sequential(Model):

    def __init__(self):
        super().__init__()

        self.layers = []
        self.input_dim = None


    def add(self, layer):
        if isinstance(layer, layers.Input):
            self.input_dim = layer.input_dim
        else:
            if self.is_eager:
                layer.build()
                self.layers.append(layer)
            else:
                self.layers.append(layer)


    def predict(self, x):
        return self._forwardpass(x)

    def _forwardpass(self, x):
        if self.is_training:
            ad = 1
        if isinstance(x, (list)):
            x = np.stack(x)
        for layer in self.layers:
            if self.is_training:
                x, ad = layer(x, ad)
            else:
                x = layer(x)
        return x
    

    def gradients(self, loss_grad):
        return self._backpropagate(loss_grad)

    def _backpropagate(self, delta_net):
        gradients = []
        for i, layer in enumerate(reversed(self.layers)):
            (dw, db), delta_net = layer.backward(delta_net)
            gradients.append((dw, db))
        return gradients

    def sub_weights(self, layers_delta_w):
        for layer, delta_w in zip(self.layers, reversed(layers_delta_w)):
            layer.sub_weights(delta_w)
        return True

    def set_weights(self, w):
        for layer in self.layers:
            layer.set_weights(w)
        return True
    
    def sub_biases(self, layers_delta_b):
        for layer, delta_b in zip(self.layers, reversed(layers_delta_b)):
            layer.sub_biases(delta_b)
        return True

    def set_biases(self, b):
        for layer in self.layers:
            layer.set_biases(b)
        return True

    def build(self, optimizer=None, loss=None, metrics=None):
        loss.add_model(self)
        optimizer.add_model(self)

        self.layers[-1].is_last = True

        if self.is_eager:
            pass
        else:
            wshape = [0, 0]
            bshape = 0
            if self.input_dim:
                wshape[1] = self.input_dim
                bshape = self.input_dim
            self.layers[0].is_first = True
            for layer in self.layers:
                wshape[0] = layer.output_dim
                bshape = layer.output_dim
                layer.build(wshape=wshape, bshape=bshape)
                wshape[1] = wshape[0]


