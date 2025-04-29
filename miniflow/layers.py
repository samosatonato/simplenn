from . import utils
from . import activations as ac
import numpy as np


class Layer:

    def __init__(self):
        
        self.initializer = None
        self.w = None
        self.b = None

    def __call__(self, x, ad=None):
        return self._call(x=x, ad=ad)

    def _call(self, x, ad=None):
        raise NotImplementedError

    def build(self, initializer: utils.Initializer = None, wshape=None, bshape=None):
        default_initializer = utils.RandomNormal()
        self.initializer = initializer or default_initializer

        self.add_weights(shape=wshape)
        self.add_biases(shape=bshape)
    
    def add_weights(self, shape=None):
        if self.w is None:
            self.w = self.initializer.new_weights(shape=shape)
        else:
            raise ValueError('Weights already initialized.')

    def add_biases(self, shape=None):
        if self.b is None:
            self.b = self.initializer.new_biases(shape=shape)
        else:
            raise ValueError('Biases already initialized.')

    def get_weights(self):
        return self.w

    def set_weights(self, nw):
        self.w = nw

    def sub_weights(self, dw):
        self.w -= dw

    def get_biases(self):
        return self.b

    def set_biases(self, nb):
        self.b = nb

    def sub_biases(self, db):
        self.b -= db

    def forward(self):
        raise NotImplementedError


class Input:

    def __init__(self, input_dim):
        self.input_dim = input_dim


class SimpleDense(Layer):

    def __init__(self, output_dim, activation, regularizer=None):
        super().__init__()

        self.output_dim = output_dim
        self.regularizer = regularizer

        if isinstance(activation, ac.Activation):
            self.activation = activation
        elif isinstance(activation, str):
            self.activation = ac.Activation.from_identifier(activation)
        else:
            raise ValueError

        self.cache_net = None
        self.cache_a = None
        self.cache_x = None

        self.is_last = False

    def _call(self, x, ad=None):
        return self.forward(x=x, ad=ad)

    def forward(self, x, ad=None):
        
        # Stores the gradient of previous (this input) activation
        self.cache_prev_ad = ad

        # Input: x shape=(N, in_dim)
        # Output: a shape=(N, out_dim)

        # shape=(N, in_dim)
        self.cache_x = None
        # shape=(N, out_dim)
        self.cache_net = None


        if self.activation is None:
            # shape=(N, out_dim)
            self.cache_net = (self.w @ x.T).T
            return self.cache_net, 1
        else:
            # TODO: handle stochastic (one datapoint) input dimensions
            # w: shape=(out_dim, in_dim)
            # xT: shape=(in_dim, N)
            # net' = wxT: shape=(out_dim, N)
            # net = (net')T: shape=(N, out_dim)
            if ad is not None:
                # shape=(N, in_dim)
                self.cache_x = x
                # shape=(N, out_dim)
                self.cache_net = (self.w @ x.T).T + self.b.T
                # shape=(N, out_dim)
                self.cache_a = self.activation(self.cache_net)
                if not self.is_last:
                    return self.cache_a, self.activation.d(self.cache_net)
                else:
                    return self.cache_a, 1       
            else:
                return self.activation((self.w @ x.T).T + self.b.T)

    def backward(self, d):
        return self.gradient(d)

    def gradient(self, delta_net):
        # dc_da: shape=(N, out_dim)
        
        # shape=(out_dim, in_dim)
        dc_dw = (delta_net.T @ self.cache_x) / delta_net.shape[0]

        if self.regularizer:
            dc_dw += self.regularizer.l2_lambda * self.w

        # shape=(1, out_dim)
        dc_db = np.sum(delta_net, axis=0, keepdims=True).T / delta_net.shape[0]

        # shape=(N, in_dim)
        dc_dx = (delta_net @ self.w) * self.cache_prev_ad

        return (dc_dw, dc_db), dc_dx

