import numpy as np


class Initializer:

    DEFAULT_RANDOM_SEED = 42

    def __init__(self, seed=DEFAULT_RANDOM_SEED):
        self.rng = np.random.default_rng(seed)

    def new_weights(self, shape):
        raise NotImplementedError
    
    def new_biases(self, shape):
        raise NotImplementedError


class RandomNormal(Initializer):

    def new_weights(self, shape):
        return self.rng.normal(size=shape)

    def new_biases(self, shape):
         return np.expand_dims(self.rng.normal(size=shape), axis=1)

class RandomUniform(Initializer):
    def new_weights(self, shape):
        pass
    def new_biases(self, shape):
        pass

# https://cs230.stanford.edu/section/4/#xavier-initialization
class GlorotUniform(Initializer):
    def new_weights(self, shape):
        # shape[0] = output_dim, shape[1] = input_dim
        limit = np.sqrt(6. / (shape[1] + shape[0]))
        return self.rng.uniform(low=-limit, high=limit, size=shape)
    def new_biases(self, shape):
         return np.zeros((shape, 1))

# https://medium.com/the-modern-scientist/exploring-the-he-normal-distribution-in-neural-network-weight-initialization-b802a53074e5
class HeNormal(Initializer):
     def new_weights(self, shape):
         # scale = sqrt(2 / input_dim)
         stddev = np.sqrt(2. / shape[1])
         return self.rng.normal(loc=0.0, scale=stddev, size=shape)
     def new_biases(self, shape):
         return np.zeros((shape, 1))

def split_data_8_2(data):
    samples = np.shape(data)[0]
    tr_data = data[:, :int(samples*0.8)]
    te_data = data[:, int(samples*0.8):]
    return tr_data, te_data


def split_data(data, mode='standard'):
    if mode == 'standard':
        samples = np.shape(data)[0]
        splt = (int(samples*0.8), int(samples*0.15), int(samples*0.05))
        tr_data = data[:, :splt[0]]
        te_data = data[:, splt[0]:splt[1]]
        v_data = data[:, splt[1]:splt[2]]
        return tr_data, te_data, v_data
    else:
        raise ValueError('Invalid split mode.')


def initialize_normal_weights(input_dim, output_dim):

    """
    Initialize weights and biases for a layer with normal distribution.
    """

    W = np.random.randn(output_dim, input_dim) * np.sqrt(2. / input_dim)
    B = np.zeros((output_dim, 1))
    
    return W, B


def initialize_uniform_weights(input_dim, output_dim):

    """
    Initialize weights and biases for a layer with uniform distribution.
    """

    W = np.random.uniform(-1, 1, (output_dim, input_dim))
    B = np.zeros((output_dim, 1))

    return W, B


def initialize_zero_weights(input_dim, output_dim):

    """
    Initialize weights and biases for a layer with zero values.
    """

    W = np.zeros((output_dim, input_dim))
    B = np.zeros((output_dim, 1))

    return W, B


