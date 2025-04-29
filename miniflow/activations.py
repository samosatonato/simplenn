import numpy as np


class Activation:

    _registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = getattr(cls, "name", cls.__name__.lower())
        Activation._registry[name] = cls

    @classmethod
    def from_identifier(cls, identifier, **kwargs):
        if isinstance(identifier, cls):
            return identifier
        elif isinstance(identifier, str):
            if identifier in cls._registry:
                return cls._registry[identifier](**kwargs)
            else:
                raise ValueError(f"Unknown identifier: {identifier}")
        else:
            raise TypeError("Expected string identifier or instance of Activation")

    def __init__(self):
        pass

    def __call__(self, x):
        return self._activation(x)
    
    def d(self, x):
        return self._activation_d(x)

    def _activation(self, x):
        raise NotImplementedError
    
    def _activation_d(self, x):
        raise NotImplementedError



class Linear(Activation):
    name = 'linear'

    def _activation(self, x):
        return x

    def _activation_d(self, x):
        return np.ones_like(x)

class Tanh(Activation):
    name = 'tanh'

    def _activation(self, x):
        return np.tanh(x)

    def _activation_d(self, x):
        return 1 - np.tanh(x) ** 2

class Heaviside(Activation):
    name = 'heaviside'

    def _activation(self, x):
        return np.heaviside(x, 0)

    def _activation_d(self, x):
        return np.zeros_like(x)

class ReLU(Activation):
    name = 'relu'

    def _activation(self, x):
        return np.maximum(0, x) 

    def _activation_d(self, x):
        return np.where(x > 0, 1, 0)

class Sigmoid(Activation):
    name = 'sigmoid'

    def _activation(self, x):
        clipped_x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-clipped_x))

    def _activation_d(self, x):
        s = self._activation(x)
        return s * (1 - s)

class Softmax(Activation):
    name = 'softmax'

    def _activation(self, x):
        exps = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exps / np.sum(exps, axis=0, keepdims=True)

    def _activation_d(self, x):
        raise NotImplementedError('Currently softmax activation supports only output layer.')
    
