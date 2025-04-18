import numpy as np

import nn.nn as nn
import nn.layers as layers
import nn.activations as activations
import mlp.training as training
import nn.evaluation as evaluation



class MLPModule(nn.Module):
    
    """
    Base class for all MLP modules.
    """

    pass



class MLPLayer(MLPModule, nn.Layer):
    pass

class Dense(MLPLayer, layers.Dense):
    pass


class MLPActivation(MLPModule, nn.Activation):
    pass

class Heaviside(MLPActivation, activations.Heaviside):
    pass

class Sigmoid(MLPActivation, activations.Sigmoid):
    pass

class Tanh(MLPActivation, activations.Tanh):
    pass

class ReLU(MLPActivation, activations.ReLU):  
    pass

class Softmax(MLPActivation, activations.Softmax):
    pass


# TODO MOVE LOSS TO LOSS MODULE
class LossFunction(nn.LossFunction):
    pass

# TODO MOVE OPTIMIZER TO OPTIMIZER MODULE
class ParameterOptimizer(nn.ParameterOptimizer):
    pass

# TODO MOVE HYPERPARAMETER OPTIMIZER TO HYPERPARAMETER OPTIMIZER MODULE
class HyperparameterOptimizer(nn.HyperparameterOptimizer):
    pass


class MLP(nn.NN):
    pass


