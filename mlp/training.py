import numpy as np



class GradientDescent():

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

        self.features_labels = None
        self.labels = None
        self.features = None
        self.model = None
        self.loss = None

        self.batch_size = None



    def _fit(self):
        pass

    def _is_correct_shapes(self, input_dim, output_dim):
        if self.features_labels is not None:
            if self.features_labels.shape[0] != input_dim + output_dim:
                raise ValueError('Features and labels shape mismatch.')
        elif self.features is not None and self.labels is not None:
            if self.features.shape[0] != input_dim or self.labels.shape[0] != output_dim:
                raise ValueError('Features and labels shape mismatch.')
        else:
            raise ValueError('No data loaded.')


    def _prepare_data(self):
        input_dim = self.model.get_input_dim()
        output_dim = self.model.get_output_dim()

        self._is_correct_shapes(input_dim, output_dim)

        if 

        if self.features_labels is not None:
            if self.features_labels.shape[0] != input_dim + output_dim:
                raise ValueError('Features and labels shape mismatch.')
            features = self.features_labels[:input_dim, :]
            labels = self.features_labels[input_dim:, :]
        elif self.features is not None and self.labels is not None:
            if self.features.shape[0] != input_dim or self.labels.shape[0] != output_dim:
                raise ValueError('Features and labels shape mismatch.')
            features = self.features
            labels = self.labels
        else:
            raise ValueError('No data loaded.')
        if features.shape[1] != labels.shape[1]:
            raise ValueError('Features and labels shape mismatch.')
        if features.shape[1] % self.batch_size != 0:
            raise ValueError('Batch size does not divide the number of samples.')
        if features.shape[1] < self.batch_size:
            raise ValueError('Batch size is greater than the number of samples.')


    def load_features_labels(self, features_labels):
        if self.labels is not None or self.features is not None:
            raise ValueError('Data already loaded.')
        self.features_labels = features_labels        
        return self.features_labels

    def load_features(self, features):
        if self.features_labels is not None:
            raise ValueError('Data already loaded.')
        self.features = features
        return self.features
    
    def load_labels(self, labels):
        if self.features_labels is not None:
            raise ValueError('Data already loaded.')
        self.labels = labels
        return self.labels

    def load_model(self, model):
        if self.model is not None:
            raise ValueError('Model already loaded.')
        self.model = model
        return self.model

    def load_lossfunction(self, lossfunction):
        if self.loss is not None:
            raise ValueError('Loss function already loaded.')
        self.loss = lossfunction
        return self.loss

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        return self.batch_size
    
    def reset(self):
        self.data = None
        self.features_labels = None
        self.labels = None
        self.features = None
        self.model = None
        self.loss = None

        self.batch_size = None


    def train(self, model, lossfunction=None):
        if model is not None and self.model is not None:
            raise ValueError('Trying to load another model, a model already loaded.')
        if model is not None:
            self.model = model
        if self.model is None:
            raise ValueError('No model loaded.')
        if self.features_labels is None or self.features is None or self.labels is None:
            raise ValueError('Data not loaded.')

        def _prepare_data():
        

        def _fit():


class StochasticGradientDescent(GradientDescent):

    def __init__(self, learning_rate):
        super().__init__(learning_rate)

        self.set_batch_size(1)


class LossFunction():
    def __init__(self):
        pass

    def loss(self, predictions, labels):
        pass

    def backward(self, predictions, labels):
        pass



class Adagrad():

    def __init__(self, model, data, learning_rate):
        self.model = model
        self.data = data
        self.learning_rate = learning_rate

    
class RMSProp():
    pass


class Adam():

    def __init__(self, model, data, learning_rate):
        self.model = model
        self.data = data
        self.learning_rate = learning_rate

