import nn.nn as nn

import numpy as np



class DataProcessor:
    """
    Base class for all data processors.
    """

    def __init__(self, preprocessor=None, loader=None):
        default_preprocessor = nn.StandardScaler()
        default_loader = nn.DataLoader()

        self.preprocessor = default_preprocessor or preprocessor
        self.loader = default_loader or loader

    def run(self):
        raise NotImplementedError



class DataLoader:
    """
    Base class for all data loaders.

    ### Shapes:
    - features: (N, input_dim) where N is the number of samples and input_dim is the dimension of features (number of features)
    - labels: (N, output_dim) where N is the number of samples and output_dim is the dimension of labels (number of labels)
    
    - features <=> inputs
    - labels <=> targets
    - features_labels <=> inputs_targets
    """

    def __init__(self):
        self.features_labels = None  # Features and labels
        self.features = None  # Features
        self.labels = None  # Labels

    """
    # Leaving this here for now, maybe will delete later.
    def load_data(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError('Data must be a numpy array.')
        try:
            if self.data is not None:
                raise ValueError('Data already loaded.')
        except ValueError:
            print('Data already loaded.')
            return self.data
    """            

    def _check_data(self) -> bool:

        if self.features_labels is not None:
            if not isinstance(self.features_labels, np.ndarray):
                print('Error: features_labels is not a numpy array.')
                return False
            if len(self.features_labels.shape) != 2:
                print('Error: features_labels is not a 2D array.')
                return False
            return True
        
        if not isinstance(self.features, np.ndarray):
            print('Error: features is not a numpy array.')
            return False
        if not isinstance(self.labels, np.ndarray):
            print('Error: labels is not a numpy array.')
            return False
        
        if self.features.shape[0] != self.labels.shape[0]:
            print('Error: features and labels have different number of samples.')
            return False

        if self.features.dtype.kind not in ("f", "i"):
            print('Error: features should be numeric.')
            return False
        
        return True


    def load_features_labels(self, features_labels):
        
        if not isinstance(features_labels, np.ndarray):
            raise ValueError('Features and labels must be a numpy array.')
        
        try:
            if self.features is not None or self.labels is not None:
                raise ValueError('Data already loaded.')
        except:
            print('Data already loaded.')
            return self.features_labels


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
    
    def _shuffle_data(self):
        indices = np.arange(len(self.features))
        np.random.shuffle(indices)
        self.features = self.features[indices]
        self.labels = self.labels[indices]

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= len(self.features):
            raise StopIteration
        start = self.current_index
        end = start + self.batch_size
        self.current_index = end
        return self.features[start:end], self.labels[start:end]


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        return self.mean, self.std

    def transform(self, data):
        return (data - self.mean) / self.std





class OneHotEncoder:
    def __init__(self, categories='auto'):
        pass


class DataPreprocessor(nn.DataLoader):
    

    def __init__(self):
        super().__init__()
        self.encoder = None
        self.scaler = None

    def load_encoder(self, encoder):
        if self.encoder is not None:
            raise ValueError('Encoder already loaded.')
        self.encoder = encoder
        return self.encoder
    