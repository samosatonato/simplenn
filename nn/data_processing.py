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

    def __call__(self, data):
        """
        Call the data processor.
        """
        raise NotImplementedError('Data processor not implemented.')


    def _shuffle_data(self):
        indices = np.arange(len(self.features))
        np.random.shuffle(indices)
        self.features = self.features[indices]
        self.labels = self.labels[indices]



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

    def load_features_labels(self, features_labels, input_dim=None, output_dim=None):
        if not isinstance(features_labels, np.ndarray):
            raise ValueError('Features and labels must be a numpy array.')
        
        try:
            if self.features is not None or self.labels is not None:
                raise ValueError('Data already loaded.')
        except:
            print('Data already loaded.')
            return self.features_labels

        if input_dim is None or output_dim is None:
            raise ValueError('Input and output dimensions must be specified when loading combined features and labels.')

        if input_dim + output_dim != features_labels.shape[1]:
            raise ValueError('Input and output dimensions do not match the shape of the features and labels array.')

        if self.labels is not None or self.features is not None:
            raise ValueError('Data already loaded.')
        
        return self._split_features_labels(input_dim, output_dim)

    def _split_features_labels(self, input_dim=None, output_dim=None):
        if self.features_labels is not None:
            self.features = self.features_labels[:, :-output_dim]
            self.labels = self.features_labels[:, -output_dim:]

        return self.features, self.labels


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
    

    def load_file(self, file_path):

        """
        Load data from a file.
        - data: path to the file
        """

        if not isinstance(file_path, str):
            raise ValueError('File path must be a string.')
        
        try:
            if self.features_labels is not None:
                raise ValueError('Data already loaded.')
        except ValueError:
            print('Data already loaded.')
            return self.features_labels
        
        # Load data from file
        self.features_labels = np.loadtxt(file_path, delimiter=',')

        return self.features_labels
    


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



class Encoder:
    """
    Base class for all encoders.
    """

    def __init__(self):
        pass

    def __call__(self, data):
        """
        Call the encoder.
        - data: data to be encoded
        """

        if not isinstance(data, np.ndarray):
            raise ValueError('Data must be a numpy array.')
        
        return self.encode(data)
    

    def encode(self, data):
        """
        Encode data.
        """
        raise NotImplementedError('Encoder not implemented.')

    def decode(self, data):
        """
        Decode data.
        """
        raise NotImplementedError('Decoder not implemented.')


# TODO: Add ecncode-decode pair-specific object logic.
# TODO: Add string support.
class OneHotEncoder1D:

    """
    One-hot encoder for 1D labels.
    - categories: categories to be used for encoding
    - auto: automatically determine categories from the labels
    - labels: labels to be encoded
    """

    def __init__(self, categorization='auto'):
        self.categorization = categorization  # Categorization to be used for encoding
        self.categories = None  # Categories to be used for encoding


    def encode(self, labels):

        """
        Encode labels using one-hot encoding.
        - labels: labels to be encoded
        """

        if not isinstance(labels, np.ndarray):
            raise ValueError('Labels must be a numpy array.')
        
        if len(labels.shape) != 1:
            raise ValueError('Labels must be a 1D array.')
        
        if self.categorization == 'auto':
            self.categories = np.unique(labels)
        
        label_to_index = {label: idx for idx, label in enumerate(self.categories)}

        one_hot_labels = np.zeros((len(labels), len(self.categories)), dtype=int)
        
        for i, label in enumerate(labels):
            col = label_to_index[label]
            one_hot_labels[i, col] = 1
        
        return one_hot_labels

    def decode(self, one_hot_labels):

        """
        Decode one-hot encoded labels.
        - one_hot_labels: one-hot encoded labels to be decoded
        """

        if not isinstance(one_hot_labels, np.ndarray):
            raise ValueError('One-hot labels must be a numpy array.')
        
        if one_hot_labels.shape[1] != len(self.categories):
            raise ValueError('One-hot labels have different number of categories.')
        
        decoded_labels = np.zeros(one_hot_labels.shape[0], dtype=int)
        
        for i in range(one_hot_labels.shape[0]):
            decoded_labels[i] = np.argmax(one_hot_labels[i])
        
        return decoded_labels

