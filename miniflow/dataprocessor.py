import numpy as np
import math


class DataProcessor:

    def __init__(self, preprocessor=None, loader=None):
        default_preprocessor = StandardScaler()
        default_loader = DataLoader()

        self.preprocessor = default_preprocessor or preprocessor
        self.loader = default_loader or loader

    def __call__(self, data):
        raise NotImplementedError('Data processor call not implemented.')

    def load_csv(self, file_path: str, label_col: int = 0, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        kwargs.setdefault('skiprows', 0)
        data = self.loader.load_csv(file_path, **kwargs)

        if data.ndim == 1:
             data = data.reshape(1, -1)
        if label_col < 0:
             label_col = data.shape[1] + label_col

        if not (0 <= label_col < data.shape[1]):
             raise ValueError()

        y = data[:, label_col]
        x = np.delete(data, label_col, axis=1)

        return x, y
    
    def load_dataset(self, file_path: str, label_col: int = 0, skiprows: int = 1, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        data = self.loader.load_dataset(file_path, skiprows=skiprows, **kwargs)

        if data.ndim == 1:
             data = data.reshape(1, -1)
        if label_col < 0:
             label_col = data.shape[1] + label_col

        if not (0 <= label_col < data.shape[1]):
             raise ValueError()

        y = data[:, label_col].astype(kwargs.get('dtype', np.float32))

        x = np.delete(data, label_col, axis=1)

        print(f"Dataset loaded: {x.shape[0]} samples, {x.shape[1]} features, labels shape {y.shape}")
        return x, y


    def batchify(self, x, y, batch_size=1):
        batches = []
        
        for i in range(math.ceil(x.shape[0] / batch_size) - 1):
            batch = (x[i*batch_size : i*batch_size+batch_size], y[i*batch_size : i*batch_size+batch_size])
            batches.append(batch)

        batches.append((x[(math.ceil(x.shape[0] / batch_size) - 1)*batch_size :], y[(math.ceil(x.shape[0] / batch_size) - 1)*batch_size :]))

        return batches

    def shuffle(self, data1=None, data2=None):
        if isinstance(data2, np.ndarray):
            data = np.hstack((data1, data2))
            np.random.shuffle(data)
            return data[:,:data1.shape[1]], data[:,data1.shape[1]:]
        else:
            return np.random.shuffle(data1)

    def encode(self, encoder, y):
        return encoder.encode(y)

    def load_csv(self, file):
        data = self.loader.load_csv(file)
        y = data[:, 0]
        x = data[:, 1:]
        return x, y



class DataPreprocessor:
    
    def __init__(self, encoder=None, scaler=None):

        self.encoder = None
        self.scaler = scaler or StandardScaler()


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

    def load_csv(self, file_path: str, **kwargs) -> np.ndarray:
        import os
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        try:
            kwargs.setdefault('delimiter', ',')
            return np.loadtxt(file_path, **kwargs)
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            raise

    def load_dataset(self, file_path: str, skiprows: int = 1, **kwargs) -> np.ndarray:
        kwargs.setdefault('skiprows', skiprows) # Set default skiprows if not passed
        return self.load_csv(file_path, **kwargs)

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

    def __call__(self, data):
        self.fit(data)
        return self.transform(data)


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


# TODO: Add ecncode-decode pa ir-specific object logic.
# TODO: Add string support.
class OneHotEncoder1D(Encoder):

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
        
        # if len(labels.shape) != 1:
        #     raise ValueError('Labels must be a 1D array.')
        
        if self.categorization == 'auto':
            # Returns the sorted unique elements of an array.
            categories = np.unique(labels)
        
        label_to_index = {round(label): idx for idx, label in np.ndenumerate(categories)}

        one_hot_labels = np.zeros((len(labels), len(categories)), dtype=int)
        
        for i, label in enumerate(labels):
            col = label_to_index[int(label)]
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

