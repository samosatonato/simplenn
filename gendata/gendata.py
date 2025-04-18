import numpy as np
import itertools as itls
from enum import Enum

RANDOM_SEED = 1
 

class Distribution(Enum):
    NORMAL = 'normal'
    UNIFORM = 'uniform'


class ClusterCenter(Enum):
    RANDOM = 'random'
    


class ClusteredClasses():

    """
    cluster_centers is shape (labels_n, dim)
    """

    def __init__(self, labels_n=None, dim=None, sample_size=None, distribution=None, cluster_centers=None, seed=None):
        self.labels_n = labels_n
        self.dim = dim
        self.sample_size = sample_size
        self.distribution = distribution
        self.cluster_centers = cluster_centers
        self.seed = seed

        self.rng = np.random.default_rng(seed)


    def _gendata(self):
        if self.labels_n is None:
            raise ValueError('labels_n must be set.')
        if self.sample_size is None:
            raise ValueError('sample_size must be set.')
        if self.dim is None:
            raise ValueError('dim must be set.')
        if len(self.sample_size) != self.labels_n:
            raise ValueError('Length of sample_size must match labels_n.')
        
        all_data = []
        
        for label in range(self.labels_n):
            if self.distribution == Distribution.NORMAL:
                current_sample_size = int(self.sample_size[label])
                
                # shape (dim, current_sample_size)
                coords = [self.rng.normal(loc=self.cluster_centers[label][dim_index], scale=1, size=current_sample_size) for dim_index in range(self.dim)]
                
                # shape (dim, current_sample_size)
                points = np.vstack(coords)
                
                # shape current_sample_size
                labels = np.full(current_sample_size, label)
                
                # shape (dim+1, current_sample_size)
                datapoints = np.vstack((points, labels))

                # shapae (current_sample_size, dim+1)
                datapoints = np.transpose(datapoints)
                
                all_data.append(datapoints)

        # shape (sum(sample_size), dim+1)
        all_data = np.vstack(all_data)
        return all_data


    def getdata(self, labels_n=None, sample_size=None, distribution=None, cluster_centers=None, seed=None):
        if labels_n is not None and self.labels_n is not None:
            raise ValueError('labels_n is already set.')
        elif labels_n is not None:
            self.labels_n = labels_n
        elif labels_n is None and self.labels_n is None:
            self.labels_n = 3

        if sample_size is not None and self.sample_size is not None:
            raise ValueError('sample_size is already set.')
        elif sample_size is not None:
            self.sample_size = sample_size
        elif sample_size is None and self.sample_size is None:
            self.sample_size = (25, 25, 25)

        if distribution is not None and self.distribution is not None:
            raise ValueError('distribution is already set.')
        elif distribution is not None:
            self.distribution = distribution
        elif distribution is None and self.distribution is None:
            self.distribution = 'Normal'

        if cluster_centers is not None and self.cluster_centers is not None:
            raise ValueError('cluster_centers is already set.')
        elif cluster_centers is not None:
            self.cluster_centers = cluster_centers
        elif cluster_centers is None and self.cluster_centers is None:
            self.cluster_centers = np.random.uniform(low=-10, high=10, size=(self.labels_n, self.dim))

        if seed is not None and self.seed is not None:
            raise ValueError('seed is already set.')
        elif seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        elif seed is None and self.seed is None:
            self.seed = RANDOM_SEED
            self.rng = np.random.default_rng(RANDOM_SEED)

        return self._gendata()


    def reset(self):
        self.labels_n = None
        self.sample_size = None
        self.cluster_centers = None
        self.distribution = 'Normal'
        self.rng = np.random.default_rng(self.seed)



class ClusteredClasses2D(ClusteredClasses):

    def __init__(self, labels_n=None, sample_size=None, distribution=None, cluster_centers=None, seed=None):
        super().__init__(labels_n=labels_n, dim=2, sample_size=sample_size, distribution=distribution, cluster_centers=cluster_centers, seed=seed)

    def _gendata(self):
        return super()._gendata()

    def getdata(self, labels_n=None, sample_size=None, distribution=None, cluster_centers=None, seed=None):
        super().getdata(labels_n=labels_n, sample_size=sample_size, distribution=distribution, cluster_centers=cluster_centers, seed=seed)
        return self._gendata()
    
    def reset(self):
        super().reset()



class ClusteredClasses3D(ClusteredClasses):

    def __init__(self, labels_n=None, sample_size=None, distribution=None, cluster_centers=None, seed=None):
        super().__init__(labels_n=labels_n, dim=3, sample_size=sample_size, distribution=distribution, cluster_centers=cluster_centers, seed=seed)

    def _gendata(self):
        return super()._gendata()

    def getdata(self, labels_n=None, sample_size=None, distribution=None, cluster_centers=None, seed=None):
        super().getdata(labels_n=labels_n, sample_size=sample_size, distribution=distribution, cluster_centers=cluster_centers, seed=seed)
        return self._gendata()
    
    def reset(self):
        super().reset()

