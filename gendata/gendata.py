import numpy as np
import itertools as itls
from enum import Enum
import math

RANDOM_SEED = 1

class Distribution(Enum):
    """Specifies the probability distribution for generating blob clusters."""
    NORMAL = 'normal'
    UNIFORM = 'uniform'

class Pattern(Enum):
    """Specifies the type of data pattern to generate."""
    BLOBS = 'blobs'      # Simple gaussian or uniform clusters
    MOONS = 'moons'      # Two interleaving half circles
    CIRCLES = 'circles'  # Two concentric circles
    SPIRAL = 'spiral'    # Two interleaved spirals



class ClusteredClasses():

    def __init__(self, labels_n=None, dim=None, sample_size=None,
                 distribution=Distribution.NORMAL, cluster_centers=None,
                 pattern=Pattern.BLOBS, noise=0.1, seed=None, split=True):
        """Initializes the ClusteredClasses generator."""
        self.labels_n = labels_n
        self.dim = dim
        self.sample_size = sample_size
        self.distribution = distribution if isinstance(distribution, Distribution) else Distribution(distribution)
        self.cluster_centers = cluster_centers # Can be None initially
        self.pattern = pattern if isinstance(pattern, Pattern) else Pattern(pattern)
        self.noise = noise
        self.seed = seed if seed is not None else RANDOM_SEED
        self.split = split

        self.rng = np.random.default_rng(self.seed)

        # Validate pattern compatibility
        if self.dim is not None and self.dim != 2 and self.pattern in [Pattern.MOONS, Pattern.CIRCLES, Pattern.SPIRAL]:
             raise ValueError(f"Pattern '{self.pattern.value}' currently only supports 2 dimensions (dim=2).")
        if self.labels_n is not None and self.labels_n != 2 and self.pattern in [Pattern.MOONS, Pattern.CIRCLES, Pattern.SPIRAL]:
             raise ValueError(f"Pattern '{self.pattern.value}' currently only supports 2 labels (labels_n=2).")

    def _validate_and_set_param(self, attr_name, value, default=None):
        current_value = getattr(self, attr_name)
        if value is not None and current_value is not None and value != current_value:

            raise ValueError(f'{attr_name} is already set to {current_value}. Cannot change to {value}. Use reset() first.')
        elif value is not None:
            setattr(self, attr_name, value)
        elif current_value is None:
            if default is None:
                 if attr_name in ['labels_n', 'dim', 'sample_size']:
                     raise ValueError(f'{attr_name} must be set.')
            else:
                setattr(self, attr_name, default)

    def getdata(self, labels_n=None, dim=None, sample_size=None,
                distribution=None, cluster_centers=None, pattern=None,
                noise=None, seed=None):

        if seed is not None:
            self._validate_and_set_param('seed', seed)
            self.rng = np.random.default_rng(self.seed)

        self._validate_and_set_param('pattern', pattern, default=Pattern.BLOBS)
        if isinstance(self.pattern, str):
             self.pattern = Pattern(self.pattern)

        self._validate_and_set_param('dim', dim)
        self._validate_and_set_param('labels_n', labels_n)

        if self.pattern in [Pattern.MOONS, Pattern.CIRCLES, Pattern.SPIRAL]:
            if self.dim is None: self.dim = 2
            elif self.dim != 2: raise ValueError(f"Pattern '{self.pattern.value}' requires dim=2.")
            if self.labels_n is None: self.labels_n = 2
            elif self.labels_n != 2: raise ValueError(f"Pattern '{self.pattern.value}' requires labels_n=2.")
        else:
            if self.dim is None: self.dim = 2
            if self.labels_n is None: self.labels_n = 3


        default_ss = tuple(25 for _ in range(self.labels_n))
        self._validate_and_set_param('sample_size', sample_size, default=default_ss)
        if len(self.sample_size) != self.labels_n:
             raise ValueError(f'Length of sample_size {len(self.sample_size)} must match labels_n {self.labels_n}.')

        self._validate_and_set_param('distribution', distribution, default=Distribution.NORMAL)
        if isinstance(self.distribution, str):
            self.distribution = Distribution(self.distribution)

        self._validate_and_set_param('noise', noise, default=0.1)

        if self.pattern == Pattern.BLOBS:
             if cluster_centers is None and self.cluster_centers is None:
                  centers = self.rng.uniform(low=-10, high=10, size=(self.labels_n, self.dim))
                  self._validate_and_set_param('cluster_centers', centers)
             elif cluster_centers is not None:
                  centers_arr = np.array(cluster_centers)
                  if centers_arr.shape != (self.labels_n, self.dim):
                      raise ValueError(f"cluster_centers shape must be ({self.labels_n}, {self.dim}), but got {centers_arr.shape}")
                  self._validate_and_set_param('cluster_centers', centers_arr)
        elif cluster_centers is not None:
             print(f"Warning: 'cluster_centers' parameter ignored for pattern '{self.pattern.value}'.")


        return self._gendata()

    def _gendata(self):

        if self.labels_n is None or self.dim is None or self.sample_size is None or self.pattern is None:
            raise ValueError("Cannot generate data: Not all required parameters are set (labels_n, dim, sample_size, pattern).")

        total_samples = sum(self.sample_size)
        x = np.zeros(shape=(total_samples, self.dim))
        y = np.zeros(shape=total_samples, dtype=int)

        current_pos = 0
        label_indices = {}

        for i, size in enumerate(self.sample_size):
            start = current_pos
            end = current_pos + size
            y[start:end] = i
            label_indices[i] = (start, end)
            current_pos = end

        if self.pattern == Pattern.BLOBS:
            if self.cluster_centers is None:
                 raise ValueError("Cluster centers required for 'blobs' pattern.")
            for i in range(self.labels_n):
                start, end = label_indices[i]
                center = self.cluster_centers[i]
                size = self.sample_size[i]

                if self.distribution == Distribution.NORMAL:
                    points = self.rng.normal(loc=center, scale=1.0 + self.noise, size=(size, self.dim)) # Noise adds to std dev
                elif self.distribution == Distribution.UNIFORM:
                    scale = 1.5 + self.noise
                    low = center - scale
                    high = center + scale
                    points = self.rng.uniform(low=low, high=high, size=(size, self.dim))
                else:
                    raise ValueError(f"Unsupported distribution: {self.distribution}")
                x[start:end] = points

        elif self.pattern == Pattern.MOONS:
            if self.dim != 2 or self.labels_n != 2:
                raise ValueError("Moons pattern requires dim=2 and labels_n=2.")

            n_samples_0 = self.sample_size[0]
            n_samples_1 = self.sample_size[1]
            total_moons = n_samples_0 + n_samples_1

            outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_0))
            outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_0))

            inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_1))
            inner_circ_y = 0.5 - np.sin(np.linspace(0, np.pi, n_samples_1))

            start0, end0 = label_indices[0]
            start1, end1 = label_indices[1]

            x[start0:end0, 0] = outer_circ_x
            x[start0:end0, 1] = outer_circ_y
            x[start1:end1, 0] = inner_circ_x
            x[start1:end1, 1] = inner_circ_y

            x += self.rng.normal(scale=self.noise, size=x.shape)

        elif self.pattern == Pattern.CIRCLES:
            if self.dim != 2 or self.labels_n != 2:
                 raise ValueError("Circles pattern requires dim=2 and labels_n=2.")

            factor = 0.5
            n_samples_0 = self.sample_size[0]
            n_samples_1 = self.sample_size[1]
            start0, end0 = label_indices[0]
            start1, end1 = label_indices[1]

            linspace0 = np.linspace(0, 2 * np.pi, n_samples_0, endpoint=False)
            linspace1 = np.linspace(0, 2 * np.pi, n_samples_1, endpoint=False)
            outer_circ_x = np.cos(linspace0)
            outer_circ_y = np.sin(linspace0)
            inner_circ_x = np.cos(linspace1) * factor
            inner_circ_y = np.sin(linspace1) * factor

            x[start0:end0, 0] = outer_circ_x
            x[start0:end0, 1] = outer_circ_y
            x[start1:end1, 0] = inner_circ_x
            x[start1:end1, 1] = inner_circ_y

            x += self.rng.normal(scale=self.noise, size=x.shape)

        elif self.pattern == Pattern.SPIRAL:
             if self.dim != 2 or self.labels_n != 2:
                 raise ValueError("Spiral pattern requires dim=2 and labels_n=2.")

             n_samples_0 = self.sample_size[0]
             n_samples_1 = self.sample_size[1]
             start0, end0 = label_indices[0]
             start1, end1 = label_indices[1]
             total_spiral = n_samples_0 + n_samples_1

             n_points = total_spiral // 2
             theta = np.sqrt(self.rng.uniform(0, 1, n_points)) * 3 * np.pi

             # Spiral 1 (positive rotation)
             r_a = 2 * theta + np.pi
             x_a = r_a * np.cos(theta)
             y_a = r_a * np.sin(theta)

             # Spiral 2 (negative rotation)
             r_b = -2 * theta - np.pi
             x_b = r_b * np.cos(theta)
             y_b = r_b * np.sin(theta)

             spiral_x = np.concatenate((x_a, x_b))
             spiral_y = np.concatenate((y_a, y_b))
             spiral_labels = np.concatenate((np.zeros(len(x_a)), np.ones(len(x_b))))

             points = np.stack((spiral_x, spiral_y), axis=-1)
             points += self.rng.normal(scale=self.noise * 5, size=points.shape)

             indices = np.arange(points.shape[0])
             self.rng.shuffle(indices)
             shuffled_points = points[indices]
             shuffled_labels = spiral_labels[indices]

             idx0 = np.where(shuffled_labels == 0)[0]
             idx1 = np.where(shuffled_labels == 1)[0]

             points0 = shuffled_points[self.rng.choice(idx0, n_samples_0, replace=False)]
             points1 = shuffled_points[self.rng.choice(idx1, n_samples_1, replace=False)]

             x[start0:end0] = points0
             x[start1:end1] = points1

        else:
            raise ValueError(f"Unsupported pattern: {self.pattern}")


        if self.split:
            # Return features x and labels y separately
            return x, y
        else:
            # y needs to be reshaped to (N, 1) for hstack
            return np.hstack((x, y.reshape(-1, 1)))


    def reset(self):
        """Resets configurable parameters"""
        self.labels_n = None
        self.dim = None
        self.sample_size = None
        self.cluster_centers = None
        self.distribution = Distribution.NORMAL
        self.pattern = Pattern.BLOBS
        self.noise = 0.1
        # Keep the seed unless asked to reset it
        self.rng = np.random.default_rng(self.seed)


class ClusteredClasses2D(ClusteredClasses):
    """Generates 2D datasets with specified patterns."""
    def __init__(self, labels_n=None, sample_size=None,
                 distribution=Distribution.NORMAL, cluster_centers=None,
                 pattern=Pattern.BLOBS, noise=0.1, seed=None, split=True):
        """Initializes the 2D generator: fixing dim=2."""
        super().__init__(labels_n=labels_n, dim=2, sample_size=sample_size,
                         distribution=distribution, cluster_centers=cluster_centers,
                         pattern=pattern, noise=noise, seed=seed, split=split)

