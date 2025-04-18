import numpy as np
import itertools as itls


RANDOM_SEED = 1


def gen_labeled_clusters(shuffle=True, dim=2, label_n=3, sample_size=(25,25,25,), distribution='normal', cluster_centers='random', vector_shape='row'):
    if dim >= 10:
        raise ValueError('Number of dimensions too high.')
    
    rng = np.random.default_rng(RANDOM_SEED)



    # Prepare indeces according to sample_size.
    sample_indices = [0] + list(itls.accumulate(sample_size))
    # print(sample_indices)

    # Prepare initial ndarray.
    if vector_shape == 'column':
        multi_labeled_random_data = np.zeros((dim + 1, np.sum(sample_size)))

        """
        | 1x1 | 2x1 | ...
        | 1x2 | 2x2 | ...
        | 1d  | 2d  | ...
        """

        if distribution == 'normal':
            for label_idx in range(label_n):
                loc = rng.uniform(low=-10, high=10, size=dim)
                scale = 1
                size = (sample_size[label_idx], dim)
                multi_labeled_random_data[sample_indices[label_idx]:sample_indices[label_idx+1],:dim] = rng.normal(loc=loc, scale=scale, size=size)
                multi_labeled_random_data[sample_indices[label_idx]:sample_indices[label_idx+1],dim] = label_idx
        else:
            raise ValueError('Incorrect distribution type.')

    elif vector_shape == 'row':
        multi_labeled_random_data = np.zeros((dim + 1, np.sum(sample_size)))

        """
        | 1x1 | 2x1 | 1d |
        | 2x1 | 2x2 | 2d |
          ...   ...   ...
        """

        if distribution == 'normal':
            for label_idx in range(label_n):
                loc = rng.uniform(low=-10, high=10, size=(1,dim))
                scale = 1
                size = (dim, sample_size[label_idx])
                multi_labeled_random_data[:dim,sample_indices[label_idx]:sample_indices[label_idx+1]] = rng.normal(loc=loc, scale=scale, size=size)
                multi_labeled_random_data[dim,sample_indices[label_idx]:sample_indices[label_idx+1]] = label_idx
        else:
            raise ValueError('Incorrect distribution type.')



    else:
        raise NameError('Incorrect vector shape type.')

    for label_idx in range(label_n):
        if distribution == 'normal':
            loc = rng.uniform(low=-10, high=10, size=dim)
            scale = 1
            size = (sample_size[label_idx], dim)
            multi_labeled_random_data[sample_indices[label_idx]:sample_indices[label_idx+1],:dim] = rng.normal(loc=loc, scale=scale, size=size)
            multi_labeled_random_data[sample_indices[label_idx]:sample_indices[label_idx+1],dim] = label_idx
            # print(multi_labeled_random_data)
        else:
            raise ValueError('Incorrect distribution type.')
    if shuffle:
        np.random.shuffle(multi_labeled_random_data)

    # Transpose to get column vectors
    return multi_labeled_random_data.T

def get_random_labeled_data(mode='cluster', **kwargs):
    if mode == 'cluster':
        return gen_labeled_clusters(**kwargs)
    

    def gen_labeled_clusters(shuffle=True, dim=2, label_n=3, sample_size=(25,25,25,), distribution='normal', cluster_centers='random', vector_shape='row'):
    if dim >= 10:
        raise ValueError('Number of dimensions too high.')
    if distribution != 'normal':
         raise ValueError('Incorrect distribution type.')
    if len(sample_size) != label_n:
        raise ValueError('Length of sample_size must match label_n.')

    rng = np.random.default_rng(RANDOM_SEED)
    total_samples = np.sum(sample_size)

    if isinstance(cluster_centers, str) and cluster_centers == 'random':
        centers = rng.uniform(low=-10, high=10, size=(label_n, dim))
    elif isinstance(cluster_centers, np.ndarray) and cluster_centers.shape == (label_n, dim):
        centers = cluster_centers
    else:
        raise ValueError('Invalid cluster_centers specification.')

    data_segments = []
    label_segments = []
    scale = 1

    for label_idx in range(label_n):
        loc = centers[label_idx, :]
        num_samples = sample_size[label_idx]
        size = (num_samples, dim)

        cluster_data = rng.normal(loc=loc, scale=scale, size=size)
        cluster_labels = np.full((num_samples, 1), label_idx)

        data_segments.append(cluster_data)
        label_segments.append(cluster_labels)

    all_data = np.vstack(data_segments)
    all_labels = np.vstack(label_segments)

    multi_labeled_random_data = np.hstack((all_data, all_labels))

    if shuffle:
        rng.shuffle(multi_labeled_random_data)

    return multi_labeled_random_data.T


def get_random_labeled_data(mode='cluster', **kwargs):
    if mode == 'cluster':
        return gen_labeled_clusters(**kwargs)
    else:
        raise ValueError(f"Mode '{mode}' not recognized.")

# Example usage:
# data = get_random_labeled_data(dim=2, label_n=3, sample_size=(30, 40, 50))
# print(data.shape)
# print(data)
    