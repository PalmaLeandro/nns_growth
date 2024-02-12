"""
This module implements a method to sample data from the noisy XOR setting studied in https://arxiv.org/pdf/2202.07626.pdf.
The method implements an optional rotation to study axis aligned biases.
"""

import numpy, scipy, torch

CLASSES, CLUSTERS_PER_CLASS = 2, 2

def get_dataloader(noise_rate:float, within_cluster_variance:float, input_dimension:int, sample_size:int, 
                   batch_size:int=1, rotation_matrix=None, *args, **kwargs):
    samples_cluster = numpy.random.choice(list(range(CLASSES * CLUSTERS_PER_CLASS)), size=sample_size)

    inputs = labels = None
    for cluster in range(CLASSES * CLUSTERS_PER_CLASS):
        cluster_num_samples = len(numpy.where(samples_cluster == cluster)[0])
        cluster_dimension = 0 if cluster < CLASSES else 1
        cluster_mean = [0.] * input_dimension 
        cluster_mean[cluster_dimension] = 1. if cluster % CLUSTERS_PER_CLASS == 0 else -1.
        
        # cluster_num_samples samples are calculated as input_dimension Gaussians with variance = within_cluster_variance
        cluster_inputs = numpy.random.normal(scale=within_cluster_variance ** 0.5, size=cluster_num_samples * input_dimension)
        cluster_inputs = cluster_inputs.reshape(cluster_num_samples, input_dimension) + cluster_mean
        cluster_labels = numpy.repeat(1. if cluster < CLASSES else -1., cluster_num_samples)
        inputs = cluster_inputs if inputs is None else numpy.concatenate([inputs, cluster_inputs])
        labels = cluster_labels if labels is None else numpy.concatenate([labels, cluster_labels])

    rotation_matrix_ = scipy.stats.special_ortho_group.rvs(input_dimension) if rotation_matrix is None else rotation_matrix
    inputs = numpy.matmul(inputs, rotation_matrix_)

    if noise_rate > 0:
        label_flipping_mask = numpy.random.choice([1., -1.], size=sample_size, p=[1 - noise_rate, noise_rate])
        labels *= label_flipping_mask

    # Transform -1, +1 labels to 0, 1 for cross entropy loss
    labels += 1.
    labels *= 0.5
    
    with torch.no_grad():
        tensor_X = torch.Tensor(inputs)
        tensor_y = torch.Tensor(labels).unsqueeze(1)

    dataset = torch.utils.data.TensorDataset(tensor_X, tensor_y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    
    return (dataloader, rotation_matrix_) if rotation_matrix is None else dataloader