from typing import List, Tuple, Any

import einops  # type: ignore
import torch
from torch.utils.data import random_split, TensorDataset, dataset


def generate_cell_Pe(data: torch.Tensor, values: List[float], cg_spacing: float) -> torch.Tensor:
    """Generates a tensor of cell-centres Peclet values.
    Parameters
    ----------
    data:
        Input data without dimensionless group.
    values:
        Values the dimensionless group can take.
    cg_spacing:
        Spacing in the coarse grid.

    Returns
    -------
    cell_Pe:
        Tensor containing Peclet values.
    """

    # Cell peclet number is delta_x / diffusivity
    num_var = data.shape[-1]
    data = einops.rearrange(data, '(sim i j) var -> (i j) sim var', sim=len(values), var=num_var, i=int(1 / cg_spacing))
    Pe_tensor = torch.cat(([(cg_spacing * torch.ones(data.shape[0]) / i) for i in values]), dim=0).unsqueeze(-1)

    return Pe_tensor


def downsampling(coarse_size: int, fine_size: int, data_fine: torch.Tensor) -> torch.Tensor:
    """Downsamples fine grid data by coarse grid cell centres.
    For this method, the ratio of fine grid to coarse grid data points must be an odd integer, >1.
    This method assumes a 2D domain.

    Parameters
    ----------
    coarse_size: int
        Number of elements along one side of a square coarse grid.
    fine_size: int
        Number of elements along one side of a square fine grid.
    data_fine: torch.Tensor
        Fine grid data tensor.

    Returns
    -------
    Downsampled fine grid data tensor, of shape = data_coarse.shape
    """

    num_var = data_fine.shape[-1]
    data_fine = einops.rearrange(data_fine, '(s i j) v -> (s v) i j', v=num_var, i=fine_size, j=fine_size)
    downsampling_ratio = int(fine_size / coarse_size)

    if downsampling_ratio % 2 == 0:
        raise ValueError('Fine and Coarse grid size incompatible with downsampling...')

    sampling_bools = generate_sampling_points(downsampling_ratio, fine_size, coarse_size)
    sampled_data = data_fine[:, sampling_bools]
    data_fine = einops.rearrange(sampled_data, '(s v) (i j) -> (s i j) v', v=num_var, i=coarse_size, j=coarse_size)

    return data_fine


def generate_sampling_points(downsampling_ratio: int, fine_size: int, coarse_size: int) -> torch.Tensor:
    """ Generates a tensor of booleans at which to downsample fine grid data.

    Parameters
    ----------
    downsampling_ratio: int
        ratio between fine_size and coarse_size.
    fine_size: int
        Number of elements along one side of a square fine grid.
    coarse_size: int
        Number of elements along one side of a square coarse grid.

    Returns
    -------
    torch.Tensor
        Tensor of booleans where sampling locations are 1 and the rest of the tensor is 0.
    """

    # TODO: Make below more legible, split it up

    sampling_indices = torch.linspace(int(int(downsampling_ratio / 2) + 1),
                                      fine_size - int(downsampling_ratio / 2),
                                      steps=coarse_size)

    # TODO: No need to iterate, just iterate over sampling_indices (itertools.combinations of i, j)
    sampling_bools = torch.zeros((fine_size, fine_size), dtype=torch.bool)
    for i in range(1, fine_size + 1):
        for j in range(1, fine_size + 1):
            if i in sampling_indices and j in sampling_indices:
                sampling_bools[i - 1, j - 1] = True

    return sampling_bools


# TODO: Below doesn't need to exist, just do it expliticitly

def normalise(features: torch.Tensor, variable: int) -> torch.Tensor:
    """Normalise data according to specified channel.

    Parameters
    ----------
    features:
        Feature tensor.
    variable:
        Channel along which to normalise.

    Returns
    -------
    Feature tensor with specified channel normalised.
    """
    mean = torch.mean(features[:, variable])
    std = torch.std(features[:, variable])
    features[:, variable] = (features[:, variable] - mean) / std
    return features


def training_val_split(data: TensorDataset, frac_train: float) -> Tuple[dataset.Subset, dataset.Subset]:
    """Generates indices for training validation split.

    Parameters
    ----------
    data: Dataset
        input data, only a single channel is required as the row numbers should be consistent across channels.
    frac_train: float
        Fraction of data required for training.

    Returns
    -------
    train_ds: dataset.Subset
        Subset of the whole dataset to be used for training.
    val_ds: dataset.Subset
        Subset of the whole dataset to be used for validation.
    """

    if frac_train > 1:
        raise ValueError('Cannot split dataset. frac_train must be <=1')

    num_train = int(data.tensors[0].shape[0] * frac_train)
    num_val = int(data.tensors[0].shape[0] - num_train)
    train_ds, val_ds = random_split(data, (num_train, num_val))

    return train_ds, val_ds
