from typing import List, Tuple, Any

import einops  # type: ignore
import torch
from torch.utils.data import random_split


def generate_cell_Pe(data: torch.Tensor,
                     values: List[float],
                     cg_spacing: float) -> torch.Tensor:
    """Generates a tensor of cell-centres Peclet values.
    Parameters
    ----------
    data:
        input data without dimensionless group.
    values:
        values the dimensionless group can take.
    cg_spacing:
        spacing in the coarse grid.

    Returns
    -------
    cell_Pe:
        Tensor containing Peclet values.
    """
    return torch.cat(([(cg_spacing * torch.ones(data.shape[0]) / i) for i in values]), dim=0).unsqueeze(-1)


def downsampling(coarse_size: int, fine_size: int, data_fine: torch.Tensor) -> torch.Tensor:
    """Downsamples fine grid data by coarse grid cell centres.
    For this method, the ratio of fine grid to coarse grid data points must be an odd integer, >1.
    This method assumes a 2D domain.

    Parameters
    ----------
    coarse_size:
        Number of elements along one side of a square coarse grid.
    fine_size:
        Number of elements along one side of a square fine grid.
    data_fine:
        Fine grid data tensor.

    Returns
    -------
    Downsampled fine grid data tensor, of shape = data_coarse.shape
    """
    num_var = data_fine[-1]
    data_fine = einops.rearrange(data_fine,
                                 '(sim row column) variable -> (sim variable) row column',
                                 variable=data_fine[-1], row=fine_size, column=fine_size)
    downsampling_ratio = int(fine_size / coarse_size)

    if downsampling_ratio % 2 == 0:
        raise ValueError('Fine and Coarse grid size incompatible with downsampling...')

    sampling_bools = generate_sampling_points(downsampling_ratio, fine_size, coarse_size)
    downsampled_data = data_fine[:, sampling_bools]

    return einops.rearrange(downsampled_data,
                            '(sim variable) row column -> (sim row column) variable',
                            variable=num_var)


def generate_sampling_points(downsampling_ratio, fine_size, coarse_size) -> torch.Tensor:
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
    sampling_indices = torch.linspace(int(int(downsampling_ratio / 2) + 1),
                                      fine_size - int(downsampling_ratio / 2),
                                      steps=coarse_size)

    sampling_bools = torch.zeros((fine_size, fine_size), dtype=torch.bool)
    for i in range(1, fine_size + 1):
        for j in range(1, fine_size + 1):
            if i in sampling_indices and j in sampling_indices:
                sampling_bools[i - 1, j - 1] = True
    return sampling_bools


def extract_features(data: torch.Tensor, indices: List[int]) -> torch.Tensor:
    """
    Parameters
    ----------
    data:
        Raw input data in form [Simulation, Variable, ...].
    indices:
        Indices selected for feature vector.

    Returns
    -------
    object
    Tensor of features.
    """
    return data[:, indices, ...]


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


def training_val_split(data: torch.Tensor, frac_train: float) -> Tuple[Any, Any]:
    """Generates indices for training validation split.

    Parameters
    ----------
    data: torch.Tensor
        input data, only a single channel is required as the row numbers should be consistent across channels.
    frac_train:
        Fraction of data required for training.

    Returns
    -------
    Tuple[List[int]]:
        Two lists of randomly generated non-overlapping indices.
    """

    if frac_train > 1:
        raise ValueError('Cannot split dataset. frac_train must be <=1')
    samples = torch.numel(data)
    train, val = random_split(data, [int(frac_train * samples), samples - int(frac_train * samples)])  # type: ignore

    return train, val
