import torch
import einops

from typing import List, Tuple


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
    return torch.cat(([(cg_spacing * torch.ones(data[i].shape[0]) / i) for i in values]), dim=0).unsqueeze(-1)


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
    downsampling_ratio = int(fine_size / coarse_size)

    if downsampling_ratio % 2 == 0:
        raise ValueError('Fine and Coarse grid size incompatible with downsampling.')

    sampling_indices = torch.linspace(int(int(downsampling_ratio / 2) + 1),
                                      fine_size - int(downsampling_ratio / 2),
                                      steps=coarse_size)

    sampling_bools = torch.zeros((fine_size, fine_size), dtype=torch.bool)
    for i in range(1, fine_size + 1):
        for j in range(1, fine_size + 1):
            if i in sampling_indices and j in sampling_indices:
                sampling_bools[i - 1, j - 1] = True

    downsampled_data: torch.Tensor = einops.rearrange(
        data_fine[:, :, sampling_bools],
        'simulation variable (row column) -> simulation variable row column',
        row=coarse_size)

    return downsampled_data


def extract_features(data: torch.Tensor, indices: Tuple) -> torch.Tensor:
    """
    Parameters
    ----------
    data:
        Raw input data in form [Simulation, Variable, ...].
    indices:
        Indices selected for feature vector.

    Returns
    -------
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


def training_val_split(n_samples: int, val_fraction: float):
    return n_val := int(val_fraction * n_samples)
