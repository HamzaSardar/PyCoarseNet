import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Tuple
from collections import OrderedDict
import sys
import pycoarsenet.data.initialise as initialise
from pycoarsenet.data.enums import eFeatures

sys.path.append('../')

from pycoarsenet.utils.config import Config

sys.path.append('../')

from src import galilean_invariance


def load_data(data_dir: Path, mode: str, config: Config) -> Tuple[Tensor, Tensor]:
    """Loads data from data_dir.

    Parameters
    ----------
    data_dir: Path
        Absolute path to CFD results in torch.Tensor file format. Use DATA_DIR for training, EVAL_DIR for eval.
    mode: str
        Training or Evaluation.
    config: Config
        YAML config file for the training.

    Returns
    -------
    data_coarse_raw: Tensor
        Coarse grid data from specified simulations.
    data_fine_raw: Tensor
        Fine grid data from specified simulations.
    """

    data_coarse_dict = OrderedDict()
    data_fine_dict = OrderedDict()

    if mode.lower()[0] == 't':
        for alpha in config.TRAINING_SIMS:
            coarse_path = data_dir / f'{alpha}_data_coarse.t'
            fine_path = data_dir / f'{alpha}_data_fine.t'

            data_coarse_dict[alpha] = torch.load(coarse_path)
            data_fine_dict[alpha] = torch.load(fine_path)

        data_coarse_raw = torch.cat(([*data_coarse_dict.values()]), dim=0)
        data_fine_raw = torch.cat(([*data_fine_dict.values()]), dim=0)

    elif mode.lower()[0] == 'e':
        for alpha in config.TEST_SIMS:
            coarse_path = data_dir / f'{alpha}_data_coarse.t'
            fine_path = data_dir / f'{alpha}_data_fine.t'

            data_coarse_dict[alpha] = torch.load(coarse_path)
            data_fine_dict[alpha] = torch.load(fine_path)

        data_coarse_raw = torch.cat(([*data_coarse_dict.values()]), dim=0)
        data_fine_raw = torch.cat(([*data_fine_dict.values()]), dim=0)

    else:
        raise ValueError('Mode must be Training or Evaluation.')

    return data_coarse_raw, data_fine_raw


def generate_features_labels(data_coarse: Tensor, data_fine: Tensor, mode: str, config: Config) -> Tensor:
    """ Preprocess data and compile into one torch.Tensor.

    Parameters
    ----------
    data_coarse: Tensor
        Simulation data from coarse grid CFD.
    data_fine: Tensor
        Simulation data from fine grid CFD.
    mode: str
        Training or Evaluation.
    config: Config
        YAML config file.

    Returns
    -------
    features_labels: Tensor
        Preprocessed tensor containing features and labels, to be converted into torch Dataset.
    """

    # generate tensor column of Peclet number
    if mode.lower()[0] == 't':
        Pe = initialise.generate_cell_Pe(data_coarse, values=config.TRAINING_SIMS, cg_spacing=config.COARSE_SPACING)
    elif mode.lower()[0] == 'e':
        Pe = initialise.generate_cell_Pe(data_coarse, values=config.TEST_SIMS, cg_spacing=config.COARSE_SPACING)
    else:
        raise ValueError('Mode must be "t" or "e".')

    # downsample the fine grid data to have tensors of matching size for the coarse and fine grid data
    data_fine_ds = initialise.downsampling(config.COARSE_SIZE, config.FINE_SIZE, data_fine)

    # take the relevant values from the raw dataset
    targets = data_fine_ds[:, [
                                  eFeatures.T.value,
                                  eFeatures.dT_dX.value,
                                  eFeatures.dT_dY.value,
                                  eFeatures.d2T_dXX.value,
                                  eFeatures.d2T_dXY.value,
                                  eFeatures.d2T_dYX.value,
                                  eFeatures.d2T_dYY.value
                              ], ...]
    features_partial = data_coarse[:, [
                                          eFeatures.T.value,
                                          eFeatures.dT_dX.value,
                                          eFeatures.dT_dY.value,
                                          eFeatures.d2T_dXX.value,
                                          eFeatures.d2T_dXY.value,
                                          eFeatures.d2T_dYX.value,
                                          eFeatures.d2T_dYY.value
                                      ], ...]

    # T is first column in targets and features_partial
    delta_var = targets[:, 0] - features_partial[:, 0]
    labels = delta_var.unsqueeze(-1)

    if config.INVARIANCE:
        # T at column 0, remove from features
        features_partial = features_partial[:, 1:]

        # convert first derivatives to magnitude
        d_mag = galilean_invariance.derivative_magnitude(features_partial[:, :2])

        # convert second derivatives to eigenvalues
        h_eigs = galilean_invariance.hessian_eigenvalues(features_partial[:, 2:])

        if config.TENSOR_INVARIANTS:
            # return tensor invariants
            invar_1, invar_2 = galilean_invariance.hessian_invariants(h_eigs)

            # create feature vector with galilean invariance
            features = torch.cat((d_mag,
                                  invar_1.unsqueeze(-1),
                                  invar_2.unsqueeze(-1)),
                                 dim=-1
                                 )

        else:
            # create feature vector with rotatational invariance
            features = torch.cat((d_mag, h_eigs), dim=-1)
    else:
        # create feature vector with no invariance
        features = features_partial[:, 1:]

    features_labels = torch.cat((features, Pe, labels), dim=-1)

    # normalise all but the labels
    for i in range(features_labels.shape[1] - 1):  # type: ignore
        features_labels = initialise.normalise(features_labels, i)

    return features_labels


def generate_dataloaders(features_labels: Tensor, config: Config) -> Tuple[DataLoader, DataLoader]:
    """ Generate required DataLoaders from preprocessed data.

    Parameters
    ----------
    features_labels: Tensor
        PyTorch Tensor containing features and labels joined at dim=-1.
    config: Config
        YAML config file.

    Returns
    -------
    train_loader: DataLoader, val_loader: DataLoader
        training and validation DataLoaders.
    """

    # convert the data tensor to a TensorDataset and split it into training and validation
    full_ds = TensorDataset(features_labels[:, :-1], features_labels[:, -1])
    train_ds, val_ds = initialise.training_val_split(full_ds, config.TRAINING_FRACTION)

    # use the TensorDataset to create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=True)

    return train_loader, val_loader

