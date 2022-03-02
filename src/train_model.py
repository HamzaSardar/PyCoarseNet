import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import einops
from typing import List, Tuple, NoReturn

import pycoarsenet.data.initialise as initialise
from pycoarsenet.training.model_setup import Network


MODEL: Network = Network([7, 10, 1])
N_EPOCHS: int = 500
DATA_DIR: str = '/home/hamza/Projects/TorchFoam/Data/changing_alpha/'
COARSE_SPACING: float = 0.05
FINE_SIZE: int = 100
COARSE_SIZE: int = 20
INDICES: List[int] = [3, 4, 5, 7, 8, 10, 11, 16]
ALPHA_VALS: List[float] = [0.001, 0.005, 0.01, 0.05]
TRAINING_FRACTION: float = 0.8
BATCH_SIZE = 32


def load_data(data_dir: str) -> Tuple[Tensor, Tensor]:
    """Loads data from data_dir.

    Parameters
    ----------
    data_dir: str
        Absolute path to CFD results in torch.Tensor file format.

    Returns
    -------
    data_coarse_raw: Tensor, data_fine_raw: Tensor
        Coarse and fine grid data from specified simulations.
    """
    data_coarse_dict = {}
    data_fine_dict = {}
    for alpha in ALPHA_VALS:
        data_coarse_dict[alpha] = torch.load(
         f'{data_dir}{alpha}_data_coarse.t')
        data_fine_dict[alpha] = torch.load(
         f'{data_dir}{alpha}_data_fine.t')
    data_coarse_raw = torch.cat(([*data_coarse_dict.values()]), dim=0)
    data_fine_raw = torch.cat(([*data_fine_dict.values()]), dim=0)
    return data_coarse_raw, data_fine_raw


def generate_features_labels(data_coarse: Tensor, data_fine: Tensor) -> Tensor:
    """ Preprocess data and compile into one torch.Tensor.

    Parameters
    ----------
    data_coarse: Tensor
        Simulation data from coarse grid CFD.
    data_fine: Tensor
        Simulation data from fine grid CFD.

    Returns
    -------
    features_labels: Tensor
        Preprocessed tensor to be converted into torch Dataset.
    """
    Pe = initialise.generate_cell_Pe(data_coarse,
                                     ALPHA_VALS,
                                     COARSE_SPACING)

    data_fine_ds = initialise.downsampling(COARSE_SIZE, FINE_SIZE,
                                           data_fine)
    targets = initialise.extract_features(data_fine_ds, INDICES[:-1])
    features_partial = initialise.extract_features(data_coarse, INDICES[:-1])
    delta = targets - features_partial
    delta_var = delta[:, 0]
    labels = einops.rearrange(delta_var,
                              'simulation row column -> (simulation row column)').unsqueeze(-1)
    features = initialise.extract_features(data_coarse, INDICES)
    for i in features.shape[1]:
        features = initialise.normalise(features, i)
    features = einops.rearrange(features,
                                'simulation variable row column -> (simulation row column) variable',
                                variable=features.shape[1]).unsqueeze(-1)
    features_labels = torch.cat((features, Pe, labels), dim=-1)
    return features_labels


def generate_dataloaders(features_labels: Tensor) -> Tuple[DataLoader, DataLoader]:
    """ Generate required DataLoaders from preprocessed data.

    Parameters
    ----------
    features_labels:
        Preprocessed dataset.
    Returns
    -------
    train_loader: DataLoader, val_loader: DataLoader
        training and validation DataLoaders.
    """
    train_indices, val_indices = initialise.training_val_split(features_labels[0], TRAINING_FRACTION)
    train_ds = SimulationDataset(features_labels[:, train_indices])
    val_ds = SimulationDataset(features_labels[:, val_indices])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, val_loader


def train(model: Network, train_loader: DataLoader, val_loader: DataLoader) -> NoReturn:
    """ Training loop.

    Parameters
    ----------
    model: Network
        Neural Network.
    train_loader: DataLoader
        DataLoader containing training data.
    val_loader: DataLoader
        DataLoader containing validation data.
    """
    # TODO: Add in epoch wise losses to address issue #6

    for epoch in range(N_EPOCHS):

        train_losses = []
        val_losses = []
        model.train()

        for feature_vector, label in train_loader:
            train_model_outputs = model(feature_vector.float())
            train_loss = model.loss_fn(train_model_outputs, label.float())
            train_losses.append(train_loss)
            model.optimiser.zero_grad()
            train_loss.backward()
            model.optimiser.step()

        model.eval()
        for feature_vector, label in val_loader:
            val_model_outputs = model(feature_vector.float())
            val_loss = model.loss_fn(val_model_outputs, label.float())
            val_losses.append(val_loss)


"""
Custom Dataset for use with CFD results in torch.Tensor format.
"""


class SimulationDataset(Dataset):

    def __init__(self, data: Tensor):
        """ Custom dataset to handle simulation results and pass information to DataLoaders.

        Parameters
        ----------
        data: Tensor
            Raw dataset.

        Returns
        ----------
        features: Tensor, label: Tensor
            Feature vector for the given row, label for the given row.
        """
        self.data = data
        self.length = data.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        features = self.data[idx, :-1]
        label = self.data[idx, -1]
        return features, label


def main():
    dc_raw, dc_fine = load_data(DATA_DIR)
    fl = generate_features_labels(dc_raw, dc_fine)
    train_loader, val_loader = generate_dataloaders(fl)
    train(MODEL, train_loader, val_loader)


if __name__ == '__main__':
    main()
