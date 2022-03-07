import csv
import sys
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import einops
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

sys.path.append('..')
import pycoarsenet.data.initialise as initialise
from pycoarsenet.model import Network


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_EPOCHS: int = 500
LEARNING_RATE: float = 1e-2
DATA_DIR: Path = Path('/home/hamza/Projects/TorchFoam/Data/changing_alpha/')
RESULTS_DIR: Path = Path('/home/hamza/Projects/PyCoarseNet/results/')
COARSE_SPACING: float = 0.05
FINE_SIZE: int = 100
COARSE_SIZE: int = 20

# TODO: replace below with an enum
INDICES: List[int] = [3, 4, 5, 7, 8, 10, 11]

ALPHA_VALS: List[float] = [0.001, 0.005, 0.01, 0.05]
TRAINING_FRACTION: float = 0.8
BATCH_SIZE: int = 32


def load_data(data_dir: Path) -> Tuple[Tensor, Tensor]:
    """Loads data from data_dir.

    Parameters
    ----------
    data_dir: Path
        Absolute path to CFD results in torch.Tensor file format.

    Returns
    -------
    data_coarse_raw: Tensor
        Coarse grid data from specified simulations.
    data_fine_raw: Tensor
        Fine grid data from specified simulations.
    """

    data_coarse_dict = OrderedDict()
    data_fine_dict = OrderedDict()

    for alpha in ALPHA_VALS:
        coarse_path = Path(data_dir / f'{alpha}_data_coarse.t')
        fine_path = Path(data_dir / f'{alpha}_data_fine.t')

        data_coarse_dict[alpha] = torch.load(coarse_path)
        data_fine_dict[alpha] = torch.load(fine_path)

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
        Preprocessed tensor containing features and labels, to be converted into torch Dataset.
    """

    Pe = initialise.generate_cell_Pe(data_coarse, ALPHA_VALS, COARSE_SPACING)
    data_fine_ds = initialise.downsampling(COARSE_SIZE, FINE_SIZE, data_fine)

    targets = data_fine_ds[:, INDICES, ...]
    features_partial = data_coarse[:, INDICES, ...]

    # TODO: An enum on the line below would tell you that 0 represents T

    delta_var = targets[:, 0] - features_partial[:, 0]
    labels = delta_var.unsqueeze(-1)

    # T at column 0, remove from features
    features = features_partial[:, 1:]

    # TODO: Check vimnote

    for i in range(features.shape[1]):  # type: ignore
        features = initialise.normalise(features, i)

    # features = einops.rearrange(features, 'sim var i j -> (sim i j) var 1', var=features.shape[1])
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

    full_ds = TensorDataset(features_labels[:, :-1], features_labels[:, -1])
    train_ds, val_ds = initialise.training_val_split(full_ds, TRAINING_FRACTION)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, val_loader


def train(model: Network, train_loader: DataLoader, val_loader: DataLoader) -> None:
    """ Training loop. Writes output to CSV file.

    Parameters
    ----------
    model: Network
        Neural Network.
    train_loader: DataLoader
        DataLoader containing training data.
    val_loader: DataLoader
        DataLoader containing validation data.
    """

    # set Optimiser
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # set loss function
    loss_fn: nn.Module = nn.MSELoss()

    # work on global(DEVICE) - GPU if available
    model.to(DEVICE)

    # prepare results file
    RESULTS_DIR.mkdir(parents=True, exist_ok=False)
    with open(RESULTS_DIR / 'results.csv', 'w+', newline='') as f_out:
        writer = csv.writer(f_out, delimiter=',')

        writer.writerow([
            'epoch',
            'train_mse_loss',
            'val_ms_loss'
        ])

    print('Training...')

    # training loop
    for epoch in range(N_EPOCHS):

        # reset epoch loss
        train_mse_loss = 0.0
        val_mse_loss = 0.0

        # training step
        model.train()
        for feature_vector, label in train_loader:
            # move data to GPU if available
            feature_vector = feature_vector.to(DEVICE)
            label = label.to(DEVICE)

            batch_t_model_outputs = model.forward(feature_vector.float())
            batch_t_loss = loss_fn(batch_t_model_outputs, label.float())

            # increment epoch loss by batch loss
            # total batch loss = batch loss * number of values in batch
            # .item() returns the value in a 1 element tensor as a number
            train_mse_loss += batch_t_loss.item() * feature_vector.size(0)

            # update the parameters
            optimiser.zero_grad()
            batch_t_loss.backward()
            optimiser.step()

        # validation step
        model.eval()
        for feature_vector, label in val_loader:
            # move data to GPU if available
            feature_vector = feature_vector.to(DEVICE)
            label = label.to(DEVICE)

            batch_v_model_outputs = model.forward(feature_vector.float())
            batch_v_loss = loss_fn(batch_v_model_outputs, label.float())

            # increment epoch loss by batch loss
            val_mse_loss += batch_v_loss.item() * feature_vector.size(0)

        # normalise loss by number of data points in DataLoader
        train_mse_loss /= len(train_loader.dataset)  # type: ignore
        val_mse_loss /= len(val_loader.dataset)  # type: ignore

        # print epoch results
        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, \n train_loss: {train_mse_loss:5f} \n val_loss: {val_mse_loss:5f} \n')

        # log results in list
        epoch_results = [epoch, train_mse_loss, val_mse_loss]

        # open results file in append mode and write results as csv
        with open(RESULTS_DIR / 'results.csv', 'a', newline='') as f_out:
            writer = csv.writer(f_out, delimiter=',')
            writer.writerow(epoch_results)

    # end training loop
    print('Training complete.')


def main():
    model: Network = Network([7, 10, 1])
    dc_raw, dc_fine = load_data(DATA_DIR)
    fl = generate_features_labels(dc_raw, dc_fine)
    train_loader, val_loader = generate_dataloaders(fl)
    train(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
