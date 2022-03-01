import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pycoarsenet.data.initialise as initialise
from pycoarsenet.training.model_setup import Network


NETWORK: Network = Network([7, 10, 1])
N_EPOCHS: int = 500
DATA_DIR: str = ''
DATA_COARSE, DATA_FINE = load_data(DATA_DIR)
COARSE_SPACING: float = 0.05
FINE_SIZE: int = 100
COARSE_SIZE: int = 20
INDICES List[int] = (3, 4, 5, 7, 8, 10, 11, 16)
ALPHA_VALS: List[float] = [0.001, 0.005, 0.01, 0.05]


def load_data(data_dir) -> torch.Tensor, torch.Tensor:
    data_coarse_dict = {}
    data_fine_dict = {}
    for alpha in ALPHA_VALS:
        data_coarse_dict[i] = torch.load(
            f'{data_dir}{i}_data_coarse.t')
        data_fine_dict[i] = torch.load(
            f'{data_dir}{i}_data_fine.t')
    data_coarse_raw = torch.cat((data_coarse_dict.values()), dim=-1)
    data_fine_raw = torch.cat((data_fine_dict.values()), dim=0)
    return data_coarse_raw, data_fine_raw


def generate_dataloader(data_coarse, data_fine):
    Pe = initialise.generate_cell_Pe(data_coarse,
                                        ALPHA_VALS,
                                        COARSE_SPACING
                                        )
    data_fine = initialise.downsampling(COARSE_SIZE, FINE_SIZE, data_fine)
    targets = initialise._extract_features(data_fine, INDICES[:-1])
    features = initialise._extract_features(data_coarse, INDICES[:-1])
    delta_var = self._extract_targets(features[:, 0], targets[:, 0])
    labels = einops.rearrange(delta_var,
                              'simulation row column -> (simulation
                              row column').unsqueeze(-1)
    for i in len(features[1].shape):
        features = initialise.normalise(features, i) 
    features = initialise._extract_features(data_coarse, INDICES) 
    features = einops.rearrange(features,
                              'simulation row column -> (simulation
                              row column').unsqueeze(-1)
def train()









if __name__ == '__main__':
    # TODO: Do more of the heavy lifting in this script i.e. make the classes cleaner with less abstraction.
    # TODO: Load in data and run the initialise functions.

    """
    def train(self, **kwargs):
        for epoch in range(self.n_epochs):
            self.model.train()
            train_features = dataset.features[dataset.train_indices, :]
            train_targets = dataset.targets[dataset.train_indices, :]

            train_model_outputs = self.model(train_features.float())
            train_loss = self.loss_fn(train_model_outputs, train_targets.float())
            self.train_losses.append(train_loss)

            self.optimiser.zero_grad()
            train_loss.backward()
            self.optimiser.step()

            self.model.eval()
            val_features = dataset.features[dataset.val_indices, :]
            val_targets = dataset.targets[dataset.val_indices, :]

            val_model_outputs = self.model(val_features.float())
            val_loss = self.loss_fn(val_model_outputs, val_targets.float())

            self.val_losses.append(val_loss)

    """

    dataset = InitialiseData(data_dir='/home/hamza/Projects/TorchFoam/Data/changing_alpha/',
                             num_sims=4,
                             list_var=[0.001, 0.005, 0.01, 0.05],
                             cg_spacing=0.05,
                             coarse_size=20,
                             fine_size=100,
                             Dimensionless='Pe')

