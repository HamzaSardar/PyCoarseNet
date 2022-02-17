import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from typing import List, Any, Dict
from functools import partial
from pycoarsenet.data.initialise_data import InitialiseData


class Network:
    def __init__(self,
                 n_epochs: int,
                 loss_fn: object = nn.MSELoss(),
                 optimiser: object = torch.optim.SGD,
                 learning_rate: float = 1e-2,
                 layers: List[Any]):
        """

        Parameters
        ----------
        layers : object
        """
        self.n_epochs = n_epochs
        self.loss_fn = loss_fn
        self.layers = layers
        _optimiser = partial(optimiser)
        self.learning_rate = learning_rate
        self.model = nn.Sequential(*self.layers)
        self.optimiser = _optimiser(self.model.parameters(), lr=self.learning_rate)

        self.train_losses = []
        self.val_losses = []

    def train(self):
        for epoch in range(self.n_epochs):
            train_features = dataset.features[dataset.train_indices, :]
            train_targets = dataset.targets[dataset.train_indices, :]
            train_size = train_features[0].shape

            train_model_outputs = self.model(train_features.float())
            train_loss = self.loss_fn(train_model_outputs, train_targets.float())
            self.train_losses.append(train_loss)

            self.optimiser.zero_grad()
            train_loss.backward()
            self.optimiser.step()

            with torch.no_grad():
                val_features = dataset.features[dataset.val_indices, :]
                val_targets = dataset.targets[dataset.val_indices, :]
                val_size = train_features[0].shape

                val_model_outputs = self.model(val_features.float())
                val_loss = self.loss_fn(val_model_outputs, val_targets.float())

                self.val_losses.append(val_loss)

    def plot(self, fig_path: str):
        x_axis = torch.linspace(1, self.n_epochs, self.n_epochs)
        plt.plot(x_axis, self.train_losses, '-b', label='Training Loss', linewidth=1)
        plt.plot(x_axis, self.val_losses, 'r', label='Validation Loss', linewidth=1, linestyle='dashed')
        plt.xlabel('Epoch #')
        plt.ylabel('MSE Loss')
        plt.suptitle('Training/Validation Loss')
        plt.title('Experiment 3a')
        plt.legend()
        plt.savefig(fig_path)
