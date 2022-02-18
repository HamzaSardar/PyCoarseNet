import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from typing import List, Any, Dict
from functools import partial
from pycoarsenet.data.initialise_data import InitialiseData


class Network(nn.Module):
    def __init__(self,
                 n_epochs: int,
                 layers: List[int],
                 loss_fn: nn.Module = nn.MSELoss(),
                 optimiser: torch.optim.Optimizer = torch.optim.SGD,
                 activation_fn: nn.Module = nn.Tanh(),
                 learning_rate: float = 1e-2):
        """Instantiates Neural Network.

        Parameters
        ----------
        n_epochs:
            Number of epochs.
        layers:
            List containing number of neurons per layer.
        loss_fn
        optimiser
        activation_fn
        learning_rate
        """
        super().__init__()
        self.n_epochs = n_epochs
        self.loss_fn = loss_fn
        self.activation = activation_fn
        self.learning_rate = learning_rate
        self.optimiser = optimiser(self.model.parameters(), lr=self.learning_rate)

        model_layers: List[nn.Module] = []
        for i in range(len(layers)):
            if i + 1 != len(layers):
                model_layers.append(nn.Linear(layers[i], layers[i+1]))
                model_layers.append(self.activation)
            else:
                model_layers.append(nn.Linear(layers[i], layers[i+1]))

        self.model = nn.Sequential(*model_layers)
        self.train_losses = []
        self.val_losses = []

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
