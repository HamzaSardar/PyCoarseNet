from typing import List

import matplotlib.pyplot as plt  # type: ignore
import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self,
                 layers: List[int],
                 loss_fn: nn.Module = nn.MSELoss(),
                 optimiser=torch.optim.SGD,
                 activation_fn: nn.Module = nn.Tanh(),
                 learning_rate: float = 1e-2):
        """Instantiates Neural Network.

        Parameters
        ----------
        layers: List[int]
            List containing number of neurons per layer.
        loss_fn: nn.Module
            Loss function.
        optimiser:
            Optimiser from `torch.optim`.
        activation_fn: nn.Module
            Activation function for the network.
        learning_rate: float
            Initial learning rate.
        """
        super().__init__()
        self.loss_fn = loss_fn
        self.activation = activation_fn
        self.learning_rate = learning_rate
        self.model = nn.Sequential(*self._create_linear_layers(layers, self.activation))
        self.optimiser = optimiser(self.model.parameters(), lr=self.learning_rate)
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    @staticmethod
    def _create_linear_layers(layers: List[int], activation: nn.Module) -> List[nn.Module]:
        """ Helper function - creates linear layers of neural network separated by activations where required.

        Parameters
        ----------
        layers: List[int]
            List containing number of neurons per layer including input and output layer.
        activation: nn.Module
            Activation function.

        Returns
        -------
        model_layers: List[nn.Module]
            List of `nn.Linear(...)` layers.
        """
        model_layers: List[nn.Module] = []
        for i in range(len(layers)):
            if i + 1 != len(layers):
                model_layers.append(nn.Linear(layers[i], layers[i + 1]))
                model_layers.append(activation)
            else:
                model_layers.append(nn.Linear(layers[i], layers[i + 1]))
        return model_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Perform a single forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Input feature vector.

        Returns
        -------
        Ouput: torch.Tensor
        """
        # TODO: Add dimensionality check before doing self.model(x).
        return self.model(x)
