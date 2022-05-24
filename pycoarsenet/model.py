from typing import List

import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self,
                 feature_size: int,
                 output_size: int,
                 num_hidden_layers: int,
                 num_neurons: int,
                 activation_fn: nn.Module = nn.Tanh()):

        """Instantiates Neural Network.

        Parameters
        ----------
        feature_size: int
            Size of feature vector.
        output_size: int
            Number of elements in the output layer, typically 1.
        num_hidden_layers: int
            Number of hidden layers.
        num_neurons: int
            Number of neurons per hidden layer.
        activation_fn: nn.Module
            Activation function for the network.
        """

        super().__init__()

        # generate layers from input
        layers = [feature_size]
        for i in range(num_hidden_layers):
            layers.append(num_neurons)
        layers.append(output_size)

        self.activation = activation_fn
        self.model = nn.Sequential(*self._create_linear_layers(layers, self.activation))

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
            # checking if the next layer will be the final layer
            if i < len(layers) - 2:
                model_layers.append(nn.Linear(layers[i], layers[i + 1]))
                model_layers.append(activation)
            elif i == len(layers) - 2:
                model_layers.append(nn.Linear(layers[i], 1))
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
