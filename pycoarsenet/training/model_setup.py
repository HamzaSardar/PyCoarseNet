import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from typing import List, Any, Dict


class Network(nn.Module):
    def __init__(self,
                 layers: List[int],
                 loss_fn: nn.Module = nn.MSELoss(),
                 optimiser: torch.optim.Optimizer = torch.optim.SGD,
                 activation_fn: nn.Module = nn.Tanh(),
                 learning_rate: float = 1e-2):
        """Instantiates Neural Network.

        Parameters
        ----------
        layers:
            List containing number of neurons per layer.
        loss_fn
        optimiser
        activation_fn
        learning_rate
        """
        super().__init__()
        self.loss_fn = loss_fn
        self.activation = activation_fn
        self.learning_rate = learning_rate
        self.optimiser = optimiser(self.model.parameters(), lr=self.learning_rate)
        self.model = nn.Sequential(*self._create_linear_layers(layers, self.activation))
        self.train_losses = []
        self.val_losses = []

    @staticmethod
    def _create_linear_layers(layers, activation) -> List[nn.Module]:
        model_layers: List[nn.Module] = []
        for i in range(len(layers)):
            if i + 1 != len(layers):
                model_layers.append(nn.Linear(layers[i], layers[i+1]))
                model_layers.append(activation)
            else:
                model_layers.append(nn.Linear(layers[i], layers[i+1]))
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
