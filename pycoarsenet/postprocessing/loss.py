from pathlib import Path
from typing import List

import torch
from matplotlib import pyplot as plt  # type: ignore


def plot_loss_history(fig_path: Path,
                      n_epochs: int,
                      train_losses: List[int],
                      val_losses: List[int],
                      loss_fn: str,
                      experiment: str):
    x_axis = torch.linspace(1, n_epochs, n_epochs)  # type: ignore
    plt.plot(x_axis, train_losses, '-b', label='Training Loss', linewidth=1)
    plt.plot(x_axis, val_losses, 'r', label='Validation Loss', linewidth=1, linestyle='dashed')
    plt.xlabel('Epoch #')
    plt.ylabel(loss_fn)
    plt.suptitle('Training/Validation Loss')
    plt.title(experiment)
    plt.legend()
    plt.savefig(fig_path)
