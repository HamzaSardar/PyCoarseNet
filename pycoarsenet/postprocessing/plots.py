from pathlib import Path
from typing import List

import torch
import numpy as np
from matplotlib import pyplot as plt  # type: ignore

plt.rcParams.update({'font.size': 20})


def plot_loss_history(fig_path: Path,
                      n_epochs: int,
                      train_losses: List[float],
                      val_losses: List[float],
                      loss_fn: str,
                      experiment: str) -> None:
    """ Plots training_validation loss against epochs.

    Parameters
    ----------
    fig_path: Path
        Path to location for saving figures.
    n_epochs: int
        Number of epochs.
    train_losses: List[float]
        Training loss history as list.
    val_losses: List[float]
        Validation loss history as list.
    loss_fn: str
        Loss function used, as string.
    experiment: str
        Brief title or description of current run.

    """

    x = torch.linspace(1, n_epochs, n_epochs)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.gca()

    ax.plot(x, train_losses, '-b', label='Training Loss', linewidth=1)
    ax.plot(x, val_losses, 'r', label='Validation Loss', linewidth=1, linestyle='dashed')
    plt.xlabel('Epoch #', axes=ax)
    plt.ylabel(loss_fn, axes=ax)
    plt.suptitle('Training/Validation Loss', axes=ax)
    plt.title(experiment, axes=ax)
    plt.legend()
    fig.savefig(fig_path)


def plot_model_evaluation(fig_path: Path,
                          model_predictions: np.ndarray,
                          actual_error: np.ndarray,
                          title: str) -> None:
    """ Plots model predictions against ground truth.

    Parameters
    ----------
    fig_path: Path
        Path to location for saving figures.
    model_predictions: np.ndarray
        Model outputs as a numpy array.
    actual_error: np.ndarray
        Ground truth as a numpy array.
    title: str
        Figure title.
    """

    fig = plt.figure(figsize=(16, 16))
    ax = fig.gca()

    ax.scatter(model_predictions, actual_error, s=100)
    ax.plot(actual_error, actual_error, 'black', label='actual_error = predicted_error')

    plt.suptitle('Grid error in cell-centred T', axes=ax)
    plt.title(title, axes=ax)
    # plt.rcParams['font.size'] = '8'
    plt.xlabel('Model-predicted error', axes=ax)
    plt.ylabel('Actual error', axes=ax)
    plt.legend()

    fig.savefig(fig_path)
