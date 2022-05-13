import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import wandb
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from wandb.sdk.lib import RunDisabled
from wandb.wandb_run import Run

import galilean_invariance

sys.path.append('..')
import pycoarsenet.data.initialise as initialise
from pycoarsenet.model import Network
from pycoarsenet.data.enums import eFeatures
from pycoarsenet.postprocessing import plots
from pycoarsenet.utils.config import Config

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAINING_VALS: List[float] = [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.05]
# EVAL_VALS: List[float] = [0.001, 0.005, 0.01, 0.05]
EVAL_VALS: List[float] = [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.05]

TRAIN_LOSSES: List[float] = []
VAL_LOSSES: List[float] = []


def load_data(data_dir: Path, mode: str) -> Tuple[Tensor, Tensor]:
    """Loads data from data_dir.

    Parameters
    ----------
    data_dir: Path
        Absolute path to CFD results in torch.Tensor file format. Use DATA_DIR for training, EVAL_DIR for eval.
    mode: str
        Training or Evaluation.

    Returns
    -------
    data_coarse_raw: Tensor
        Coarse grid data from specified simulations.
    data_fine_raw: Tensor
        Fine grid data from specified simulations.
    """

    data_coarse_dict = OrderedDict()
    data_fine_dict = OrderedDict()

    if mode.lower()[0] == 't':
        for alpha in TRAINING_VALS:
            coarse_path = data_dir / f'{alpha}_data_coarse.t'
            fine_path = data_dir / f'{alpha}_data_fine.t'

            data_coarse_dict[alpha] = torch.load(coarse_path)
            data_fine_dict[alpha] = torch.load(fine_path)

        data_coarse_raw = torch.cat(([*data_coarse_dict.values()]), dim=0)
        data_fine_raw = torch.cat(([*data_fine_dict.values()]), dim=0)

    elif mode.lower()[0] == 'e':
        for alpha in EVAL_VALS:
            coarse_path = data_dir / f'{alpha}_data_coarse.t'
            fine_path = data_dir / f'{alpha}_data_fine.t'

            data_coarse_dict[alpha] = torch.load(coarse_path)
            data_fine_dict[alpha] = torch.load(fine_path)

        data_coarse_raw = torch.cat(([*data_coarse_dict.values()]), dim=0)
        data_fine_raw = torch.cat(([*data_fine_dict.values()]), dim=0)

    else:
        raise ValueError('Mode must be Training or Evaluation.')

    return data_coarse_raw, data_fine_raw


def generate_features_labels(data_coarse: Tensor, data_fine: Tensor, mode: str, config: Config) -> Tensor:
    """ Preprocess data and compile into one torch.Tensor.

    Parameters
    ----------
    data_coarse: Tensor
        Simulation data from coarse grid CFD.
    data_fine: Tensor
        Simulation data from fine grid CFD.
    mode: str
        Training or Evaluation.
    config: Config
        YAML config file.

    Returns
    -------
    features_labels: Tensor
        Preprocessed tensor containing features and labels, to be converted into torch Dataset.
    """
    if mode.lower()[0] == 't':
        Pe = initialise.generate_cell_Pe(data_coarse, values=TRAINING_VALS, cg_spacing=config.COARSE_SPACING)
    elif mode.lower()[0] == 'e':
        Pe = initialise.generate_cell_Pe(data_coarse, values=EVAL_VALS, cg_spacing=config.COARSE_SPACING)
    else:
        raise ValueError('Mode must be "t" or "e".')

    data_fine_ds = initialise.downsampling(config.COARSE_SIZE, config.FINE_SIZE, data_fine)

    targets = data_fine_ds[:, [
                                  eFeatures.T.value,
                                  eFeatures.dT_dX.value,
                                  eFeatures.dT_dY.value,
                                  eFeatures.d2T_dXX.value,
                                  eFeatures.d2T_dXY.value,
                                  eFeatures.d2T_dYX.value,
                                  eFeatures.d2T_dYY.value
                              ], ...]
    features_partial = data_coarse[:, [
                                          eFeatures.T.value,
                                          eFeatures.dT_dX.value,
                                          eFeatures.dT_dY.value,
                                          eFeatures.d2T_dXX.value,
                                          eFeatures.d2T_dXY.value,
                                          eFeatures.d2T_dYX.value,
                                          eFeatures.d2T_dYY.value
                                      ], ...]

    # T is first column in targets and features_partial
    delta_var = targets[:, 0] - features_partial[:, 0]
    labels = delta_var.unsqueeze(-1)

    if config.INVARIANCE:
        # T at column 0, remove from features
        features_partial = features_partial[:, 1:]

        # convert first derivatives to magnitude
        d_mag = galilean_invariance.derivative_magnitude(features_partial[:, 2])

        # convert second derivatives to eigenvalues
        h_eigs = galilean_invariance.hessian_eigenvalues(features_partial[:, 2:])

        if config.TENSOR_INVARIANTS:
            # return tensor invariants
            invar_1, invar_2 = galilean_invariance.hessian_invariants(h_eigs)

            """Experiment 1 features"""
            features = torch.cat((d_mag,
                                  invar_1.unsqueeze(-1),
                                  invar_2.unsqueeze(-1)),
                                 dim=-1
                                 )

            """Experiment 2 features"""
            # features = torch.cat((d_mag, h_eigs), dim=-1)

        else:
            # join d_mag and h_eigs to make features
            # features = torch.cat((features_partial[:, :2], h_eigs), dim=-1)
            """Uncomment below for both derivatives to be invariant"""
            # features = torch.cat((d_mag, h_eigs), dim=-1)
    else:
        features = features_partial[:, 1:]

    features_labels = torch.cat((features, Pe, labels), dim=-1)

    """ removing points with error O(10e-6) and below"""

    # zero_indices = []
    #
    # for i in range(labels.shape[0]):
    #     if abs(labels[i]) <= 10e-6:
    #         zero_indices.append(i)
    #
    # def th_delete(tensor, indices):
    #     mask = torch.ones(tensor[:, 0].numel(), dtype=torch.bool)
    #     mask[indices] = False
    #     return tensor[mask]
    #
    # features_labels = th_delete(features_labels, zero_indices)
    # print(f'Number of data points used: {features_labels.shape[0]}')

    """ statistical or physical normalisation: """
    # normalise all but the labels
    for i in range(features_labels.shape[1] - 1):  # type: ignore
        features_labels = initialise.normalise(features_labels, i)

    # # normalise gradients with grid spacing
    # features_labels[:, 0] *= COARSE_SPACING
    # features_labels[:, (1, 2)] *= (COARSE_SPACING ** 2)
    #
    # # normalise Pe statistically
    # features_labels = initialise.normalise(features_labels, -1)

    return features_labels


def generate_dataloaders(features_labels: Tensor, config: Config) -> Tuple[DataLoader, DataLoader]:
    """ Generate required DataLoaders from preprocessed data.

    Parameters
    ----------
    features_labels: Tensor
        PyTorch Tensor containing features and labels joined at dim=-1.
    config: Config
        YAML config file.

    Returns
    -------
    train_loader: DataLoader, val_loader: DataLoader
        training and validation DataLoaders.
    """

    full_ds = TensorDataset(features_labels[:, :-1], features_labels[:, -1])
    train_ds, val_ds = initialise.training_val_split(full_ds, config.TRAINING_FRACTION)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=True)

    return train_loader, val_loader


def train(model: Network,
          train_loader: DataLoader,
          val_loader: DataLoader,
          config: Config,
          wandb_run: Union[Run, RunDisabled]) -> None:
    """ Training loop. Writes output to CSV file and logs to wandb if required.

    Parameters
    ----------
    model: Network
        Neural Network.
    train_loader: DataLoader
        DataLoader containing training data.
    val_loader: DataLoader
        DataLoader containing validation data.
    config: Config
        YAML config file.
    wandb_run: Union[Run, RunDisabled]
        Current training run on wandb if enabled.
    """

    # set Optimiser
    optimiser = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # set Cyclical Learning Rate scheduler
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer=optimiser,
        base_lr=0.0001,
        max_lr=0.001,
        step_size_up=10000,
        cycle_momentum=False
    )

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
            'val_mse_loss'
        ])

    print('Training...')

    # training loop
    for epoch in range(config.N_EPOCHS):

        # reset epoch loss
        train_mse_loss = 0.0
        val_mse_loss = 0.0

        # training step
        model.train()
        for feature_vector, label in train_loader:
            # def closure():
            #     if torch.is_grad_enabled():
            #         optimiser.zero_grad()
            #     output = model.forward(feature_vector.float())
            #     loss = loss_fn(output, label.unsqueeze(-1).float())
            #     if loss.requires_grad:
            #         loss.backward()
            #     return loss
            #
            # optimiser.step(closure)

            # move data to GPU if available
            feature_vector = feature_vector.to(DEVICE)
            label = label.to(DEVICE)

            batch_t_model_outputs = model.forward(feature_vector.float())
            batch_t_loss = loss_fn(batch_t_model_outputs, label.unsqueeze(-1).float())

            # increment epoch loss by batch loss
            # total batch loss = batch loss * number of values in batch
            # .item() returns the value in a 1 element tensor as a number
            train_mse_loss += batch_t_loss.item() * feature_vector.size(0)

            # update the parameters
            optimiser.zero_grad()
            batch_t_loss.backward()
            optimiser.step()
            scheduler.step()

        # validation step
        model.eval()
        for feature_vector, label in val_loader:
            # move data to GPU if available
            feature_vector = feature_vector.to(DEVICE)
            label = label.to(DEVICE)

            batch_v_model_outputs = model.forward(feature_vector.float())
            batch_v_loss = loss_fn(batch_v_model_outputs, label.unsqueeze(-1).float())

            # increment epoch loss by batch loss
            val_mse_loss += batch_v_loss.item() * feature_vector.size(0)

        # normalise loss by number of data points in DataLoader
        train_mse_loss /= len(train_loader.dataset)  # type: ignore
        val_mse_loss /= len(val_loader.dataset)  # type: ignore

        # print epoch results
        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, \n train_loss: {train_mse_loss:5f} \n val_loss: {val_mse_loss:5f} \n')

        # add losses to global trackers
        TRAIN_LOSSES.append(train_mse_loss)
        VAL_LOSSES.append(val_mse_loss)

        # log results in list to write to csv
        epoch_results = [epoch, train_mse_loss, val_mse_loss]

        # open results file in append mode and write results as csv
        with open(RESULTS_DIR / 'results.csv', 'a', newline='') as f_out:
            writer = csv.writer(f_out, delimiter=',')
            writer.writerow(epoch_results)

        # log results to wandb
        if isinstance(wandb_run, Run):
            results_dict = {'epoch': epoch, 'train_mse_loss': train_mse_loss, 'val_mse_loss': val_mse_loss}
            wandb_run.log(results_dict)

    print('Training complete.')


def main(args: argparse.Namespace) -> None:

    # toggle training
    # switch to false to load a saved model
    model_train = True

    # load config file
    train_config = Config()
    train_config.load_config(args.config_path)

    if model_train:
        # initialise weights and biases - will create a random name for the run
        run = wandb.init(
            config=train_config.config,
            project="cg-cfd with CLR",
            entity="hamzasardar"
        )

        # give local results the same name as the wandb run
        global RESULTS_DIR
        RESULTS_DIR = args.results_path / run.name

        # initialise network, load training data, and run training
        model: Network = Network([4, 10, 10, 10, 10, 1], activation_fn=nn.Tanh())
        dc_raw, df_raw = load_data(args.data_path, 't')
        fl = generate_features_labels(dc_raw, df_raw, 't', train_config)
        train_loader, val_loader = generate_dataloaders(fl, train_config)
        train(model,  train_loader, val_loader, config=train_config, wandb_run=run)

        # plotting loss history
        plots.plot_loss_history(
            fig_path=RESULTS_DIR / 'loss_history.png',
            n_epochs=train_config.N_EPOCHS,
            train_losses=TRAIN_LOSSES,
            val_losses=VAL_LOSSES,
            loss_fn='MSE Loss',
            experiment='Galilean Invariance - varying alpha'
        )

        # Evaluation on training data

        model.eval()

        model_predictions = model.forward(val_loader.dataset.dataset.tensors[0].float())
        labels = train_loader.dataset.dataset.tensors[1]

        np_model_predictions = model_predictions.detach().numpy()
        np_labels = labels.detach().numpy()

        # plotting model performance on training data
        plots.plot_model_evaluation(
            fig_path=RESULTS_DIR / f'model_evaluation_v.png',
            model_predictions=np_model_predictions,
            actual_error=np_labels,
            title='Model Evalation - training flows'
        )

        if train_config.EVALUATE:
            # Evaluation on evaluative flows
            dc_raw, df_raw = load_data(args.eval_path, 'e')
            fl = generate_features_labels(dc_raw, df_raw, 'e.png', train_config)
            train_loader, val_loader = generate_dataloaders(fl, train_config)

            model_predictions = model.forward(train_loader.dataset.dataset.tensors[0].float())
            labels = train_loader.dataset.dataset.tensors[1]

            np_model_predictions = model_predictions.detach().numpy()
            np_labels = labels.detach().numpy()

            # plotting model performance on training data
            plots.plot_model_evaluation(
                fig_path=RESULTS_DIR / f'model_evaluation_e.png',
                model_predictions=np_model_predictions,
                actual_error=np_labels,
                title='Model Evaluation - Interpolative Evaluation'
            )

        # save model locally
        torch.save(model.state_dict(), RESULTS_DIR / f'model_{run.name}.h5')

        # initialise model as wandb artifact
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(RESULTS_DIR / f'model_{run.name}.h5')

        # save model to wandb
        run.log_artifact(artifact)
        run.finish()

    else:
        # specify and load a previously saved model
        model_path = args.model_path
        model_name = args.model_name
        wandb_model = wandb.restore(name=model_name, run_path=model_path)
        model = torch.load(wandb_model)

        # initialise network, load training data, and run training
        dc_raw, df_raw = load_data(DATA_DIR, 't')
        fl = generate_features_labels(dc_raw, df_raw, 't')
        train_loader, val_loader = generate_dataloaders(fl)

        # pass data into loaded model
        # model_outputs = model.forward(feature_vector.float())
        # model_loss = loss_fn(_model_outputs, label.unsqueeze(-1).float())


if __name__ == '__main__':

    # pass cl arguments
    parser = argparse.ArgumentParser(description='PyCoarseNet: CG-CFD Error Prediction')

    # path arguments
    parser.add_argument('-dp', '--data-path', type=Path, required=True)
    parser.add_argument('-cp', '--config-path', type=Path, required=True)
    parser.add_argument('-rp', '--results-path', type=Path, required=True)
    parser.add_argument('-ep', '--eval-path', type=Path, required=False)

    parsed_args = parser.parse_args()

    main(parsed_args)
