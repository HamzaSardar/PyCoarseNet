import argparse
import csv
import sys
from pathlib import Path
from typing import List, Union

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, TensorDataset
from wandb.sdk.lib import RunDisabled
from wandb.wandb_run import Run

sys.path.append('..')
from pycoarsenet.model import Network
import pycoarsenet.data.preprocessing as preprocessing
from pycoarsenet.postprocessing import plots
from pycoarsenet.utils.config import Config

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_LOSSES: List[float] = []
VAL_LOSSES: List[float] = []


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
        base_lr=config.MIN_LEARNING_RATE,
        max_lr=config.LEARNING_RATE,
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

        # early stopping
        if val_mse_loss <= 0.00002:
            config.N_EPOCHS = epoch + 1
            break

    print('Training complete.')


def main(args: argparse.Namespace) -> None:
    # toggle profile plots
    compare_profiles = False

    # toggle training
    model_train = True

    # load config file
    train_config = Config()
    train_config.load_config(args.config_path)

    if model_train:
        # initialise weights and biases - will create a random name for the run
        run = wandb.init(
            config=train_config.config,
            project="CG-CFD",
            entity="hamzasardar"
        )

        # give local results the same name as the wandb run
        global RESULTS_DIR
        RESULTS_DIR = args.results_path / run.name

        # initialise network, load training data, and run training
        network: Network = Network(
            feature_size=4,
            output_size=1,
            num_hidden_layers=train_config.NUM_HIDDEN_LAYERS,
            num_neurons=train_config.NUM_NEURONS,
            activation_fn=nn.Tanh()
        )
        dc_raw, df_raw = preprocessing.load_data(args.data_path, 't', train_config)
        fl = preprocessing.generate_features_labels(dc_raw, df_raw, 't', train_config)
        train_loader, val_loader = preprocessing.generate_dataloaders(fl, train_config)
        train(network, train_loader, val_loader, config=train_config, wandb_run=run)

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

        network.eval()

        # below now accesses the right datasets
        # model_predictions = network.forward(fl[val_loader.dataset.indices, :-1].float())
        model_predictions = network.forward(fl[:, :-1].float())
        # labels = fl[val_loader.dataset.indices, -1]
        labels = fl[:, -1]

        np_model_predictions = model_predictions.detach().numpy()
        np_labels = labels.detach().numpy()

        # plotting model performance on training data
        plots.plot_model_evaluation(
            fig_path=RESULTS_DIR / f'model_evaluation_v.png',
            model_predictions=np_model_predictions,
            actual_error=np_labels,
            title='Model Evalation - Validation Data'
        )

        if train_config.EVALUATE:
            # Evaluation on evaluative flows
            dc_raw, df_raw = preprocessing.load_data(args.eval_path, 'e', train_config)
            fl = preprocessing.generate_features_labels(dc_raw, df_raw, 'e.png', train_config)
            train_loader, val_loader = preprocessing.generate_dataloaders(fl, train_config)

            # model_predictions = network.forward(fl[train_loader.dataset.indices, :-1].float())
            model_predictions = network.forward(fl[:, :-1].float())
            # labels = fl[train_loader.dataset.indices, -1]
            labels = fl[:, -1]

            np_model_predictions = model_predictions.detach().numpy()
            np_labels = labels.detach().numpy()

            # plotting model performance on training data
            plots.plot_model_evaluation(
                fig_path=RESULTS_DIR / f'model_evaluation_e.png',
                model_predictions=np_model_predictions,
                actual_error=np_labels,
                title='Model Evaluation - Evaluation of Invariance'
            )

        # save model locally
        torch.save(network.model.state_dict(), RESULTS_DIR / f'model_{run.name}.h5')

        # initialise model as wandb artifact
        artifact = wandb.Artifact('network', type='model')
        artifact.add_file(RESULTS_DIR / f'model_{run.name}.h5')

        # save model to wandb
        run.log_artifact(artifact)
        run.finish()

    else:
        pass

    if compare_profiles:
        c_pt = torch.load('/Users/user/Data/changing_alpha/evaluation_flows/0.0005_data_coarse.t')
        f_pt = torch.load('/Users/user/Data/changing_alpha/evaluation_flows/0.0005_data_fine.t')
        fl = preprocessing.generate_features_labels(c_pt, f_pt, 't', train_config)

        best_model = nn.Sequential(
            nn.Linear(4, 59),
            nn.Tanh(),
            nn.Linear(59, 59),
            nn.Tanh(),
            nn.Linear(59, 59),
            nn.Tanh(),
            nn.Linear(59, 59),
            nn.Tanh(),
            nn.Linear(59, 59),
            nn.Tanh(),
            nn.Linear(59, 59),
            nn.Tanh(),
            nn.Linear(59, 59),
            nn.Tanh(),
            nn.Linear(59, 1)
        )

        best_model.load_state_dict(
            torch.load(
                '/Users/user/Projects/PyCoarseNet/results/CLR/report/morning-tree-169/model_morning-tree-169.h5')
        )

        # run inference and correct the coarse data
        disc_e = best_model.forward(fl[:, :-1].float())
        corrected_c = c_pt[:, 3].unsqueeze(-1) + disc_e

        # use mask to extract data along line x=0.525
        mask1 = c_pt[:, 0] == 0.525
        mask_indices1 = torch.nonzero(mask1)
        corrected_c_sampled525 = corrected_c[mask_indices1].squeeze()
        torch.save(corrected_c_sampled525, '/Users/user/Data/corrected_c_525.t')

        # use mask to extract data along line x=0.475
        mask2 = c_pt[:, 0] == 0.475
        mask_indices2 = torch.nonzero(mask2)
        corrected_c_sampled475 = corrected_c[mask_indices2].squeeze()
        torch.save(corrected_c_sampled475, '/Users/user/Data/corrected_c_475.t')


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
