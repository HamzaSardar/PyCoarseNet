import torch
import torch.nn as nn

from pycoarsenet.data.initialise_data import InitialiseData
from pycoarsenet.training.model_setup import Network

if __name__ == '__main__':
    dataset = InitialiseData(data_dir='/home/hamza/Projects/TorchFoam/Data/changing_alpha/',
                             num_sims=4,
                             list_var=[0.001, 0.005, 0.01, 0.05],
                             cg_spacing=0.05,
                             coarse_size=20,
                             fine_size=100,
                             Dimensionless='Pe')

