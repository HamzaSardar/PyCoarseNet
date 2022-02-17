import torch
import einops

from typing import List, Dict, Tuple, NoReturn


class InitialiseData:
    def __init__(self, data_dir: str, num_sims: int, list_var: List[float], cg_spacing: float, coarse_size: int,
                 fine_size: int, *args: object, **kwargs: object) -> None:

        """ Preprocessing class to take the data output from TorchFoam and organise it into required format for
        Corse Grid disretisation error prediction. Assumes results are varying one parameter while others remain fixed.

        Parameters
        ----------
        data_dir:
            File path to the folder containing data output from TorchFoam.
        num_sims:
            Number of simulations in the dataset.
        list_var:
            List of values the varied parameter from the experiments takes.
        cg_spacing:
            Distance between cell centres in coarse grid.
        coarse_size:
            Number of cells along one side of a square coarse grid.
        fine_size:
            Number of cells along one side of a square fine grid.
        args: str
            The letters denoting any dimensionless groups required in the data, as comma-separated strings.
            Example: 'Pe' for Peclet Number.

        Returns
        -------
        object
        """
        self.data_dir = data_dir
        self.num_sims = num_sims
        self.list_var = list_var
        self.cg_spacing = cg_spacing
        self.coarse_size = coarse_size
        self.fine_size = fine_size
        self.data_coarse = None
        self.data_fine = None
        # self.features = None
        self.targets = None
        self.means = []
        self.stds = []
        self.num_features = None
        self.train_indices = None
        self.val_indices = None
        self.train_indices = None
        self.val_indices = None
        data_coarse_dict = {}
        data_fine_dict = {}

        # Use num_sims to load the relevant number of torch Tensors from the data_dir
        # or use list_var, depending on tensor naming convention
        # following code for varying thermal diffusivity:

        if 'Pe' in kwargs.values():
            cell_Pe_dict: Dict[float, torch.Tensor] = {}
            for i in list_var:
                data_coarse_dict[i] = torch.load(self.data_dir + f'{i}_data_coarse.t')
                data_fine_dict[i] = torch.load(self.data_dir + f'{i}_data_fine.t')

            for key in data_coarse_dict.keys():
                cell_Pe_dict[key] = (cg_spacing * torch.ones(data_coarse_dict[key].shape[0])) / key

            data_coarse_no_Pe = torch.cat(([tensor for tensor in data_coarse_dict.values()]), dim=0)
            cell_Pe = torch.cat([tensor for tensor in cell_Pe_dict.values()], dim=0).unsqueeze(-1)
            # Cell_Pe is final column of data_coarse
            data_coarse_raw = torch.cat((data_coarse_no_Pe, cell_Pe), dim=-1)
            data_fine_raw = torch.cat(([tensor for tensor in data_fine_dict.values()]), dim=0)

            self.data_coarse = einops.rearrange(
                data_coarse_raw,
                f'(num_sims row column) variable ->num_sims variable row column',
                num_sims=self.num_sims, row=self.coarse_size)
            # fine grid data needs to be downsampled after this
            data_fine = einops.rearrange(
                data_fine_raw,
                f'(num_sims row column) variable -> num_sims variable row column',
                num_sims=self.num_sims, row=self.fine_size)

            self.data_fine = self.downsampling(self.coarse_size, self.fine_size, data_fine)

            # getting and reshaping features and targets
            features = self._extract_features(self.data_coarse, indices=(3, 4, 5, 7, 8, 10, 11, 16))
            targets = self._extract_features(self.data_fine, indices=(3, 4, 5, 7, 8, 10, 11))

            delta_var = self._extract_targets(features[:, 0], targets[:, 0])

            self.features: torch.Tensor = einops.rearrange(features,
                                             'simulation variable row column -> (simulation row column) variable',
                                             variable=8)[:, 1:]
            self.targets = einops.rearrange(delta_var,
                                            'simulation row column -> (simulation row column)').unsqueeze(-1)
            self.num_features = int(self.features.shape[-1])
            self.normalise()
            self.train_indices, self.val_indices = self.training_val_split()

    @staticmethod
    def downsampling(coarse_size, fine_size, data_fine) -> torch.Tensor:
        """Downsampling fine data points by overlapping cell centres with coarse mesh.
           For this method, the ratio of fine grid to coarse grid data points must be an odd integer, >1.
           This method assumes a 2D domain.

        Returns
        -------
        torch.Tensor containing downsampled fine grid data.
        """

        downsampling_ratio = int(fine_size / coarse_size)

        if isinstance(downsampling_ratio, int) and downsampling_ratio % 2 == 1:
            sampling_indices = torch.linspace(int(int(downsampling_ratio / 2) + 1),
                                              fine_size - int(downsampling_ratio / 2),
                                              steps=coarse_size)
            #print('indices:', sampling_indices)
            sampling_bools = torch.zeros((fine_size, fine_size), dtype=torch.bool)
            for i in range(1, fine_size + 1):
                for j in range(1, fine_size + 1):
                    if i in sampling_indices and j in sampling_indices:
                        sampling_bools[i - 1, j - 1] = True
            #print(sampling_bools)
        else:
            raise ValueError('Fine and Coarse grids incompatible with selected downsampling.')

        return einops.rearrange(data_fine[:, :, sampling_bools],
                                'simulation variable (row column) -> simulation variable row column',
                                row=coarse_size)

    @staticmethod
    def _extract_features(data: torch.Tensor, indices: Tuple) -> torch.Tensor:
        if indices is None:
            indices = input("Please input the required indices:")
        return data[:, indices, :, :]

    @staticmethod
    def _extract_targets(coarse_data: torch.Tensor, fine_data: torch.Tensor) -> torch.Tensor:

        return fine_data - coarse_data

    def normalise(self): # -> torch.Tensor:
        for i in range(self.num_features):
            mean = torch.mean(self.features[:, i])
            self.means.append(mean)
            std = torch.std(self.features[:, i])
            self.stds.append(std)
            self.features[:, i] = (self.features[:, i] - mean) / std

    def training_val_split(self):
        n_samples = torch.numel(self.features[:, 0])
        n_val = int(0.2 * n_samples)

        shuffled_indices = torch.randperm(n_samples)
        train_indices = shuffled_indices[:-n_val]
        val_indices = shuffled_indices[-n_val:]

        return train_indices, val_indices
