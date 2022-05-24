from pathlib import Path
from typing import Any, Dict

import yaml
from dataclasses import dataclass, field, fields


@dataclass
class Config:
    _config: Dict[str, Any] = field(default_factory=dict, repr=False)

    # simulation parameters
    COARSE_SPACING: float = field(init=False)
    FINE_SIZE: int = field(init=False)
    COARSE_SIZE: int = field(init=False)

    # network parameters
    NUM_HIDDEN_LAYERS: int = field(init=False)
    NUM_NEURONS: int = field(init=False)

    # training parameters
    N_EPOCHS: int = field(init=False)
    LEARNING_RATE: float = field(init=False)
    MIN_LEARNING_RATE: float = field(init=False)
    EVALUATE: bool = field(init=False)
    INVARIANCE: bool = field(init=False)
    TENSOR_INVARIANTS: bool = field(init=False)
    TRAINING_FRACTION: float = field(init=False)
    BATCH_SIZE: int = field(init=False)
 
    def load_config(self, config_path: Path) -> None:

        """Load values from .yml file into class attributes."""

        # load yaml file to dictionary
        with open(config_path, 'r') as f:
            tmp_config = yaml.load(stream=f, Loader=yaml.Loader)

        # assuming values are dictionaries
        for v in tmp_config.values():
            self._config.update(v)

        # generate a set of existing field names
        field_names = set(map(lambda x: x.name, fields(self)))

        # remove any private fields
        _private_fields = [f for f in field_names if f.startswith('_')]
        field_names.difference_update(_private_fields)

        # update class attributes
        for k, v in self._config.items():

            if k not in field_names:
                raise ValueError(f'Invalid Field: {k} with value {v}')

            k_field = next(filter(lambda x: x.name == k, fields(self)))
            setattr(self, k, k_field.type(v))

            field_names.remove(k)

        if len(field_names) > 0:

            msg = 'Missing values in config file: \n'
            for fname in field_names:
                _field = next(filter(lambda x: x.name == fname, fields(self)))
                msg += f'{_field.name}: {_field.type.__name__}\n'

            raise ValueError(msg)

    @property
    def config(self) -> Dict[str, Any]:
        return self._config
