from pathlib import Path
from typing import Any, Dict

import yaml
from dataclasses import dataclass, field, fields

from .enums import eCorruption, eDecoder, eSolverFunction
from .exceptions import SolverConsistencyError, SolverConsistencyWarning


@dataclass
class Config:
    _config: Dict[str, Any] = field(default_factory=dict, repr=False)

    # simulation parameters
    COARSE_SPACING: float = field(init=False) 
    FINE_SIZE: int = field(init=False) 100
    COARSE_SIZE: int = field(init=False) 20

    # training parameters
    N_EPOCHS: int = field(init=False) 100
    LEARNING_RATE: float = field(init=False) 0.001
    EVALUATE: bool = field(init=False) True
    INVARIANCE: bool = field(init=False) True
    TENSOR_INVARIANTS: bool = field(init=False) True
    TRAINING_FRACTION: float = field(init=False) 0.8
    BATCH_SIZE: int = field(init=False) 1
 
    def load_config(self, config_path: Path) -> None:

        """Load values from .yml file into class attributes."""

        # load yaml file to dictionary
        with open(config_path, 'r') as f:
            tmp_config = yaml.load(stream=f, Loader=yaml.CLoader)

        # assuming values are dictionaries
        for v in tmp_config.values():
            self._config.update(v)

        # generate a set of existing field names
        field_names = set(map(lambda x: x.name, fields(self)))

        # remove any private fields
        _private_fields = [f for f in field_names if f.startswith('_')]







