import yaml
import random
import numpy as np


# base_config
config = {
    'SIMULATION': {
        'COARSE_SPACING': 0.05,
        'FINE_SIZE': 100,
        'COARSE_SIZE': 20
    },
    'NETWORK': {
        'NUM_HIDDEN_LAYERS': 4,
        'NUM_NEURONS': 10
    },
    'TRAINING': {
        'N_EPOCHS': 500,
        'LEARNING_RATE': 0.001,
        'MIN_LEARNING_RATE': 0.0001,
        'EVALUATE': True,
        'INVARIANCE': True,
        'TENSOR_INVARIANTS': True,
        'TRAINING_FRACTION': 0.8,
        'BATCH_SIZE': 1
    }
}


if __name__ == '__main__':

    # generate n random config files
    for i in range(24):
        # take random samples
        log2_batch_size = random.sample(range(0, 5), 1)
        log2_num_hidden_layers = random.sample(range(0, 4), 1)
        log10_num_neurons = np.random.uniform(np.log10(10), np.log10(100), 1)

        # update values in base config
        config['TRAINING']['BATCH_SIZE'] = int(2 ** log2_batch_size[0])
        config['NETWORK']['NUM_HIDDEN_LAYERS'] = int(2 ** log2_num_hidden_layers[0])
        config['NETWORK']['NUM_NEURONS'] = int(10 ** log10_num_neurons[0])

        # write to a new config file
        with open(f'./config/config_{i}.yml', 'w') as config_file:
            yaml.dump(config, config_file, default_flow_style=False)

        print(f'config file {i} generated.')
