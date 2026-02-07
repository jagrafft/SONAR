"""
Config generators and CONFIGS registry for GraphPropPred.
cartesian_product yields hyperparameter combinations; config_BlockSONAR_GraphProp
yields BlockSONAR configs. CONFIGS maps model name to (config_fun, model_class).
"""
import itertools
from models import *


def cartesian_product(params):
    """Yield dicts from the cartesian product of param values (key -> list of values)."""
    keys = params.keys()
    vals = params.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def config_BlockSONAR_GraphProp(num_features, num_classes, task):
    """Yield full config dicts (model + optim) for BlockSONAR over the hyperparameter grid."""
    grid = {
        'hidden_dim': [30], #20, 10
        'num_iters': [20, 10, 5],
        'epsilon': [1., 1e-1, 1e-3],
        'normalization': [None, 'sym', 'rw'], 
        'use_dissipation': [True, False],
        'use_forcing': [True, False],
        'fix_resistance': [True, False],
        'num_blocks': [1, 2]
    }
    
    # Iterate through the cartesian product of the grid
    for conf in cartesian_product(grid):
        fixed_hyperparams =  {
            'model': {
                'input_dim': num_features,
                'output_dim': num_classes,
                'edge_dim': 0, # there is no edge information in the GPP tasks
                'activ_fun': 'Tanh',
                #'train_weights': True, 
                'bias': True,
            },
            'optim': {
                'lr': 0.003,
                'weight_decay': 1e-6
            }
        }
        
        fixed_hyperparams['model'].update(conf)
        yield fixed_hyperparams


#sonar_ = lambda num_features, num_classes, task: config_SONAR_GraphProp(num_features, num_classes, task)
blocksonar_ = lambda num_features, num_classes, task: config_BlockSONAR_GraphProp(num_features, num_classes, task)

CONFIGS = {
    #'SONAR': (sonar_, SONAR),
    'BlockSONAR': (blocksonar_, BlockSONAR),
}


