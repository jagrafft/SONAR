"""
Shared utilities for the graph transfer task: seeding, CSV updates, cartesian product.
Used by conf.py and main.py.
"""
import torch
import numpy as np
import random
import pandas
import itertools


def set_seed(seed):
    """Set seed for torch, numpy, and random."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def update_csv(df, best_train_loss, best_val_loss, best_test_loss, best_epoch, model_params, exp_args, path):
    """Append one result row (losses, epoch, model_params, exp_args) to df and save CSV to path. Returns updated df."""
    tmp = {
        'train_loss': best_train_loss, 
        'val_loss': best_val_loss, 
        'test_loss': best_test_loss,
        'convergence time (epochs)': best_epoch
    }
    tmp.update({f'model_{k}':v for k,v in model_params.items()})
    tmp.update(exp_args)
    df.append(tmp)
    pandas.DataFrame(df).to_csv(path, index=False)
    return df


def cartesian_product(params):
    """
    Yield dicts from the cartesian product of param values.
    params: dict of key -> list of values. Yields one dict per combination.
    """
    keys = params.keys()
    vals = params.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))