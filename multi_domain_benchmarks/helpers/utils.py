"""
Utilities: cartesian_product, optimizer_to, set_seed, get_split_mask, join_dataset_splits.
Used by experiments, model_selection, and dataset code.
"""
import torch
import random
import numpy as np
import os
from torch_geometric.data import Data
from typing import Any, List, Tuple, Union
from torch import Tensor
import itertools


def cartesian_product(params):
    """Yield dicts from the cartesian product of param values (key -> list of values)."""
    keys = params.keys()
    vals = params.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def optimizer_to(optim, device):
    """Move optimizer state tensors to device (e.g. after loading checkpoint)."""
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def set_seed(seed):
    """Set random seed for Python, numpy, torch, CUDA; disable cudnn non-determinism."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False


def get_split_mask(data: Data, batch_size: int, split_mask_name: str) -> Tuple[Tensor, Tensor]:
    """Return (batch_mask, node_mask) for the given split (e.g. train_mask)."""
    if hasattr(data, split_mask_name):
        return getattr(data, split_mask_name), getattr(data, split_mask_name)
    else:
        return torch.ones(size=(batch_size,), dtype=torch.bool), torch.ones(size=(data.x.shape[0],), dtype=torch.bool)


'''
Adapted from https://github.com/hamed1375/Exphormer.git
'''


def join_dataset_splits(datasets: List) -> Any:
    """Join train, val, test datasets into one dataset object.

    Args:
        datasets: list of 3 PyG datasets to merge

    Returns:
        joint dataset with `split_idxs` property storing the split indices
    """
    assert len(datasets) == 3, "Expecting train, val, test datasets"

    n1, n2, n3 = len(datasets[0]), len(datasets[1]), len(datasets[2])
    data_list = [datasets[0].get(i) for i in range(n1)] + \
                [datasets[1].get(i) for i in range(n2)] + \
                [datasets[2].get(i) for i in range(n3)]

    datasets[0]._indices = None
    datasets[0]._data_list = data_list
    datasets[0].data, datasets[0].slices = datasets[0].collate(data_list)
    split_idxs = [list(range(n1)),
                  list(range(n1, n1 + n2)),
                  list(range(n1 + n2, n1 + n2 + n3))]
    datasets[0].split_idxs = split_idxs

    return datasets[0]
