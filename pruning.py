import torch
import torch.nn as nn
import copy
from model import CIFAR10Model
from data import get_data_loaders
from utils import print_model_stats, evaluate_model, count_parameters
import os
import torch.nn.utils.prune as prune

def apply_pruning(model, amount=0.2, min_channels=32):
    """
    Applies structured L2-norm pruning (n=2) to large Conv2d and Linear layers.
    Only prunes layers with enough output channels/neurons (min_channels).
    """
    m = copy.deepcopy(model)
    for name, module in m.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Get output dimension (dim=0 for output channels/neurons)
            out_dim = module.weight.shape[0]
            if out_dim >= min_channels:
                try:
                    prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
                    prune.remove(module, 'weight')
                    print(f"Pruned {name}: {amount*100:.0f}% of {out_dim} outputs")
                except Exception as e:
                    print(f"Skipping {name} due to error: {e}")
    return m