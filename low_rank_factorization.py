import torch
import torch.nn as nn
import copy
import os
from data import get_data_loaders


def apply_low_rank_factorization(model, fraction=0.5):
    """Apply truncated SVD to the model's fc1 and fc2 layers."""
    m = copy.deepcopy(model)
    for name in ['fc1', 'fc2']:
        orig = getattr(m, name)
        W = orig.weight.data
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        k = max(1, int(min(W.shape) * fraction))
        # Truncate SVD components
        U_k = U[:, :k]
        S_k = S[:k]
        Vh_k = Vh[:k, :]
        # in_features -> k
        lin1 = nn.Linear(orig.in_features, k, bias=False)
        lin1.weight.data.copy_(Vh_k)
        # k -> out_features (preserve bias here)
        lin2 = nn.Linear(k, orig.out_features, bias=(orig.bias is not None))
        lin2.weight.data.copy_(U_k * S_k.unsqueeze(0))
        if orig.bias is not None:
            lin2.bias.data.copy_(orig.bias.data)
        setattr(m, name, nn.Sequential(lin1, lin2))
    return m
