"""
A few common weight initialization helpers.
Useful when you want more control over how your model starts training.
Weight init helpers — small stuff I got tired of rewriting.
"""


import torch
import torch.nn as nn


def xavier(model: nn.Module):
    # xavier for all linear layers
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def kaiming_init(model: nn.Module):
    # kaiming is usually good for relu networks
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def normal_init(model: nn.Module, std: float = 0.02):
    # quick normal init for small models
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, mean=0.0, std=std)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
