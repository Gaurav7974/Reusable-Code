"""
A small, flexible MLP you can reuse for tabular or simple feature data.
Nothing fancy here — just a basic fully-connected network.
"""

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()

        # basic 2-layer MLP; tweak as needed
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
