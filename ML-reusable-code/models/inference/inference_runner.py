"""
A small wrapper for running inference consistently.
Keeps evaluation code cleaner and avoids repeated no_grad blocks.
"""

import torch
from typing import Union


class InferenceRunner:
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device

    def predict(self, x: Union[torch.Tensor, list]):
        # convert lists to Tensors automatically if needed
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        x = x.to(self.device)

        # basic inference pass
        self.model.eval()
        with torch.no_grad():
            out = self.model(x)

        return out.cpu()
