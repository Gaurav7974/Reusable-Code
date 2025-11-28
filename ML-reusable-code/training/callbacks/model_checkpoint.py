"""
ModelCheckpoint callback:
Saves the model whenever validation loss improves.
"""

import torch
import numpy as np


class ModelCheckpoint:
    def __init__(self, filepath: str = "best_model.pt"):
        self.filepath = filepath
        self.best_loss = np.inf

    def on_train_begin(self):
        self.best_loss = np.inf

    def on_epoch_end(self, epoch, train_loss, val_loss, model):
        if val_loss is None:
            return

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.filepath)
            print(f"Model improved at epoch {epoch+1}, saved to {self.filepath}")

    def on_train_end(self):
        pass
