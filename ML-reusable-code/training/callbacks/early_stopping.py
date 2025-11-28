"""
Early stopping callback.
Stops training when validation loss stops improving.
This prevents overfitting and saves time when the model stops learning.
"""

import numpy as np

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """
        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum improvement to count as progress.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.should_stop = False

    def on_train_begin(self):
        self.counter = 0
        self.best_loss = np.inf
        self.should_stop = False

    def on_epoch_end(self, epoch, train_loss, val_loss, model):
        # Early stopping only monitors val_loss
        if val_loss is None:
            return

        # If improvement is too small, count it as no improvement
        if val_loss > (self.best_loss - self.min_delta):
            self.counter += 1
        else:
            self.counter = 0
            self.best_loss = val_loss

        if self.counter >= self.patience:
            print("Early stopping triggered.")
            self.should_stop = True

    def on_train_end(self):
        pass
