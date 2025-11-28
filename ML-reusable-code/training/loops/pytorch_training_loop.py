"""
Generic PyTorch training loop.
Designed to be a clean, reusable template across multiple ML projects.

This loop handles:
- forward + backward pass
- optimizer stepping
- validation pass
- metric tracking
- callback hooks (early stopping, checkpointing, etc.)
"""

import torch
from typing import Callable, Optional, Dict, Any


def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: str = "cpu",
    callbacks: Optional[list] = None,
) -> Dict[str, Any]:
    """
    A reusable PyTorch training loop.

    Args:
        model: Your PyTorch model.
        train_loader: Dataloader for training data.
        val_loader: Optional dataloader for validation.
        criterion: Loss function.
        optimizer: Optimizer for training.
        num_epochs: Total number of epochs to train.
        device: CPU/GPU.
        callbacks: List of callback objects.

    Returns:
        Dictionary containing loss history and metrics.
    """

    # Move model to device once
    model = model.to(device)

    history = {"train_loss": [], "val_loss": []}

    if callbacks:
        for cb in callbacks:
            cb.on_train_begin()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            # Expecting batch = (inputs, labels)
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # Validation step
        avg_val_loss = None
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            history["val_loss"].append(avg_val_loss)

        # Run callbacks per epoch
        if callbacks:
            for cb in callbacks:
                cb.on_epoch_end(epoch, avg_train_loss, avg_val_loss, model)

        print(
            f"Epoch {epoch+1}/{num_epochs} "
            f"- Train Loss: {avg_train_loss:.4f} "
            f"- Val Loss: {avg_val_loss:.4f}"
        )

    if callbacks:
        for cb in callbacks:
            cb.on_train_end()

    return history
