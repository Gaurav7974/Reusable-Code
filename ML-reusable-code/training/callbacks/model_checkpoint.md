# ModelCheckpoint - Code Explanation

## What It Does
Automatically saves your model to a file whenever validation loss improves.

---

## Code Breakdown

### 1. `__init__` - Setup
```python
def __init__(self, filepath: str = "best_model.pt"):
    self.filepath = filepath
    self.best_loss = np.inf
```

- `filepath`: Where to save the model file
- `best_loss`: Initialized to infinity so first epoch always saves

---

### 2. `on_train_begin` - Reset
```python
def on_train_begin(self):
    self.best_loss = np.inf
```

Resets `best_loss` to infinity at training start.

---

### 3. `on_epoch_end` - Main Logic
```python
def on_epoch_end(self, epoch, train_loss, val_loss, model):
    if val_loss is None:
        return
```
If no validation loss, exit early.

```python
    if val_loss < self.best_loss:
        self.best_loss = val_loss
        torch.save(model.state_dict(), self.filepath)
        print(f"Model improved at epoch {epoch+1}, saved to {self.filepath}")
```

If current validation loss is better (lower) than best:
1. Update `best_loss`
2. Save model weights to file using `torch.save()`
3. Print confirmation message

---

### 4. `on_train_end` - Cleanup
```python
def on_train_end(self):
    pass
```

Does nothing currently. Placeholder for future cleanup code.

---

## How It Works

```
Epoch 1: val_loss = 0.8 → 0.8 < inf → Save model (best_loss = 0.8)
Epoch 2: val_loss = 0.6 → 0.6 < 0.8 → Save model (best_loss = 0.6)  
Epoch 3: val_loss = 0.7 → 0.7 > 0.6 → Don't save
Epoch 4: val_loss = 0.5 → 0.5 < 0.6 → Save model (best_loss = 0.5)
```

Final file contains the model from Epoch 4 with val_loss = 0.5.

---

## Key Points

- Only saves when validation loss **improves** (gets lower)
- Overwrites the file each time it saves
- `state_dict()` saves only the model weights, not the architecture
- Helps prevent keeping an overfitted model