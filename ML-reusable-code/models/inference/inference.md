# InferenceRunner

A small wrapper to make inference cleaner.  
Saves you from repeating the same `no_grad()` block and device handling.

### What it handles
- Moves input to the right device
- Converts lists to tensors automatically
- Runs `.eval()` mode
- Wraps inference inside `torch.no_grad()`

### When to use it
Anytime you want a simple `predict()` interface for a PyTorch model.

### Example
```python
runner = InferenceRunner(model, device="cuda")
preds = runner.predict(data)
```
Read this also
https://pytorch.org/docs/stable/notes/autograd.html#inference-mode