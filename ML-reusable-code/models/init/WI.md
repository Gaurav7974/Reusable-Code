# Weight Initialization Helpers

Small helpers for common initialization schemes.  
Useful when you want to control how your model starts training.

### Available functions
- `xavier(model)` — good for many feedforward networks  
- `kaiming_init(model)` — often best for ReLU networks  
- `normal_init(model, std)` — simple normal distribution init

### Why use custom initialization?
- Can help early training stability  
- Avoids dead neurons  
- Often speeds up convergence

### Example
```python
model = SimpleMLP(32, 64, 1)
xavier(model)  # apply xavier init
```
Good reads
Xavier & He initialization explained simply:
https://cs231n.github.io/neural-networks-2/