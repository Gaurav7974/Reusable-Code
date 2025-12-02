# SimpleMLP

A small MLP you can reuse for tabular data or any basic feature input.  
It's a straightforward 2-layer network — nothing fancy, but enough for quick experiments.

### What it does
- Takes an input feature vector  
- Runs it through a hidden layer + ReLU  
- Outputs a prediction (regression or classification)

### When to use it
- Baseline models  
- Quick prototypes  
- Cases where a full deep model is unnecessary

### Quick example
```python
model = SimpleMLP(input_dim=32, hidden_dim=64, output_dim=1)
output = model(x)
```
Read this for better understanding
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

