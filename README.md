# ML Reusable Code Library

> A production-ready collection of reusable machine learning utilities for data processing, model training, and evaluation.

---

## Table of Contents

- [Overview](#overview)
- [Current Structure](#current-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [Design Principles](#design-principles)
- [Coding Standards](#coding-standards)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

This repository provides **modular, well-documented ML components** to eliminate repetitive coding and accelerate development workflows.

**Goal:** Build a standardized library of reusable utilities that work across multiple ML projects.

---

## Current Structure

```
ml-reusable-code/
│
├── data/                   # Data operations
│   ├── loaders/           # Dataset loaders
│   ├── preprocessing/     # Data cleaning & transformation
│   ├── augmentation/      # Data augmentation
│   └── splits/            # Train/val/test splitting
│
├── training/              # Training utilities
│   ├── loops/            # Training loops
│   ├── callbacks/        # Early stopping, logging
│   ├── checkpoints/      # Model checkpointing
│   └── schedulers/       # Learning rate schedulers
│
├── tests/                # Unit tests
└── README.md
```

---

## Features

### Data Module
* **Dataset Loaders** - CSV, images, and custom formats
* **Preprocessing** - Cleaning, normalization, standardization
* **Augmentation** - Image and data augmentation functions
* **Splitting** - Train/validation/test split utilities

### Training Module
* **Training Loops** - PyTorch/TensorFlow compatible loops
* **Callbacks** - Early stopping, logging, custom callbacks
* **Checkpointing** - Save/load model states
* **Schedulers** - Learning rate scheduling wrappers

---

## Installation

**Clone the repository:**

```bash
git clone https://github.com/yourusername/ml-reusable-code.git
cd ml-reusable-code
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## Usage Examples

### Load a CSV Dataset

```python
from data.loaders.load_csv_dataset import load_csv_dataset

# Load and clean CSV data
df = load_csv_dataset("dataset.csv", dropna=True)
print(df.head())
```

### Split Data

```python
from data.splits.train_val_test_split import split_data

# Split data into train/val/test
train, val, test = split_data(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
```

### Training Loop

```python
from training.loops.pytorch_train_loop import train_model

# Train a PyTorch model
model = train_model(
    model=my_model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    device="cuda"
)
```

---

## Design Principles

| Principle | Description |
|-----------|-------------|
| **Modular** | Each utility serves a single, clear purpose |
| **Framework-Agnostic** | Minimal dependencies, works with PyTorch/TensorFlow |
| **Well-Documented** | Type hints and docstrings for all functions |
| **Production-Ready** | Tested, reliable code for real-world projects |

---

## Coding Standards

* **PEP8 compliance** - Clean, readable code
* **Type hints required** - All function signatures typed
* **Docstrings** - Google/Numpydoc format
* **No hardcoded paths** - All paths parameterized
* **Framework-agnostic** - Avoid unnecessary lock-in

---

## Contributing

**Before contributing, ensure your code:**

1. Is generic and reusable across multiple projects
2. Includes complete type hints and docstrings
3. Has unit tests in `/tests` directory
4. Includes usage examples

**Example contribution:**

```python
from typing import List
import pandas as pd

def load_csv_dataset(filepath: str, dropna: bool = False) -> pd.DataFrame:
    """
    Load a CSV dataset with optional preprocessing.
    
    Args:
        filepath: Path to CSV file
        dropna: Whether to drop rows with missing values
        
    Returns:
        Loaded DataFrame
        
    Example:
        >>> df = load_csv_dataset("data.csv", dropna=True)
    """
    df = pd.read_csv(filepath)
    if dropna:
        df = df.dropna()
    return df
```

---

## Roadmap

- [x] Data loading utilities
- [x] Training loop foundations
- [ ] Evaluation metrics and visualization
- [ ] Model architectures library
- [ ] Experiment configuration management
- [ ] Comprehensive test coverage (>80%)
- [ ] CI/CD pipeline setup

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## Version

**Current Version:** `v0.1.0`  
**Last Updated:** November 2025

---

<div align="center">

**[Star this repo](https://github.com/yourusername/ml-reusable-code)** if you find it useful!

Made with care for the ML community

</div>
