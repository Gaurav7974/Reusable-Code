# ML Reusable Code Library

A high-quality, production-ready collection of **reusable machine learning utilities** designed to standardize and accelerate real-world ML development.  
This repository contains modular, documented, framework-agnostic components for data processing, model training, evaluation, and experimentation.

The goal is simple:  
**Eliminate repetitive coding and provide stable, clean, reusable ML components that can be dropped into any project.**

---

## Table of Contents
- [Overview](#overview)
- [Scope](#scope)
- [Folder Structure](#folder-structure)
- [Modules](#modules)
  - [Data](#data)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Models](#models)
  - [Utils](#utils)
- [Design Principles](#design-principles)
- [Coding Standards](#coding-standards)
- [Usage Examples](#usage-examples)
- [Contributing Guidelines](#contributing-guidelines)
- [Versioning](#versioning)
- [License](#license)

---

## Overview

Modern ML workflows repeatedly use the same fundamental tasks:
- dataset loading  
- cleaning and preprocessing  
- splitting and augmentation  
- training loops  
- metrics and evaluation  
- model saving, inference, and utilities  
- reproducibility and configuration  

This repository centralizes these components into **a unified, reliable, consistent codebase** so you never rewrite the same functionality again.

Everything in this repo:
- is **modular**
- is **fully documented**
- uses **type hints**
- avoids framework lock-in whenever possible
- follows strict quality standards

---

## Scope

This repository includes reusable code for:

### 1. Data Handling
- dataset loaders  
- preprocessing utilities  
- normalization/standardization  
- augmentation functions  
- train/val/test splits  

### 2. Training Utilities
- PyTorch/TensorFlow-compatible training loops  
- early stopping  
- checkpoint managers  
- LR scheduling wrappers  
- logging helpers  

### 3. Evaluation Tools
- classification metrics  
- regression metrics  
- visualization utilities (confusion matrix, ROC, etc.)  

### 4. Model Utilities
- model architecture templates  
- initialization helpers  
- inference wrappers  
- save/load utilities  

### 5. Experiment Utilities
- config loaders  
- seed setters  
- experiment folder management  

Anything outside these categories does **not** belong here.

---

## Folder Structure

```plaintext
ml-reusable-code/
│
├── data/
│   ├── loaders/
│   ├── preprocessing/
│   ├── augmentation/
│   └── splits/
│
├── training/
│   ├── loops/
│   ├── callbacks/
│   ├── schedulers/
│   └── checkpoints/
│
├── evaluation/
│   ├── classification/
│   ├── regression/
│   └── visualization/
│
├── models/
│   ├── architectures/
│   ├── init/
│   └── inference/
│
├── utils/
│   ├── config/
│   └── seeds/
│
├── tests/
└── docs/
```
## Folder Rules

Each folder contains:
- clean utilities  
- example usage files  
- individual README files  
- strict file naming rules (snake_case, no duplicates, one responsibility per file)

---

## Modules

### Data
Reusable utilities for:
- CSV loaders  
- data cleaning  
- normalization & standardization  
- missing value handling  
- outlier removal  
- image augmentation  
- robust train/validation/test splitting  

---

### Training
Reusable code for:
- PyTorch training loops  
- callbacks (early stopping, logging, checkpointing)  
- checkpoint saving/loading  
- LR scheduler wrappers  

---

### Evaluation
Contains tools for:
- classification metrics  
- regression metrics  
- visualization (confusion matrix, ROC, distributions)  
- reusable evaluation pipelines  

---

### Models
Includes:
- model architectures  
- weight initialization schemes  
- inference wrappers  

---

### Utils
Utility helpers for:
- reproducibility (seed setting)  
- config handling  
- experiment directory utilities (auto folder creation, logging, etc.)  

---

## Design Principles

### Reuse > Rewrite
Code must be generic and reusable across multiple projects.

### Framework-Agnostic
Avoid unnecessary lock-in to PyTorch/TensorFlow.  
Use core Python, NumPy, and lightweight wrappers whenever possible.

### Strict Modularity
**One file = one responsibility.**  
No merging unrelated utilities.

### Documentation First
Every utility must include:
- type hints  
- proper docstring (Google/Numpydoc format)  
- a minimal example  
- a folder-level README  

### Consistency
Naming, folder structure, API design, and coding style must remain uniform across the entire repository.

---

## Coding Standards

### Python Rules
- PEP8 compliance  
- Type hints required on all functions  
- Google/Numpydoc docstrings  
- No dead code or commented-out code  
- Functions must be deterministic unless explicitly stated  

### Repository Rules
- No experimental or incomplete code  
- No dataset/model-specific logic  
- No hardcoded paths  
- No large Jupyter notebooks saved to the repo  

---

## Testing

All critical utilities must include minimal unit tests located in the `/tests` directory.  
Tests should validate:
- expected output shape  
- determinism  
- correct error handling  
- reproducibility where applicable  

---

## Usage Examples

### Load a CSV
```python
from data.loaders.load_csv_dataset import load_csv_dataset

df = load_csv_dataset("dataset.csv", dropna=True)
```
## Contributing Guidelines

Before adding **ANY** utility, you must verify the following:

1. The utility is reused across **3 or more** ML projects.  
2. It is fully **generic** and not tied to any dataset, model architecture, or experiment.  
3. It includes the following requirements:
   - complete docstring  
   - full type hints  
   - a minimal usage example  
   - tests in the `/tests` directory  
   - a folder-level README documenting the module  

Pull requests that do **not** meet these criteria will be rejected immediately.

---

## Versioning

This project follows **semantic versioning (SemVer)**:

- **v0.1.0** — initial stable release  
- **v0.2.x** — additional utilities, minor improvements, no breaking changes  
- **v1.x.x** — major release with guaranteed backward compatibility and a mature API  

Every update must be recorded in the `CHANGELOG.md` file.

---

## License

The recommended license for this repository is the **MIT License**, due to its:
- broad compatibility  
- commercial and open-source friendliness  
- simplicity and minimal restrictions  

Include a `LICENSE` file in the repository root with the standard MIT text to finalize licensing.

