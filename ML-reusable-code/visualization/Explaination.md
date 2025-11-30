# Confusion Matrix Plotter - Code Explanation

## Overview

A confusion matrix is a table that summarizes how well your classification model performs. This code implements a lightweight visualization tool that builds and plots confusion matrices without requiring scikit-learn, keeping everything transparent and easy to understand.

---

## Class Structure

```python
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[list] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix"
):
```

This function acts as a **classification evaluator and visualizer** that turns predictions into insights.

---

## 1. Class Detection

```python
classes = np.unique(y_true) if labels is None else labels
num_classes = len(classes)
```

### What happens here:
Identifies all unique classes in your data or uses custom labels you provide.

### Line-by-line breakdown:

| Variable | Purpose |
|----------|---------|
| `classes` | Array of unique class values (or custom labels) |
| `num_classes` | How many different classes exist |

### Example:
```python
y_true = np.array([0, 1, 2, 0, 1, 2])
classes = np.unique(y_true)  # Result: [0, 1, 2]
num_classes = 3
```

---

## 2. Matrix Initialization

```python
mat = np.zeros((num_classes, num_classes), dtype=float)
```

### What happens here:
Creates an empty confusion matrix (all zeros) with shape `(num_classes, num_classes)`.

### Why this shape?
- **Rows** = True classes (what actually happened)
- **Columns** = Predicted classes (what the model said)

### Visual example for 3 classes:
```
       Predicted
         0  1  2
Actual 0 [0  0  0]
       1 [0  0  0]
       2 [0  0  0]
```

---

## 3. Matrix Construction - The Core Logic

```python
for t, p in zip(y_true, y_pred):
    ti = classes.index(t) if labels else np.where(classes == t)[0][0]
    pi = classes.index(p) if labels else np.where(classes == p)[0][0]
    mat[ti, pi] += 1
```

This loop builds the confusion matrix by counting predictions. Let's break it down:

### Step 1: Iterate through each prediction

```python
for t, p in zip(y_true, y_pred):
```

**What this means:**
- `t` = true label for this sample
- `p` = predicted label for this sample
- Process all samples one at a time

---

### Step 2: Convert true label to matrix index

```python
ti = classes.index(t) if labels else np.where(classes == t)[0][0]
```

**What this means:**
- If you provided custom labels (list), find position using `.index()`
- Otherwise, find position in numpy array using `np.where()`
- `ti` = row index for this sample

**Example:**
```python
classes = np.array(['cat', 'dog', 'bird'])
t = 'dog'
ti = np.where(classes == 'dog')[0][0]  # Result: 1
```

---

### Step 3: Convert predicted label to matrix index

```python
pi = classes.index(p) if labels else np.where(classes == p)[0][0]
```

**Identical logic to Step 2, but for predictions:**
- `pi` = column index for this sample

---

### Step 4: Increment the matrix cell

```python
mat[ti, pi] += 1
```

**What this means:**
- Go to row `ti` (true class), column `pi` (predicted class)
- Add 1 to that cell

**The pattern:**
- **Diagonal cells** `mat[i, i]` = correct predictions
- **Off-diagonal cells** = misclassifications

### Practical example:

```python
y_true = np.array(['cat', 'dog', 'cat'])
y_pred = np.array(['cat', 'dog', 'dog'])
labels = ['cat', 'dog']

# Iteration 1: t='cat', p='cat'
# ti=0, pi=0 → mat[0,0] += 1

# Iteration 2: t='dog', p='dog'
# ti=1, pi=1 → mat[1,1] += 1

# Iteration 3: t='cat', p='dog'
# ti=0, pi=1 → mat[0,1] += 1

# Result:
# mat = [[1, 1],
#        [0, 1]]
```

---

## 4. Normalization (Optional)

```python
if normalize:
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid div-by-zero
    mat = mat / row_sums
```

### What happens here:
Converts raw counts to percentages per class.

### Step 1: Calculate row sums

```python
row_sums = mat.sum(axis=1, keepdims=True)
```

**What this means:**
- Sum across columns (axis=1) for each row
- `keepdims=True` maintains 2D shape for broadcasting

**Example:**
```python
mat = [[1, 1],
       [0, 1]]
row_sums = [[2],    # Row 0: 1+1=2
            [1]]    # Row 1: 0+1=1
```

---

### Step 2: Prevent division by zero

```python
row_sums[row_sums == 0] = 1
```

**What this means:**
- If a row has no samples (sum=0), set it to 1
- This prevents `x/0` errors

---

### Step 3: Divide each row

```python
mat = mat / row_sums
```

**What this means:**
- Each cell becomes a percentage of its row's total
- Now each row sums to 1.0

**Example:**
```python
mat = [[1, 1],     mat = [[0.5, 0.5],
       [0, 1]]            [0.0, 1.0]]

# Row 0: 1/2=0.5, 1/2=0.5
# Row 1: 0/1=0.0, 1/1=1.0
```

**Why is this useful?**
- Shows **per-class accuracy** (diagonal values)
- Easy to compare performance across imbalanced classes

---

## 5. Visualization Setup

```python
plt.figure(figsize=(6, 5))
plt.imshow(mat, cmap="Blues")
plt.title(title)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
```

### What happens here:
Creates the heatmap visualization.

| Line | Purpose |
|------|---------|
| `plt.figure()` | Create new plot, set size |
| `plt.imshow()` | Display matrix as color grid |
| `plt.title()` | Add main title |
| `plt.xlabel/ylabel()` | Label axes |
| `plt.colorbar()` | Show color scale |

---

## 6. Cell Annotations

```python
for i in range(num_classes):
    for j in range(num_classes):
        val = mat[i, j]
        plt.text(j, i, f"{val:.2f}" if normalize else int(val),
                 ha="center", va="center", color="black")
```

### What happens here:
Adds numbers inside each cell so you can see exact values.

### Step 1: Iterate through all cells

```python
for i in range(num_classes):
    for j in range(num_classes):
```

**What this means:**
- `i` = row (y-coordinate)
- `j` = column (x-coordinate)
- Visit every cell in the matrix

---

### Step 2: Format the value

```python
val = mat[i, j]
f"{val:.2f}" if normalize else int(val)
```

**What this means:**
- If normalized: format as 2 decimal places (e.g., `0.95`)
- If raw counts: format as integer (e.g., `42`)

---

### Step 3: Place text on the plot

```python
plt.text(j, i, ..., ha="center", va="center", color="black")
```

**What this means:**
- Place text at column `j`, row `i`
- Center it horizontally and vertically
- Make it black for contrast

---

## Complete Flow Diagram

```
Input: y_true, y_pred
    ↓
Detect Classes
    ↓
Create Empty Matrix (num_classes × num_classes)
    ↓
┌─────────────────────────────────────────┐
│ For each (true, pred) sample:           │
│   Find true class row index             │
│   Find predicted class column index     │
│   Increment mat[row, col]               │
└─────────────────────────────────────────┘
    ↓
Should Normalize?
    ├─ YES → Divide each row by its sum
    └─ NO  → Keep raw counts
    ↓
Visualize with matplotlib
    ├─ Create heatmap
    ├─ Add title & labels
    ├─ Add colorbar
    └─ Annotate each cell
    ↓
Display plot
```

---

## Practical Example

```python
import numpy as np
import matplotlib.pyplot as plt

# Sample data: 3-class classification
y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1])
y_pred = np.array([0, 1, 1, 1, 2, 0, 0, 2])

# Call the function
plot_confusion_matrix(y_true, y_pred, 
                     labels=[0, 1, 2],
                     title="Model Performance")
```

**What happens step-by-step:**

```
Sample 1: true=0, pred=0 → mat[0,0] += 1
Sample 2: true=0, pred=1 → mat[0,1] += 1
Sample 3: true=1, pred=1 → mat[1,1] += 1
Sample 4: true=1, pred=1 → mat[1,1] += 1
Sample 5: true=2, pred=2 → mat[2,2] += 1
Sample 6: true=2, pred=0 → mat[2,0] += 1
Sample 7: true=0, pred=0 → mat[0,0] += 1
Sample 8: true=1, pred=2 → mat[1,2] += 1

Result:
mat = [[2, 1, 0],
       [0, 2, 1],
       [1, 0, 1]]

Visualization:
       Predicted
         0  1  2
Actual 0 [2  1  0]
       1 [0  2  1]
       2 [1  0  1]

Interpretation:
- Class 0: 2 correct, 1 wrongly predicted as 1
- Class 1: 2 correct, 1 wrongly predicted as 2
- Class 2: 1 correct, 1 wrongly predicted as 0
```

---

## Key Concepts

### 1. What does the diagonal tell you?

**Diagonal cells** = **correct predictions**
- `mat[0,0]` = predictions where true=0 AND pred=0 (correct!)
- `mat[1,1]` = predictions where true=1 AND pred=1 (correct!)
- Larger diagonal values = better model

---

### 2. What do off-diagonal cells tell you?

**Off-diagonal cells** = **misclassifications**
- `mat[0,1]` = true class 0, predicted as 1 (confusion between classes)
- `mat[2,0]` = true class 2, predicted as 0 (specific error pattern)
- These show you where the model struggles

---

### 3. Why normalize?

**Raw counts can be misleading:**
```python
# Class 0: 100 samples (2 errors)
# Class 1: 10 samples (2 errors)

# Raw: Both look equally bad (2 errors each)
# Normalized: Class 0 is 98% accurate, Class 1 is 80% accurate
```

---

### 4. Why use `keepdims=True`?

```python
# Without keepdims:
row_sums = [2, 3, 2]  # Shape: (3,)

# With keepdims:
row_sums = [[2],      # Shape: (3, 1)
            [3],
            [2]]

# numpy broadcasting requires compatible shapes
# (3, 3) / (3, 1) works ✓
# (3, 3) / (3,) may cause issues ✗
```

---

## Integration with Training

```python
# After training your model
y_true_test = test_dataset.labels
y_pred_test = model.predict(test_dataset.images)

# Visualize performance
plot_confusion_matrix(y_true_test, y_pred_test, 
                     labels=["cat", "dog", "bird"],
                     normalize=False,
                     title="Test Set Results")

# Also show normalized version for clarity
plot_confusion_matrix(y_true_test, y_pred_test, 
                     labels=["cat", "dog", "bird"],
                     normalize=True,
                     title="Test Set Results (Normalized)")
```

---

## Summary

| Step | Purpose | Key Line |
|------|---------|----------|
| Class Detection | Find all classes | `classes = np.unique(y_true)` |
| Matrix Init | Create empty grid | `mat = np.zeros((num_classes, num_classes))` |
| Construction | Fill with counts | `mat[ti, pi] += 1` |
| Normalization | Convert to percentages | `mat = mat / row_sums` |
| Visualization | Create heatmap | `plt.imshow(mat)` |
| Annotation | Add values | `plt.text(j, i, val)` |

**The core logic:** Count how many times each (true class, predicted class) pair occurs, visualize it as a color grid, annotate with numbers!
