# Classification Metrics - Code Explanation

## What It Does
Provides common metrics to evaluate classification models: accuracy, precision, recall, and F1 score.

---

## Function 1: `accuracy`

```python
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())
```

### What it does:
Calculates what percentage of predictions were correct.

### How it works:
1. `y_true == y_pred`: Creates boolean array (True where predictions match actual labels)
2. `.mean()`: Calculates the average (True=1, False=0)
3. `float()`: Converts to float number

### Example:
```
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

Comparison: [T, T, F, T, T]
Mean: 4/5 = 0.8
Accuracy: 80%
```

---

## Function 2: `precision_recall_f1`

```python
def precision_recall_f1(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    positive_label: int = 1
) -> Tuple[float, float, float]:
```

### Parameters:
- `y_true`: Actual labels
- `y_pred`: Predicted labels
- `positive_label`: Which label to treat as "positive" (default: 1)

### Returns:
Tuple of (precision, recall, f1)

---

### Step 1: Calculate True Positives (TP)

```python
tp = ((y_pred == positive_label) & (y_true == positive_label)).sum()
```

Counts cases where:
- Model predicted positive AND
- Actual label is positive

**Meaning**: Correct positive predictions

---

### Step 2: Calculate False Positives (FP)

```python
fp = ((y_pred == positive_label) & (y_true != positive_label)).sum()
```

Counts cases where:
- Model predicted positive AND
- Actual label is NOT positive

**Meaning**: Incorrect positive predictions (false alarms)

---

### Step 3: Calculate False Negatives (FN)

```python
fn = ((y_pred != positive_label) & (y_true == positive_label)).sum()
```

Counts cases where:
- Model predicted NOT positive AND
- Actual label IS positive

**Meaning**: Missed positive cases

---

### Step 4: Calculate Precision

```python
prec = tp / (tp + fp) if (tp + fp) else 0.0
```

**Formula**: Precision = TP / (TP + FP)

**Meaning**: Of all positive predictions, how many were correct?

**Avoids division by zero**: If no positive predictions made, precision = 0

---

### Step 5: Calculate Recall

```python
rec = tp / (tp + fn) if (tp + fn) else 0.0
```

**Formula**: Recall = TP / (TP + FN)

**Meaning**: Of all actual positives, how many did we find?

**Avoids division by zero**: If no actual positives exist, recall = 0

---

### Step 6: Calculate F1 Score

```python
if prec + rec == 0:
    f1 = 0.0
else:
    f1 = 2 * (prec * rec) / (prec + rec)
```

**Formula**: F1 = 2 × (Precision × Recall) / (Precision + Recall)

**Meaning**: Harmonic mean of precision and recall (balanced metric)

**Avoids division by zero**: If both precision and recall are 0, F1 = 0

---

### Step 7: Return Results

```python
return float(prec), float(rec), float(f1)
```

Returns all three metrics as floats.

---

## Complete Example

```
Actual:    [1, 0, 1, 1, 0, 1, 0, 0]
Predicted: [1, 0, 1, 0, 0, 1, 1, 0]

TP = 3  (positions 0, 2, 5: predicted 1, actual 1)
FP = 1  (position 6: predicted 1, actual 0)
FN = 1  (position 3: predicted 0, actual 1)

Precision = 3 / (3 + 1) = 3/4 = 0.75
Recall    = 3 / (3 + 1) = 3/4 = 0.75
F1        = 2 × (0.75 × 0.75) / (0.75 + 0.75) = 0.75
```

---
