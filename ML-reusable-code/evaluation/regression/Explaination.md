# Regression Metrics - Code Explanation

## `mse` - Mean Squared Error

```python
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_true - y_pred
    return float((diff ** 2).mean())
```

- Calculate difference between actual and predicted values
- Square each difference
- Return average of squared differences

---

## `mae` - Mean Absolute Error

```python
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).mean())
```

- Calculate absolute difference between actual and predicted
- Return average of absolute differences

---

## `rmse` - Root Mean Squared Error

```python
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))
```

- Calculate MSE
- Take square root
- Returns error in same units as original data

---

## `r2_score` - R² Score

```python
mean_y = y_true.mean()
```
Calculate mean of actual values

```python
ss_res = ((y_true - y_pred) ** 2).sum()
```
Sum of Squared Residuals: Total squared error of predictions

```python
ss_tot = ((y_true - mean_y) ** 2).sum()
```
Total Sum of Squares: Total variance in actual data

```python
if ss_tot == 0:
    return 0.0
```
Handle case where all values are identical

```python
return float(1 - ss_res / ss_tot)
```
R² = 1 - (prediction error / total variance)
- R² = 1: Perfect predictions
- R² = 0: Predictions no better than using mean
- R² < 0: Predictions worse than using mean