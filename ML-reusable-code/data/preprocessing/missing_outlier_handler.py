import numpy as np
from typing import Literal


def fill_missing(
    X: np.ndarray,
    strategy: Literal["mean", "median", "zero"] = "mean"
) -> np.ndarray:
    """
    Fill missing (NaN) values in a numeric array using simple imputation.
    
    This handles the most common missing data scenarios. For more sophisticated
    imputation (KNN, iterative, etc.), consider scikit-learn's imputers.
    
    Strategies:
        - 'mean': Replace NaNs with column average (sensitive to outliers)
        - 'median': Replace NaNs with column median (robust to outliers)
        - 'zero': Replace NaNs with 0 (use when 0 has semantic meaning)
    
    Warning: This computes statistics on the ENTIRE array. In train/test splits,
    you should fit the imputation values on training data only, then apply to test.
    For that use case, wrap this in a class with fit/transform methods.
    
    Args:
        X: 2D array of shape (n_samples, n_features) with potential NaN values
        strategy: Imputation method. Options: 'mean', 'median', 'zero'
    
    Returns:
        Copy of X with NaN values filled
        
    Example:
        >>> X = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, np.nan]])
        >>> X_clean = fill_missing(X, strategy='median')
        >>> print(X_clean)  # NaNs replaced with column medians
        
    Raises:
        ValueError: If strategy is not one of the allowed options
    """
    # Make a copy so we don't modify the original array
    X = X.copy()
    
    # Process each column independently
    for col in range(X.shape[1]):
        col_data = X[:, col]
        
        # Compute the fill value based on strategy
        # np.nanmean/nanmedian ignore NaN values when computing statistics
        if strategy == "mean":
            value = np.nanmean(col_data)
        elif strategy == "median":
            value = np.nanmedian(col_data)
        elif strategy == "zero":
            value = 0.0
        else:
            raise ValueError(
                f"Invalid strategy '{strategy}'. "
                f"Must be one of: 'mean', 'median', 'zero'"
            )
        
        # Replace all NaNs in this column with the computed value
        col_data[np.isnan(col_data)] = value
        X[:, col] = col_data
    
    return X


def remove_outliers(
    X: np.ndarray,
    z_threshold: float = 3.0
) -> np.ndarray:
    """
    Remove outlier rows using z-score method (standard deviations from mean).
    
    A row is considered an outlier if ANY of its features has |z-score| >= threshold.
    Common thresholds: 3.0 (99.7% of data), 2.5 (stricter), 4.0 (more lenient).
    
    IMPORTANT: This method assumes your data is roughly normally distributed.
    For skewed data, consider using IQR method or log-transforming first.
    
    Note: This removes ENTIRE ROWS, so you'll lose samples. If you have labels (y),
    make sure to filter them with the same mask to keep X and y aligned.
    
    Args:
        X: 2D array of shape (n_samples, n_features)
        z_threshold: Maximum allowed z-score. Typical range: 2.5 to 4.0
                     3.0 means "keep data within 3 standard deviations"
    
    Returns:
        Filtered array with outlier rows removed (n_filtered_samples, n_features)
        
    Example:
        >>> X = np.array([[1, 2], [2, 3], [100, 3], [2, 2]])  # row 2 is outlier
        >>> X_clean = remove_outliers(X, z_threshold=3.0)
        >>> print(X_clean.shape)  # Fewer rows than original
        
        >>> # If you have labels, filter them too:
        >>> mask = get_outlier_mask(X, z_threshold=3.0)  # you'd need to extract this
        >>> X_clean = X[mask]
        >>> y_clean = y[mask]
    
    Pro tip: Consider using this BEFORE train/test split to avoid leaking statistics.
    Or better yet, only remove outliers from training data.
    """
    # Compute mean and std for each feature (column)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    
    # Calculate z-scores: how many standard deviations away from mean
    # Add small epsilon to avoid division by zero if std=0 (constant feature)
    z_scores = np.abs((X - mean) / (std + 1e-8))
    
    # Keep only rows where ALL features are below threshold
    # np.all(axis=1) means "check if all columns in this row pass the test"
    mask = np.all(z_scores < z_threshold, axis=1)
    
    return X[mask]


# Bonus helper function for users who need the mask separately
def get_outlier_mask(X: np.ndarray, z_threshold: float = 3.0) -> np.ndarray:
    """
    Get boolean mask indicating which rows are NOT outliers.
    
    Useful when you need to filter both X and y with the same mask.
    
    Args:
        X: 2D array of shape (n_samples, n_features)
        z_threshold: Maximum allowed z-score
        
    Returns:
        Boolean array of shape (n_samples,) - True for non-outliers
        
    Example:
        >>> mask = get_outlier_mask(X, z_threshold=3.0)
        >>> X_clean = X[mask]
        >>> y_clean = y[mask]  # Keep X and y aligned
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    z_scores = np.abs((X - mean) / (std + 1e-8))
    return np.all(z_scores < z_threshold, axis=1)