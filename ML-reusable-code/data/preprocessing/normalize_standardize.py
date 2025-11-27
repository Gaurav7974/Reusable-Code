import numpy as np
from typing import Tuple


class Normalizer:
    """
    MinMax normalization - scales features to [0, 1] range.
    
    This is useful when you want to squash all your features into the same scale
    without changing the distribution shape. Common for neural networks, KNN, and
    algorithms sensitive to feature magnitudes.
    
    Formula: (X - X_min) / (X_max - X_min)
    
    Important: Always fit on training data only, then transform both train and test.
    Otherwise you're leaking information from your test set.
    
    Example:
        >>> normalizer = Normalizer()
        >>> X_train_scaled = normalizer.fit_transform(X_train)
        >>> X_test_scaled = normalizer.transform(X_test)  # Use same min/max from training
    """
    
    def __init__(self):
        """Initialize normalizer. Stats will be computed when you call fit()."""
        self.min = None  # Will store min value per feature
        self.max = None  # Will store max value per feature
    
    def fit(self, X: np.ndarray):
        """
        Learn min and max values from your training data.
        
        Args:
            X: Training data, shape (n_samples, n_features)
        """
        self.min = np.min(X, axis=0)  # axis=0 means "per column"
        self.max = np.max(X, axis=0)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale data using previously learned min/max values.
        
        Args:
            X: Data to transform, shape (n_samples, n_features)
            
        Returns:
            Normalized data in [0, 1] range
            
        Raises:
            ValueError: If you call transform before fit
        """
        if self.min is None or self.max is None:
            raise ValueError("You must call fit() before transform(). Normalizer needs to learn min/max first.")
        
        # Add small epsilon (1e-8) to avoid division by zero if min == max
        return (X - self.min) / (self.max - self.min + 1e-8)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Convenience method: fit and transform in one call.
        
        Only use this on training data. For test/validation, use transform() separately.
        
        Args:
            X: Training data to fit and transform
            
        Returns:
            Normalized training data
        """
        self.fit(X)
        return self.transform(X)


class Standardizer:
    """
    Z-score standardization - centers data to mean=0, std=1.
    
    This is the go-to normalization for most ML algorithms, especially linear models
    and algorithms that assume normally distributed features. Unlike MinMax, this
    doesn't bound your data to a fixed range - outliers will still be outliers.
    
    Formula: (X - mean) / std
    
    When to use this vs Normalizer:
    - Use Standardizer for: Linear regression, logistic regression, SVM, PCA
    - Use Normalizer for: Neural networks, KNN, algorithms needing bounded input
    
    Example:
        >>> standardizer = Standardizer()
        >>> X_train_scaled = standardizer.fit_transform(X_train)
        >>> X_test_scaled = standardizer.transform(X_test)  # Use same mean/std from training
    """
    
    def __init__(self):
        """Initialize standardizer. Mean and std will be computed when you call fit()."""
        self.mean = None  # Will store mean per feature
        self.std = None   # Will store standard deviation per feature
    
    def fit(self, X: np.ndarray):
        """
        Learn mean and standard deviation from your training data.
        
        Args:
            X: Training data, shape (n_samples, n_features)
        """
        self.mean = np.mean(X, axis=0)  # axis=0 means "per column"
        self.std = np.std(X, axis=0)    # using population std (default)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize data using previously learned mean/std.
        
        Args:
            X: Data to transform, shape (n_samples, n_features)
            
        Returns:
            Standardized data with mean≈0, std≈1
            
        Raises:
            ValueError: If you call transform before fit
        """
        if self.mean is None or self.std is None:
            raise ValueError("You must call fit() before transform(). Standardizer needs to learn mean/std first.")
        
        # Add small epsilon (1e-8) to avoid division by zero if std == 0 (constant feature)
        return (X - self.mean) / (self.std + 1e-8)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Convenience method: fit and transform in one call.
        
        Only use this on training data. For test/validation, use transform() separately.
        
        Args:
            X: Training data to fit and transform
            
        Returns:
            Standardized training data
        """
        self.fit(X)
        return self.transform(X)