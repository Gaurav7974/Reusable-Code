import pandas as pd
from typing import Optional, List

def load_csv_dataset(
    path: str,
    usecols: Optional[List[str]] = None,
    dropna: bool = False,
    dtype: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame with basic cleaning options.
    
    This is a simple wrapper around pd.read_csv that handles common preprocessing
    steps you'd normally repeat across projects. Keeps things consistent and saves
    you from writing the same 5 lines every time.
    
    Args:
        path: File path to the CSV (can be local path or URL)
        usecols: List of column names to load. Useful when you only need a subset
                 of columns from a large file. If None, loads everything.
        dropna: If True, removes any rows with missing values. Use carefully - 
                you might want more sophisticated missing data handling instead.
        dtype: Dictionary mapping column names to data types. Helps with memory
               and prevents pandas from guessing types incorrectly.
               Example: {'age': int, 'price': float}
    
    Returns:
        DataFrame with the loaded (and optionally cleaned) data
        
    Example:
        >>> df = load_csv_dataset('data.csv', usecols=['id', 'label'], dropna=True)
        >>> print(df.shape)
    """
    # Load the CSV - let pandas handle parsing
    df = pd.read_csv(path, usecols=usecols, dtype=dtype)
    
    # Drop rows with any missing values if requested
    # Note: This removes the ENTIRE row if ANY column has NaN
    if dropna:
        df = df.dropna().reset_index(drop=True)  # reset_index keeps indices clean
    
    return df