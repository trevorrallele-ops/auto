"""Utility functions for validation and error handling."""
import os
import logging
from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_file_path(path: Union[str, Path]) -> Path:
    """Validate and sanitize file path."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    return path

def validate_csv_data(df: pd.DataFrame) -> None:
    """Validate CSV data structure."""
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    required_cols = ['Close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.isnull().all().any():
        raise ValueError("DataFrame contains columns with all null values")

def safe_model_load(path: Union[str, Path]) -> object:
    """Safely load model with validation."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    # Check file size (basic safety check)
    if path.stat().st_size > 100 * 1024 * 1024:  # 100MB limit
        raise ValueError(f"Model file too large: {path}")
    
    # Check file extension for safety
    if path.suffix not in ['.joblib', '.pkl', '.pickle']:
        raise ValueError(f"Unsafe file extension: {path.suffix}")
    
    try:
        import joblib
        model = joblib.load(path)
        # Basic validation that it's a sklearn-like model
        if not hasattr(model, 'predict'):
            raise ValueError("Loaded object is not a valid model")
        logger.info(f"Successfully loaded model from {path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {path}: {e}")
        raise

def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def validate_numeric_range(value: float, min_val: float, max_val: float, name: str) -> float:
    """Validate numeric value is within range."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value)}")
    if not min_val <= value <= max_val:
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
    return float(value)

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default for zero denominator."""
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
        return default
    return numerator / denominator if abs(denominator) > 1e-10 else default

def safe_model_save(model: object, path: Union[str, Path]) -> None:
    """Safely save model with validation."""
    if not hasattr(model, 'predict'):
        raise ValueError("Object is not a valid model")
    
    path = Path(path)
    ensure_dir(path.parent)
    
    try:
        import joblib
        joblib.dump(model, path)
        logger.info(f"Successfully saved model to {path}")
    except Exception as e:
        logger.error(f"Failed to save model to {path}: {e}")
        raise