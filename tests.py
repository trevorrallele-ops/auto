"""Basic unit tests for core functions."""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from features import load_data, add_technical_indicators, prepare_features
from models import build_classifiers, evaluate_model
from backtest import backtest_signals
from utils import validate_file_path, validate_csv_data, safe_divide

class TestFeatures(unittest.TestCase):
    
    def setUp(self):
        """Create test data."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        self.test_data = pd.DataFrame({
            'Date': dates,
            'Open': np.random.uniform(100, 110, 100),
            'High': np.random.uniform(110, 120, 100),
            'Low': np.random.uniform(90, 100, 100),
            'Close': np.random.uniform(100, 110, 100),
            'Volume': np.random.randint(1000, 10000, 100)
        })
        
    def test_add_technical_indicators(self):
        """Test technical indicator calculation."""
        df = self.test_data.copy()
        df.set_index('Date', inplace=True)
        result = add_technical_indicators(df)
        
        # Check required columns exist
        required_cols = ['return_1d', 'sma_5', 'ema_12', 'rsi_14', 'macd', 'bb_upper', 'atr_14']
        for col in required_cols:
            self.assertIn(col, result.columns)
        
        # Check no infinite values
        self.assertFalse(np.isinf(result.select_dtypes(include=[np.number])).any().any())

class TestUtils(unittest.TestCase):
    
    def test_validate_csv_data(self):
        """Test CSV validation."""
        # Valid data
        valid_df = pd.DataFrame({'Close': [100, 101, 102]})
        validate_csv_data(valid_df)  # Should not raise
        
        # Invalid data - empty
        with self.assertRaises(ValueError):
            validate_csv_data(pd.DataFrame())
        
        # Invalid data - missing Close column
        with self.assertRaises(ValueError):
            validate_csv_data(pd.DataFrame({'Open': [100, 101]}))
    
    def test_safe_divide(self):
        """Test safe division."""
        self.assertEqual(safe_divide(10, 2), 5.0)
        self.assertEqual(safe_divide(10, 0), 0.0)
        self.assertEqual(safe_divide(10, 0, default=99), 99)

class TestModels(unittest.TestCase):
    
    def test_build_classifiers(self):
        """Test classifier building."""
        models = build_classifiers()
        self.assertIsInstance(models, dict)
        self.assertGreater(len(models), 0)
        
        # Check all models have predict method
        for name, model in models.items():
            self.assertTrue(hasattr(model, 'predict'))

class TestBacktest(unittest.TestCase):
    
    def test_backtest_signals(self):
        """Test basic backtest functionality."""
        prices = pd.Series([100, 101, 99, 102, 98], 
                          index=pd.date_range('2020-01-01', periods=5))
        signals = pd.Series([1, 1, 0, 1, 0], index=prices.index)
        
        bt_df, perf = backtest_signals(prices, signals)
        
        # Check output structure
        self.assertIsInstance(bt_df, pd.DataFrame)
        self.assertIsInstance(perf, dict)
        self.assertIn('total_return', perf)
        self.assertIn('sharpe', perf)

if __name__ == '__main__':
    unittest.main()