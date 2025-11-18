#!/usr/bin/env python3
"""Test the compare predictions functionality."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

from compare_predictions import load_price_data, load_predictions, match_predictions_to_actuals
from utils import logger

def test_compare_functionality():
    """Test that compare script properly pulls latest data and compares predictions."""
    
    # Check if we have the required files
    if not os.path.exists('GLD_daily.csv'):
        logger.error("GLD_daily.csv not found")
        return False
    
    if not os.path.exists('figures/trade_ready_signals.csv'):
        logger.error("figures/trade_ready_signals.csv not found")
        return False
    
    try:
        # Load current price data
        price_df = load_price_data('GLD_daily.csv')
        logger.info(f"Loaded price data: {len(price_df)} rows, latest date: {price_df.index[-1]}")
        
        # Load predictions
        predictions_df = load_predictions('figures/trade_ready_signals.csv')
        logger.info(f"Loaded predictions: {len(predictions_df)} rows")
        
        # Check if predictions are recent
        latest_pred_date = predictions_df['as_of_date'].max()
        latest_price_date = price_df.index[-1]
        
        logger.info(f"Latest prediction date: {latest_pred_date}")
        logger.info(f"Latest price date: {latest_price_date}")
        
        # Match predictions to actuals
        results_df = match_predictions_to_actuals(predictions_df, price_df)
        
        # Check how many predictions have actual outcomes
        with_actuals = results_df.dropna(subset=['realized_direction'])
        logger.info(f"Predictions with actual outcomes: {len(with_actuals)}/{len(results_df)}")
        
        if len(with_actuals) > 0:
            accuracy = with_actuals['prediction_correct'].sum() / len(with_actuals)
            logger.info(f"Overall accuracy: {accuracy:.2%}")
            
            # Show sample results
            sample = with_actuals[['target_date', 'model', 'predicted_action', 'realized_direction', 'prediction_correct']].head()
            logger.info(f"Sample results:\n{sample}")
            
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == '__main__':
    success = test_compare_functionality()
    if success:
        logger.info("✅ Compare functionality test passed")
    else:
        logger.error("❌ Compare functionality test failed")