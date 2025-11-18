#!/usr/bin/env python3
"""Update GLD data from Yahoo Finance and run comparison."""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

from utils import logger, validate_file_path
from config import FIGURES_DIR

def update_gld_data(symbol: str = "GLD", output_file: str = "GLD_daily.csv") -> bool:
    """Download latest GLD data from Yahoo Finance."""
    try:
        # Get existing data to determine start date
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file, parse_dates=[0], index_col=0)
            start_date = existing_df.index[-1] + timedelta(days=1)
            logger.info(f"Updating from {start_date}")
        else:
            start_date = "2004-01-01"  # GLD inception
            logger.info(f"Downloading full history from {start_date}")
        
        # Download data
        ticker = yf.Ticker(symbol)
        new_data = ticker.history(start=start_date, end=None)
        
        if len(new_data) == 0:
            logger.info("No new data available")
            return True
        
        # Format columns to match existing format
        new_data = new_data.rename(columns={
            'Open': 'Open',
            'High': 'High', 
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        })
        new_data['Adj Close'] = new_data['Close']  # Yahoo Finance adjusted close
        new_data = new_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        
        if os.path.exists(output_file):
            # Append new data
            existing_df = pd.read_csv(output_file, parse_dates=[0], index_col=0)
            combined_df = pd.concat([existing_df, new_data])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        else:
            combined_df = new_data
        
        # Save updated data
        combined_df.index.name = 'Date'
        combined_df.to_csv(output_file)
        
        logger.info(f"Updated {output_file} with {len(new_data)} new rows")
        logger.info(f"Latest date: {combined_df.index[-1]}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update data: {e}")
        return False

def run_full_update_and_compare():
    """Update data and run comparison."""
    logger.info("Starting data update and comparison...")
    
    # Update GLD data
    if not update_gld_data():
        return False
    
    # Run comparison if predictions exist
    pred_file = f"{FIGURES_DIR}/trade_ready_signals.csv"
    if os.path.exists(pred_file):
        logger.info("Running prediction comparison...")
        os.system(f"python compare_predictions.py GLD_daily.csv --predictions {pred_file}")
    else:
        logger.warning(f"No predictions file found at {pred_file}")
    
    return True

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='GLD', help='Yahoo Finance symbol')
    parser.add_argument('--output', default='GLD_daily.csv', help='Output CSV file')
    parser.add_argument('--compare', action='store_true', help='Also run comparison')
    args = parser.parse_args()
    
    if args.compare:
        run_full_update_and_compare()
    else:
        update_gld_data(args.symbol, args.output)