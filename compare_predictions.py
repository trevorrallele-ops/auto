#!/usr/bin/env python3
"""
Compare previous trade-ready predictions against actual realized outcomes.

Usage:
  python compare_predictions.py GLD_daily.csv --predictions figures/trade_ready_signals.csv --out figures/prediction_backtest.csv

This script:
1. Loads the current daily price data (with new data since last prediction date)
2. Matches each prediction with its target_date's actual close price
3. Computes realized direction (UP/DOWN) and calculates if prediction was correct
4. Outputs a comparison CSV and summary statistics
"""
import argparse
import warnings
from pathlib import Path

import pandas as pd
import numpy as np


def load_price_data(csv_path):
    """Load historical price CSV and return DataFrame indexed by date."""
    df = pd.read_csv(csv_path, parse_dates=['Date'] if 'Date' in pd.read_csv(csv_path, nrows=1).columns else [0])
    
    # Auto-detect date column
    date_cols = [c for c in df.columns if c.lower() in ('date', 'datetime', 'timestamp')]
    if date_cols:
        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
        df.set_index(date_cols[0], inplace=True)
    else:
        # try first column if it looks like date
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    
    df.index.name = 'Date'
    return df.sort_index()


def load_predictions(pred_csv):
    """Load trade_ready_signals.csv."""
    df = pd.read_csv(pred_csv)
    df['target_date'] = pd.to_datetime(df['target_date'])
    df['as_of_date'] = pd.to_datetime(df['as_of_date'])
    return df


def match_predictions_to_actuals(predictions_df, price_df):
    """Match each prediction to actual close price on target_date."""
    # Normalize price_df index
    if not isinstance(price_df.index, pd.DatetimeIndex):
        try:
            price_df.index = pd.to_datetime(price_df.index)
        except Exception:
            pass
    
    close_col = 'Adj Close' if 'Adj Close' in price_df.columns else 'Close'
    
    # Get reference close (as_of_date close) for each prediction
    rows = []
    for idx, row in predictions_df.iterrows():
        target_dt = row['target_date']
        as_of_dt = row['as_of_date']
        
        # Get reference close price (as-of date)
        try:
            ref_close = float(price_df.loc[as_of_dt, close_col])
        except (KeyError, TypeError):
            ref_close = np.nan
        
        # Get target close price
        try:
            target_close = float(price_df.loc[target_dt, close_col])
        except (KeyError, TypeError):
            target_close = np.nan
        
        # Realized direction
        realized_direction = None
        if not np.isnan(ref_close) and not np.isnan(target_close):
            realized_direction = 'UP' if target_close > ref_close else 'DOWN'
        
        # Was prediction correct?
        correct = None
        if realized_direction is not None:
            predicted_action = row['action']
            if predicted_action == 'BUY' and realized_direction == 'UP':
                correct = True
            elif predicted_action == 'SELL' and realized_direction == 'DOWN':
                correct = True
            elif predicted_action == 'HOLD':
                # HOLD is considered correct if it doesn't lose (small move)
                pct_move = abs(target_close - ref_close) / ref_close
                correct = pct_move < 0.01  # less than 1% move
            else:
                correct = False
        
        rows.append({
            'as_of_date': as_of_dt,
            'target_date': target_dt,
            'horizon_days': row['horizon_days'],
            'model': row['model'],
            'predicted_action': row['action'],
            'prob': row['prob'],
            'val_auc': row['val_auc'],
            'ref_close': ref_close,
            'target_close': target_close,
            'realized_direction': realized_direction,
            'pct_return': ((target_close - ref_close) / ref_close * 100) if not np.isnan(ref_close) and not np.isnan(target_close) else np.nan,
            'prediction_correct': correct,
        })
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Compare predictions vs actuals')
    parser.add_argument('csv', help='Current daily price data CSV')
    parser.add_argument('--predictions', default='figures/trade_ready_signals.csv', help='Predictions CSV to compare')
    parser.add_argument('--out', default='figures/prediction_backtest.csv', help='Output CSV path')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading prices from {args.csv}")
    price_df = load_price_data(args.csv)
    
    print(f"Loading predictions from {args.predictions}")
    predictions_df = load_predictions(args.predictions)
    
    print(f"Matching {len(predictions_df)} predictions to actuals...")
    results_df = match_predictions_to_actuals(predictions_df, price_df)
    
    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    
    print(f"\nWrote: {out_path}")
    print(f"\nSummary Statistics:")
    print(f"Total predictions: {len(results_df)}")
    
    # Filter to predictions with realized outcomes
    with_actuals = results_df.dropna(subset=['realized_direction'])
    print(f"Predictions with actual data: {len(with_actuals)}")
    
    if len(with_actuals) > 0:
        accuracy = with_actuals['prediction_correct'].sum() / len(with_actuals)
        print(f"Prediction accuracy: {accuracy:.2%}")
        
        # Per-model accuracy
        print("\nAccuracy by model:")
        for model, grp in with_actuals.groupby('model'):
            acc = grp['prediction_correct'].sum() / len(grp)
            print(f"  {model:12s}: {acc:6.2%} ({grp['prediction_correct'].sum()}/{len(grp)})")
        
        # Per-horizon accuracy
        print("\nAccuracy by horizon (days):")
        for h, grp in with_actuals.groupby('horizon_days'):
            acc = grp['prediction_correct'].sum() / len(grp)
            print(f"  Horizon {h:2d}d: {acc:6.2%} ({grp['prediction_correct'].sum()}/{len(grp)})")
        
        # Average return by action
        print("\nAverage return by predicted action:")
        for action, grp in with_actuals.groupby('predicted_action'):
            avg_ret = grp['pct_return'].mean()
            print(f"  {action:6s}: {avg_ret:+.2f}%")
    
    print(f"\nFirst 10 comparisons:")
    print(with_actuals[['as_of_date', 'target_date', 'model', 'predicted_action', 'prob', 
                        'ref_close', 'target_close', 'pct_return', 'realized_direction', 'prediction_correct']].head(10))


if __name__ == '__main__':
    main()
