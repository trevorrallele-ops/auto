#!/usr/bin/env python3
"""
Rolling prediction workflow: compare old predictions, retrain on updated data, and generate new predictions.

Usage:
  python rolling_predict.py GLD_daily.csv --new-horizons 1 2 3 4 5 --retrain --compare

Steps:
  1. (Optional) Compare previous predictions in figures/trade_ready_signals.csv against actual data
  2. Retrain all models on the updated historical data (with --retrain flag)
  3. Generate new predictions for the next n trading days
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and report status."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: Command failed with exit code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Rolling prediction: compare old predictions, retrain, and generate new ones'
    )
    parser.add_argument('csv', help='Updated daily price CSV (with new data appended)')
    parser.add_argument('--new-horizons', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                        help='Horizons for new predictions')
    parser.add_argument('--compare', action='store_true', help='Compare old predictions vs actuals')
    parser.add_argument('--retrain', action='store_true', help='Retrain models on new data')
    parser.add_argument('--old-predictions', default='figures/trade_ready_signals.csv',
                        help='Path to old predictions CSV for comparison')
    parser.add_argument('--symbol', default='GLD', help='Ticker symbol for export')
    parser.add_argument('--out', default='figures/trade_ready_signals_rolling.csv',
                        help='Output path for new predictions')
    parser.add_argument('--perm-repeats', type=int, default=5, help='Permutation importance repeats')
    parser.add_argument('--stop-mult', type=float, default=1.5, help='ATR multiplier for stop')
    parser.add_argument('--limit-mult', type=float, default=2.0, help='ATR multiplier for limit')
    parser.add_argument('--no-shap', action='store_true', help='Disable SHAP')
    args = parser.parse_args()
    
    success = True
    
    # Step 1: Compare old predictions to actuals
    if args.compare and Path(args.old_predictions).exists():
        cmd = [
            'python3', 'compare_predictions.py', args.csv,
            '--predictions', args.old_predictions,
            '--out', 'figures/prediction_comparison.csv'
        ]
        if not run_command(cmd, "STEP 1: Comparing old predictions vs actual data"):
            success = False
    
    # Step 2: Retrain models (optional)
    if args.retrain:
        cmd = ['python3', 'run_tuned_experiments.py', args.csv]
        if not run_command(cmd, "STEP 2: Retraining models on updated data"):
            print("WARNING: Model retraining had issues; proceeding with existing models")
    
    # Step 3: Generate new predictions
    horizons_str = ' '.join(map(str, args.new_horizons))
    cmd = [
        'python3', 'trade_ready_signals.py', args.csv,
        '--horizons'] + list(map(str, args.new_horizons)) + [
        '--train-if-missing',
        '--perm-repeats', str(args.perm_repeats),
        '--stop-mult', str(args.stop_mult),
        '--limit-mult', str(args.limit_mult),
        '--symbol', args.symbol,
        '--consensus',
        '--out', args.out
    ]
    if args.no_shap:
        cmd.append('--no-shap')
    
    if not run_command(cmd, f"STEP 3: Generating new predictions for horizons {horizons_str}"):
        success = False
    
    # Summary
    print(f"\n{'='*70}")
    if success:
        print("✓ Rolling prediction workflow completed successfully!")
        print(f"\nOutputs:")
        print(f"  Predictions: {args.out}")
        print(f"  Consensus orders: figures/broker_orders_consensus.csv")
        if args.compare and Path(args.old_predictions).exists():
            print(f"  Prediction comparison: figures/prediction_comparison.csv")
    else:
        print("✗ Rolling prediction workflow had some issues; check output above")
        sys.exit(1)
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
