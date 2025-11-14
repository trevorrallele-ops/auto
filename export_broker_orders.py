#!/usr/bin/env python3
"""
Export broker-ready CSV with columns: symbol, side, qty, order_type, target_date, model

Reads `figures/trade_ready_signals.csv` (or provided --input) and writes `figures/broker_orders.csv`.

Usage:
  python export_broker_orders.py --symbol IAU --input figures/trade_ready_signals.csv --out figures/broker_orders.csv

By default only rows with action BUY or SELL are exported.
"""
import argparse
import re
from pathlib import Path
import pandas as pd


def parse_qty(row):
    # prefer suggested_size column
    if 'suggested_size' in row and pd.notna(row['suggested_size']):
        try:
            return int(float(row['suggested_size']))
        except Exception:
            pass
    # fallback: try to parse copy_trade like 'BUY 100 @market (prob=0.123)'
    ct = row.get('copy_trade', '')
    m = re.search(r"(BUY|SELL)\s+(\d+)", str(ct))
    if m:
        return int(m.group(2))
    # last resort: None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='GLD', help='Ticker symbol for broker orders')
    parser.add_argument('--input', default='figures/trade_ready_signals.csv')
    parser.add_argument('--out', default='figures/broker_orders.csv')
    parser.add_argument('--order-type', default='market', help='Order type to suggest (market/limit)')
    parser.add_argument('--min-prob', type=float, default=0.0, help='Minimum prob to include')
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(f"Input file not found: {inp}")

    df = pd.read_csv(inp, parse_dates=['as_of_date','target_date'])

    # filter by action column if present
    if 'action' in df.columns:
        df = df[df['action'].isin(['BUY','SELL'])]
    else:
        # derive action from prob threshold >0.5
        df['action'] = df['prob'].apply(lambda p: 'BUY' if p>0.5 else 'SELL')

    # filter by min_prob
    df = df[df['prob'] >= args.min_prob]

    rows = []
    for _, r in df.iterrows():
        qty = parse_qty(r)
        if qty is None or qty == 0:
            # skip rows without quantity info
            continue
        rows.append({
            'symbol': args.symbol,
            'side': r['action'],
            'qty': int(qty),
            'order_type': args.order_type,
            'target_date': r['target_date'],
            'model': r.get('model','')
        })

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(outp, index=False)
    print('Wrote', outp)


if __name__ == '__main__':
    main()
