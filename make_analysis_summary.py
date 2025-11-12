#!/usr/bin/env python3
"""
Merge backtest metrics (figures/results_summary.csv) with trade-ready signals
(figures/trade_ready_signals.csv) to produce figures/analysis_summary.csv and a
plain-text copy-trade report `figures/copy_trade_report.txt` for the next N days.

Usage: python make_analysis_summary.py --days 5
"""
import argparse
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--days", type=int, default=5, help="Number of upcoming trading days to include in the report")
args = parser.parse_args()

fig_dir = Path('figures')
fig_dir.mkdir(exist_ok=True)

# Load results summary if present
results_path = fig_dir / 'results_summary.csv'
if results_path.exists():
    base = pd.read_csv(results_path, index_col=0)
else:
    base = pd.DataFrame()

signals_path = fig_dir / 'trade_ready_signals.csv'
if not signals_path.exists():
    raise FileNotFoundError(f"{signals_path} not found; run trade_ready_signals.py first")

signals = pd.read_csv(signals_path, parse_dates=['as_of_date','target_date'])
# merge on model (left join base metrics to signals)
if not base.empty:
    merged = signals.merge(base.reset_index().rename(columns={'index':'model'}), on='model', how='left')
else:
    merged = signals.copy()

# save analysis_summary.csv
merged.to_csv(fig_dir / 'analysis_summary.csv', index=False)
print('Wrote figures/analysis_summary.csv')

# generate a plain-text copy-trade report for next N days
report_lines = []
report_lines.append('Copy-Trade Report')
report_lines.append('=================')

upcoming = merged.sort_values('target_date').head(args.days * len(merged['model'].unique()))
# group by target_date
for td, group in upcoming.groupby('target_date'):
    report_lines.append(f"\nTarget date: {td.date()}")
    for _, r in group.iterrows():
        report_lines.append(f"- Model: {r['model']:8} | Horizon: {int(r['horizon_days'])} | Action: {r.get('action','HOLD'):4} | Prob: {r['prob']:.3f} | Size: {int(r.get('suggested_size',0))} | {r.get('copy_trade','')}")

report_file = fig_dir / 'copy_trade_report.txt'
report_file.write_text('\n'.join(report_lines))
print('Wrote', report_file)

print('\nSample of merged analysis:')
print(merged.head())
