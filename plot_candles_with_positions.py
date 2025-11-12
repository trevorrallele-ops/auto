#!/usr/bin/env python3
"""
Plot yearly candlestick charts and overlay traded positions from
`figures/trade_ready_signals.csv`.

Saves PNG files under `figures/candles_{year}.png`.
"""
import pandas as pd
from pathlib import Path
import mplfinance as mpf

fig_dir = Path('figures')
fig_dir.mkdir(exist_ok=True)

# load data
import os
csv = 'GLD_daily.csv'
if not Path(csv).exists():
    raise FileNotFoundError(csv)

df = pd.read_csv(csv, parse_dates=[0])
df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
if 'Adj Close' not in df.columns and 'Close' in df.columns:
    df['Adj Close'] = df['Close']

# prepare OHLC DataFrame for mplfinance
df = df.set_index('Date').sort_index()
try:
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, 'tz', None) is not None:
        df.index = df.index.tz_convert(None)
except Exception:
    # fallback: coerce to naive datetime
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
# ensure required columns exist: Open High Low Close Volume
for c in ['Open','High','Low','Adj Close','Volume']:
    if c not in df.columns:
        df[c] = df['Adj Close']
# ensure 'Close' column exists for mplfinance
if 'Close' not in df.columns:
    df['Close'] = df['Adj Close']

signals_path = Path('figures/trade_ready_signals.csv')
if not signals_path.exists():
    print('No trade_ready_signals.csv found; run trade_ready_signals.py first')
    trades = pd.DataFrame()
else:
    trades = pd.read_csv(signals_path, parse_dates=['target_date'])

# For each year, create a candlestick plot and overlay markers for BUY/SELL
years = sorted(pd.DatetimeIndex(df.index).year.unique())
for y in years:
    dfy = df[df.index.year == y]
    if dfy.empty:
        continue
    addplots = []
    if not trades.empty:
        # select trades with target_date in this year
        tr_y = trades[trades['target_date'].dt.year == y]
        if not tr_y.empty:
            buys = tr_y[tr_y['action']=='BUY']
            sells = tr_y[tr_y['action']=='SELL']
            if not buys.empty:
                buy_dates = pd.to_datetime(buys['target_date']).dt.date
                buy_vals = [ df.loc[pd.to_datetime(d), 'Adj Close'] if pd.to_datetime(d) in df.index else None for d in buy_dates ]
                buy_tuples = [(pd.to_datetime(d), v) for d,v in zip(buy_dates, buy_vals) if v is not None]
                if buy_tuples:
                    x = [t[0] for t in buy_tuples]
                    yv = [t[1] for t in buy_tuples]
                    addplots.append(mpf.make_addplot(pd.Series(yv, index=x), type='scatter', markersize=80, marker='^', color='g'))
            if not sells.empty:
                sell_dates = pd.to_datetime(sells['target_date']).dt.date
                sell_vals = [ df.loc[pd.to_datetime(d), 'Adj Close'] if pd.to_datetime(d) in df.index else None for d in sell_dates ]
                sell_tuples = [(pd.to_datetime(d), v) for d,v in zip(sell_dates, sell_vals) if v is not None]
                if sell_tuples:
                    x = [t[0] for t in sell_tuples]
                    yv = [t[1] for t in sell_tuples]
                    addplots.append(mpf.make_addplot(pd.Series(yv, index=x), type='scatter', markersize=80, marker='v', color='r'))
    out = fig_dir / f'candles_{y}.png'
    mpf.plot(dfy[['Open','High','Low','Close']], type='candle', style='charles', volume=False, addplot=addplots, title=f'GLD {y}', savefig=dict(fname=out, dpi=150))
    print('Saved', out)
