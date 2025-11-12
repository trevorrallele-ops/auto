"""Simple backtest utilities for signal-based strategies.

The backtest here is intentionally simple: it assumes 1 unit position when signal==1, 0 when signal==0.
It calculates cumulative returns and annualized Sharpe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_signals(prices: pd.Series, signals: pd.Series, transaction_cost: float = 0.0):
    """Return DataFrame with daily strategy returns and performance metrics.

    prices: pd.Series indexed by date
    signals: pd.Series of 0/1 positions aligned with prices (use predicted signal for t to trade at t+1 price)
    transaction_cost: fraction cost applied when position changes
    """
    prices = prices.sort_index()
    signals = signals.reindex(prices.index).fillna(0).astype(float)
    # daily returns
    ret = prices.pct_change().fillna(0)
    # strategy returns: position_t * return_t
    strat_ret = signals.shift(0).fillna(0) * ret
    # apply transaction costs when position changes
    trades = signals.diff().abs().fillna(0)
    strat_ret = strat_ret - trades * transaction_cost
    cum = (1 + strat_ret).cumprod()
    total_return = cum.iloc[-1] - 1
    # annualized metrics (assume daily data ~252 trading days)
    ann_ret = (1 + total_return) ** (252 / len(strat_ret)) - 1 if len(strat_ret) > 0 else np.nan
    ann_vol = strat_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else np.nan

    out = pd.DataFrame({"price": prices, "signal": signals, "strat_return": strat_ret, "cum": cum})
    perf = {"total_return": float(total_return), "ann_ret": float(ann_ret), "ann_vol": float(ann_vol), "sharpe": float(sharpe)}
    return out, perf


def backtest_advanced(prices: pd.Series,
                      signals: pd.Series,
                      prob: pd.Series | None = None,
                      transaction_cost: float = 0.0005,
                      slippage: float = 0.0005,
                      leverage: float = 1.0,
                      long_short: bool = False,
                      sizing_method: str = "proportional"):
    """More realistic daily backtest.

    - signals: expected to be in {-1,0,1} for short/flat/long if long_short True, else {0,1}
    - prob: optional probability/confidence scores used for position sizing
    - transaction_cost: fraction charged on trade notional when changing position
    - slippage: additional fraction lost on entry
    - leverage: maximum gross exposure (e.g., 1.0 = 100% long only, 2.0 = up to 2x)
    - sizing_method: 'proportional' uses prob to size position, 'fixed' uses full allocation

    Returns (df, perf) where df contains daily columns and perf summary.
    """
    prices = prices.sort_index()
    signals = signals.reindex(prices.index).fillna(0)
    if prob is None:
        prob = pd.Series(1.0, index=prices.index)
    else:
        prob = prob.reindex(prices.index).fillna(0.5)

    # Normalize signals to -1..1
    if long_short:
        pos = signals.clip(-1, 1).astype(float)
    else:
        pos = signals.clip(0, 1).astype(float)

    # size positions
    if sizing_method == "proportional":
        # map prob [0.5,1] to [0,1] when binary; if long_short, center at 0
        if long_short:
            size = (prob - 0.5) * 2.0
        else:
            size = (prob - 0.5) * 2.0
        size = size.clip(-1, 1)
        position = pos * size
    else:
        position = pos

    # apply leverage cap
    position = position * leverage
    position = position.clip(-leverage, leverage)

    # returns
    ret = prices.pct_change().fillna(0)

    # PnL before transaction costs
    daily_pnl = position.shift(0).fillna(0) * ret

    # transaction costs on turnover (notional change)
    turnover = (position - position.shift(1)).abs().fillna(position.abs())
    tc = turnover * transaction_cost

    # slippage applied on new trades (approximate)
    slip = turnover * slippage

    strat_ret = daily_pnl - tc - slip

    cum = (1 + strat_ret).cumprod()
    total_return = cum.iloc[-1] - 1
    ann_ret = (1 + total_return) ** (252 / len(strat_ret)) - 1 if len(strat_ret) > 0 else float("nan")
    ann_vol = strat_ret.std() * (252 ** 0.5)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else float("nan")

    out = pd.DataFrame({"price": prices, "signal": signals, "position": position, "strat_return": strat_ret, "cum": cum})
    perf = {"total_return": float(total_return), "ann_ret": float(ann_ret), "ann_vol": float(ann_vol), "sharpe": float(sharpe)}
    return out, perf


if __name__ == "__main__":
    print("Backtest helpers. Import backtest_signals in run_experiments.py to evaluate strategy returns.")
