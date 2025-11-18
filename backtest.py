"""Simple backtest utilities for signal-based strategies.

The backtest here is intentionally simple: it assumes 1 unit position when signal==1, 0 when signal==0.
It calculates cumulative returns and annualized Sharpe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional

from config import TRADING_DAYS_PER_YEAR, DEFAULT_TRANSACTION_COST, DEFAULT_SLIPPAGE, DEFAULT_LEVERAGE
from utils import safe_divide, validate_numeric_range, logger


def backtest_signals(prices: pd.Series, signals: pd.Series, transaction_cost: float = DEFAULT_TRANSACTION_COST) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Backtest signals with validation and proper error handling."""
    if len(prices) == 0 or len(signals) == 0:
        raise ValueError("Prices and signals cannot be empty")
    
    transaction_cost = validate_numeric_range(transaction_cost, 0.0, 0.1, "transaction_cost")
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
    # annualized metrics
    ann_ret = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / len(strat_ret)) - 1 if len(strat_ret) > 0 else np.nan
    ann_vol = strat_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = safe_divide(ann_ret, ann_vol, np.nan)

    out = pd.DataFrame({"price": prices, "signal": signals, "strat_return": strat_ret, "cum": cum})
    perf = {"total_return": float(total_return), "ann_ret": float(ann_ret), "ann_vol": float(ann_vol), "sharpe": float(sharpe)}
    return out, perf


def backtest_advanced(prices: pd.Series,
                      signals: pd.Series,
                      prob: Optional[pd.Series] = None,
                      transaction_cost: float = DEFAULT_TRANSACTION_COST,
                      slippage: float = DEFAULT_SLIPPAGE,
                      leverage: float = DEFAULT_LEVERAGE,
                      long_short: bool = False,
                      sizing_method: str = "proportional",
                      volatility_target: Optional[float] = None,
                      stop_loss: Optional[float] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Advanced backtest with validation and proper error handling."""
    if len(prices) == 0 or len(signals) == 0:
        raise ValueError("Prices and signals cannot be empty")
    
    # Validate parameters
    transaction_cost = validate_numeric_range(transaction_cost, 0.0, 0.1, "transaction_cost")
    slippage = validate_numeric_range(slippage, 0.0, 0.1, "slippage")
    leverage = validate_numeric_range(leverage, 0.1, 10.0, "leverage")
    
    if volatility_target is not None:
        volatility_target = validate_numeric_range(volatility_target, 0.01, 1.0, "volatility_target")
    if stop_loss is not None:
        stop_loss = validate_numeric_range(stop_loss, 0.001, 0.5, "stop_loss")
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
    if volatility_target is not None:
        # use realized vol to scale position
        ret_series = prices.pct_change().fillna(0)
        realized_vol = ret_series.rolling(14).std() * (TRADING_DAYS_PER_YEAR ** 0.5)
        size = (volatility_target / (realized_vol + 1e-9)).clip(0, leverage)
        position = pos * size
    else:
        if sizing_method == "proportional":
            # map prob [0.5,1] to [0,1] when binary; if long_short, center at 0
            if prob is None:
                size = pd.Series(1.0, index=pos.index)
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

    # apply stop-loss: if a day's pnl is less than -stop_loss, cut it to -stop_loss (clip)
    if stop_loss is not None:
        # interpret stop_loss as fractional loss per day (e.g., 0.05 = -5%)
        strat_ret = strat_ret.clip(lower=-stop_loss)

    cum = (1 + strat_ret).cumprod()
    total_return = cum.iloc[-1] - 1
    ann_ret = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / len(strat_ret)) - 1 if len(strat_ret) > 0 else float("nan")
    ann_vol = strat_ret.std() * (TRADING_DAYS_PER_YEAR ** 0.5)
    sharpe = safe_divide(ann_ret, ann_vol, float("nan"))

    out = pd.DataFrame({"price": prices, "signal": signals, "position": position, "strat_return": strat_ret, "cum": cum})
    perf = {"total_return": float(total_return), "ann_ret": float(ann_ret), "ann_vol": float(ann_vol), "sharpe": float(sharpe)}
    return out, perf


if __name__ == "__main__":
    print("Backtest helpers. Import backtest_signals in run_experiments.py to evaluate strategy returns.")
