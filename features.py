"""Feature engineering utilities for GLD data.

Functions:
 - load_data(path): loads CSV and returns DataFrame with Date index
 - add_technical_indicators(df): computes SMA, EMA, RSI, MACD, Bollinger Bands, ATR
 - prepare_features(path): convenience wrapper to load and return X,y for next-day direction
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[0])
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df.set_index("Date", inplace=True)
    return df.sort_index()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=window).mean()
    ma_down = down.rolling(window=window).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(series: pd.Series, window: int = 20, n_std: int = 2):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    return upper, lower


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    # require High, Low, Close
    high = df.get("High")
    low = df.get("Low")
    close = df.get("Close")
    if high is None or low is None or close is None:
        # fallback to zero series if columns missing
        return pd.Series(index=df.index, data=0.0)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    price = df.get("Adj Close") if "Adj Close" in df.columns else df.get("Close")
    df["return_1d"] = price.pct_change()
    df["sma_5"] = sma(price, 5)
    df["sma_10"] = sma(price, 10)
    df["ema_12"] = ema(price, 12)
    df["ema_26"] = ema(price, 26)
    df["rsi_14"] = rsi(price, 14)
    macd_line, signal_line, hist = macd(price)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist
    bb_up, bb_low = bollinger_bands(price)
    df["bb_upper"] = bb_up
    df["bb_lower"] = bb_low
    df["atr_14"] = atr(df, 14)
    # percentage distance from bands
    df["bb_pct"] = (price - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # lagged features
    for lag in (1, 2, 3):
        df[f"return_{lag}d"] = price.pct_change(lag)

    # drop rows with NA
    df = df.dropna()
    return df


def prepare_features(path: str, target_horizon: int = 1):
    df = load_data(path)
    df = add_technical_indicators(df)
    price = df.get("Adj Close") if "Adj Close" in df.columns else df.get("Close")
    # Binary target: next-day up or down
    df["target_up"] = (price.shift(-target_horizon) > price).astype(int)
    df = df.dropna()
    X = df.drop(columns=[c for c in df.columns if c.startswith("target_")])
    y = df["target_up"]
    return X, y, df


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "GLD_daily.csv"
    X, y, df = prepare_features(path)
    print("Prepared features:", X.shape)
