"""
Data loading utilities for the backtester.

Two ways to get price data in:

1. load_csv(path) - load real OHLCV data. Works with CSVs exported from
   Yahoo Finance, Stooq, your broker, or `yfinance` run locally:

       import yfinance as yf
       yf.download("AAPL", start="2015-01-01").to_csv("AAPL.csv")

   The CSV just needs a Date column and a Close column (case-insensitive).

2. generate_synthetic(...) - generates a realistic-looking price series
   using geometric Brownian motion with a slow drift and volatility
   clustering, for testing the engine when you don't have real data handy.
"""

import numpy as np
import pandas as pd


def load_csv(path: str, date_col: str = "Date", close_col: str = "Close") -> pd.DataFrame:
    """Load OHLCV data from a CSV and return a DataFrame indexed by date
    with at least a 'close' column."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    date_col, close_col = date_col.lower(), close_col.lower()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df = df.rename(columns={close_col: "close"})
    return df[["close"]]


def generate_synthetic(
    n_days: int = 1000,
    start_price: float = 100.0,
    annual_drift: float = 0.07,
    annual_vol: float = 0.20,
    vol_cluster_strength: float = 0.35,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic daily price series using GBM with a GARCH-like
    volatility clustering overlay, so it behaves a bit more like a real
    equity series than pure GBM (calm periods and turbulent periods).
    """
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    mu = annual_drift
    sigma = annual_vol

    # simple volatility clustering: vol follows its own mean-reverting process
    vol = np.zeros(n_days)
    vol[0] = sigma
    for t in range(1, n_days):
        shock = rng.normal(0, 1)
        vol[t] = max(
            0.05,
            sigma
            + vol_cluster_strength * (vol[t - 1] - sigma)
            + 0.02 * shock,
        )

    shocks = rng.normal(0, 1, n_days)
    log_returns = (mu - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * shocks
    prices = start_price * np.exp(np.cumsum(log_returns))

    # Anchor the range on the most recent business day so `periods` always
    # yields exactly n_days regardless of what day "today" happens to be.
    today = pd.Timestamp.today().normalize()
    last_business_day = today - pd.tseries.offsets.CustomBusinessDay(0)
    dates = pd.bdate_range(end=last_business_day, periods=n_days)
    return pd.DataFrame({"close": prices}, index=dates)
