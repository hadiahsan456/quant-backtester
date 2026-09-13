"""
Strategies produce a 'position' column: +1 (long), -1 (short), or 0 (flat)
for each day, based only on information available up to that day (no
lookahead bias - signals are shifted forward one day before being applied
to returns in the backtester).
"""

import pandas as pd


def momentum_strategy(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Go long if price is above its N-day moving average (uptrend),
    short if below (downtrend). Classic trend-following signal.
    """
    ma = df["close"].rolling(lookback).mean()
    position = pd.Series(0, index=df.index)
    position[df["close"] > ma] = 1
    position[df["close"] < ma] = -1
    return position


def mean_reversion_strategy(
    df: pd.DataFrame, window: int = 20, num_std: float = 1.5
) -> pd.Series:
    """Bollinger-band style mean reversion: go long when price is
    num_std standard deviations below its rolling mean (oversold, bet on
    bounce-back), short when it's num_std above (overbought).
    """
    ma = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std

    # Stateful pass: enter when price breaks a band, exit when it crosses
    # back through the rolling mean, otherwise hold whatever we had.
    positions = []
    current = 0
    for price, mean, lo, hi in zip(df["close"], ma, lower, upper):
        if pd.isna(mean):
            current = 0
        elif current == 1 and price >= mean:
            current = 0
        elif current == -1 and price <= mean:
            current = 0
        elif current == 0 and price < lo:
            current = 1
        elif current == 0 and price > hi:
            current = -1
        positions.append(current)

    return pd.Series(positions, index=df.index)
