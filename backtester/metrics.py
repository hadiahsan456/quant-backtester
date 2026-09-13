"""
Standard quant performance metrics computed off a daily returns series.
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio from a series of daily strategy returns."""
    excess = daily_returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    if excess.std() == 0:
        return 0.0
    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * excess.mean() / excess.std())


def max_drawdown(equity_curve: pd.Series) -> float:
    """Largest peak-to-trough decline in the equity curve, as a negative
    fraction (e.g. -0.23 means a 23% drawdown at the worst point)."""
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def win_rate(daily_returns: pd.Series) -> float:
    """Fraction of active trading days (nonzero return) that were profitable."""
    active = daily_returns[daily_returns != 0]
    if len(active) == 0:
        return 0.0
    return float((active > 0).sum() / len(active))


def annualized_return(equity_curve: pd.Series) -> float:
    n_days = len(equity_curve)
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = n_days / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return 0.0
    return float(total_return ** (1 / years) - 1)


def summarize(daily_returns: pd.Series, equity_curve: pd.Series) -> dict:
    return {
        "Annualized Return": f"{annualized_return(equity_curve):.2%}",
        "Sharpe Ratio": f"{sharpe_ratio(daily_returns):.2f}",
        "Max Drawdown": f"{max_drawdown(equity_curve):.2%}",
        "Win Rate": f"{win_rate(daily_returns):.2%}",
    }
