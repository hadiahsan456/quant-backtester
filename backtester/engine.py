"""
Core backtesting engine: turns a position series into a P&L series,
accounting for transaction costs on position changes.
"""

import pandas as pd


class Backtester:
    def __init__(self, df: pd.DataFrame, transaction_cost_bps: float = 5.0):
        """
        df: DataFrame with a 'close' column, datetime index.
        transaction_cost_bps: round-trip cost in basis points charged
            whenever the position changes (models slippage + commission).
        """
        self.df = df
        self.transaction_cost = transaction_cost_bps / 10_000

    def run(self, position: pd.Series) -> pd.DataFrame:
        """
        position: series of {-1, 0, 1}, aligned to self.df's index.
        Returns a DataFrame with columns: close, position, market_return,
        strategy_return, equity_curve.
        """
        result = self.df.copy()
        result["position"] = position

        # Shift position forward one day: you can only act on a signal
        # the day AFTER you observe it, otherwise this is lookahead bias.
        result["position"] = result["position"].shift(1).fillna(0)

        result["market_return"] = result["close"].pct_change().fillna(0)
        result["strategy_return"] = result["position"] * result["market_return"]

        # charge transaction cost whenever position changes
        position_change = result["position"].diff().abs().fillna(0)
        result["strategy_return"] -= position_change * self.transaction_cost

        result["equity_curve"] = (1 + result["strategy_return"]).cumprod()
        return result
