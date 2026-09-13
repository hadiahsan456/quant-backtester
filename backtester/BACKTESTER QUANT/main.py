"""
Run both strategies against the same price series and compare performance.

Usage:
    python main.py                     # synthetic demo data
    python main.py --csv AAPL.csv      # your own real OHLCV data
"""

import argparse
import matplotlib.pyplot as plt

from backtester import Backtester, data, strategies, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to OHLCV CSV file")
    parser.add_argument("--lookback", type=int, default=20, help="Momentum lookback window")
    parser.add_argument("--mr-window", type=int, default=20, help="Mean-reversion window")
    parser.add_argument("--mr-std", type=float, default=1.5, help="Mean-reversion band width (std devs)")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="Transaction cost in bps")
    args = parser.parse_args()

    if args.csv:
        prices = data.load_csv(args.csv)
        label = args.csv
    else:
        prices = data.generate_synthetic()
        label = "synthetic demo data"

    bt = Backtester(prices, transaction_cost_bps=args.cost_bps)

    momentum_pos = strategies.momentum_strategy(prices, lookback=args.lookback)
    mr_pos = strategies.mean_reversion_strategy(prices, window=args.mr_window, num_std=args.mr_std)

    momentum_result = bt.run(momentum_pos)
    mr_result = bt.run(mr_pos)

    buy_and_hold = (1 + prices["close"].pct_change().fillna(0)).cumprod()

    print(f"\nBacktest on: {label}\n")
    print("Momentum strategy:")
    for k, v in metrics.summarize(momentum_result["strategy_return"], momentum_result["equity_curve"]).items():
        print(f"  {k}: {v}")

    print("\nMean-reversion strategy:")
    for k, v in metrics.summarize(mr_result["strategy_return"], mr_result["equity_curve"]).items():
        print(f"  {k}: {v}")

    print("\nBuy-and-hold benchmark:")
    for k, v in metrics.summarize(prices["close"].pct_change().fillna(0), buy_and_hold).items():
        print(f"  {k}: {v}")

    # Plot equity curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(momentum_result.index, momentum_result["equity_curve"], label="Momentum")
    ax.plot(mr_result.index, mr_result["equity_curve"], label="Mean Reversion")
    ax.plot(prices.index, buy_and_hold, label="Buy & Hold", linestyle="--", color="gray")
    ax.set_title(f"Strategy Comparison — {label}")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("equity_curves.png", dpi=150)
    print("\nSaved chart to equity_curves.png")


if __name__ == "__main__":
    main()
