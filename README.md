# Quantitative Trading Strategy Backtester

A lightweight backtesting engine for evaluating rules-based trading strategies
against historical equity price data. Built to benchmark momentum and
mean-reversion strategies against buy-and-hold, with standard risk/return
metrics (Sharpe ratio, max drawdown, win rate, annualized return).

## Why this exists

This project backs one of the resume lines for quant trading/research
internship applications — a from-scratch backtester rather than a black-box
library, so the mechanics (signal generation, lookahead-bias avoidance,
transaction costs, performance metrics) are all visible and explainable in
an interview.

## Structure

```
quant-backtester/
├── backtester/
│   ├── data.py         # load real CSV data or generate synthetic price series
│   ├── strategies.py   # momentum + mean-reversion signal generators
│   ├── engine.py        # Backtester class: turns signals into P&L
│   └── metrics.py       # Sharpe, max drawdown, win rate, annualized return
├── main.py               # runnable script: compares strategies, saves a chart
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
python main.py                    # runs on synthetic demo data
```

This prints performance metrics for both strategies plus a buy-and-hold
benchmark, and saves `equity_curves.png`.

## Using real market data

The engine currently runs on synthetic data because it was built in a
sandboxed environment without market-data access. To use it with real
data, get a CSV with `Date` and `Close` columns — for example:

```bash
pip install yfinance
python -c "import yfinance as yf; yf.download('AAPL', start='2018-01-01').to_csv('AAPL.csv')"
python main.py --csv AAPL.csv
```

Any CSV with those two columns works (Yahoo Finance, Stooq, your broker's
export, etc.) — `load_csv()` lowercases and matches column names, so it's
tolerant of minor formatting differences.

## How the strategies work

- **Momentum**: long when price is above its N-day moving average
  (uptrend), short when below. Classic trend-following.
- **Mean reversion**: Bollinger-band style — go long when price drops
  more than *k* standard deviations below its rolling mean (bet on a
  bounce), short when it's that far above. Exits when price crosses back
  through the mean.

Both signals are shifted forward one day before being applied to returns,
so the backtest never trades on information it couldn't have had yet
(no lookahead bias) — a detail worth mentioning in interviews.

## Roadmap / next steps

- [ ] Validate against real historical data across multiple tickers and
      market regimes (not just one synthetic path)
- [ ] Add a simple Black-Scholes options-pricing module to extend into
      derivatives
- [ ] Parameter sweep / walk-forward optimization for lookback windows
- [ ] Position sizing beyond a flat +1/-1 (e.g. volatility-scaled sizing)
