import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from factors.value import compute_value_factor
from factors.momentum import compute_momentum_factor
from factors.low_vol import compute_low_vol_factor
from optimization.mean_variance import optimize
from backtest.performance import *

# ---------------- DATA DOWNLOAD ----------------
universe = pd.read_csv("data/raw/universe.csv")
tickers = universe["ticker"].tolist()

prices = yf.download(
    tickers,
    start="2021-01-01",
    end="2024-12-31",
    auto_adjust=True,
    progress=False
)["Close"]

prices.to_csv("data/processed/daily_prices.csv")

monthly_prices = prices.resample("M").last()
monthly_returns = monthly_prices.pct_change().dropna()

monthly_prices.to_csv("data/processed/monthly_prices.csv")
monthly_returns.to_csv("data/processed/monthly_returns.csv")

# Market cap snapshot
mcap = {t: yf.Ticker(t).info.get("marketCap") for t in tickers}
mcap = pd.Series(mcap)
mcap.to_csv("data/processed/market_cap.csv")

# ---------------- FACTORS ----------------
book = pd.read_csv("data/raw/book_value.csv", index_col=0)["book_value"]

value = compute_value_factor(monthly_returns, book, mcap)
momentum = compute_momentum_factor(monthly_returns)
lowvol = compute_low_vol_factor(monthly_returns)

factors = pd.concat([value, momentum, lowvol], axis=1).dropna()
factors.to_csv("data/processed/factor_returns.csv")

# ---------------- PORTFOLIO ----------------
mu = factors.mean() * 12
cov = factors.cov() * 12

weights = optimize(mu.values, cov.values)
opt_port = factors @ weights
eq_port = factors.mean(axis=1)

# ---------------- RESULTS ----------------
results = {
    "Equal": eq_port,
    "Optimized": opt_port
}

for k, r in results.items():
    print(f"\n{k} Portfolio")
    print("CAGR:", round(cagr(r), 3))
    print("Vol:", round(volatility(r), 3))
    print("Sharpe:", round(sharpe(r), 2))
    print("Max DD:", round(max_drawdown(r), 2))

(1 + pd.DataFrame(results)).cumprod().plot(
    title="Factor Portfolio Performance", figsize=(10,6)
)
plt.show()
