import numpy as np

def cagr(r):
    return (1 + r).prod() ** (12 / len(r)) - 1

def volatility(r):
    return r.std() * np.sqrt(12)

def sharpe(r):
    return r.mean() / r.std() * np.sqrt(12)

def max_drawdown(r):
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    return ((cum - peak) / peak).min()