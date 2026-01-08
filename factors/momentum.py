import pandas as pd

def compute_momentum_factor(returns, q=0.3):
    signal = (1 + returns).rolling(11).apply(lambda x: x.prod()) - 1
    signal = signal.shift(1)

    factor_returns = []

    for date in signal.dropna().index:
        r = returns.loc[date].dropna()
        s = signal.loc[date].dropna()

        common = r.index.intersection(s.index)
        ranked = s[common].rank(pct=True)

        long = ranked[ranked >= 1-q].index
        short = ranked[ranked <= q].index

        factor_returns.append(r[long].mean() - r[short].mean())

    return pd.Series(factor_returns, index=signal.dropna().index, name="Momentum")
