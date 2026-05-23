import pandas as pd

def compute_value_factor(returns, book, mcap, q=0.3):
    value_score = book / mcap
    value_score = value_score.dropna()

    factor_returns = []

    for date in returns.index:
        r = returns.loc[date].dropna()
        scores = value_score.reindex(r.index).dropna()
        r = r.loc[scores.index]

        ranked = scores.rank(pct=True)
        long = ranked[ranked >= 1-q].index
        short = ranked[ranked <= q].index

        factor_returns.append(r[long].mean() - r[short].mean())

    return pd.Series(factor_returns, index=returns.index, name="Value")
