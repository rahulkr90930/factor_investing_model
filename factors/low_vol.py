import pandas as pd

def compute_low_vol_factor(returns, q=0.3):
    vol = returns.rolling(6).std()

    factor_returns = []

    for date in vol.dropna().index:
        r = returns.loc[date].dropna()
        v = vol.loc[date].dropna()

        common = r.index.intersection(v.index)
        ranked = v[common].rank(pct=True)

        long = ranked[ranked <= q].index
        short = ranked[ranked >= 1-q].index

        factor_returns.append(r[long].mean() - r[short].mean())

    return pd.Series(factor_returns, index=vol.dropna().index, name="LowVol")
