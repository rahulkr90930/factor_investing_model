import numpy as np
from scipy.optimize import minimize

def optimize(mu, cov):
    n = len(mu)

    def neg_sharpe(w):
        return -(w @ mu) / np.sqrt(w @ cov @ w)

    cons = {"type": "eq", "fun": lambda w: w.sum() - 1}
    bounds = [(0, 0.5)] * n
    init = np.ones(n) / n

    res = minimize(neg_sharpe, init, bounds=bounds, constraints=cons)
    return res.x
