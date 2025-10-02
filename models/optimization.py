# models/optimization.py
import numpy as np
import pandas as pd
import gurobipy as gp

def optimize_mean_variance(returns_df: pd.DataFrame):
    """
    Classic MV:  min w' Σ w  s.t.  w' μ = 1,  w >= 0
    Returns normalized weights or None if infeasible.
    """
    cov = returns_df.cov().values
    mu = returns_df.mean().values
    d = len(returns_df.columns)

    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.setParam('DualReductions', 0)
        env.start()
        with gp.Model(env=env, name="mv_portfolio") as m:
            w = m.addMVar(d, name="w", lb=0.0, ub=gp.GRB.INFINITY)
            port_ret = w @ mu
            port_var = w @ cov @ w

            # If all expected returns are negative, the old code returned None.
            if np.all(mu < 0):
                return None

            m.setObjective(port_var, gp.GRB.MINIMIZE)
            m.addConstr(port_ret == 1.0)
            m.optimize()

            if m.status == gp.GRB.OPTIMAL:
                sol = w.X
                s = sol.sum()
                return sol / s if s != 0 else np.ones(d) / d
            return None


def optimize_black_litterman(returns_df: pd.DataFrame,
                             bl_mean,
                             bl_cov,
                             num_indicators: int | None = None):
    """
    BL optimizer to match how you call it:
        bl = optimize_black_litterman(window_sub, bl_mean, bl_cov, num_indicators)

    Objective/constraint identical to MV, but using BL μ, Σ.
    We regularize Σ slightly for numerical stability and always return a valid weight vector.
    """
    d = len(returns_df.columns)
    mu = np.asarray(bl_mean).reshape(-1)
    Sigma = np.asarray(bl_cov)

    # Sanity checks
    if mu.shape[0] != d:
        # align by trunc/pad to d to avoid crashes
        tmp = np.zeros(d)
        k = min(d, mu.shape[0])
        tmp[:k] = mu[:k]
        mu = tmp
    if Sigma.shape != (d, d):
        S = np.zeros((d, d))
        k = min(d, Sigma.shape[0] if Sigma.ndim > 0 else 0)
        if Sigma.ndim == 2:
            k = min(d, Sigma.shape[0], Sigma.shape[1])
            S[:k, :k] = Sigma[:k, :k]
        Sigma = S

    # If all negative, fall back to equal weight
    if np.all(mu < 0):
        return np.ones(d) / d

    # Regularize covariance
    tr = float(np.trace(Sigma)) if Sigma.size else 0.0
    ridge = (1e-8 * tr / max(d, 1)) if np.isfinite(tr) and tr > 0 else 1e-8
    Sigma = Sigma + np.eye(d) * ridge

    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.setParam('DualReductions', 0)
        env.setParam('TimeLimit', 10)
        env.start()
        with gp.Model(env=env, name="bl_portfolio") as m:
            w = m.addMVar(d, name="w", lb=0.0, ub=gp.GRB.INFINITY)
            port_ret = w @ mu
            port_var = w @ Sigma @ w

            m.setObjective(port_var, gp.GRB.MINIMIZE)
            m.addConstr(port_ret == 1.0)
            m.optimize()

            if m.status == gp.GRB.OPTIMAL:
                sol = w.X
                s = sol.sum()
                return sol / s if s != 0 else np.ones(d) / d
            # infeasible/unbounded → equal weight fallback
            return np.ones(d) / d
