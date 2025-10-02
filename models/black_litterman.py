# models/blacklitterman.py
import numpy as np
import pandas as pd
from sklearn.svm import SVR

__all__ = [
    "compute_black_litterman_estimates",
    "estimate_svr_parameters",
    "perform_rolling_training",
]

# ---------- helpers ----------

def estimate_svr_parameters(
    price_panel_window: pd.DataFrame,
    features_by_stock: dict[str, pd.DataFrame],
    stock_name: str,
    num_indicators: int,
):
    """
    Old behavior: linear SVR on features -> Adj Close.
    Robustness added:
      - reindex X to y index
      - drop rows with NaN in either X or y
      - if too few samples, return zeros (shape-stable) to keep block structure intact
    """
    # price_panel_window has MultiIndex columns: (Field, Symbol)
    if ('Adj Close', stock_name) not in price_panel_window.columns:
        # No price → return neutral blocks
        return np.zeros((num_indicators, 1)), np.array([0.0]), np.ones(num_indicators)

    y = price_panel_window[('Adj Close', stock_name)].copy()
    y.index = pd.to_datetime(y.index)

    if stock_name not in features_by_stock:
        return np.zeros((num_indicators, 1)), np.array([0.0]), np.ones(num_indicators)

    X = features_by_stock[stock_name].reindex(y.index)

    # keep rows where both X and y exist
    mask = X.notna().all(axis=1) & y.notna()
    Xc = X.loc[mask]
    yc = y.loc[mask]

    # need at least a few rows to fit SVR
    min_rows = max(20, num_indicators * 3)
    if Xc.shape[0] < min_rows:
        return np.zeros((num_indicators, 1)), np.array([0.0]), np.ones(num_indicators)

    svr = SVR(kernel="linear")
    svr.fit(Xc.values, yc.values)

    # weights: (num_indicators, 1)
    w = np.asarray(svr.coef_, dtype=float).reshape(1, -1).T
    if w.shape[0] != num_indicators:
        w = np.resize(w, (num_indicators, 1))

    b = np.asarray(svr.intercept_, dtype=float)
    var = np.var(Xc.values, axis=0)
    if var.ndim == 0:
        var = np.array([var], dtype=float)

    # ensure length == num_indicators (pad/truncate)
    if var.shape[0] != num_indicators:
        tmp = np.ones(num_indicators, dtype=float)
        k = min(num_indicators, var.shape[0])
        tmp[:k] = var[:k]
        var = tmp

    return w, b, var


def perform_rolling_training(
    returns_df: pd.DataFrame,
    features_by_stock: dict[str, pd.DataFrame],
    price_panel_window: pd.DataFrame,
    num_indicators: int,
):
    """
    Old behavior preserved:
      - For each stock, estimate SVR weights & intercepts
      - Build regression_weights (d x d*m), intercept vector (d,), and Λ (d x d) via plug-in h*Var.
    Robustness:
      - Only use symbols that exist in panel/features
      - Light ridge on Λ for invertibility
    """
    # shapes
    n = len(returns_df.index)
    cols = list(returns_df.columns)
    d = len(cols)

    if d == 0 or n == 0:
        # degenerate
        return np.eye(d), np.zeros((d, d * num_indicators)), np.zeros(d)

    # regression matrix and intercept vector
    W = np.zeros((d, d * num_indicators), dtype=float)
    intercept = np.zeros(d, dtype=float)

    # bandwidth factor (same functional form as original)
    h_factor = (4 / (d * num_indicators + 2)) ** (2 / (d * num_indicators + 4)) * n ** (
        -2 / (d * num_indicators + 4)
    )
    h_vals = []

    # ensure panel is sorted and aligned to the training window
    price_panel_window = price_panel_window.loc[returns_df.index]
    price_panel_window = price_panel_window.sort_index(axis=1)

    # estimate per asset
    for j, sym in enumerate(cols):
        w, b, var = estimate_svr_parameters(price_panel_window, features_by_stock, sym, num_indicators)
        start = j * num_indicators
        W[j, start : start + num_indicators] = w.flatten()
        intercept[j] = float(b[0]) if b.size else 0.0
        h_vals.extend(list(h_factor * var))

    # Λ = W H W'
    H_tilde = np.diag(np.asarray(h_vals, dtype=float))
    Lambda = W @ H_tilde @ W.T

    # Light ridge for numerical stability
    ridge = 1e-6 * (np.trace(Lambda) / max(d, 1))
    if not np.isfinite(ridge) or ridge <= 0:
        ridge = 1e-6
    Lambda = Lambda + np.eye(d) * ridge

    return Lambda, W, intercept


# ---------- main API ----------

def compute_black_litterman_estimates(
    returns_df: pd.DataFrame,
    features_by_stock: dict[str, pd.DataFrame],
    price_panel_window: pd.DataFrame,
    num_indicators: int,
):
    """
    Black–Litterman with SVR-driven uncertainty, faithful to the older code but robust.

    Parameters
    ----------
    returns_df : (T x d) DataFrame of asset returns (window)
    features_by_stock : dict[symbol] -> (T x m) DataFrame of technical indicators
                         (m should == num_indicators; we pad/truncate if not)
    price_panel_window : (T x (Field, Symbol)) MultiIndex columns containing at least ('Adj Close', sym)
    num_indicators : m in the older code (e.g., 9)

    Returns
    -------
    adjusted_mean : (d,) ndarray
    adjusted_covariance : (d x d) ndarray
    """
    # Prior from sample
    mu = returns_df.mean().values.astype(float)       # (d,)
    Sigma = returns_df.cov().values.astype(float)     # (d x d)
    d = len(returns_df.columns)
    I = np.eye(d, dtype=float)
    tau = 1.0

    # Early exits
    if d == 0 or returns_df.shape[0] == 0:
        return mu, Sigma

    # Normalize/align features per symbol to window; pad/truncate to num_indicators
    aligned_feats: dict[str, pd.DataFrame] = {}
    for s in returns_df.columns:
        f = features_by_stock.get(s, None)
        if f is None or not isinstance(f, pd.DataFrame):
            continue
        f = f.reindex(returns_df.index).ffill().bfill()
        # enforce m columns
        if f.shape[1] > num_indicators:
            f = f.iloc[:, :num_indicators]
        elif f.shape[1] < num_indicators:
            extra = num_indicators - f.shape[1]
            for k in range(extra):
                f[f"PAD_{k}"] = 0.0
        aligned_feats[s] = f

    # If too few have features, just return prior
    if len(aligned_feats) < 2:
        return mu, Sigma

    # Rolling training on the whole window (as in the old code)
    Lambda, B, beta = perform_rolling_training(
        returns_df, aligned_feats, price_panel_window, num_indicators
    )

    # BL closed-form
    # (Σ + [(Στ)^-1 + I' Λ^-1 I]^-1)
    Sigma_tau_inv = _pinv(Sigma * tau)
    Lambda_inv = _pinv(Lambda)
    middle_inv = _pinv(Sigma_tau_inv + I.T @ Lambda_inv @ I)
    adjusted_cov = Sigma + middle_inv

    # Build stacked mean(features) vector consistent with B @ mstack
    mstack = _feature_mean_vector(aligned_feats, list(returns_df.columns), num_indicators)

    rhs = Sigma_tau_inv @ mu + I.T @ Lambda_inv @ (beta + B @ mstack)
    adjusted_mean = adjusted_cov @ rhs

    return adjusted_mean, adjusted_cov


# ---------- small utils ----------

def _pinv(M: np.ndarray) -> np.ndarray:
    """Numerically safe pseudo-inverse with tiny ridge if needed."""
    try:
        return np.linalg.pinv(M)
    except Exception:
        d = M.shape[0]
        ridge = 1e-8 * (np.trace(M) / max(d, 1))
        if not np.isfinite(ridge) or ridge <= 0:
            ridge = 1e-8
        return np.linalg.pinv(M + np.eye(d) * ridge)


def _feature_mean_vector(features: dict[str, pd.DataFrame], cols, num_indicators: int) -> np.ndarray:
    """
    Stack per-asset feature means into a long vector of length d * num_indicators,
    matching the block layout of B (d x d*m) in perform_rolling_training.
    """
    means = []
    for c in cols:
        if c in features:
            v = features[c].mean().values
        else:
            v = np.zeros(num_indicators, dtype=float)
        # ensure exactly num_indicators
        if v.shape[0] != num_indicators:
            vv = np.zeros(num_indicators, dtype=float)
            k = min(num_indicators, v.shape[0])
            vv[:k] = v[:k]
            v = vv
        means.extend(v.tolist())
    return np.array(means, dtype=float)
