# strategies/execute.py
import time
from datetime import datetime
import numpy as np
import pandas as pd

from models.black_litterman import compute_black_litterman_estimates
from models.optimization import optimize_mean_variance, optimize_black_litterman


MAX_BL_SECS = 3.0          # hard cap per rebalance for the whole BL step
DIAG_ENABLE = True         # turn diagnostics on/off


def execute_strategy(price_df, returns_df, components, update_dates, features_dict, price_ohlcv_panel, lookback, num_indicators):
    print(f"Running strategy with lookback {lookback} days.")
    monthly_rebalance_idx = price_df.groupby(price_df.index.strftime('%Y-%m')).head(1).index
    mv_w = pd.DataFrame(index=price_df.index, columns=price_df.columns)
    bl_w = pd.DataFrame(index=price_df.index, columns=price_df.columns)

    strat_ret = returns_df.copy()
    strat_ret['portfolio_mv'] = 0.0
    strat_ret['portfolio_bl'] = 0.0
    
    # ---- before you build `rebals` or right after you have returns_df/price_df/feats ----
    master_idx = returns_df.index  # global timeline
    # symbols that exist in both: Adj Close panel & features dict & returns universe
    adj_syms = set(price_ohlcv_panel.columns.get_level_values(1)
                [price_ohlcv_panel.columns.get_level_values(0) == 'Adj Close'])
    feat_syms = set(features_dict.keys())
    universe_syms = list(set(price_df.columns) & adj_syms & feat_syms)

    # Build a boolean "valid-row" matrix: row is a date, col is a symbol; True if all features
    # for that symbol are non-NA on that date.
    # Use compact dtype to save RAM.
    valid_matrix = pd.DataFrame(
        False, index=master_idx, columns=universe_syms, dtype="bool"
    )

    for s in universe_syms:
        f = features_dict[s].reindex(master_idx)            # align once
        valid_matrix[s] = f.notna().all(axis=1).values      # True if that day has all features

    # Assemble rebalance info
    rebals = []
    for current_date in price_df.index:
        current_idx = price_df.index.get_loc(current_date)
        if current_idx > lookback and current_date in monthly_rebalance_idx:
            dt_py = pd.to_datetime(current_date).to_pydatetime()
            # latest component set
            for ud in sorted(update_dates, reverse=True):
                if datetime.strptime(ud, "%Y-%m-%d") <= dt_py:
                    curr_comps = sorted(components[ud])
                    break
            rebals.append((current_date, current_idx, curr_comps))

    # Compute window returns for each rebalance
    for reb_date, idx, comps in rebals:
        comps_in_data = [c for c in comps if c in returns_df.columns]
        if not comps_in_data:
            mv_w.loc[reb_date] = 0
            bl_w.loc[reb_date] = 0
            continue

        window = returns_df[comps_in_data].iloc[idx - lookback: idx].dropna(axis=1, how="all")
        if window.shape[1] == 0 or window.shape[0] < 5:
            mv_w.loc[reb_date] = 0
            bl_w.loc[reb_date] = 0
            continue

        print(f"Lookback={lookback}. Decision on {reb_date.date()}: {window.shape[1]} assets in window.")

        # ===== Mean–Variance =====
        try:
            mv = optimize_mean_variance(window)
        except Exception as e:
            print(f"[MV] optimize failed on {reb_date.date()}: {e}")
            mv = None
        if mv is not None:
            mv_w.loc[reb_date, window.columns] = mv
        
        mv_w.loc[reb_date] = mv_w.loc[reb_date].fillna(0)
        # ===== Black–Litterman (fast, vectorized) =====
        # intersect window names with precomputed universe
        cands = window.columns.intersection(valid_matrix.columns)
        if len(cands) == 0:
            valid_syms = []
        else:
            # count per-symbol valid rows within the window via a single slice + column sum
            counts = valid_matrix.loc[window.index, cands].sum(axis=0)  # boolean sum
            valid_syms = list(counts[counts >= 5].index)

        print(f"[BL] {len(valid_syms)} valid symbols on {reb_date.date()}.")

        used_mv_fallback = False

        if len(valid_syms) >= 2:
            window_sub = window[valid_syms]
            features_sub = {
                c: features_dict[c].reindex(window_sub.index).ffill().bfill()
                for c in valid_syms
            }
            price_sub = price_ohlcv_panel.loc[window_sub.index, (slice(None), valid_syms)]

            try:
                bl_mean, bl_cov = compute_black_litterman_estimates(
                    window_sub, features_sub, price_sub, num_indicators
                )
                bl = optimize_black_litterman(window_sub, bl_mean, bl_cov, num_indicators)
            except Exception as e:
                print(f"[BL] failed on {reb_date.date()} ({len(valid_syms)} syms): {e}")
                bl = None

            if bl is not None:
                bl_w.loc[reb_date, window_sub.columns] = bl
            else:
                # Fallback to MV weights for this date
                bl_w.loc[reb_date] = mv_w.loc[reb_date]
                used_mv_fallback = True
        else:
            # Not enough names: fallback to MV
            bl_w.loc[reb_date] = mv_w.loc[reb_date]
            used_mv_fallback = True

        if used_mv_fallback:
            print(f"[BL] fallback → MV weights on {reb_date.date()}.")
        bl_w.loc[reb_date] = bl_w.loc[reb_date].fillna(0)

    # 若整段 BL 仍全 0（極端情況），整段回退 MV
    if (bl_w.sum(axis=1) == 0).all():
        print("[BL] entire series empty → fallback to MV series.")
        bl_w = mv_w.copy()

    mv_w = mv_w.ffill().fillna(0)
    bl_w = bl_w.ffill().fillna(0)

    # Daily portfolio returns
    for i in range(len(price_df) - 1):
        cur = price_df.index[i]
        nxt = price_df.index[i + 1]
        strat_ret.loc[nxt, 'portfolio_mv'] = float((mv_w.loc[cur] * returns_df.loc[nxt]).sum())
        strat_ret.loc[nxt, 'portfolio_bl'] = float((bl_w.loc[cur] * returns_df.loc[nxt]).sum())

    return strat_ret, mv_w, bl_w
