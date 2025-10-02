import warnings
import numpy as np
import pandas as pd
import quantstats as qs
from pathlib import Path

from config import (
    BENCHMARKS, COMPONENT, COMPONENT_NAME,
    DATA_START_DATE, DECISION_START_DATE, END_DATE,
    NUM_INDICATORS, CACHE_DIR
)
from dataio.yf_cache import install_dataset
from features.indicators import compute_technical_indicators
from strategies.execute import execute_strategy
from plotting.plots import plot_cumulative_returns, plot_allocation, plot_linear_cumulative_assets, plot_turnover_rate

warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")
warnings.filterwarnings("ignore", category=FutureWarning, module="quantstats")
warnings.filterwarnings("ignore", message="dropping on a non-lexsorted multi-index")
pd.set_option('future.no_silent_downcasting', True)


def build_features(price_panel_full: pd.DataFrame, stock_list):
    """price_panel_full has MultiIndex cols: (Field, Symbol)."""
    feat = {}
    for s in stock_list:
        if ('Adj Close', s) not in price_panel_full.columns:
            continue
        df_s = price_panel_full.xs(s, axis=1, level=1).copy()  # Open/High/Low/Close/Adj Close/Volume
        feat[s] = compute_technical_indicators(df_s, stock_label=s)
    return feat

def main():
    components = COMPONENT
    update_dates = list(components.keys())[:-2]
    stock_list = sorted(components['Union'])

    print("Installing dataset (OHLCV, prices/returns, benchmarks)...")
    (
        price_panel_full,      # (Field, Symbol)
        price_data,            # Adj Close (decision window)
        returns_data,          # daily returns (decision window)
        bench_price,           # benchmarks Adj Close
        bench_rets             # benchmarks returns
    ) = install_dataset(
        stock_list=stock_list,
        benchmarks=BENCHMARKS,
        data_start_date=DATA_START_DATE,
        decision_start_date=DECISION_START_DATE,
        end_date=END_DATE,
        auto_adjust=False
    )

    # Decision-window Adj Close (wide)
    adj_close_dec = price_data

    print("Computing technical indicators...")
    feats = {}
    for s in stock_list:
        if ('Adj Close', s) not in price_panel_full.columns:
            continue
        df_s = price_panel_full.xs(s, axis=1, level=1).copy()
        feats[s] = compute_technical_indicators(df_s, stock_label=s)

    LOOKBACKS = [50, 80, 100, 120, 150]
    base_out = Path("result") / COMPONENT_NAME
    base_out.mkdir(parents=True, exist_ok=True)

    # optional: one-time “stocks vs benchmarks” plot
    plot_linear_cumulative_assets(
        asset_returns=returns_data,
        benchmark_returns=bench_rets,
        save_path=str(base_out / "stocks_benchmarks_cumulative.png"),
        legend_title="Assets"
    )

    # ===== NEW: collectors for the all-in-one plot =====
    all_windows_curves = {}   # each key -> pd.Series aligned to strategy index
    all_mv_weights = {}       # collect MV weights for turnover analysis
    all_bl_weights = {}       # collect BL weights for turnover analysis

    for lb in LOOKBACKS:
        print(f"\n=== Running window {lb}d ===")
        out_dir = base_out / f"{lb}d"
        out_dir.mkdir(parents=True, exist_ok=True)

        # ---------- run strategy ----------
        strategy_returns, mv_w, bl_w = execute_strategy(
            adj_close_dec.copy(),
            returns_data.copy(),
            components, update_dates, feats,
            price_panel_full,
            lb, NUM_INDICATORS
        )

        # ---------- build report df for this window ----------
        df_bl = pd.DataFrame(index=strategy_returns.index)
        df_bl["EQW"] = returns_data.mean(axis=1).reindex(df_bl.index)
        for b in BENCHMARKS:
            if b in bench_rets.columns:
                df_bl[b] = bench_rets[b].reindex(df_bl.index)


        mv_col = f"MV ({lb}d)"
        bl_col = f"BL ({lb}d)"
        df_bl[mv_col] = strategy_returns["portfolio_mv"]
        df_bl[bl_col] = strategy_returns["portfolio_bl"]

        # ---------- save CSVs & plots per-window ----------
        (out_dir / "csv").mkdir(exist_ok=True)
        df_bl.to_csv(out_dir / "csv" / "benchmark_and_strat_returns.csv")
        strategy_returns[["portfolio_mv", "portfolio_bl"]].to_csv(out_dir / "csv" / "strategy_returns_only.csv")
        mv_w.to_csv(out_dir / "csv" / "weights_mv.csv")
        bl_w.to_csv(out_dir / "csv" / "weights_bl.csv")

        plot_cumulative_returns(df_bl, save_path=str(out_dir / "cumulative_returns.png"))
        plot_allocation(mv_w, save_path=str(out_dir / "allocation_mv.png"))
        plot_allocation(bl_w, save_path=str(out_dir / "allocation_bl.png"))

        # Plot turnover rates for both MV and BL strategies
        plot_turnover_rate(mv_w, save_path=str(out_dir / "turnover_rate_mv.png"))
        plot_turnover_rate(bl_w, save_path=str(out_dir / "turnover_rate_bl.png"))

        plot_linear_cumulative_assets(
            asset_returns=returns_data.reindex(df_bl.index),
            benchmark_returns=bench_rets.reindex(df_bl.index) if not bench_rets.empty else None,
            save_path=str(out_dir / "stocks_benchmarks_cumulative.png"),
            legend_title=f"Assets ({lb}d subset)"
        )


        # ===== collect for global plot =====
        all_windows_curves[mv_col] = df_bl[mv_col]
        all_windows_curves[bl_col] = df_bl[bl_col]
        all_mv_weights[f"MV_{lb}d"] = mv_w
        all_bl_weights[f"BL_{lb}d"] = bl_w

    # ===== build one DataFrame for ALL lookbacks & plot once =====
    all_dir = base_out / "all_windows"
    all_dir.mkdir(exist_ok=True, parents=True)

    # inner-join on common index (or use outer then fillna(0) if you prefer)
    all_curves_df = pd.DataFrame(all_windows_curves).sort_index()
    all_curves_df["EQW"] = returns_data.mean(axis=1).reindex(all_curves_df.index)
    for b in BENCHMARKS:
        if b in bench_rets.columns:
            all_curves_df[b] = bench_rets[b].reindex(all_curves_df.index)


    # One big PNG with all MV/BL curves
    plot_cumulative_returns(all_curves_df, save_path=str(all_dir / "cumulative_returns_all_windows.png"))

    # ===== Generate combined turnover rate analysis =====
    print("Generating combined turnover rate analysis...")
    
    # Create turnover rate comparison plots for all windows
    if all_mv_weights:
        # Plot individual MV turnover rates by window
        for window_name, weights in all_mv_weights.items():
            plot_turnover_rate(
                weights, 
                save_path=str(all_dir / f"turnover_rate_{window_name}.png")
            )
    
    if all_bl_weights:
        # Plot individual BL turnover rates by window  
        for window_name, weights in all_bl_weights.items():
            plot_turnover_rate(
                weights, 
                save_path=str(all_dir / f"turnover_rate_{window_name}.png")
            )

    print("\nDone. Check:")
    print(" - Per-window folders in result/<window>d/")
    print(" - Combined plot in result/all_windows/cumulative_returns_all_windows.png")
    print(" - Turnover rate plots in each window folder and result/all_windows/")
    print(" - Individual files: turnover_rate_mv.png, turnover_rate_bl.png per window")


if __name__ == "__main__":
    main()
