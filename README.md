
# Portfolio Optimization (MV & Black–Litterman) — Modular Version with Local yfinance Cache

This refactor modularizes your single-file script into a small package and adds a **local cache layer for yfinance** so you don't call the API every run.

## Key ideas

- `dataio/yf_cache.py`: persistent per-ticker parquet cache. On each run we **only fetch missing dates**, append, and write back.
- `features/indicators.py`: computes technical indicators (ta library) for each ticker.
- `models/black_litterman.py`: computes BL-adjusted mean/cov using an SVR-based uncertainty estimator.
- `models/optimization.py`: Gurobi-based MV and BL optimizers.
- `strategies/execute.py`: rolling rebalancing loop.
- `plotting/plots.py`: plotting helpers.
- `config.py`: tickers, date ranges, and historical constituents.
- `main.py`: entry point.

## Quick start

```bash
pip install -r requirements.txt
python main.py --config config_dow_jones.yaml
python main.py --config config_spdr.yaml
python main.py --config config_msci_world.yaml
```

The first run will populate `./data/yf_cache/*.parquet`. Later runs will only fetch new data after the last cached date.

## Notes

- Gurobi is optional at import time but required when running the optimizers. If you do not have a license, you can replace the Gurobi solver with a CVXOPT/OSQP version.
- The cache uses **daily data** (`interval='1d'`). Adjust if needed.
