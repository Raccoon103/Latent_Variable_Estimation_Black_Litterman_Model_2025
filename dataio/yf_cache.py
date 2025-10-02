# dataio/yf_cache.py
from __future__ import annotations

import os
from typing import Iterable, Tuple, List
import pandas as pd
import numpy as np
import yfinance as yf

_YF_FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


# ------------------------- small helpers -------------------------

def _safe_dates(start, end) -> Tuple[pd.Timestamp, pd.Timestamp]:
    s = pd.Timestamp(start).tz_localize(None).normalize()
    e = pd.Timestamp(end).tz_localize(None).normalize()
    if e <= s:
        e = s + pd.Timedelta(days=1)
    # yfinance end is exclusive; push +1 day so we get intended last day
    return s, e + pd.Timedelta(days=1)

def _ensure_date_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    # sometimes parquet restores index name as None; enforce 'Date'
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df

def _normalize_single(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=_YF_FIELDS)

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    keep = [c for c in _YF_FIELDS if c in df.columns]
    df = df[keep].copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_index()
    df.index.name = "Date"
    return df

def _normalize_multi(df: pd.DataFrame, symbols: Iterable[str]) -> pd.DataFrame:
    if df is None or df.empty:
        cols = pd.MultiIndex.from_product([_YF_FIELDS, []], names=[None, None])
        return pd.DataFrame(columns=cols)

    if not isinstance(df.columns, pd.MultiIndex):
        sym = list(symbols)[0]
        single = _normalize_single(df)
        single.columns = pd.MultiIndex.from_product([single.columns, [sym]])
        return single

    df = df.copy()
    # some yfinance versions return (symbol, field)
    names = df.columns.names
    if len(names) == 2 and names[0] and names[0].lower() in ("ticker", "symbol"):
        df = df.swaplevel(0, 1, axis=1)

    fields_present = [f for f in _YF_FIELDS if f in df.columns.get_level_values(0)]
    syms_present = [s for s in symbols if s in df.columns.get_level_values(1)]
    if not syms_present:
        cols = pd.MultiIndex.from_product([_YF_FIELDS, []], names=[None, None])
        return pd.DataFrame(columns=cols, index=df.index)

    df = df.loc[:, (fields_present, syms_present)].sort_index(axis=1)
    for f, s in df.columns:
        df[(f, s)] = pd.to_numeric(df[(f, s)], errors="coerce")
    df.index.name = "Date"
    return df


# ------------------------- cache helpers -------------------------

def _sym_cache_path(cache_dir: str | None, sym: str, kind: str = "ohlcv") -> str | None:
    if cache_dir is None:
        return None
    sub = "benchmarks" if kind == "bench" else "ohlcv"
    path = os.path.join(cache_dir, sub)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, f"{sym}.parquet")

def _load_cached_sym(cache_dir: str | None, sym: str, kind: str) -> pd.DataFrame | None:
    path = _sym_cache_path(cache_dir, sym, kind)
    if path is None or not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        return _ensure_date_index(df)
    except Exception:
        return None

def _save_cached_sym(cache_dir: str | None, sym: str, kind: str, df: pd.DataFrame) -> None:
    path = _sym_cache_path(cache_dir, sym, kind)
    if path is None:
        return
    df = _ensure_date_index(df)
    df.to_parquet(path)

def _slice_or_extend_cached(sym: str,
                            kind: str,
                            start: pd.Timestamp,
                            end: pd.Timestamp,
                            auto_adjust: bool,
                            cache_dir: str | None) -> pd.DataFrame:
    """
    Return OHLCV for [start,end) for one symbol, using cache if possible.
    If cache is missing or doesn't cover full range, download the missing part,
    merge, and rewrite cache.
    """
    cached = _load_cached_sym(cache_dir, sym, kind)
    need_download = True
    if cached is not None and not cached.empty:
        cov_start, cov_end = cached.index.min(), cached.index.max()
        # if fully covered, just slice
        if cov_start <= start and cov_end >= (end - pd.Timedelta(days=1)):
            return cached.loc[start:end]
        need_download = True

    # Download requested window
    df_new = yf.download(
        sym,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        group_by="column",
        progress=False,
        threads=True,
    )
    df_new = _normalize_single(df_new)
    if df_new is None:
        df_new = pd.DataFrame(columns=_YF_FIELDS)

    # Merge with cache if present
    if cached is not None and not cached.empty:
        merged = pd.concat([cached, df_new], axis=0)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = df_new

    # write back cache and return requested slice
    _save_cached_sym(cache_dir, sym, kind, merged)
    return merged.loc[start:end]


# ------------------------- public functions -------------------------

def download_global_price_data(stock_list: List[str],
                               data_start_date: str,
                               end_date: str,
                               *,
                               cache_dir: str | None = None,
                               auto_adjust: bool = False) -> pd.DataFrame:
    """
    Build a (Field, Symbol) panel from per-symbol cached OHLCV.
    """
    s, e = _safe_dates(data_start_date, end_date)
    parts = []
    for sym in stock_list:
        df = _slice_or_extend_cached(sym, "ohlcv", s, e, auto_adjust, cache_dir)
        if df is None or df.empty:
            continue
        # expand to (Field, Symbol)
        block = df.copy()
        block.columns = pd.MultiIndex.from_product([block.columns, [sym]])
        parts.append(block)
    if not parts:
        cols = pd.MultiIndex.from_product([_YF_FIELDS, []], names=[None, None])
        return pd.DataFrame(columns=cols)
    panel = pd.concat(parts, axis=1).sort_index(axis=1)
    panel.index.name = "Date"
    return panel


def download_raw_concat(stock_list: List[str],
                        decision_start_date: str,
                        end_date: str,
                        *,
                        cache_dir: str | None = None,
                        auto_adjust: bool = False) -> pd.DataFrame:
    """
    Decision-window OHLCV in LONG form, with per-symbol caching.
    """
    s, e = _safe_dates(decision_start_date, end_date)
    frames = []
    for sym in stock_list:
        try:
            df = _slice_or_extend_cached(sym, "ohlcv", s, e, auto_adjust, cache_dir)
            if df.empty:
                continue
            first_idx = df.index.min()
            if pd.notna(first_idx):
                print(f"{sym} start from {first_idx}")
            tmp = df.copy()
            tmp["Symbol"] = sym
            frames.append(tmp)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=_YF_FIELDS + ["Symbol"])
    raw = pd.concat(frames, axis=0)
    raw.index.name = "Date"
    return raw


def build_price_and_returns(raw_data: pd.DataFrame,
                            decision_start_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if raw_data is None or raw_data.empty:
        return pd.DataFrame(), pd.DataFrame()

    if "Adj Close" not in raw_data.columns or "Symbol" not in raw_data.columns:
        raise ValueError(f"raw_data missing required columns. Got: {list(raw_data.columns)}")

    piv = (raw_data.reset_index()
                    .pivot_table(index="Date", columns="Symbol", values="Adj Close", aggfunc="last")
                    .sort_index())
    piv = piv.loc[pd.to_datetime(decision_start_date):]
    rets = piv.pct_change().iloc[1:].copy()
    piv = piv.loc[rets.index]
    return piv, rets


def download_benchmarks(benchmarks: List[str],
                        start_index: pd.Timestamp,
                        end_index: pd.Timestamp,
                        *,
                        cache_dir: str | None = None,
                        auto_adjust: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    s, e = _safe_dates(start_index, end_index)
    frames = []
    for b in benchmarks:
        try:
            df = _slice_or_extend_cached(b, "bench", s, e, auto_adjust, cache_dir)
            if df.empty:
                continue
            tmp = df.copy()
            tmp["Symbol"] = b
            frames.append(tmp)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    raw_b = pd.concat(frames, axis=0)
    raw_b.index.name = "Date"

    piv = (raw_b.reset_index()
                 .pivot_table(index="Date", columns="Symbol", values="Adj Close", aggfunc="last")
                 .sort_index())

    rets = piv.pct_change().iloc[1:]
    piv = piv.loc[rets.index]
    return piv, rets


# ------------------------- one-shot installer -------------------------

def install_dataset(stock_list: List[str],
                    benchmarks: List[str],
                    data_start_date: str,
                    decision_start_date: str,
                    end_date: str,
                    *,
                    cache_dir: str | None = None,
                    auto_adjust: bool = False):
    """
    Returns:
      - global_price_data: (Field, Symbol) OHLCV panel over data window
      - price_data: wide Adj Close (decision window)
      - returns_data: daily returns aligned to price_data
      - benchmark_data: wide Adj Close for benchmarks
      - benchmark_returns_data: daily returns for benchmarks
    """
    global_price_data = download_global_price_data(
        stock_list, data_start_date, end_date, cache_dir=cache_dir, auto_adjust=auto_adjust
    )

    raw_data = download_raw_concat(
        stock_list, decision_start_date, end_date, cache_dir=cache_dir, auto_adjust=auto_adjust
    )

    price_data, returns_data = build_price_and_returns(raw_data, decision_start_date)

    if not price_data.empty:
        bench_start = price_data.index.min()
        bench_end = price_data.index.max()
    else:
        bench_start = pd.to_datetime(decision_start_date)
        bench_end = pd.to_datetime(end_date)

    benchmark_data, benchmark_returns_data = download_benchmarks(
        benchmarks, bench_start, bench_end, cache_dir=cache_dir, auto_adjust=auto_adjust
    )

    return (
        global_price_data,
        price_data,
        returns_data,
        benchmark_data,
        benchmark_returns_data,
    )
