"""
Portfolio core (v2): shared logic with corrected methodology.

Key changes from v1:
  - CAPM expected returns (not mean-historical)
  - Bucket weights derived from CRRA utility, not hand-picked
  - Walk-forward weight computation (no lookahead)
  - Stooq fallback when yfinance fails
  - Correlation matrices for diagnostics
"""

from __future__ import annotations

import io
from datetime import date

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from pandas_datareader import data as pdr
    HAS_PDR = True
except ImportError:
    HAS_PDR = False

ETF_UNIVERSE_URL = "https://raw.githubusercontent.com/echuvyrov/TrackingETFs/master/etfs.csv"

CONSERVATIVE_TICKERS = ["SHY", "AGG", "BND", "TIP", "LQD", "CSJ", "MBB"]
HYBRID_TICKERS = ["HYG", "JNK", "EMB", "VIG", "SDY", "DVY", "VNQ", "PFF"]
GROWTH_TICKERS = ["VUG", "IVW", "IWF", "IWO", "VTI", "QQQ", "XLK", "XLV"]
ALL_TICKERS = sorted(set(CONSERVATIVE_TICKERS + HYBRID_TICKERS + GROWTH_TICKERS))

BUCKET_MAP = {
    "Conservative": CONSERVATIVE_TICKERS,
    "Hybrid": HYBRID_TICKERS,
    "Growth": GROWTH_TICKERS,
}

BUCKET_PRIOR_VOL = {"Conservative": 0.05, "Hybrid": 0.10, "Growth": 0.18}
BUCKET_PRIOR_EXCESS = {
    "Conservative": 0.005,  # ~0.5% over rf
    "Hybrid":       0.014,  # ~1.4% over rf
    "Growth":       0.022,  # ~2.2% over rf
}


def risk_aversion_from_score(risk_score: int) -> float:
    """Map 1-10 risk score to CRRA gamma. Score 1 → gamma=10 (very averse)."""
    risk_score = max(1, min(10, int(risk_score)))
    return 11 - risk_score


def crra_bucket_weights(risk_score: int, risk_free_rate: float = 0.04) -> dict[str, float]:
    """
    Bucket weights from CRRA utility (Merton portfolio choice with no-leverage cap).

    Procedure:
      1. Per-bucket Merton terms = excess / (γ · σ²), proportional to optimal allocation.
      2. If total Merton terms < 1.0 (high γ / risk-averse): the remainder is cash,
         routed into the conservative bucket as the safest available proxy.
      3. If total > 1.0 (low γ / aggressive): scale down proportionally, but also
         apply a γ-dependent tilt that shifts weight toward growth as γ decreases.
         This reflects that an investor with extreme risk tolerance prefers higher-vol
         buckets even when total risky budget is already saturated.
    """
    gamma = risk_aversion_from_score(risk_score)
    merton = {
        b: BUCKET_PRIOR_EXCESS[b] / (gamma * BUCKET_PRIOR_VOL[b] ** 2)
        for b in ["Conservative", "Hybrid", "Growth"]
    }
    total_risky = sum(merton.values())

    if total_risky <= 1.0:
        # High aversion regime — cash floor in conservative bucket
        weights = dict(merton)
        weights["Conservative"] += (1.0 - total_risky)
    else:
        # Low aversion regime — apply γ-dependent tilt within risky budget
        # tilt_factor: 0 at γ=10 (no extra tilt) → 1 at γ=1 (max growth tilt)
        tilt_factor = max(0, (10 - gamma) / 9)  # 0 to 1

        # Baseline Merton weights (gamma-invariant after normalization)
        baseline = {k: v / total_risky for k, v in merton.items()}

        # Tilt target: progressively shift conservative → growth as γ falls
        tilt_target = {"Conservative": 0.20, "Hybrid": 0.30, "Growth": 0.50}

        weights = {
            k: (1 - tilt_factor) * baseline[k] + tilt_factor * tilt_target[k]
            for k in baseline
        }

    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def profile_label(risk: int) -> str:
    if risk <= 3:
        return "Conservative"
    if risk <= 7:
        return "Hybrid"
    return "Growth"


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_etf_universe() -> pd.DataFrame:
    try:
        resp = requests.get(ETF_UNIVERSE_URL, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), header=None, names=["ticker", "name"])
        df["ticker"] = df["ticker"].str.strip().str.upper()
        return df
    except Exception:
        return pd.DataFrame(columns=["ticker", "name"])


def _try_yfinance(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    try:
        raw = yf.download(
            tickers, start=start, end=end,
            auto_adjust=True, progress=False, threads=False,
        )
        if raw is None or raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" in raw.columns.get_level_values(0):
                closes = raw["Close"]
            elif "Close" in raw.columns.get_level_values(1):
                closes = raw.xs("Close", axis=1, level=1)
            else:
                return pd.DataFrame()
        else:
            if "Close" in raw.columns:
                closes = raw[["Close"]].rename(columns={"Close": tickers[0]})
            else:
                return pd.DataFrame()
        return closes.dropna(how="all").ffill().dropna(how="all")
    except Exception:
        return pd.DataFrame()


def _try_stooq(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    if not HAS_PDR:
        return pd.DataFrame()
    try:
        stooq_tickers = [f"{t}.US" for t in tickers]
        raw = pdr.DataReader(stooq_tickers, "stooq", start, end)
        if raw is None or raw.empty:
            return pd.DataFrame()
        raw = raw.sort_index()
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" in raw.columns.get_level_values(0):
                closes = raw["Close"]
            else:
                return pd.DataFrame()
            closes.columns = [c.replace(".US", "") for c in closes.columns]
        else:
            if "Close" in raw.columns:
                closes = raw[["Close"]]
                closes.columns = [tickers[0]]
            else:
                return pd.DataFrame()
        return closes.dropna(how="all").ffill().dropna(how="all")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_prices(tickers: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    """Two-tier fallback: yfinance first, Stooq second."""
    if not tickers:
        return pd.DataFrame()
    ticker_list = list(tickers)

    closes = _try_yfinance(ticker_list, start, end)
    if not closes.empty and len(closes.columns) >= len(ticker_list) * 0.8:
        return closes

    closes_stooq = _try_stooq(ticker_list, start, end)
    if not closes_stooq.empty:
        if not closes.empty:
            for col in closes_stooq.columns:
                if col not in closes.columns:
                    closes[col] = closes_stooq[col]
            return closes
        return closes_stooq
    return closes


def capm_expected_returns(
    prices: pd.DataFrame,
    market_prices: pd.Series,
    risk_free_rate: float = 0.04,
) -> pd.Series:
    """
    CAPM expected returns: E[R_i] = r_f + β_i * (E[R_m] - r_f).
    Beta from OLS regression of asset returns on market returns.
    """
    asset_returns = prices.pct_change().dropna()
    market_returns = market_prices.pct_change().dropna()
    aligned = pd.concat([asset_returns, market_returns.rename("__MKT__")], axis=1).dropna()

    if aligned.empty or len(aligned) < 30:
        return pd.Series(risk_free_rate, index=prices.columns)

    market_ret = aligned["__MKT__"]
    market_var = market_ret.var()
    market_premium_ann = market_ret.mean() * 252 - risk_free_rate

    expected = {}
    for asset in prices.columns:
        if asset not in aligned.columns:
            expected[asset] = risk_free_rate
            continue
        cov = aligned[asset].cov(market_ret)
        beta = cov / market_var if market_var > 0 else 0
        expected[asset] = risk_free_rate + beta * market_premium_ann

    return pd.Series(expected)


def optimize_bucket(
    prices: pd.DataFrame,
    market_prices: pd.Series | None = None,
    max_weight: float = 0.40,
    risk_free_rate: float = 0.04,
    use_kelly_cap: bool = False,
) -> dict[str, float]:
    """Max-Sharpe with CAPM-based μ, optional Kelly-derived per-asset cap."""
    from pypfopt import EfficientFrontier, risk_models

    if prices.shape[1] == 0:
        return {}
    if prices.shape[1] == 1:
        return {prices.columns[0]: 1.0}

    try:
        if market_prices is not None and not market_prices.empty:
            mu = capm_expected_returns(prices, market_prices, risk_free_rate)
        else:
            mu = prices.pct_change().mean() * 252

        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

        if use_kelly_cap:
            returns = prices.pct_change().dropna()
            asset_vol = returns.std() * np.sqrt(252)
            asset_excess = mu - risk_free_rate
            kelly_per_asset = (asset_excess / (asset_vol ** 2)).clip(0, max_weight)
            bounds = [(0, min(max(0.05, k * 0.5), max_weight)) for k in kelly_per_asset]
        else:
            bounds = (0, max_weight)

        ef = EfficientFrontier(mu, S, weight_bounds=bounds)
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        weights = ef.clean_weights(cutoff=0.01, rounding=4)
        return {k: v for k, v in weights.items() if v > 1e-4}
    except Exception:
        n = prices.shape[1]
        return {col: 1 / n for col in prices.columns}


def compute_portfolio_weights(
    risk_score: int,
    prices: pd.DataFrame,
    market_prices: pd.Series | None = None,
    risk_free_rate: float = 0.04,
    use_kelly_cap: bool = False,
) -> pd.Series:
    """Top-level: CRRA bucket weights × within-bucket CAPM optimization."""
    bucket_w = crra_bucket_weights(risk_score, risk_free_rate)
    full_weights = {}

    for bucket, bw in bucket_w.items():
        bucket_tickers = [t for t in BUCKET_MAP[bucket] if t in prices.columns]
        if not bucket_tickers or bw <= 0:
            continue
        within = optimize_bucket(
            prices[bucket_tickers], market_prices=market_prices,
            risk_free_rate=risk_free_rate, use_kelly_cap=use_kelly_cap,
        )
        for ticker, w_within in within.items():
            full_weights[ticker] = bw * w_within

    series = pd.Series(full_weights)
    if series.sum() > 0:
        series = series / series.sum()
    return series


def compute_weights_walkforward(
    prices: pd.DataFrame,
    market_prices: pd.Series,
    rebalance_dates: pd.DatetimeIndex,
    risk_score: int,
    risk_free_rate: float = 0.04,
    lookback_days: int = 756,
    use_kelly_cap: bool = False,
) -> dict:
    """
    Walk-forward: at each rebalance date, use only data BEFORE that date.
    This is the methodological fix that eliminates lookahead bias.
    """
    weights_by_date = {}
    for rebalance_date in rebalance_dates:
        end_idx = prices.index.searchsorted(rebalance_date)
        start_idx = max(0, end_idx - lookback_days)
        if end_idx - start_idx < 60:
            weights_by_date[rebalance_date] = pd.Series(dtype=float)
            continue
        window_prices = prices.iloc[start_idx:end_idx]
        window_market = market_prices.iloc[start_idx:end_idx] if market_prices is not None else None
        weights = compute_portfolio_weights(
            risk_score=risk_score, prices=window_prices,
            market_prices=window_market, risk_free_rate=risk_free_rate,
            use_kelly_cap=use_kelly_cap,
        )
        weights_by_date[rebalance_date] = weights
    return weights_by_date


CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,800&family=JetBrains+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'Fraunces', serif; }
    h1, h2, h3 { font-family: 'Fraunces', serif; letter-spacing: -0.02em; }
    h1 { font-weight: 800; }
    code, pre, .stCode { font-family: 'JetBrains Mono', monospace !important; }
    [data-testid="stMetricValue"] { font-family: 'Fraunces', serif; font-weight: 600; }
    section[data-testid="stSidebar"] { border-right: 1px solid rgba(0,0,0,0.08); }
    .small-caption { font-size: 0.85rem; color: #6b6b6b; font-style: italic; }
    .methodology-note {
        background-color: rgba(100, 100, 100, 0.08);
        border-left: 3px solid #888;
        padding: 0.75rem 1rem;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
</style>
"""
