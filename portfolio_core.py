"""
Shared constants, ticker buckets, and helper functions used across
the Streamlit pages. Kept in a single module so risk logic stays consistent
between the portfolio builder and the backtester.
"""

from __future__ import annotations

import io
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------------------------
# GitHub raw-CSV endpoints
# ---------------------------------------------------------------------------
ETF_UNIVERSE_URL = (
    "https://raw.githubusercontent.com/echuvyrov/TrackingETFs/master/etfs.csv"
)
SP500_URL = (
    "https://raw.githubusercontent.com/Ate329/top-us-stock-tickers/main/tickers/sp500.csv"
)

# ---------------------------------------------------------------------------
# Bucket ticker definitions
# ---------------------------------------------------------------------------
CONSERVATIVE_TICKERS = [
    "SHY", "AGG", "BND", "TIP", "LQD", "CSJ", "MBB",
]

HYBRID_TICKERS = [
    "HYG", "JNK", "EMB", "VIG", "SDY", "DVY", "VNQ", "PFF",
]

GROWTH_TICKERS = [
    "VUG", "IVW", "IWF", "IWO", "VTI", "QQQ", "XLK", "XLV",
]

ALL_TICKERS = sorted(set(CONSERVATIVE_TICKERS + HYBRID_TICKERS + GROWTH_TICKERS))

BUCKET_MAP = {
    "Conservative": CONSERVATIVE_TICKERS,
    "Hybrid": HYBRID_TICKERS,
    "Growth": GROWTH_TICKERS,
}

# ---------------------------------------------------------------------------
# Risk -> bucket weight curve (hand-rolled for transparency)
# ---------------------------------------------------------------------------
def bucket_weights_for_risk(risk: int) -> dict[str, float]:
    """Map a 1-10 risk score to weights across the three buckets."""
    risk = max(1, min(10, int(risk)))
    if risk <= 5:
        t = (risk - 1) / 4
        cons = 0.90 + t * (0.40 - 0.90)
        hyb = 0.10 + t * (0.35 - 0.10)
        grw = 0.00 + t * (0.25 - 0.00)
    else:
        t = (risk - 5) / 5
        cons = 0.40 + t * (0.05 - 0.40)
        hyb = 0.35 + t * (0.20 - 0.35)
        grw = 0.25 + t * (0.75 - 0.25)
    total = cons + hyb + grw
    return {
        "Conservative": cons / total,
        "Hybrid": hyb / total,
        "Growth": grw / total,
    }


def profile_label(risk: int) -> str:
    if risk <= 3:
        return "Conservative"
    if risk <= 7:
        return "Hybrid"
    return "Growth"


# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_etf_universe() -> pd.DataFrame:
    try:
        resp = requests.get(ETF_UNIVERSE_URL, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), header=None,
                         names=["ticker", "name"])
        df["ticker"] = df["ticker"].str.strip().str.upper()
        return df
    except Exception as e:
        st.warning(f"Could not load ETF universe from GitHub: {e}")
        return pd.DataFrame(columns=["ticker", "name"])


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_sp500_universe() -> pd.DataFrame:
    try:
        df = pd.read_csv(SP500_URL)
        df.columns = [c.lower() for c in df.columns]
        if "symbol" in df.columns:
            df = df.rename(columns={"symbol": "ticker"})
        df["ticker"] = df["ticker"].str.replace("/", "-", regex=False)
        return df
    except Exception as e:
        st.warning(f"Could not load S&P 500 list from GitHub: {e}")
        return pd.DataFrame(columns=["ticker"])


@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_prices(tickers: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    """Download adjusted close prices via yfinance, with retry and diagnostics."""
    if not tickers:
        return pd.DataFrame()

    try:
        raw = yf.download(
            list(tickers),
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,  # threads=True can cause issues on Streamlit Cloud
        )
    except Exception as e:
        st.error(f"yfinance download failed: {e}")
        return pd.DataFrame()

    if raw is None or raw.empty:
        st.error(
            f"yfinance returned no data for {len(tickers)} tickers. "
            "This is usually a temporary Yahoo Finance rate limit on Streamlit Cloud. "
            "Try refreshing in a minute, or test locally first."
        )
        return pd.DataFrame()

    # Normalize: extract Close prices regardless of column structure
    if isinstance(raw.columns, pd.MultiIndex):
        # Multi-ticker download — columns are (field, ticker) by default in newer yfinance
        if "Close" in raw.columns.get_level_values(0):
            closes = raw["Close"]
        elif "Close" in raw.columns.get_level_values(1):
            closes = raw.xs("Close", axis=1, level=1)
        else:
            st.error(f"Unexpected yfinance column structure: {raw.columns}")
            return pd.DataFrame()
    else:
        # Single ticker
        if "Close" in raw.columns:
            closes = raw[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            return pd.DataFrame()

    closes = closes.dropna(how="all").ffill().dropna(how="all")
    if closes.empty:
        st.error("Price data downloaded but contained no valid rows after cleaning.")
        return pd.DataFrame()

    return closes


# ---------------------------------------------------------------------------
# Optimization (re-used by builder and backtester)
# ---------------------------------------------------------------------------
def optimize_bucket(prices: pd.DataFrame, max_weight: float = 0.40,
                    risk_free_rate: float = 0.04) -> dict[str, float]:
    """Max-Sharpe optimization within a bucket. Falls back to equal-weight on failure."""
    from pypfopt import EfficientFrontier, expected_returns, risk_models

    if prices.shape[1] == 0:
        return {}
    if prices.shape[1] == 1:
        return {prices.columns[0]: 1.0}
    try:
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        weights = ef.clean_weights()
        return {k: v for k, v in weights.items() if v > 1e-4}
    except Exception:
        n = prices.shape[1]
        return {col: 1 / n for col in prices.columns}


def compute_portfolio_weights(
    risk_score: int,
    prices: pd.DataFrame,
    risk_free_rate: float = 0.04,
) -> pd.Series:
    """
    Compute the full target weight vector across all tickers given a risk score
    and the price history. Returns a Series indexed by ticker, summing to 1.
    """
    bucket_w = bucket_weights_for_risk(risk_score)
    full_weights = {}

    for bucket, bw in bucket_w.items():
        bucket_tickers = [t for t in BUCKET_MAP[bucket] if t in prices.columns]
        if not bucket_tickers:
            continue
        within = optimize_bucket(prices[bucket_tickers], risk_free_rate=risk_free_rate)
        for ticker, w_within in within.items():
            full_weights[ticker] = bw * w_within

    series = pd.Series(full_weights)
    if series.sum() > 0:
        series = series / series.sum()  # normalize for missing tickers
    return series


# ---------------------------------------------------------------------------
# Page styling shared across pages
# ---------------------------------------------------------------------------
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
</style>
"""
