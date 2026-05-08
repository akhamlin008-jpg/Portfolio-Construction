"""
Portfolio Construction Model — Streamlit App
=============================================

Builds a target portfolio for one of three investor profiles
(Conservative, Hybrid, Growth) given:
  - Risk tolerance score (1-10)
  - Monthly contribution budget (quarterly cap = monthly * 3)

Universe sources (pulled live from GitHub):
  - echuvyrov/TrackingETFs   -> curated ETF universe (bonds + growth)
  - Ate329/top-us-stock-tickers -> daily-updated S&P 500 constituents

Pricing:
  - yfinance for historical adjusted closes

Optimization:
  - Bucket-level allocation: hand-rolled (transparent risk -> weight mapping)
  - Within-bucket allocation: PyPortfolioOpt max-Sharpe with bounds
"""

from __future__ import annotations

import io
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, expected_returns, risk_models

# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Portfolio Construction Model",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Typography — editorial / refined feel */
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Fraunces', serif;
    }
    h1, h2, h3 {
        font-family: 'Fraunces', serif;
        letter-spacing: -0.02em;
    }
    h1 { font-weight: 800; }
    code, pre, .stCode {
        font-family: 'JetBrains Mono', monospace !important;
    }
    /* Tighter metric blocks */
    [data-testid="stMetricValue"] {
        font-family: 'Fraunces', serif;
        font-weight: 600;
    }
    /* Sidebar accent */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(0,0,0,0.08);
    }
    .small-caption {
        font-size: 0.85rem;
        color: #6b6b6b;
        font-style: italic;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants — risk-profile bucket weights & ticker universe definitions
# ---------------------------------------------------------------------------

# GitHub raw-CSV endpoints
ETF_UNIVERSE_URL = (
    "https://raw.githubusercontent.com/echuvyrov/TrackingETFs/master/etfs.csv"
)
SP500_URL = (
    "https://raw.githubusercontent.com/Ate329/top-us-stock-tickers/main/tickers/sp500.csv"
)

# Sub-buckets within each broad asset class
# (Carved up by hand using the TrackingETFs CSV — these are the tickers
# that map cleanly to each risk role.)
CONSERVATIVE_TICKERS = [
    "SHY",   # 1-3 Year Treasury
    "AGG",   # Aggregate Bond
    "BND",   # Total Bond Market
    "TIP",   # TIPS (inflation-protected)
    "LQD",   # Investment Grade Corporate
    "CSJ",   # 1-3 Year Credit
    "MBB",   # Mortgage-Backed
]

HYBRID_TICKERS = [
    "HYG",   # High Yield Corporate Bond
    "JNK",   # SPDR High Yield
    "EMB",   # Emerging Markets Bond
    "VIG",   # Dividend Appreciation
    "SDY",   # SPDR Dividend
    "DVY",   # Dow Jones Select Dividend
    "VNQ",   # REIT
    "PFF",   # Preferred Stock
]

GROWTH_TICKERS = [
    "VUG",   # Vanguard Growth
    "IVW",   # S&P 500 Growth
    "IWF",   # Russell 1000 Growth
    "IWO",   # Russell 2000 Growth
    "VTI",   # Total Stock Market
    "QQQ",   # Nasdaq-100 (added — not in CSV but a staple)
    "XLK",   # Technology Select
    "XLV",   # Health Care Select
]

# Bucket weight schedule by risk score (1 = ultra-conservative, 10 = aggressive)
# Hand-rolled — this is the part you want full transparency on.
def bucket_weights_for_risk(risk: int) -> dict[str, float]:
    """Map a 1-10 risk score to weights across the three buckets."""
    risk = max(1, min(10, int(risk)))
    # Linear interpolation between the endpoints
    # risk=1  -> 90% conservative, 10% hybrid,  0% growth
    # risk=5  -> 40% conservative, 35% hybrid, 25% growth
    # risk=10 ->  5% conservative, 20% hybrid, 75% growth
    if risk <= 5:
        t = (risk - 1) / 4  # 0 -> 1
        cons = 0.90 + t * (0.40 - 0.90)
        hyb = 0.10 + t * (0.35 - 0.10)
        grw = 0.00 + t * (0.25 - 0.00)
    else:
        t = (risk - 5) / 5  # 0 -> 1
        cons = 0.40 + t * (0.05 - 0.40)
        hyb = 0.35 + t * (0.20 - 0.35)
        grw = 0.25 + t * (0.75 - 0.25)
    # Normalize to handle float drift
    total = cons + hyb + grw
    return {
        "Conservative": cons / total,
        "Hybrid": hyb / total,
        "Growth": grw / total,
    }


# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_etf_universe() -> pd.DataFrame:
    """Pull the curated ETF list from echuvyrov/TrackingETFs."""
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
    """Pull the daily-updated S&P 500 constituents from Ate329."""
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
    """Download adjusted close prices via yfinance. Tuple input for cache hashability."""
    if not tickers:
        return pd.DataFrame()
    raw = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    # Normalize: we want a (date x ticker) frame of close prices
    if isinstance(raw.columns, pd.MultiIndex):
        closes = pd.DataFrame({t: raw[t]["Close"] for t in tickers if t in raw.columns.levels[0]})
    else:
        # Single ticker case
        closes = pd.DataFrame({tickers[0]: raw["Close"]})
    closes = closes.dropna(how="all").ffill().dropna()
    return closes


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

def optimize_bucket(prices: pd.DataFrame, max_weight: float = 0.40) -> dict[str, float]:
    """
    Run max-Sharpe optimization on a bucket's prices.
    Returns dict of {ticker: weight} summing to 1.

    Uses PyPortfolioOpt (library-based) for the heavy lifting because the
    bucketing logic upstream is where our differentiation lives.
    """
    if prices.shape[1] < 2:
        # Single asset → 100% allocation
        return {prices.columns[0]: 1.0} if prices.shape[1] == 1 else {}

    try:
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
        ef.max_sharpe(risk_free_rate=0.04)
        weights = ef.clean_weights()
        # Drop zero-weights
        return {k: v for k, v in weights.items() if v > 1e-4}
    except Exception:
        # Fallback: equal weight if optimizer fails (e.g. singular cov matrix)
        n = prices.shape[1]
        return {col: 1 / n for col in prices.columns}


def build_portfolio(
    risk_score: int,
    quarterly_budget: float,
    lookback_years: int,
) -> dict:
    """
    Top-level builder: bucket weights -> within-bucket optimization ->
    dollar allocation -> share counts.
    """
    end = date.today()
    start = end - timedelta(days=int(lookback_years * 365.25))

    bucket_w = bucket_weights_for_risk(risk_score)

    bucket_tickers = {
        "Conservative": CONSERVATIVE_TICKERS,
        "Hybrid": HYBRID_TICKERS,
        "Growth": GROWTH_TICKERS,
    }

    all_tickers = tuple(sorted({t for lst in bucket_tickers.values() for t in lst}))
    with st.spinner(f"Pulling price history for {len(all_tickers)} tickers via yfinance..."):
        prices = fetch_prices(all_tickers, start, end)

    if prices.empty:
        st.error("Could not retrieve price data. Check your network connection.")
        return {}

    # Latest prices for share-count calculation
    latest_prices = prices.iloc[-1]

    rows = []
    bucket_summary = []

    for bucket, weight in bucket_w.items():
        tickers_in_bucket = [t for t in bucket_tickers[bucket] if t in prices.columns]
        if not tickers_in_bucket:
            continue
        bucket_prices = prices[tickers_in_bucket]
        within_weights = optimize_bucket(bucket_prices)

        bucket_dollars = quarterly_budget * weight
        bucket_summary.append({
            "Bucket": bucket,
            "Target Weight": f"{weight:.1%}",
            "Allocation ($)": f"${bucket_dollars:,.2f}",
            "# Holdings": len(within_weights),
        })

        for ticker, w_within in within_weights.items():
            target_weight = weight * w_within
            target_dollars = quarterly_budget * target_weight
            price = float(latest_prices[ticker])
            shares = int(np.floor(target_dollars / price)) if price > 0 else 0
            actual_dollars = shares * price
            rows.append({
                "Bucket": bucket,
                "Ticker": ticker,
                "Latest Price": price,
                "Target Weight": target_weight,
                "Within-Bucket Wt": w_within,
                "Target $": target_dollars,
                "Whole Shares": shares,
                "Actual $": actual_dollars,
            })

    allocation_df = pd.DataFrame(rows).sort_values(
        ["Bucket", "Target Weight"], ascending=[True, False]
    ).reset_index(drop=True)

    summary_df = pd.DataFrame(bucket_summary)

    return {
        "bucket_weights": bucket_w,
        "summary": summary_df,
        "allocation": allocation_df,
        "prices": prices,
        "total_invested": allocation_df["Actual $"].sum() if not allocation_df.empty else 0,
        "cash_remaining": quarterly_budget - (allocation_df["Actual $"].sum() if not allocation_df.empty else 0),
    }


# ---------------------------------------------------------------------------
# UI — Sidebar inputs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ◆ Inputs")

    monthly_budget = st.number_input(
        "Monthly contribution budget ($)",
        min_value=100.0,
        max_value=100_000.0,
        value=1_000.0,
        step=100.0,
        help="Quarterly allocation cap = monthly × 3",
    )
    quarterly_budget = monthly_budget * 3
    st.caption(f"Quarterly cap: **${quarterly_budget:,.2f}**")

    st.markdown("---")

    risk_score = st.slider(
        "Risk tolerance",
        min_value=1, max_value=10, value=5, step=1,
        help="1 = ultra-conservative (capital preservation) · 10 = aggressive growth",
    )

    # Show the resulting profile label
    if risk_score <= 3:
        profile = "Conservative"
    elif risk_score <= 7:
        profile = "Hybrid"
    else:
        profile = "Growth"
    st.markdown(f"**Profile:** {profile}")

    st.markdown("---")

    lookback_years = st.select_slider(
        "Historical lookback for optimization",
        options=[1, 2, 3, 5, 7, 10],
        value=3,
        help="Years of price history used to estimate returns & covariance",
    )

    st.markdown("---")
    run_button = st.button("Build Portfolio", type="primary", use_container_width=True)

    with st.expander("Data sources"):
        st.markdown(
            """
            - **ETF universe** — [echuvyrov/TrackingETFs](https://github.com/echuvyrov/TrackingETFs)
            - **S&P 500 list** — [Ate329/top-us-stock-tickers](https://github.com/Ate329/top-us-stock-tickers)
            - **Prices** — yfinance (Yahoo Finance)
            """
        )


# ---------------------------------------------------------------------------
# UI — Main panel
# ---------------------------------------------------------------------------

st.title("Portfolio Construction Model")
st.markdown(
    "<p class='small-caption'>Risk-bucketed allocation across conservative bonds, "
    "income/dividend hybrids, and growth equities — sized to a quarterly contribution.</p>",
    unsafe_allow_html=True,
)

if not run_button:
    # Pre-run preview — show the universe so the user sees what's available
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Bucket weights at your risk score")
        preview_w = bucket_weights_for_risk(risk_score)
        preview_df = pd.DataFrame({
            "Bucket": list(preview_w.keys()),
            "Weight": [f"{v:.1%}" for v in preview_w.values()],
            "Dollar Target": [f"${quarterly_budget * v:,.2f}" for v in preview_w.values()],
        })
        st.dataframe(preview_df, hide_index=True, use_container_width=True)

        st.subheader("Risk → Allocation curve")
        curve_data = pd.DataFrame(
            [bucket_weights_for_risk(r) for r in range(1, 11)],
            index=range(1, 11),
        )
        st.area_chart(curve_data, height=240)

    with col2:
        st.subheader("Universe preview")
        with st.spinner("Fetching ticker lists from GitHub..."):
            etf_df = load_etf_universe()
            sp500_df = load_sp500_universe()
        c1, c2 = st.columns(2)
        c1.metric("ETF universe", len(etf_df))
        c2.metric("S&P 500 constituents", len(sp500_df))

        tab_cons, tab_hyb, tab_grw = st.tabs(["Conservative", "Hybrid", "Growth"])
        for tab, tickers in zip(
            [tab_cons, tab_hyb, tab_grw],
            [CONSERVATIVE_TICKERS, HYBRID_TICKERS, GROWTH_TICKERS],
        ):
            with tab:
                bucket_view = etf_df[etf_df["ticker"].isin(tickers)]
                # Add any missing ones (e.g. QQQ) with placeholder names
                missing = [t for t in tickers if t not in bucket_view["ticker"].values]
                if missing:
                    bucket_view = pd.concat([
                        bucket_view,
                        pd.DataFrame({"ticker": missing, "name": ["—"] * len(missing)}),
                    ], ignore_index=True)
                st.dataframe(bucket_view, hide_index=True, use_container_width=True)

    st.info("Adjust inputs in the sidebar, then click **Build Portfolio**.")
    st.stop()


# ---------------------------------------------------------------------------
# Run the model
# ---------------------------------------------------------------------------

result = build_portfolio(risk_score, quarterly_budget, lookback_years)

if not result:
    st.stop()

# Top-line metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Quarterly budget", f"${quarterly_budget:,.0f}")
m2.metric("Invested (whole shares)", f"${result['total_invested']:,.2f}")
m3.metric("Cash remaining", f"${result['cash_remaining']:,.2f}")
m4.metric("Profile", profile)

st.markdown("---")

# Bucket summary
st.subheader("Bucket allocation")
st.dataframe(result["summary"], hide_index=True, use_container_width=True)

# Allocation table
st.subheader("Holdings")
display_df = result["allocation"].copy()
display_df["Latest Price"] = display_df["Latest Price"].apply(lambda x: f"${x:,.2f}")
display_df["Target Weight"] = display_df["Target Weight"].apply(lambda x: f"{x:.2%}")
display_df["Within-Bucket Wt"] = display_df["Within-Bucket Wt"].apply(lambda x: f"{x:.2%}")
display_df["Target $"] = display_df["Target $"].apply(lambda x: f"${x:,.2f}")
display_df["Actual $"] = display_df["Actual $"].apply(lambda x: f"${x:,.2f}")
st.dataframe(display_df, hide_index=True, use_container_width=True)

# Visualizations
st.markdown("---")
v1, v2 = st.columns([1, 1])

with v1:
    st.subheader("Allocation by ticker")
    chart_df = result["allocation"].set_index("Ticker")[["Actual $"]]
    st.bar_chart(chart_df, height=360)

with v2:
    st.subheader("Allocation by bucket")
    bucket_chart = result["allocation"].groupby("Bucket")["Actual $"].sum()
    st.bar_chart(bucket_chart, height=360)

# Historical performance preview
st.markdown("---")
st.subheader("Historical performance — equal-weight bucket benchmark")
prices_norm = result["prices"] / result["prices"].iloc[0]
bucket_perf = pd.DataFrame()
for bucket, tickers in [
    ("Conservative", CONSERVATIVE_TICKERS),
    ("Hybrid", HYBRID_TICKERS),
    ("Growth", GROWTH_TICKERS),
]:
    cols = [t for t in tickers if t in prices_norm.columns]
    if cols:
        bucket_perf[bucket] = prices_norm[cols].mean(axis=1)
st.line_chart(bucket_perf, height=360)

st.caption(
    "Note: This is a model output for demonstration. Not investment advice. "
    "Run quarterly; reassess risk score if life circumstances change."
)

# Download button for the allocation
csv_bytes = result["allocation"].to_csv(index=False).encode("utf-8")
st.download_button(
    "Download allocation as CSV",
    data=csv_bytes,
    file_name=f"allocation_risk{risk_score}_q{date.today().isoformat()}.csv",
    mime="text/csv",
)
