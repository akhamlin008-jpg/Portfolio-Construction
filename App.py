"""
Portfolio Construction Model — Streamlit App (Page 1: Builder)
==============================================================

Builds a target portfolio for one of three investor profiles
(Conservative, Hybrid, Growth) given:
  - Risk tolerance score (1-10)
  - Monthly contribution budget (quarterly cap = monthly * 3)

See the Backtest & Analysis page (sidebar) for performance testing,
Monte Carlo, Kelly criterion, and sensitivity analysis.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from portfolio_core import (
    ALL_TICKERS,
    BUCKET_MAP,
    CONSERVATIVE_TICKERS,
    CUSTOM_CSS,
    GROWTH_TICKERS,
    HYBRID_TICKERS,
    bucket_weights_for_risk,
    fetch_prices,
    load_etf_universe,
    load_sp500_universe,
    optimize_bucket,
    profile_label,
)

st.set_page_config(
    page_title="Portfolio Construction Model",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def build_portfolio(
    risk_score: int,
    quarterly_budget: float,
    lookback_years: int,
    risk_free_rate: float,
) -> dict:
    end = date.today()
    start = end - timedelta(days=int(lookback_years * 365.25))
    bucket_w = bucket_weights_for_risk(risk_score)

    with st.spinner(f"Pulling price history for {len(ALL_TICKERS)} tickers via yfinance..."):
        prices = fetch_prices(tuple(ALL_TICKERS), start, end)

    if prices.empty:
        st.error("Could not retrieve price data. Check your network connection.")
        return {}

    latest_prices = prices.iloc[-1]
    rows, bucket_summary = [], []

    for bucket, weight in bucket_w.items():
        tickers_in_bucket = [t for t in BUCKET_MAP[bucket] if t in prices.columns]
        if not tickers_in_bucket:
            continue
        within = optimize_bucket(prices[tickers_in_bucket], risk_free_rate=risk_free_rate)
        bucket_dollars = quarterly_budget * weight
        bucket_summary.append({
            "Bucket": bucket,
            "Target Weight": f"{weight:.1%}",
            "Allocation ($)": f"${bucket_dollars:,.2f}",
            "# Holdings": len(within),
        })
        for ticker, w_within in within.items():
            target_weight = weight * w_within
            target_dollars = quarterly_budget * target_weight
            price = float(latest_prices[ticker])
            shares = int(np.floor(target_dollars / price)) if price > 0 else 0
            rows.append({
                "Bucket": bucket,
                "Ticker": ticker,
                "Latest Price": price,
                "Target Weight": target_weight,
                "Within-Bucket Wt": w_within,
                "Target $": target_dollars,
                "Whole Shares": shares,
                "Actual $": shares * price,
            })

    allocation_df = pd.DataFrame(rows).sort_values(
        ["Bucket", "Target Weight"], ascending=[True, False]
    ).reset_index(drop=True)

    return {
        "bucket_weights": bucket_w,
        "summary": pd.DataFrame(bucket_summary),
        "allocation": allocation_df,
        "prices": prices,
        "total_invested": allocation_df["Actual $"].sum() if not allocation_df.empty else 0,
        "cash_remaining": quarterly_budget - (allocation_df["Actual $"].sum() if not allocation_df.empty else 0),
    }


# Sidebar
with st.sidebar:
    st.markdown("### ◆ Inputs")
    monthly_budget = st.number_input(
        "Monthly contribution budget ($)",
        min_value=100.0, max_value=100_000.0, value=1_000.0, step=100.0,
    )
    quarterly_budget = monthly_budget * 3
    st.caption(f"Quarterly cap: **${quarterly_budget:,.2f}**")
    st.markdown("---")
    risk_score = st.slider("Risk tolerance", 1, 10, 5, 1)
    profile = profile_label(risk_score)
    st.markdown(f"**Profile:** {profile}")
    st.markdown("---")
    risk_free_rate = st.number_input(
        "Risk-free rate (annual)",
        min_value=0.0, max_value=0.20, value=0.04, step=0.005, format="%.3f",
    )
    lookback_years = st.select_slider(
        "Historical lookback (years)", options=[1, 2, 3, 5, 7, 10], value=3
    )
    st.markdown("---")
    run_button = st.button("Build Portfolio", type="primary", use_container_width=True)

    if "shared_state" not in st.session_state:
        st.session_state.shared_state = {}

    with st.expander("Data sources"):
        st.markdown("""
        - **ETF universe** — [echuvyrov/TrackingETFs](https://github.com/echuvyrov/TrackingETFs)
        - **S&P 500 list** — [Ate329/top-us-stock-tickers](https://github.com/Ate329/top-us-stock-tickers)
        - **Prices** — yfinance
        """)

st.title("Portfolio Construction Model")
st.markdown(
    "<p class='small-caption'>Risk-bucketed allocation across conservative bonds, "
    "income/dividend hybrids, and growth equities — sized to a quarterly contribution. "
    "Use the <b>Backtest & Analysis</b> page in the sidebar for performance testing.</p>",
    unsafe_allow_html=True,
)

if not run_button:
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
        tab_c, tab_h, tab_g = st.tabs(["Conservative", "Hybrid", "Growth"])
        for tab, tickers in zip(
            [tab_c, tab_h, tab_g],
            [CONSERVATIVE_TICKERS, HYBRID_TICKERS, GROWTH_TICKERS],
        ):
            with tab:
                bucket_view = etf_df[etf_df["ticker"].isin(tickers)]
                missing = [t for t in tickers if t not in bucket_view["ticker"].values]
                if missing:
                    bucket_view = pd.concat([
                        bucket_view,
                        pd.DataFrame({"ticker": missing, "name": ["—"] * len(missing)}),
                    ], ignore_index=True)
                st.dataframe(bucket_view, hide_index=True, use_container_width=True)
    st.info("Adjust inputs in the sidebar, then click **Build Portfolio**.")
    st.stop()

result = build_portfolio(risk_score, quarterly_budget, lookback_years, risk_free_rate)
if not result:
    st.stop()

# Persist for the backtester page
st.session_state.shared_state.update({
    "risk_score": risk_score,
    "monthly_budget": monthly_budget,
    "quarterly_budget": quarterly_budget,
    "risk_free_rate": risk_free_rate,
    "lookback_years": lookback_years,
    "allocation": result["allocation"],
    "bucket_weights": result["bucket_weights"],
    "prices": result["prices"],
    "profile": profile,
})

m1, m2, m3, m4 = st.columns(4)
m1.metric("Quarterly budget", f"${quarterly_budget:,.0f}")
m2.metric("Invested (whole shares)", f"${result['total_invested']:,.2f}")
m3.metric("Cash remaining", f"${result['cash_remaining']:,.2f}")
m4.metric("Profile", profile)

st.markdown("---")
st.subheader("Bucket allocation")
st.dataframe(result["summary"], hide_index=True, use_container_width=True)

st.subheader("Holdings")
display_df = result["allocation"].copy()
display_df["Latest Price"] = display_df["Latest Price"].apply(lambda x: f"${x:,.2f}")
display_df["Target Weight"] = display_df["Target Weight"].apply(lambda x: f"{x:.2%}")
display_df["Within-Bucket Wt"] = display_df["Within-Bucket Wt"].apply(lambda x: f"{x:.2%}")
display_df["Target $"] = display_df["Target $"].apply(lambda x: f"${x:,.2f}")
display_df["Actual $"] = display_df["Actual $"].apply(lambda x: f"${x:,.2f}")
st.dataframe(display_df, hide_index=True, use_container_width=True)

st.markdown("---")
v1, v2 = st.columns([1, 1])
with v1:
    st.subheader("Allocation by ticker")
    st.bar_chart(result["allocation"].set_index("Ticker")[["Actual $"]], height=360)
with v2:
    st.subheader("Allocation by bucket")
    st.bar_chart(result["allocation"].groupby("Bucket")["Actual $"].sum(), height=360)

st.markdown("---")
st.subheader("Historical performance — equal-weight bucket benchmark")
prices_norm = result["prices"] / result["prices"].iloc[0]
bucket_perf = pd.DataFrame()
for bucket, tickers in BUCKET_MAP.items():
    cols = [t for t in tickers if t in prices_norm.columns]
    if cols:
        bucket_perf[bucket] = prices_norm[cols].mean(axis=1)
st.line_chart(bucket_perf, height=360)

st.success("Portfolio built. Open **Backtest & Analysis** in the left sidebar to run regime tests, Monte Carlo, and risk metrics.")
st.caption("Note: Demonstration model only. Not investment advice.")

csv_bytes = result["allocation"].to_csv(index=False).encode("utf-8")
st.download_button(
    "Download allocation as CSV",
    data=csv_bytes,
    file_name=f"allocation_risk{risk_score}_q{date.today().isoformat()}.csv",
    mime="text/csv",
)
