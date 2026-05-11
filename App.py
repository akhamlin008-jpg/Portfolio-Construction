"""
Portfolio Construction Model (v2) — Builder

Methodology changes from v1:
  - Bucket weights derived from CRRA utility (Merton solution), not hand-picked.
  - Expected returns via CAPM (β-anchored), not noisy historical means.
  - Correlation diagnostics and factor-style exposure proxy added.
  - Stooq fallback for price data when yfinance fails.
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
    capm_expected_returns,
    compute_portfolio_weights,
    crra_bucket_weights,
    fetch_prices,
    load_etf_universe,
    optimize_bucket,
    profile_label,
    risk_aversion_from_score,
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
    use_kelly_cap: bool,
) -> dict:
    end = date.today()
    start = end - timedelta(days=int(lookback_years * 365.25))

    # Pull all bucket tickers PLUS SPY for CAPM market series
    needed = tuple(sorted(set(ALL_TICKERS) | {"SPY"}))
    with st.spinner(f"Loading price history for {len(needed)} tickers..."):
        prices = fetch_prices(needed, start, end)

    if prices.empty or "SPY" not in prices.columns:
        st.error(
            "Could not load required price data. Tried yfinance and Stooq fallback. "
            "This is usually a temporary upstream issue — try refreshing in a minute."
        )
        return {}

    market_prices = prices["SPY"]
    portfolio_prices = prices[[c for c in prices.columns if c != "SPY"]]

    bucket_w = crra_bucket_weights(risk_score, risk_free_rate)
    latest_prices = portfolio_prices.iloc[-1]
    rows, bucket_summary = [], []

    for bucket, weight in bucket_w.items():
        tickers_in_bucket = [t for t in BUCKET_MAP[bucket] if t in portfolio_prices.columns]
        if not tickers_in_bucket or weight <= 0:
            continue
        within = optimize_bucket(
            portfolio_prices[tickers_in_bucket],
            market_prices=market_prices,
            risk_free_rate=risk_free_rate,
            use_kelly_cap=use_kelly_cap,
        )
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
        "prices": portfolio_prices,
        "market_prices": market_prices,
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
    gamma = risk_aversion_from_score(risk_score)
    st.markdown(f"**Profile:** {profile}  ·  **CRRA γ:** {gamma}")

    st.markdown("---")
    risk_free_rate = st.number_input(
        "Risk-free rate (annual)",
        min_value=0.0, max_value=0.20, value=0.04, step=0.005, format="%.3f",
    )
    lookback_years = st.select_slider(
        "Historical lookback (years)", options=[1, 2, 3, 5, 7, 10], value=3,
    )
    use_kelly_cap = st.checkbox(
        "Use Kelly-derived weight caps",
        value=False,
        help="Cap each asset's weight at half its univariate Kelly fraction.",
    )

    st.markdown("---")
    run_button = st.button("Build Portfolio", type="primary", use_container_width=True)

    if "shared_state" not in st.session_state:
        st.session_state.shared_state = {}

    with st.expander("Methodology"):
        st.markdown("""
        - **Bucket weights:** CRRA utility maximization (Merton solution) with γ derived from risk score.
        - **Expected returns:** CAPM-based, anchored to SPY as the market proxy.
        - **Covariance:** Ledoit-Wolf shrinkage to reduce estimation noise.
        - **Optimization:** Max-Sharpe within each bucket, then scaled by CRRA bucket weight.
        - **Price data:** yfinance with Stooq fallback (via pandas-datareader).
        """)

st.title("Portfolio Construction Model")
st.markdown(
    "<p class='small-caption'>Risk-bucketed allocation across conservative bonds, "
    "income/dividend hybrids, and growth equities, derived from CRRA utility and CAPM "
    "expected returns. See <b>Backtest & Analysis</b> for walk-forward performance testing.</p>",
    unsafe_allow_html=True,
)

if not run_button:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("CRRA-derived bucket weights")
        preview_w = crra_bucket_weights(risk_score, risk_free_rate)
        preview_df = pd.DataFrame({
            "Bucket": list(preview_w.keys()),
            "Weight": [f"{v:.1%}" for v in preview_w.values()],
            "Dollar Target": [f"${quarterly_budget * v:,.2f}" for v in preview_w.values()],
        })
        st.dataframe(preview_df, hide_index=True, use_container_width=True)
        st.markdown(
            "<div class='methodology-note'>"
            f"With risk aversion γ={gamma}, the Merton solution allocates "
            f"<b>{preview_w['Conservative']:.0%}</b> to conservative, "
            f"<b>{preview_w['Hybrid']:.0%}</b> to hybrid, and "
            f"<b>{preview_w['Growth']:.0%}</b> to growth. "
            "Lower γ ⇒ more growth tilt. These weights respond to changes in the risk-free rate "
            "because higher rates reduce the bucket excess returns."
            "</div>", unsafe_allow_html=True,
        )

    with col2:
        st.subheader("Universe preview")
        with st.spinner("Loading ETF list..."):
            etf_df = load_etf_universe()
        st.metric("ETFs available", len(etf_df))
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

result = build_portfolio(
    risk_score, quarterly_budget, lookback_years, risk_free_rate, use_kelly_cap
)
if not result:
    st.stop()

# Persist for backtester
st.session_state.shared_state.update({
    "risk_score": risk_score,
    "monthly_budget": monthly_budget,
    "quarterly_budget": quarterly_budget,
    "risk_free_rate": risk_free_rate,
    "lookback_years": lookback_years,
    "use_kelly_cap": use_kelly_cap,
    "allocation": result["allocation"],
    "bucket_weights": result["bucket_weights"],
    "prices": result["prices"],
    "market_prices": result["market_prices"],
    "profile": profile,
})

m1, m2, m3, m4 = st.columns(4)
m1.metric("Quarterly budget", f"${quarterly_budget:,.0f}")
m2.metric("Invested", f"${result['total_invested']:,.2f}")
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

# CAPM expected returns table — shows what's actually driving the optimization
st.markdown("---")
st.subheader("CAPM expected returns (β-implied)")
st.markdown(
    "<p class='small-caption'>Each asset's expected return = r_f + β × market premium. "
    "These (not historical means) drive the max-Sharpe optimization within each bucket.</p>",
    unsafe_allow_html=True,
)
held_tickers = list(result["allocation"]["Ticker"])
held_prices = result["prices"][held_tickers]
capm_mu = capm_expected_returns(held_prices, result["market_prices"], risk_free_rate)

asset_returns = held_prices.pct_change().dropna()
mkt_returns = result["market_prices"].pct_change().dropna()
aligned = pd.concat([asset_returns, mkt_returns.rename("__MKT__")], axis=1).dropna()
mkt_var = aligned["__MKT__"].var()
betas = {
    t: aligned[t].cov(aligned["__MKT__"]) / mkt_var if mkt_var > 0 else 0
    for t in held_tickers
}

capm_table = pd.DataFrame({
    "Ticker": held_tickers,
    "Beta (vs SPY)": [betas[t] for t in held_tickers],
    "CAPM Expected Return": [capm_mu[t] for t in held_tickers],
    "Realized (annualized)": [asset_returns[t].mean() * 252 for t in held_tickers],
})
capm_table["Beta (vs SPY)"] = capm_table["Beta (vs SPY)"].apply(lambda x: f"{x:.3f}")
capm_table["CAPM Expected Return"] = capm_table["CAPM Expected Return"].apply(lambda x: f"{x:.2%}")
capm_table["Realized (annualized)"] = capm_table["Realized (annualized)"].apply(lambda x: f"{x:.2%}")
st.dataframe(capm_table, hide_index=True, use_container_width=True)

# Correlation diagnostic
st.markdown("---")
st.subheader("Correlation diagnostics")
st.markdown(
    "<p class='small-caption'>Pairwise correlations across holdings. A bond-equity portfolio that "
    "diversifies well in normal times can lose its diversification benefit in stress periods "
    "(2008, March 2020) when correlations spike toward 1.</p>",
    unsafe_allow_html=True,
)
corr_matrix = asset_returns.corr()
# Render as a styled dataframe (Streamlit doesn't have a built-in heatmap, but
# styling with a colormap is cleaner than dragging in matplotlib)
st.dataframe(
    corr_matrix.style
        .background_gradient(cmap="RdYlGn_r", vmin=-1, vmax=1)
        .format("{:.2f}"),
    use_container_width=True,
)
st.caption(
    f"Average pairwise correlation across all holdings: **{corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.2f}**. "
    "Lower is better for diversification."
)

# Allocation visuals
st.markdown("---")
v1, v2 = st.columns([1, 1])
with v1:
    st.subheader("Allocation by ticker")
    st.bar_chart(result["allocation"].set_index("Ticker")[["Actual $"]], height=360)
with v2:
    st.subheader("Allocation by bucket")
    st.bar_chart(result["allocation"].groupby("Bucket")["Actual $"].sum(), height=360)

st.success(
    "Portfolio built. Open **Backtest & Analysis** in the left sidebar to run "
    "walk-forward backtesting (no lookahead bias), see how the portfolio behaved "
    "across regimes, and compare against SPY and a 60/40 benchmark."
)

st.markdown(
    "<div class='methodology-note'>"
    "<b>Known limitations of this build:</b> "
    "(1) The universe is current-listed ETFs only, so pre-inception backtests are not possible for newer ETFs (HYG: 2007, QQQ: 1999, VUG: 2004). "
    "(2) CAPM is a single-factor model; multi-factor (Fama-French) attribution is not included. "
    "(3) The CRRA bucket priors are static; they don't adapt to changing yield environments beyond what flows through the risk-free rate input. "
    "(4) Whole-share rounding produces small cash drag, reported transparently above."
    "</div>",
    unsafe_allow_html=True,
)

csv_bytes = result["allocation"].to_csv(index=False).encode("utf-8")
st.download_button(
    "Download allocation as CSV",
    data=csv_bytes,
    file_name=f"allocation_risk{risk_score}_q{date.today().isoformat()}.csv",
    mime="text/csv",
)
