"""
Backtest & Analysis (v2)
========================

Methodology changes from v1:
  - Walk-forward weight computation (no lookahead bias). At each rebalance date,
    weights are derived from price history strictly PRIOR to that date.
  - Quarterly rebalancing back to target weights, not buy-and-forget.
  - 60/40 (VTI/AGG) benchmark alongside SPY.
  - Calmar ratio and Ulcer Index (proper drawdown metrics).
  - Block bootstrap for Monte Carlo (preserves volatility clustering).
  - Treynor dropped (meaningless for bond-heavy portfolios).
  - Kelly used as a constraint diagnostic, not a separate exhibit.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from portfolio_core import (
    ALL_TICKERS,
    BUCKET_MAP,
    CUSTOM_CSS,
    compute_portfolio_weights,
    compute_weights_walkforward,
    fetch_prices,
    profile_label,
)

st.set_page_config(
    page_title="Backtest & Analysis",
    page_icon="◇",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("Backtest & Analysis")
st.markdown(
    "<p class='small-caption'>Walk-forward backtest with quarterly rebalancing. Each rebalance "
    "date uses only data available <i>before</i> that date — no lookahead bias. Compared against "
    "two benchmarks: SPY (equity-only) and a rebalanced 60/40 (VTI/AGG) portfolio.</p>",
    unsafe_allow_html=True,
)

state = st.session_state.get("shared_state", {})

with st.sidebar:
    st.markdown("### ◇ Backtest Inputs")
    if state:
        st.caption(f"Loaded from builder: risk={state.get('risk_score')}, monthly=${state.get('monthly_budget'):.0f}")
        default_risk = state.get("risk_score", 5)
        default_monthly = state.get("monthly_budget", 1000.0)
        default_rfr = state.get("risk_free_rate", 0.04)
        default_kelly = state.get("use_kelly_cap", False)
    else:
        st.info("Run the builder first to inherit settings.")
        default_risk = 5
        default_monthly = 1000.0
        default_rfr = 0.04
        default_kelly = False

    risk_score = st.slider("Risk tolerance", 1, 10, default_risk, 1)
    monthly_budget = st.number_input(
        "Monthly contribution ($)", min_value=100.0, max_value=100_000.0,
        value=float(default_monthly), step=100.0,
    )
    quarterly_budget = monthly_budget * 3
    st.caption(f"Quarterly contribution: **${quarterly_budget:,.2f}**")

    st.markdown("---")
    risk_free_rate = st.number_input(
        "Risk-free rate (annual)", min_value=0.0, max_value=0.20,
        value=float(default_rfr), step=0.005, format="%.3f",
    )
    backtest_years = st.select_slider(
        "Backtest window (years)", options=[3, 5, 7, 10, 15], value=7,
    )
    lookback_days = st.select_slider(
        "Walk-forward lookback (days)",
        options=[252, 504, 756, 1008, 1260], value=756,
        help="Days of history used to compute weights at each rebalance date.",
    )

    st.markdown("---")
    st.markdown("**Transaction costs**")
    cost_bps = st.number_input(
        "Cost per trade (bps)", min_value=0.0, max_value=100.0, value=5.0, step=1.0,
    )
    cost_fixed = st.number_input(
        "Fixed commission per ticker ($)", min_value=0.0, max_value=20.0, value=0.0, step=0.5,
    )

    st.markdown("---")
    use_kelly_cap = st.checkbox("Apply Kelly cap to weights", value=default_kelly)

    st.markdown("---")
    run_backtest = st.button("Run Backtest", type="primary", use_container_width=True)


if not run_backtest:
    st.info("Configure parameters in the sidebar and click **Run Backtest**.")
    st.markdown("""
    **What this page does:**
    - **Walk-forward backtest** — weights recomputed every quarter using only data available at that point in time.
    - **Quarterly rebalancing** — portfolio drifts back to target weights on contribution dates.
    - **Two benchmarks** — SPY buy-and-hold and a rebalanced 60/40 (VTI/AGG) portfolio, both with matched DCA dates.
    - **Regime analysis** — Bull / Bear / High-Vol classification of SPY history.
    - **Risk metrics** — Sharpe, Sortino, Beta, Alpha, Max Drawdown, **Calmar ratio**, **Ulcer Index**.
    - **Monte Carlo forecast** — block bootstrap (preserves volatility clustering) for forward 1–10 year projections.
    """)
    st.stop()


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
end = date.today()
start = end - timedelta(days=int(backtest_years * 365.25))

# Need everything in ALL_TICKERS, plus SPY (benchmark + CAPM market), plus AGG (for 60/40)
needed = tuple(sorted(set(ALL_TICKERS) | {"SPY", "AGG", "VTI"}))
with st.spinner(f"Pulling {backtest_years}y of data for {len(needed)} tickers..."):
    prices = fetch_prices(needed, start, end)

if prices.empty or "SPY" not in prices.columns:
    st.error("Could not load required price data. Try refreshing in a minute.")
    st.stop()

spy_prices = prices["SPY"]
agg_prices = prices.get("AGG")
vti_prices = prices.get("VTI")
portfolio_prices = prices[[c for c in prices.columns if c != "SPY"]]


# ---------------------------------------------------------------------------
# Rebalance schedule
# ---------------------------------------------------------------------------
def quarterly_dates(prices: pd.DataFrame) -> pd.DatetimeIndex:
    quarters = prices.index.to_series().groupby(
        [prices.index.year, prices.index.quarter]
    ).min()
    return pd.DatetimeIndex(quarters.values).sort_values()

rebalance_dates = quarterly_dates(prices)


# ---------------------------------------------------------------------------
# WALK-FORWARD: compute weights for every rebalance date using only prior data
# ---------------------------------------------------------------------------
with st.spinner("Computing walk-forward weights (no lookahead bias)..."):
    weights_schedule = compute_weights_walkforward(
        portfolio_prices, spy_prices, rebalance_dates,
        risk_score=risk_score, risk_free_rate=risk_free_rate,
        lookback_days=lookback_days, use_kelly_cap=use_kelly_cap,
    )


# ---------------------------------------------------------------------------
# Backtest engine with rebalancing
# ---------------------------------------------------------------------------
def simulate_portfolio_walkforward(
    prices: pd.DataFrame,
    weights_schedule: dict,
    rebalance_dates: pd.DatetimeIndex,
    quarterly_amount: float,
    cost_bps: float,
    cost_fixed: float,
) -> dict:
    """
    Walk-forward simulation with quarterly rebalancing.

    On each rebalance date:
      1. Add the quarterly contribution (in cash).
      2. Compute the current dollar value of each holding.
      3. Compute the target dollar value per asset using the walk-forward weights for that date.
      4. Buy/sell to bring each holding to target.
      5. Apply transaction costs proportional to dollar volume traded.

    Between rebalance dates, holdings drift with market prices.
    """
    all_tickers = sorted({t for w in weights_schedule.values() for t in w.index})
    shares = pd.Series(0.0, index=all_tickers)
    cash = 0.0
    total_costs = 0.0
    total_contributed = 0.0
    total_turnover = 0.0
    daily_value = pd.Series(index=prices.index, dtype=float)

    for current_date in prices.index:
        if current_date in rebalance_dates:
            day_prices = prices.loc[current_date].reindex(all_tickers).fillna(0)
            cash += quarterly_amount
            total_contributed += quarterly_amount

            target_weights = weights_schedule.get(current_date, pd.Series(dtype=float))
            if not target_weights.empty:
                # Current holdings value + cash = total to deploy
                current_holdings_value = (shares * day_prices).sum()
                total_value = current_holdings_value + cash

                target_shares = pd.Series(0.0, index=all_tickers)
                for ticker in all_tickers:
                    target_w = target_weights.get(ticker, 0)
                    target_dollars = total_value * target_w
                    if day_prices[ticker] > 0 and target_dollars > day_prices[ticker]:
                        target_shares[ticker] = np.floor(target_dollars / day_prices[ticker])

                # Trade to target
                share_delta = target_shares - shares
                dollar_traded = (share_delta.abs() * day_prices).sum()
                total_turnover += dollar_traded

                cost_today = dollar_traded * (cost_bps / 10_000)
                # Fixed commission per ticker that traded
                cost_today += cost_fixed * (share_delta.abs() > 0).sum()
                total_costs += cost_today

                shares = target_shares
                cash = total_value - (shares * day_prices).sum() - cost_today

        daily_value[current_date] = (shares * prices.loc[current_date].reindex(all_tickers).fillna(0)).sum() + cash

    return {
        "value_series": daily_value.dropna(),
        "shares_final": shares,
        "cash_final": cash,
        "total_contributed": total_contributed,
        "total_costs": total_costs,
        "total_turnover": total_turnover,
    }


def simulate_spy(spy_prices, rebalance_dates, quarterly_amount, cost_bps, cost_fixed):
    """Simple DCA-into-SPY benchmark."""
    shares = 0.0
    cash = 0.0
    total_costs = 0.0
    total_contributed = 0.0
    daily_value = pd.Series(index=spy_prices.index, dtype=float)
    for current_date in spy_prices.index:
        if current_date in rebalance_dates:
            available = quarterly_amount + cash
            price = spy_prices.loc[current_date]
            n = int(np.floor(available / price)) if price > 0 else 0
            spent = n * price
            tcost = spent * (cost_bps / 10_000) + (cost_fixed if n > 0 else 0)
            shares += n
            cash = available - spent - tcost
            total_costs += tcost
            total_contributed += quarterly_amount
        daily_value[current_date] = shares * spy_prices.loc[current_date] + cash
    return {
        "value_series": daily_value.dropna(),
        "total_contributed": total_contributed,
        "total_costs": total_costs,
    }


def simulate_6040(vti_prices, agg_prices, rebalance_dates, quarterly_amount, cost_bps, cost_fixed):
    """60% VTI / 40% AGG with quarterly rebalancing — the practitioner default."""
    if vti_prices is None or agg_prices is None:
        return None
    aligned = pd.concat([vti_prices.rename("VTI"), agg_prices.rename("AGG")], axis=1).dropna()
    shares = {"VTI": 0.0, "AGG": 0.0}
    cash = 0.0
    total_costs = 0.0
    total_contributed = 0.0
    daily_value = pd.Series(index=aligned.index, dtype=float)
    target = {"VTI": 0.6, "AGG": 0.4}
    for current_date in aligned.index:
        prices_today = aligned.loc[current_date]
        if current_date in rebalance_dates:
            current_holdings = shares["VTI"] * prices_today["VTI"] + shares["AGG"] * prices_today["AGG"]
            total_value = current_holdings + cash + quarterly_amount
            target_shares = {}
            for tkr in ["VTI", "AGG"]:
                target_dollars = total_value * target[tkr]
                target_shares[tkr] = np.floor(target_dollars / prices_today[tkr]) if prices_today[tkr] > 0 else 0
            dollar_traded = sum(abs(target_shares[t] - shares[t]) * prices_today[t] for t in ["VTI", "AGG"])
            total_turnover_cost = dollar_traded * (cost_bps / 10_000)
            n_trades = sum(1 for t in ["VTI", "AGG"] if abs(target_shares[t] - shares[t]) > 0)
            total_turnover_cost += cost_fixed * n_trades
            total_costs += total_turnover_cost
            shares = target_shares
            cash = total_value - sum(shares[t] * prices_today[t] for t in ["VTI", "AGG"]) - total_turnover_cost
            total_contributed += quarterly_amount
        daily_value[current_date] = sum(shares[t] * prices_today[t] for t in ["VTI", "AGG"]) + cash
    return {
        "value_series": daily_value.dropna(),
        "total_contributed": total_contributed,
        "total_costs": total_costs,
    }


with st.spinner("Running walk-forward backtest..."):
    portfolio_bt = simulate_portfolio_walkforward(
        portfolio_prices, weights_schedule, rebalance_dates,
        quarterly_budget, cost_bps, cost_fixed,
    )
    spy_bt = simulate_spy(spy_prices, rebalance_dates, quarterly_budget, cost_bps, cost_fixed)
    sixty_forty_bt = simulate_6040(vti_prices, agg_prices, rebalance_dates, quarterly_budget, cost_bps, cost_fixed)


# ---------------------------------------------------------------------------
# Top-line summary
# ---------------------------------------------------------------------------
p_final = portfolio_bt["value_series"].iloc[-1]
s_final = spy_bt["value_series"].iloc[-1]
sf_final = sixty_forty_bt["value_series"].iloc[-1] if sixty_forty_bt else None
contributed = portfolio_bt["total_contributed"]

st.subheader("Walk-forward backtest results")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Contributed", f"${contributed:,.0f}")
m2.metric("Portfolio", f"${p_final:,.0f}", delta=f"${p_final - contributed:+,.0f}")
m3.metric("SPY", f"${s_final:,.0f}", delta=f"${s_final - contributed:+,.0f}")
if sf_final is not None:
    m4.metric("60/40", f"${sf_final:,.0f}", delta=f"${sf_final - contributed:+,.0f}")
else:
    m4.metric("60/40", "—", help="VTI or AGG unavailable")

# Combined chart with all three lines
combined = pd.DataFrame({
    "Portfolio": portfolio_bt["value_series"],
    "SPY (100%)": spy_bt["value_series"],
})
if sixty_forty_bt:
    combined["60/40 (VTI/AGG)"] = sixty_forty_bt["value_series"]
combined["Contributed"] = pd.Series(
    np.linspace(0, contributed, len(portfolio_bt["value_series"])),
    index=portfolio_bt["value_series"].index,
)
st.line_chart(combined, height=420)

c1, c2, c3 = st.columns(3)
c1.metric("Portfolio costs", f"${portfolio_bt['total_costs']:,.2f}")
c1.caption(f"Turnover: ${portfolio_bt['total_turnover']:,.0f}")
c2.metric("SPY costs", f"${spy_bt['total_costs']:,.2f}")
if sixty_forty_bt:
    c3.metric("60/40 costs", f"${sixty_forty_bt['total_costs']:,.2f}")


# ---------------------------------------------------------------------------
# Walk-forward weight evolution
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Walk-forward weight evolution")
st.markdown(
    "<p class='small-caption'>How the portfolio's target weights have changed across the backtest. "
    "Each column is a rebalance date; each row is a holding. Computed from the lookback window "
    "ending the day before — no lookahead.</p>",
    unsafe_allow_html=True,
)
weight_history = pd.DataFrame(weights_schedule).fillna(0).T
weight_history.index.name = "Rebalance Date"
if not weight_history.empty:
    st.area_chart(weight_history, height=300)


# ---------------------------------------------------------------------------
# Regime breakdown
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Regime breakdown")
st.markdown(
    "<p class='small-caption'>Bear = 20%+ drawdown from running peak. High-Vol = SPY 30-day annualized "
    "vol > 25% (and not in Bear). Bull = everything else.</p>",
    unsafe_allow_html=True,
)


def classify_regimes(spy_prices):
    returns = spy_prices.pct_change()
    rolling_vol = returns.rolling(30).std() * np.sqrt(252)
    running_max = spy_prices.cummax()
    drawdown = (spy_prices - running_max) / running_max
    regime = pd.Series("Bull", index=spy_prices.index)
    regime[drawdown <= -0.20] = "Bear"
    regime[(rolling_vol > 0.25) & (drawdown > -0.20)] = "High-Vol"
    return regime


regimes = classify_regimes(spy_prices)
port_rets = portfolio_bt["value_series"].pct_change().fillna(0)
spy_rets = spy_bt["value_series"].pct_change().fillna(0)
sf_rets = sixty_forty_bt["value_series"].pct_change().fillna(0) if sixty_forty_bt else None

regime_rows = []
for regime_name in ["Bull", "Bear", "High-Vol"]:
    mask = regimes.reindex(port_rets.index).fillna("Bull") == regime_name
    days = mask.sum()
    if days < 5:
        continue
    def ann_ret(rets):
        if (1 + rets[mask]).prod() <= 0:
            return np.nan
        return (1 + rets[mask]).prod() ** (252 / days) - 1
    row = {
        "Regime": regime_name,
        "# Days": int(days),
        "Portfolio": f"{ann_ret(port_rets):+.2%}",
        "SPY": f"{ann_ret(spy_rets):+.2%}",
    }
    if sf_rets is not None:
        row["60/40"] = f"{ann_ret(sf_rets):+.2%}"
    regime_rows.append(row)
st.dataframe(pd.DataFrame(regime_rows), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Risk metrics")


def compute_metrics(value_series, benchmark_returns, rfr):
    returns = value_series.pct_change().dropna()
    if len(returns) < 30:
        return {}
    excess = returns - rfr / 252
    sharpe = (excess.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    downside = returns[returns < 0].std()
    sortino = (excess.mean() / downside) * np.sqrt(252) if downside > 0 else 0

    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    aligned.columns = ["port", "bench"]
    bench_var = aligned["bench"].var()
    beta = aligned["port"].cov(aligned["bench"]) / bench_var if bench_var > 0 else 0
    bench_ann = aligned["bench"].mean() * 252
    port_ann = aligned["port"].mean() * 252
    alpha = port_ann - (rfr + beta * (bench_ann - rfr))

    n_years = (value_series.index[-1] - value_series.index[0]).days / 365.25
    if n_years > 0 and value_series.iloc[0] > 0:
        cagr = (value_series.iloc[-1] / max(value_series.iloc[0], 1)) ** (1 / n_years) - 1
    else:
        cagr = 0
    running_max = value_series.cummax()
    drawdown_series = (value_series - running_max) / running_max
    max_dd = drawdown_series.min()

    # Calmar = CAGR / |max DD|
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

    # Ulcer Index: RMS of drawdowns (always-positive, penalizes sustained losses)
    ulcer = np.sqrt((drawdown_series ** 2).mean())

    return {
        "CAGR": cagr,
        "Annualized Vol": returns.std() * np.sqrt(252),
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Beta (vs SPY)": beta,
        "Alpha (annual)": alpha,
        "Max Drawdown": max_dd,
        "Calmar Ratio": calmar,
        "Ulcer Index": ulcer,
    }


port_metrics = compute_metrics(portfolio_bt["value_series"], spy_rets, risk_free_rate)
spy_metrics = compute_metrics(spy_bt["value_series"], spy_rets, risk_free_rate)
sf_metrics = compute_metrics(sixty_forty_bt["value_series"], spy_rets, risk_free_rate) if sixty_forty_bt else {}

metrics_dict = {"Portfolio": port_metrics, "SPY": spy_metrics}
if sf_metrics:
    metrics_dict["60/40"] = sf_metrics
metrics_df = pd.DataFrame(metrics_dict)


def fmt_metric(name, val):
    if pd.isna(val):
        return "—"
    if name in ("CAGR", "Annualized Vol", "Alpha (annual)", "Max Drawdown", "Ulcer Index"):
        return f"{val:.2%}"
    return f"{val:.3f}"


metrics_display = metrics_df.copy()
for col in metrics_display.columns:
    metrics_display[col] = [fmt_metric(idx, v) for idx, v in metrics_display[col].items()]
st.dataframe(metrics_display, use_container_width=True)

st.caption(
    "**Sharpe** = (return − r_f) / total vol · "
    "**Sortino** = (return − r_f) / downside vol · "
    "**Beta** = covariance with SPY / SPY variance · "
    "**Alpha** = realized return minus CAPM-expected · "
    "**Calmar** = CAGR / |max DD|, the return-per-unit-of-drawdown · "
    "**Ulcer Index** = RMS of drawdown series, penalizes long underwater periods."
)


# ---------------------------------------------------------------------------
# Drawdown chart
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Drawdown comparison")
dd_df = pd.DataFrame()
for name, series in [
    ("Portfolio", portfolio_bt["value_series"]),
    ("SPY", spy_bt["value_series"]),
] + ([("60/40", sixty_forty_bt["value_series"])] if sixty_forty_bt else []):
    rm = series.cummax()
    dd_df[name] = (series - rm) / rm
st.area_chart(dd_df, height=300)
st.caption(
    "Drawdown = current value as a percent below the running peak. "
    "Portfolios with consistently shallower drawdowns are easier to hold through bear markets."
)


# ---------------------------------------------------------------------------
# Monte Carlo (block bootstrap)
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Monte Carlo forecast (block bootstrap)")
st.markdown(
    "<p class='small-caption'>Forward-only simulation using <b>block bootstrap</b>, which samples "
    "contiguous blocks of historical returns to preserve volatility clustering and autocorrelation. "
    "Plain IID bootstrap (used in v1) destroys these features and underestimates tail risk.</p>",
    unsafe_allow_html=True,
)

held_tickers_now = list(weights_schedule[rebalance_dates[-1]].index) if rebalance_dates[-1] in weights_schedule else []
held_tickers_now = [t for t in held_tickers_now if t in portfolio_prices.columns]
held_returns = portfolio_prices[held_tickers_now].pct_change().dropna() if held_tickers_now else pd.DataFrame()

if held_returns.empty or len(held_returns) < 60:
    st.warning("Not enough data for Monte Carlo. Try a longer backtest window.")
    st.stop()

mc_cols = st.columns(3)
mc_horizon_years = mc_cols[0].selectbox("Horizon (years)", [1, 3, 5, 10], index=2)
mc_paths = mc_cols[1].selectbox("# paths", [500, 1000, 2500], index=1)
block_size = mc_cols[2].selectbox(
    "Block size (days)", [5, 10, 20, 60], index=1,
    help="Length of contiguous return blocks resampled. Larger = preserves more autocorrelation.",
)


def block_bootstrap_mc(
    held_returns, held_prices_today, weights, horizon_years, n_paths,
    quarterly_amount, block_size,
):
    """Monte Carlo using overlapping block bootstrap."""
    days = int(horizon_years * 252)
    n_assets = len(weights)
    weights_arr = weights.values
    weights_arr = weights_arr / weights_arr.sum() if weights_arr.sum() > 0 else weights_arr

    returns_arr = held_returns.values  # (T, n_assets)
    T = returns_arr.shape[0]
    contribution_indices = list(range(0, days, 63))

    paths_value = np.zeros((days + 1, n_paths))

    for sim in range(n_paths):
        # Build a return path by stitching together random blocks
        sim_returns = np.zeros((days, n_assets))
        pos = 0
        while pos < days:
            block_start = np.random.randint(0, max(1, T - block_size))
            actual_block_len = min(block_size, days - pos, T - block_start)
            sim_returns[pos:pos + actual_block_len] = returns_arr[block_start:block_start + actual_block_len]
            pos += actual_block_len

        shares = np.zeros(n_assets)
        cash = 0.0
        prices_path = held_prices_today.values.copy()

        for d in range(days):
            prices_path = prices_path * (1 + sim_returns[d])
            if d in contribution_indices:
                cash += quarterly_amount
                # Rebalance to target on each contribution
                total_value = (shares * prices_path).sum() + cash
                for i in range(n_assets):
                    target_dollars = total_value * weights_arr[i]
                    if prices_path[i] > 0 and target_dollars > prices_path[i]:
                        shares[i] = np.floor(target_dollars / prices_path[i])
                cash = total_value - (shares * prices_path).sum()
            paths_value[d + 1, sim] = (shares * prices_path).sum() + cash

    return paths_value


with st.spinner(f"Running {mc_paths} block bootstrap paths..."):
    held_prices_today = portfolio_prices[held_tickers_now].iloc[-1]
    held_weights = weights_schedule[rebalance_dates[-1]][held_tickers_now]
    paths = block_bootstrap_mc(
        held_returns, held_prices_today, held_weights,
        mc_horizon_years, mc_paths, quarterly_budget, block_size,
    )

final_values = paths[-1, :]
total_contrib = quarterly_budget * (mc_horizon_years * 4)

mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Total contributed", f"${total_contrib:,.0f}")
mc2.metric("Median outcome", f"${np.median(final_values):,.0f}",
           delta=f"${np.median(final_values) - total_contrib:+,.0f}")
mc3.metric("5th percentile", f"${np.percentile(final_values, 5):,.0f}",
           delta=f"${np.percentile(final_values, 5) - total_contrib:+,.0f}")
mc4.metric("95th percentile", f"${np.percentile(final_values, 95):,.0f}",
           delta=f"${np.percentile(final_values, 95) - total_contrib:+,.0f}")

percentiles = np.percentile(paths, [5, 25, 50, 75, 95], axis=1)
days_axis = np.arange(paths.shape[0]) / 252
fan_df = pd.DataFrame({
    "5th": percentiles[0], "25th": percentiles[1], "Median": percentiles[2],
    "75th": percentiles[3], "95th": percentiles[4],
}, index=days_axis)
fan_df.index.name = "Years forward"
st.line_chart(fan_df, height=400)

prob_loss = (final_values < total_contrib).mean()
st.metric(
    "Probability of underperforming total contributions", f"{prob_loss:.1%}",
    help="Share of paths where final value < total dollars contributed.",
)

st.caption(
    f"Block bootstrap with {block_size}-day blocks. Sampling contiguous blocks rather than "
    "individual days preserves the autocorrelation and volatility clustering that exist in "
    "real returns — IID bootstrap (v1) destroys these features and underestimates tail risk."
)

st.markdown(
    "<div class='methodology-note'>"
    "<b>What's still imperfect:</b> "
    "(1) Backtest universe is constrained to currently-listed ETFs, biasing toward survivors. "
    "(2) CAPM uses SPY as a single-factor market proxy — multi-factor (FF) would be more rigorous. "
    "(3) Monte Carlo assumes the future return distribution resembles the lookback window."
    "</div>",
    unsafe_allow_html=True,
)
