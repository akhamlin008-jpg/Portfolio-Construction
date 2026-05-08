"""
Backtest & Analysis (Page 2)
=============================

Provides:
  1. Quarterly-contribution backtest vs SPY buy-and-hold (same dates, same dollars)
  2. Regime breakdown — bull / bear / high-vol periods classified from SPY history
  3. Transaction cost modeling (bps + per-trade fixed)
  4. Risk metrics — Sharpe, Treynor, Beta, alpha, Sortino, max drawdown
  5. Kelly criterion (full + fractional) for the optimized portfolio
  6. Sensitivity analysis — how outputs respond to each input
  7. Monte Carlo forecast — GBM + bootstrap, forward-looking
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
    bucket_weights_for_risk,
    compute_portfolio_weights,
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
    "<p class='small-caption'>Test the portfolio across market regimes against SPY buy-and-hold "
    "with matched contribution dates, then run risk metrics, Kelly sizing, sensitivity, and a "
    "Monte Carlo forward forecast.</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Inputs (sidebar)
# ---------------------------------------------------------------------------
state = st.session_state.get("shared_state", {})

with st.sidebar:
    st.markdown("### ◇ Backtest Inputs")

    if state:
        st.caption(f"Loaded from builder: risk={state.get('risk_score')}, "
                   f"monthly=${state.get('monthly_budget'):.0f}")
        default_risk = state.get("risk_score", 5)
        default_monthly = state.get("monthly_budget", 1000.0)
        default_rfr = state.get("risk_free_rate", 0.04)
    else:
        st.warning("Run the builder on the main page first to inherit settings.")
        default_risk = 5
        default_monthly = 1000.0
        default_rfr = 0.04

    risk_score = st.slider("Risk tolerance", 1, 10, default_risk, 1)
    monthly_budget = st.number_input(
        "Monthly contribution ($)",
        min_value=100.0, max_value=100_000.0,
        value=float(default_monthly), step=100.0,
    )
    quarterly_budget = monthly_budget * 3
    st.caption(f"Quarterly contribution: **${quarterly_budget:,.2f}**")

    st.markdown("---")
    risk_free_rate = st.number_input(
        "Risk-free rate (annual)",
        min_value=0.0, max_value=0.20, value=float(default_rfr),
        step=0.005, format="%.3f",
    )

    backtest_years = st.select_slider(
        "Backtest window (years)", options=[2, 3, 5, 7, 10, 15], value=5
    )

    st.markdown("---")
    st.markdown("**Transaction costs**")
    cost_bps = st.number_input(
        "Cost per trade (bps)",
        min_value=0.0, max_value=100.0, value=5.0, step=1.0,
        help="Basis points of trade value. 5 bps = 0.05%.",
    )
    cost_fixed = st.number_input(
        "Fixed commission per ticker ($)",
        min_value=0.0, max_value=20.0, value=0.0, step=0.5,
    )

    st.markdown("---")
    run_backtest = st.button("Run Backtest", type="primary", use_container_width=True)


if not run_backtest:
    st.info("Configure backtest parameters in the sidebar and click **Run Backtest**.")
    st.markdown("""
    **What this page computes:**
    - **Backtest:** Quarterly contributions on the first trading day of each quarter, applied identically to your portfolio and a SPY-only benchmark.
    - **Regimes:** Bull / Bear / High-Vol classification of SPY history.
    - **Risk metrics:** Sharpe, Sortino, Treynor, Beta, alpha, max drawdown.
    - **Kelly criterion:** Full and fractional Kelly weights as a sizing reference.
    - **Sensitivity:** Output sensitivity to risk score, lookback, risk-free rate, and cost.
    - **Monte Carlo:** Forward 5-year forecast using both GBM and bootstrapped historical returns.
    """)
    st.stop()

# ---------------------------------------------------------------------------
# Pull data
# ---------------------------------------------------------------------------
end = date.today()
start = end - timedelta(days=int(backtest_years * 365.25))

needed_tickers = tuple(sorted(set(ALL_TICKERS) | {"SPY"}))
with st.spinner(f"Pulling {backtest_years}y of price data for {len(needed_tickers)} tickers..."):
    prices = fetch_prices(needed_tickers, start, end)

if prices.empty or "SPY" not in prices.columns:
    st.error("Could not load required price data (need SPY + bucket tickers).")
    st.stop()

spy_prices = prices["SPY"]
portfolio_prices = prices[[c for c in prices.columns if c != "SPY"]]

# Compute target weights using the same logic as the builder
target_weights = compute_portfolio_weights(risk_score, portfolio_prices, risk_free_rate)
if target_weights.empty:
    st.error("Could not compute portfolio weights.")
    st.stop()

# ---------------------------------------------------------------------------
# Quarterly contribution schedule
# ---------------------------------------------------------------------------
def quarterly_contribution_dates(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """Return the first trading day of each quarter in the price history."""
    quarters = prices.index.to_series().groupby(
        [prices.index.year, prices.index.quarter]
    ).min()
    return pd.DatetimeIndex(quarters.values).sort_values()


contribution_dates = quarterly_contribution_dates(prices)

# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
def simulate_dca_portfolio(
    prices: pd.DataFrame,
    weights: pd.Series,
    contribution_dates: pd.DatetimeIndex,
    quarterly_amount: float,
    cost_bps: float,
    cost_fixed: float,
) -> dict:
    """
    Simulate a DCA portfolio. On each contribution date, buy whole shares
    according to target weights; track value daily.
    """
    tickers = [t for t in weights.index if t in prices.columns]
    weights = weights[tickers] / weights[tickers].sum()  # renormalize

    shares = pd.Series(0.0, index=tickers)
    cash = 0.0
    total_costs = 0.0
    total_contributed = 0.0

    daily_value = pd.Series(index=prices.index, dtype=float)
    contribution_log = []

    for current_date in prices.index:
        # If this is a contribution day, buy
        if current_date in contribution_dates:
            day_prices = prices.loc[current_date, tickers]
            available = quarterly_amount + cash
            shares_to_buy = pd.Series(0, index=tickers)

            for ticker in tickers:
                target_dollars = available * weights[ticker]
                price = day_prices[ticker]
                if price > 0 and target_dollars > price:
                    n = int(np.floor(target_dollars / price))
                    shares_to_buy[ticker] = n
                    trade_value = n * price
                    transaction_cost = trade_value * (cost_bps / 10_000) + (cost_fixed if n > 0 else 0)
                    total_costs += transaction_cost

            shares = shares + shares_to_buy
            spent = (shares_to_buy * day_prices).sum()
            cost_today = (shares_to_buy * day_prices).sum() * (cost_bps / 10_000) + \
                         cost_fixed * (shares_to_buy > 0).sum()
            cash = available - spent - cost_today
            total_contributed += quarterly_amount
            contribution_log.append({
                "date": current_date,
                "spent": spent,
                "costs": cost_today,
                "cash_after": cash,
            })

        # Mark to market
        daily_value[current_date] = (shares * prices.loc[current_date, tickers]).sum() + cash

    return {
        "value_series": daily_value.dropna(),
        "shares_final": shares,
        "cash_final": cash,
        "total_contributed": total_contributed,
        "total_costs": total_costs,
        "contribution_log": pd.DataFrame(contribution_log),
    }


def simulate_dca_spy(
    spy_prices: pd.Series,
    contribution_dates: pd.DatetimeIndex,
    quarterly_amount: float,
    cost_bps: float,
    cost_fixed: float,
) -> dict:
    """Same DCA mechanics, but 100% SPY."""
    shares = 0.0
    cash = 0.0
    total_costs = 0.0
    total_contributed = 0.0

    daily_value = pd.Series(index=spy_prices.index, dtype=float)

    for current_date in spy_prices.index:
        if current_date in contribution_dates:
            available = quarterly_amount + cash
            price = spy_prices.loc[current_date]
            n = int(np.floor(available / price)) if price > 0 else 0
            spent = n * price
            transaction_cost = spent * (cost_bps / 10_000) + (cost_fixed if n > 0 else 0)
            shares += n
            cash = available - spent - transaction_cost
            total_costs += transaction_cost
            total_contributed += quarterly_amount
        daily_value[current_date] = shares * spy_prices.loc[current_date] + cash

    return {
        "value_series": daily_value.dropna(),
        "shares_final": shares,
        "cash_final": cash,
        "total_contributed": total_contributed,
        "total_costs": total_costs,
    }


with st.spinner("Running backtest..."):
    portfolio_bt = simulate_dca_portfolio(
        portfolio_prices, target_weights, contribution_dates,
        quarterly_budget, cost_bps, cost_fixed,
    )
    spy_bt = simulate_dca_spy(
        spy_prices, contribution_dates, quarterly_budget, cost_bps, cost_fixed,
    )

# ---------------------------------------------------------------------------
# Backtest summary
# ---------------------------------------------------------------------------
st.subheader("Backtest results")

p_final = portfolio_bt["value_series"].iloc[-1]
s_final = spy_bt["value_series"].iloc[-1]
contributed = portfolio_bt["total_contributed"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total contributed", f"${contributed:,.0f}")
m2.metric("Portfolio final value", f"${p_final:,.0f}",
          delta=f"${p_final - contributed:+,.0f}")
m3.metric("SPY final value", f"${s_final:,.0f}",
          delta=f"${s_final - contributed:+,.0f}")
diff = p_final - s_final
m4.metric("Portfolio vs SPY", f"${diff:+,.0f}",
          delta=f"{(p_final/s_final - 1)*100:+.2f}%" if s_final > 0 else "—")

# Combined chart
combined = pd.DataFrame({
    "Portfolio": portfolio_bt["value_series"],
    "SPY (buy & hold)": spy_bt["value_series"],
    "Total Contributed": pd.Series(
        np.linspace(0, contributed, len(portfolio_bt["value_series"])),
        index=portfolio_bt["value_series"].index,
    ),
})
st.line_chart(combined, height=400)

c1, c2 = st.columns(2)
c1.metric("Portfolio transaction costs", f"${portfolio_bt['total_costs']:,.2f}")
c2.metric("SPY transaction costs", f"${spy_bt['total_costs']:,.2f}")

# ---------------------------------------------------------------------------
# Regime analysis
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Regime breakdown")
st.markdown(
    "<p class='small-caption'>Regimes are classified from SPY: Bear = 20%+ drawdown from running peak; "
    "High-Vol = SPY 30-day annualized vol above 25%; Bull = everything else.</p>",
    unsafe_allow_html=True,
)


def classify_regimes(spy_prices: pd.Series) -> pd.Series:
    """Return a Series of regime labels per date."""
    returns = spy_prices.pct_change()
    rolling_vol = returns.rolling(30).std() * np.sqrt(252)
    running_max = spy_prices.cummax()
    drawdown = (spy_prices - running_max) / running_max

    regime = pd.Series("Bull", index=spy_prices.index)
    regime[drawdown <= -0.20] = "Bear"
    regime[(rolling_vol > 0.25) & (drawdown > -0.20)] = "High-Vol"
    return regime


regimes = classify_regimes(spy_prices)

# Compute returns per regime
def daily_returns(value_series: pd.Series) -> pd.Series:
    return value_series.pct_change().fillna(0)


port_rets = daily_returns(portfolio_bt["value_series"])
spy_rets = daily_returns(spy_bt["value_series"])

regime_rows = []
for regime_name in ["Bull", "Bear", "High-Vol"]:
    mask = regimes == regime_name
    days = mask.sum()
    if days < 5:
        continue
    p_ann = (1 + port_rets[mask]).prod() ** (252 / days) - 1
    s_ann = (1 + spy_rets[mask]).prod() ** (252 / days) - 1
    p_vol = port_rets[mask].std() * np.sqrt(252)
    s_vol = spy_rets[mask].std() * np.sqrt(252)
    regime_rows.append({
        "Regime": regime_name,
        "# Days": days,
        "Portfolio Ann. Return": f"{p_ann:.2%}",
        "SPY Ann. Return": f"{s_ann:.2%}",
        "Portfolio Vol": f"{p_vol:.2%}",
        "SPY Vol": f"{s_vol:.2%}",
        "Outperformance": f"{(p_ann - s_ann):+.2%}",
    })
regime_df = pd.DataFrame(regime_rows)
st.dataframe(regime_df, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Risk metrics — Portfolio vs SPY")


def compute_metrics(value_series: pd.Series, benchmark_returns: pd.Series,
                    rfr: float) -> dict:
    """Compute Sharpe, Sortino, Treynor, Beta, alpha, max DD, CAGR."""
    returns = value_series.pct_change().dropna()
    if len(returns) < 30:
        return {}

    excess = returns - rfr / 252
    sharpe = (excess.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    downside = returns[returns < 0].std()
    sortino = (excess.mean() / downside) * np.sqrt(252) if downside > 0 else 0

    # Beta (vs benchmark) and alpha
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    aligned.columns = ["port", "bench"]
    bench_var = aligned["bench"].var()
    beta = aligned["port"].cov(aligned["bench"]) / bench_var if bench_var > 0 else 0
    bench_ann = aligned["bench"].mean() * 252
    port_ann = aligned["port"].mean() * 252
    alpha = port_ann - (rfr + beta * (bench_ann - rfr))
    treynor = (port_ann - rfr) / beta if beta != 0 else np.nan

    # CAGR + max drawdown
    n_years = (value_series.index[-1] - value_series.index[0]).days / 365.25
    if n_years > 0 and value_series.iloc[0] > 0:
        cagr = (value_series.iloc[-1] / max(value_series.iloc[0], 1)) ** (1 / n_years) - 1
    else:
        cagr = 0
    running_max = value_series.cummax()
    max_dd = ((value_series - running_max) / running_max).min()

    return {
        "CAGR": cagr,
        "Annualized Vol": returns.std() * np.sqrt(252),
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Beta (vs SPY)": beta,
        "Alpha (annual)": alpha,
        "Treynor Ratio": treynor,
        "Max Drawdown": max_dd,
    }


port_metrics = compute_metrics(portfolio_bt["value_series"], spy_rets, risk_free_rate)
spy_metrics = compute_metrics(spy_bt["value_series"], spy_rets, risk_free_rate)

if port_metrics and spy_metrics:
    metrics_df = pd.DataFrame({
        "Portfolio": port_metrics,
        "SPY (buy & hold)": spy_metrics,
    })

    def fmt_metric(name, val):
        if pd.isna(val):
            return "—"
        if name in ("CAGR", "Annualized Vol", "Alpha (annual)", "Max Drawdown"):
            return f"{val:.2%}"
        return f"{val:.3f}"

    metrics_display = metrics_df.copy()
    for col in metrics_display.columns:
        metrics_display[col] = [fmt_metric(idx, v) for idx, v in metrics_display[col].items()]
    st.dataframe(metrics_display, use_container_width=True)

    st.caption(
        "**Sharpe** = (return − r_f) / total volatility · "
        "**Sortino** = (return − r_f) / downside volatility · "
        "**Beta** = covariance with benchmark / benchmark variance · "
        "**Treynor** = (return − r_f) / beta · "
        "**Alpha** = realized return minus CAPM-expected return."
    )

# ---------------------------------------------------------------------------
# Kelly criterion
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Kelly criterion sizing")

st.markdown(
    "<p class='small-caption'>Multivariate Kelly: f* = Σ⁻¹ · (μ − r_f). Full Kelly maximizes log-wealth "
    "growth but is volatile — practitioners use ½ or ¼ Kelly. Negative weights mean the asset would be "
    "shorted in an unconstrained Kelly portfolio. Total Kelly fraction > 1 means leverage; < 1 means cash position.</p>",
    unsafe_allow_html=True,
)


def kelly_weights(returns: pd.DataFrame, rfr: float) -> pd.Series:
    """Multivariate Kelly: f* = Σ⁻¹ · (μ − r_f)."""
    mu = returns.mean() * 252
    cov = returns.cov() * 252
    excess = mu - rfr
    try:
        inv_cov = np.linalg.pinv(cov.values)
        kelly = pd.Series(inv_cov @ excess.values, index=cov.index)
    except Exception:
        kelly = pd.Series(0, index=cov.index)
    return kelly


held_tickers = [t for t in target_weights.index if t in portfolio_prices.columns]
held_returns = portfolio_prices[held_tickers].pct_change().dropna()
kelly_full = kelly_weights(held_returns, risk_free_rate)

kelly_df = pd.DataFrame({
    "Optimizer Weight": target_weights[held_tickers],
    "Full Kelly": kelly_full,
    "Half Kelly": kelly_full * 0.5,
    "Quarter Kelly": kelly_full * 0.25,
}).round(4)

k1, k2, k3 = st.columns(3)
k1.metric("Full Kelly total exposure", f"{kelly_full.sum():.2f}x")
k2.metric("Half Kelly total exposure", f"{kelly_full.sum() * 0.5:.2f}x")
k3.metric("Quarter Kelly total exposure", f"{kelly_full.sum() * 0.25:.2f}x")
st.dataframe(kelly_df, use_container_width=True)

# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Sensitivity analysis")
st.markdown(
    "<p class='small-caption'>Holding everything else constant, how does the final portfolio value "
    "change if we vary one input at a time? Calculations use the historical return drift assumption.</p>",
    unsafe_allow_html=True,
)


def simulate_value_for_risk(risk_score: int) -> float:
    weights = compute_portfolio_weights(risk_score, portfolio_prices, risk_free_rate)
    if weights.empty:
        return np.nan
    bt = simulate_dca_portfolio(
        portfolio_prices, weights, contribution_dates,
        quarterly_budget, cost_bps, cost_fixed,
    )
    return bt["value_series"].iloc[-1]


def simulate_value_for_cost(bps: float) -> float:
    bt = simulate_dca_portfolio(
        portfolio_prices, target_weights, contribution_dates,
        quarterly_budget, bps, cost_fixed,
    )
    return bt["value_series"].iloc[-1]


def simulate_value_for_budget(monthly: float) -> float:
    bt = simulate_dca_portfolio(
        portfolio_prices, target_weights, contribution_dates,
        monthly * 3, cost_bps, cost_fixed,
    )
    return bt["value_series"].iloc[-1]


with st.spinner("Running sensitivities (this takes a moment)..."):
    risk_sweep = pd.DataFrame({
        "Risk Score": list(range(1, 11)),
        "Final Value": [simulate_value_for_risk(r) for r in range(1, 11)],
    })
    cost_sweep = pd.DataFrame({
        "Cost (bps)": [0, 1, 5, 10, 20, 50, 100],
        "Final Value": [simulate_value_for_cost(b) for b in [0, 1, 5, 10, 20, 50, 100]],
    })
    budget_sweep = pd.DataFrame({
        "Monthly Budget ($)": [250, 500, 1000, 2000, 5000],
        "Final Value": [simulate_value_for_budget(b) for b in [250, 500, 1000, 2000, 5000]],
    })

s1, s2, s3 = st.columns(3)
with s1:
    st.markdown("**Risk score → final value**")
    st.line_chart(risk_sweep.set_index("Risk Score"), height=200)
with s2:
    st.markdown("**Trans. cost (bps) → final value**")
    st.line_chart(cost_sweep.set_index("Cost (bps)"), height=200)
with s3:
    st.markdown("**Monthly budget → final value**")
    st.line_chart(budget_sweep.set_index("Monthly Budget ($)"), height=200)

# ---------------------------------------------------------------------------
# Monte Carlo forecast
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Monte Carlo forward forecast")
st.markdown(
    "<p class='small-caption'>Forward-looking simulation using the optimized weights. "
    "Two methods: (1) <b>GBM</b> assumes lognormal returns from historical drift and covariance; "
    "(2) <b>Bootstrap</b> samples actual historical days with replacement, preserving fat tails. "
    "Both apply the same quarterly contribution schedule going forward.</p>",
    unsafe_allow_html=True,
)

mc_cols = st.columns(3)
mc_horizon_years = mc_cols[0].selectbox("Forecast horizon (years)", [1, 3, 5, 10], index=2)
mc_paths = mc_cols[1].selectbox("# simulations", [500, 1000, 2500, 5000], index=1)
mc_method = mc_cols[2].selectbox("Method", ["Bootstrap (historical)", "GBM (parametric)", "Both"], index=2)


def monte_carlo_forecast(
    held_returns: pd.DataFrame,
    held_prices_today: pd.Series,
    weights: pd.Series,
    horizon_years: int,
    n_paths: int,
    quarterly_amount: float,
    method: str,
) -> dict:
    """Forward-only Monte Carlo. Returns array of shape (days, n_paths)."""
    days = int(horizon_years * 252)
    n_assets = len(weights)
    tickers = list(weights.index)

    # Quarterly contribution dates (every ~63 trading days)
    contribution_indices = list(range(0, days, 63))

    paths_value = np.zeros((days + 1, n_paths))
    paths_value[0, :] = 0  # start with 0 invested (DCA from scratch)

    for sim in range(n_paths):
        if method == "GBM (parametric)":
            mu = held_returns.mean().values
            cov = held_returns.cov().values
            try:
                L = np.linalg.cholesky(cov + 1e-10 * np.eye(n_assets))
            except np.linalg.LinAlgError:
                L = np.diag(np.sqrt(np.diag(cov)))
            shocks = np.random.randn(days, n_assets) @ L.T
            sim_returns = mu + shocks
        else:  # bootstrap
            sample_idx = np.random.randint(0, len(held_returns), size=days)
            sim_returns = held_returns.values[sample_idx]

        shares = np.zeros(n_assets)
        cash = 0.0
        prices_path = held_prices_today.values.copy()

        for d in range(days):
            # Apply daily returns to prices
            prices_path = prices_path * (1 + sim_returns[d])

            # Contribution at start of quarter
            if d in contribution_indices:
                available = quarterly_amount + cash
                for i in range(n_assets):
                    target_dollars = available * weights.iloc[i]
                    if prices_path[i] > 0 and target_dollars > prices_path[i]:
                        n_shares = int(np.floor(target_dollars / prices_path[i]))
                        shares[i] += n_shares
                        cash_used = n_shares * prices_path[i]
                        available -= cash_used
                cash = available

            paths_value[d + 1, sim] = (shares * prices_path).sum() + cash

    return {"paths": paths_value, "horizon_years": horizon_years}


held_prices_today = portfolio_prices[held_tickers].iloc[-1]
held_weights_normalized = target_weights[held_tickers] / target_weights[held_tickers].sum()

with st.spinner(f"Running {mc_paths} Monte Carlo paths..."):
    if mc_method == "Both":
        mc_boot = monte_carlo_forecast(
            held_returns, held_prices_today, held_weights_normalized,
            mc_horizon_years, mc_paths // 2, quarterly_budget, "Bootstrap (historical)",
        )
        mc_gbm = monte_carlo_forecast(
            held_returns, held_prices_today, held_weights_normalized,
            mc_horizon_years, mc_paths // 2, quarterly_budget, "GBM (parametric)",
        )
        # Combine
        all_paths = np.concatenate([mc_boot["paths"], mc_gbm["paths"]], axis=1)
    else:
        mc = monte_carlo_forecast(
            held_returns, held_prices_today, held_weights_normalized,
            mc_horizon_years, mc_paths, quarterly_budget, mc_method,
        )
        all_paths = mc["paths"]

# Summary statistics
final_values = all_paths[-1, :]
total_contrib = quarterly_budget * (mc_horizon_years * 4)

mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Total contributed", f"${total_contrib:,.0f}")
mc2.metric("Median outcome", f"${np.median(final_values):,.0f}",
           delta=f"${np.median(final_values) - total_contrib:+,.0f}")
mc3.metric("5th percentile", f"${np.percentile(final_values, 5):,.0f}",
           delta=f"${np.percentile(final_values, 5) - total_contrib:+,.0f}")
mc4.metric("95th percentile", f"${np.percentile(final_values, 95):,.0f}",
           delta=f"${np.percentile(final_values, 95) - total_contrib:+,.0f}")

# Fan chart: percentiles over time
percentiles = np.percentile(all_paths, [5, 25, 50, 75, 95], axis=1)
days_axis = np.arange(all_paths.shape[0]) / 252
fan_df = pd.DataFrame({
    "5th": percentiles[0],
    "25th": percentiles[1],
    "Median": percentiles[2],
    "75th": percentiles[3],
    "95th": percentiles[4],
}, index=days_axis)
fan_df.index.name = "Years forward"

st.markdown(f"**Path distribution over {mc_horizon_years} years ({mc_paths} paths)**")
st.line_chart(fan_df, height=400)

# Probability of beating contributions (loss probability)
prob_loss = (final_values < total_contrib).mean()
st.metric("Probability of underperforming total contributions",
          f"{prob_loss:.1%}",
          help="Share of MC paths where final value is less than total dollars contributed.")

st.caption(
    "Monte Carlo assumes the future return distribution resembles the lookback window. "
    "It does not predict regime shifts, structural breaks, or black swans not present in history. "
    "Use the bootstrap method for fatter-tailed outcomes and the GBM method for smoother analytical results."
)
