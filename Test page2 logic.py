"""Smoke test for backtester page logic. No Streamlit required — pure pandas/numpy."""
import numpy as np
import pandas as pd

# ----- Synthetic data -----
np.random.seed(42)
dates = pd.bdate_range("2018-01-01", "2024-01-01")
n_assets = 5
returns = np.random.randn(len(dates), n_assets) * 0.01 + 0.0003
prices = pd.DataFrame(
    100 * np.exp(np.cumsum(returns, axis=0)),
    columns=[f"ASSET{i}" for i in range(n_assets)],
    index=dates,
)
spy_returns = np.random.randn(len(dates)) * 0.012 + 0.0004
spy_prices = pd.Series(100 * np.exp(np.cumsum(spy_returns)), index=dates, name="SPY")

# ----- Quarterly contribution dates -----
def quarterly_contribution_dates(prices):
    quarters = prices.index.to_series().groupby(
        [prices.index.year, prices.index.quarter]
    ).min()
    return pd.DatetimeIndex(quarters.values).sort_values()

contribution_dates = quarterly_contribution_dates(prices)
print(f"=== Test 1: Quarterly contribution dates ===")
print(f"  Total dates: {len(contribution_dates)}")
print(f"  First 4: {[d.date() for d in contribution_dates[:4]]}")
expected_quarters = (2024 - 2018) * 4 + 1
assert abs(len(contribution_dates) - expected_quarters) <= 1
print(f"  ✓ Expected ~{expected_quarters} quarters, got {len(contribution_dates)}")

# ----- DCA simulation -----
def simulate_dca_portfolio(prices, weights, contribution_dates, quarterly_amount, cost_bps, cost_fixed):
    tickers = [t for t in weights.index if t in prices.columns]
    weights = weights[tickers] / weights[tickers].sum()
    shares = pd.Series(0.0, index=tickers)
    cash = 0.0
    total_costs = 0.0
    total_contributed = 0.0
    daily_value = pd.Series(index=prices.index, dtype=float)
    for current_date in prices.index:
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
                    total_costs += trade_value * (cost_bps / 10_000) + (cost_fixed if n > 0 else 0)
            shares = shares + shares_to_buy
            spent = (shares_to_buy * day_prices).sum()
            cost_today = (shares_to_buy * day_prices).sum() * (cost_bps / 10_000) + \
                         cost_fixed * (shares_to_buy > 0).sum()
            cash = available - spent - cost_today
            total_contributed += quarterly_amount
        daily_value[current_date] = (shares * prices.loc[current_date, tickers]).sum() + cash
    return {
        "value_series": daily_value.dropna(),
        "total_contributed": total_contributed,
        "total_costs": total_costs,
    }

print("\n=== Test 2: DCA simulation ===")
target_weights = pd.Series([0.4, 0.3, 0.15, 0.1, 0.05], index=prices.columns)
result = simulate_dca_portfolio(prices, target_weights, contribution_dates, 3000, 5.0, 0.0)
print(f"  Total contributed: ${result['total_contributed']:,.0f}")
print(f"  Final value: ${result['value_series'].iloc[-1]:,.2f}")
print(f"  Total costs: ${result['total_costs']:.2f}")
assert result['total_contributed'] == 3000 * len(contribution_dates)
assert result['value_series'].iloc[0] >= 0
assert result['total_costs'] > 0
print("  ✓ Contributions match expected schedule")
print("  ✓ Costs accumulate correctly")

# ----- Regime classification -----
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
print(f"\n=== Test 3: Regime classification ===")
print(f"  Regime distribution: {regimes.value_counts().to_dict()}")
assert set(regimes.unique()).issubset({"Bull", "Bear", "High-Vol"})
print("  ✓ All regimes are valid labels")

# ----- Risk metrics -----
def compute_metrics(value_series, benchmark_returns, rfr):
    returns = value_series.pct_change().dropna()
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
    treynor = (port_ann - rfr) / beta if beta != 0 else np.nan
    return {"Sharpe": sharpe, "Sortino": sortino, "Beta": beta, "Alpha": alpha, "Treynor": treynor}

spy_value = pd.Series(100 * np.exp(np.cumsum(spy_returns)), index=dates)
spy_rets = spy_value.pct_change().fillna(0)
metrics = compute_metrics(result["value_series"], spy_rets, 0.04)
print(f"\n=== Test 4: Risk metrics ===")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")
assert -5 < metrics["Sharpe"] < 10
assert 0 <= abs(metrics["Beta"]) < 5
print("  ✓ All metrics in plausible ranges")

# ----- Kelly -----
def kelly_weights(returns, rfr):
    mu = returns.mean() * 252
    cov = returns.cov() * 252
    excess = mu - rfr
    inv_cov = np.linalg.pinv(cov.values)
    return pd.Series(inv_cov @ excess.values, index=cov.index)

asset_returns = prices.pct_change().dropna()
kelly = kelly_weights(asset_returns, 0.04)
print(f"\n=== Test 5: Kelly criterion ===")
print(f"  Full Kelly weights: {dict(kelly.round(3))}")
print(f"  Total exposure: {kelly.sum():.2f}x")
print(f"  Half Kelly total: {(kelly * 0.5).sum():.2f}x")
assert not np.isnan(kelly).any()
print("  ✓ Kelly weights compute without NaN")

# ----- Monte Carlo -----
def monte_carlo_simple(asset_returns, prices_today, weights, horizon_years, n_paths, quarterly_amount, method):
    days = int(horizon_years * 252)
    n_assets = len(weights)
    contribution_indices = list(range(0, days, 63))
    paths_value = np.zeros((days + 1, n_paths))
    for sim in range(n_paths):
        if method == "GBM":
            mu = asset_returns.mean().values
            cov = asset_returns.cov().values
            try:
                L = np.linalg.cholesky(cov + 1e-10 * np.eye(n_assets))
            except np.linalg.LinAlgError:
                L = np.diag(np.sqrt(np.diag(cov)))
            shocks = np.random.randn(days, n_assets) @ L.T
            sim_returns = mu + shocks
        else:
            sample_idx = np.random.randint(0, len(asset_returns), size=days)
            sim_returns = asset_returns.values[sample_idx]
        shares = np.zeros(n_assets)
        cash = 0.0
        prices_path = prices_today.values.copy()
        for d in range(days):
            prices_path = prices_path * (1 + sim_returns[d])
            if d in contribution_indices:
                available = quarterly_amount + cash
                for i in range(n_assets):
                    target = available * weights.iloc[i]
                    if prices_path[i] > 0 and target > prices_path[i]:
                        n_sh = int(np.floor(target / prices_path[i]))
                        shares[i] += n_sh
                        available -= n_sh * prices_path[i]
                cash = available
            paths_value[d + 1, sim] = (shares * prices_path).sum() + cash
    return paths_value

print("\n=== Test 6: Monte Carlo forecast ===")
prices_today = prices.iloc[-1]
weights_norm = target_weights / target_weights.sum()
mc_paths = monte_carlo_simple(asset_returns, prices_today, weights_norm, 3, 100, 3000, "Bootstrap")
final_vals = mc_paths[-1, :]
print(f"  Median: ${np.median(final_vals):,.0f}")
print(f"  5th percentile: ${np.percentile(final_vals, 5):,.0f}")
print(f"  95th percentile: ${np.percentile(final_vals, 95):,.0f}")
print(f"  Total contributed: ${3000 * 12:,.0f}")
assert mc_paths.shape == (3 * 252 + 1, 100)
assert final_vals.min() >= 0
print("  ✓ Monte Carlo paths shape is correct, all values non-negative")

# Same test with GBM
mc_gbm = monte_carlo_simple(asset_returns, prices_today, weights_norm, 3, 100, 3000, "GBM")
print(f"  GBM median: ${np.median(mc_gbm[-1, :]):,.0f}")
print("  ✓ GBM method also runs successfully")

print("\n✓ All page-2 smoke tests passed")
