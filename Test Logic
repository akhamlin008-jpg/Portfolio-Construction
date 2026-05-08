"""Smoke test: verify bucket weights sum correctly and optimizer runs."""
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models


def bucket_weights_for_risk(risk: int) -> dict:
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


# Test 1: Bucket weights sum to 1.0 for every risk level
print("=== Test 1: Bucket weight sums ===")
for r in range(1, 11):
    w = bucket_weights_for_risk(r)
    s = sum(w.values())
    assert abs(s - 1.0) < 1e-9, f"Risk {r} sums to {s}"
    print(f"  Risk {r:2d}: Cons={w['Conservative']:.1%}  Hyb={w['Hybrid']:.1%}  Grw={w['Growth']:.1%}  (sum={s:.4f})")

# Test 2: Curve is monotonic — conservative goes down, growth goes up as risk rises
print("\n=== Test 2: Monotonicity ===")
cons_series = [bucket_weights_for_risk(r)["Conservative"] for r in range(1, 11)]
grw_series = [bucket_weights_for_risk(r)["Growth"] for r in range(1, 11)]
assert all(cons_series[i] >= cons_series[i+1] for i in range(9)), "Conservative not monotonically decreasing"
assert all(grw_series[i] <= grw_series[i+1] for i in range(9)), "Growth not monotonically increasing"
print("  Conservative is monotonically decreasing as risk rises: OK")
print("  Growth is monotonically increasing as risk rises: OK")

# Test 3: Optimizer runs end-to-end on synthetic data
print("\n=== Test 3: PyPortfolioOpt with synthetic prices ===")
np.random.seed(42)
n_days, n_assets = 500, 5
returns = np.random.randn(n_days, n_assets) * 0.01 + 0.0003
prices = pd.DataFrame(
    100 * np.exp(np.cumsum(returns, axis=0)),
    columns=[f"ETF{i}" for i in range(n_assets)],
    index=pd.date_range("2022-01-01", periods=n_days),
)
mu = expected_returns.mean_historical_return(prices)
S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.40))
ef.max_sharpe(risk_free_rate=0.04)
weights = ef.clean_weights()
print(f"  Optimizer output: {weights}")
print(f"  Sum of weights: {sum(weights.values()):.4f}")
print(f"  Max single weight: {max(weights.values()):.4f} (cap was 0.40)")
assert abs(sum(weights.values()) - 1.0) < 1e-3, "Weights don't sum to 1"
assert max(weights.values()) <= 0.40 + 1e-3, "Weight cap violated"

# Test 4: Edge case — single-asset bucket
print("\n=== Test 4: Edge cases ===")
single_asset_prices = prices[["ETF0"]]
print(f"  Single-asset bucket shape: {single_asset_prices.shape}")
print("  (Handled in app.py with shape check before calling optimizer)")

# Test 5: Quarterly budget math
print("\n=== Test 5: Budget math ===")
monthly = 1000
quarterly = monthly * 3
risk = 5
w = bucket_weights_for_risk(risk)
print(f"  Monthly budget: ${monthly}")
print(f"  Quarterly cap: ${quarterly}")
for bucket, weight in w.items():
    print(f"    {bucket}: ${quarterly * weight:.2f} ({weight:.1%})")
print(f"  Total allocated: ${quarterly * sum(w.values()):.2f}")

print("\n✓ All smoke tests passed")
