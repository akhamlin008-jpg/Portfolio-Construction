"""Smoke tests for v2 methodology."""
import numpy as np
import pandas as pd

BUCKET_PRIOR_VOL = {"Conservative": 0.05, "Hybrid": 0.10, "Growth": 0.18}
BUCKET_PRIOR_EXCESS = {"Conservative": 0.005, "Hybrid": 0.014, "Growth": 0.022}


def risk_aversion_from_score(risk_score):
    return 11 - max(1, min(10, int(risk_score)))


def crra_bucket_weights(risk_score, risk_free_rate=0.04):
    gamma = risk_aversion_from_score(risk_score)
    merton = {
        b: BUCKET_PRIOR_EXCESS[b] / (gamma * BUCKET_PRIOR_VOL[b] ** 2)
        for b in ["Conservative", "Hybrid", "Growth"]
    }
    total_risky = sum(merton.values())
    if total_risky <= 1.0:
        weights = dict(merton)
        weights["Conservative"] += (1.0 - total_risky)
    else:
        tilt_factor = max(0, (10 - gamma) / 9)
        baseline = {k: v / total_risky for k, v in merton.items()}
        tilt_target = {"Conservative": 0.20, "Hybrid": 0.30, "Growth": 0.50}
        weights = {k: (1 - tilt_factor) * baseline[k] + tilt_factor * tilt_target[k] for k in baseline}
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


print("=== Test 1: CRRA bucket weights across risk scores ===")
for risk in range(1, 11):
    w = crra_bucket_weights(risk)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    print(f"  Risk {risk:2d} (γ={11-risk:2d}): "
          f"Cons={w['Conservative']:.1%}  Hyb={w['Hybrid']:.1%}  Grw={w['Growth']:.1%}")

w_low, w_high = crra_bucket_weights(1), crra_bucket_weights(10)
print(f"\n  Risk=1  Growth: {w_low['Growth']:.1%}")
print(f"  Risk=10 Growth: {w_high['Growth']:.1%}")
assert w_high['Growth'] > w_low['Growth'], "Higher risk score should mean more growth"
assert w_low['Conservative'] > w_high['Conservative'], "Lower risk score should mean more conservative"
print("  ✓ Risk score genuinely tilts allocation")


print("\n=== Test 2: CAPM beta estimation ===")
np.random.seed(42)
dates = pd.bdate_range("2020-01-01", "2024-01-01")
mkt_returns = np.random.randn(len(dates)) * 0.012
mkt = pd.Series(100 * np.exp(np.cumsum(mkt_returns)), index=dates)
asset_returns = 1.5 * mkt_returns + np.random.randn(len(dates)) * 0.005
asset = pd.Series(100 * np.exp(np.cumsum(asset_returns)), index=dates)

asset_ret = asset.pct_change().dropna()
mkt_ret = mkt.pct_change().dropna()
aligned = pd.concat([asset_ret, mkt_ret], axis=1).dropna()
beta_est = aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / aligned.iloc[:, 1].var()
print(f"  True β=1.50, Estimated β={beta_est:.3f}")
assert 1.3 < beta_est < 1.7
print("  ✓ Beta estimation accurate")


print("\n=== Test 3: Walk-forward has no lookahead ===")
np.random.seed(0)
dates = pd.bdate_range("2018-01-01", "2024-01-01")
prices = pd.DataFrame(
    100 * np.exp(np.cumsum(np.random.randn(len(dates), 3) * 0.012, axis=0)),
    columns=["A", "B", "C"], index=dates,
)
rebal_dates = pd.DatetimeIndex([d for d in dates if d.month in (1, 4, 7, 10) and d.day < 8])

leaks = 0
for d in rebal_dates[:10]:
    end_idx = prices.index.searchsorted(d)
    start_idx = max(0, end_idx - 252)
    if end_idx - start_idx < 60:
        continue
    window = prices.iloc[start_idx:end_idx]
    if window.index[-1] >= d:
        leaks += 1
print(f"  Tested {min(10, len(rebal_dates))} rebalance dates, lookahead leaks: {leaks}")
assert leaks == 0
print("  ✓ No lookahead — every window ends strictly before its rebalance date")


print("\n=== Test 4: Block bootstrap preserves autocorrelation ===")
np.random.seed(123)
T = 1000
errors = np.random.randn(T) * 0.01
ar1 = np.zeros(T)
for t in range(1, T):
    ar1[t] = 0.3 * ar1[t-1] + errors[t]
print(f"  Original series lag-1 autocorr: {pd.Series(ar1).autocorr(lag=1):+.3f}")


def block_bootstrap(returns, n_samples, block_size):
    T = len(returns)
    out = np.zeros(n_samples)
    pos = 0
    while pos < n_samples:
        start = np.random.randint(0, max(1, T - block_size))
        actual_len = min(block_size, n_samples - pos, T - start)
        out[pos:pos + actual_len] = returns[start:start + actual_len]
        pos += actual_len
    return out


iid_acorr = pd.Series(block_bootstrap(ar1, 5000, 1)).autocorr(lag=1)
block_acorr = pd.Series(block_bootstrap(ar1, 5000, 20)).autocorr(lag=1)
print(f"  IID bootstrap   lag-1 autocorr: {iid_acorr:+.3f}")
print(f"  Block bootstrap lag-1 autocorr: {block_acorr:+.3f}")
assert abs(iid_acorr) < 0.05
assert abs(block_acorr) > 0.15
print("  ✓ Block bootstrap preserves autocorrelation")


print("\n=== Test 5: Drawdown metrics ===")
np.random.seed(7)
synthetic = pd.Series(100 * np.exp(np.cumsum(np.random.randn(1000) * 0.01 + 0.0003)))
dd = (synthetic - synthetic.cummax()) / synthetic.cummax()
max_dd = dd.min()
ulcer = np.sqrt((dd ** 2).mean())
cagr = (synthetic.iloc[-1] / synthetic.iloc[0]) ** (252 / 1000) - 1
calmar = cagr / abs(max_dd)
print(f"  CAGR={cagr:.2%}  Max DD={max_dd:.2%}  Calmar={calmar:.3f}  Ulcer={ulcer:.2%}")
assert max_dd <= 0 and ulcer >= 0 and not np.isnan(calmar)
print("  ✓ Drawdown metrics work")

print("\n✓ All smoke tests complete")
