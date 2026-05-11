# Portfolio Construction Model — v2

A Streamlit app that builds a quarterly contribution portfolio with **defensible methodology**:
walk-forward backtesting, CAPM expected returns, CRRA-derived bucket weights, block bootstrap
Monte Carlo, and 60/40 + SPY benchmarks.

## What changed from v1 (and why)

| v1 issue | v2 fix |
|----------|--------|
| Mean-historical-return for μ — known to be a noisy estimator | **CAPM expected returns** anchored to SPY as market proxy |
| Bucket weights hand-picked from intuition | **CRRA utility-derived** bucket weights (Merton solution with no-leverage cap) |
| Optimizer used full price history (lookahead bias) | **Walk-forward** weight computation — every weight uses only data prior to its date |
| Buy-and-forget portfolio drifted from target weights | **Quarterly rebalancing** back to target |
| Only SPY benchmark (a softball) | **60/40 VTI/AGG benchmark** added — the practitioner default |
| Sortino + Treynor (Treynor meaningless for bond-heavy portfolios) | Treynor dropped; **Calmar ratio** and **Ulcer Index** added |
| IID bootstrap Monte Carlo destroyed autocorrelation | **Block bootstrap** preserves volatility clustering |
| GBM Monte Carlo had thin tails | Removed; block bootstrap is statistically better-founded for portfolio sims |
| No correlation diagnostics | Pairwise correlation heatmap added |
| No view of what's driving expected returns | CAPM β table shown on builder page |
| Single price source (yfinance) — fragile on Streamlit Cloud | **Stooq fallback** via pandas-datareader |
| Dead SP500 universe loader | Removed |

## File structure

```
portfolio_v2/
├── app.py                          # Page 1: Builder
├── portfolio_core.py               # Shared methodology
├── pages/
│   └── 2_Backtest_and_Analysis.py  # Page 2: Walk-forward backtest
├── requirements.txt
├── README.md
└── test_v2_methodology.py          # Smoke tests for the new methodology
```

## Setup

```bash
cd portfolio_v2
pip install -r requirements.txt
streamlit run app.py
```

## Methodology details

### Bucket weights — CRRA utility (Merton portfolio choice)

The risk slider maps to a CRRA risk aversion parameter γ (10 = very conservative, 1 = aggressive).
The Merton solution allocates each bucket b proportional to:

```
w_b ∝ excess_return_b / (γ · σ_b²)
```

With a no-leverage cap (sum ≤ 1). When the cap binds at low γ, an additional growth tilt is
applied to reflect that risk-tolerant investors prefer higher-vol exposure even at full investment.

Bucket priors are static excess-return assumptions (~0.5% conservative, ~1.4% hybrid, ~2.2% growth)
that anchor the framework. Within each bucket, the optimizer uses **actual realized data** through
CAPM, so the within-bucket allocation is fully data-driven.

### Within-bucket optimization — Max-Sharpe with CAPM μ

Within each bucket, weights are determined by max-Sharpe optimization (PyPortfolioOpt):
- Expected returns from CAPM: μ_i = r_f + β_i · (E[R_m] - r_f)
- Covariance: Ledoit-Wolf shrinkage estimator
- Constraint: no asset > 40% of bucket
- Optional: Kelly-derived per-asset upper bounds

### Backtest — walk-forward, no lookahead

At each quarterly rebalance date:
1. Window = the prior N days (configurable, default 756 = ~3 years)
2. Compute weights using **only this window**
3. Apply weights forward to the next rebalance date
4. Trade to target with transaction costs

This is the single most important fix. v1 optimized on the full history and "backtested" on the same
data — that's curve-fitting. v2's weights only use information that would have been available at
each historical point.

### Monte Carlo — block bootstrap

Resamples **contiguous blocks** of historical returns (default 10 days) rather than individual days.
This preserves:
- Volatility clustering (high-vol days tend to follow high-vol days)
- Autocorrelation
- Empirical distribution shape (no Gaussian assumption)

IID bootstrap (v1) destroys these features and systematically understates tail risk.

### Risk metrics

- **Sharpe** — return per unit of total volatility (standard)
- **Sortino** — return per unit of downside volatility (penalizes only bad vol)
- **Beta vs SPY** — market sensitivity (kept as a sanity check, less meaningful for bond-heavy portfolios)
- **Alpha** — CAPM-implied excess return
- **Max Drawdown** — worst peak-to-trough decline
- **Calmar Ratio** — CAGR / |Max DD| — how much return per unit of drawdown
- **Ulcer Index** — RMS of drawdown series — penalizes sustained underwater periods

Treynor ratio is intentionally dropped: it assumes systematic risk dominates, which fails for bonds.

### Data sources

Two-tier fallback for price data:
1. **yfinance** (primary)
2. **Stooq** via pandas-datareader (fallback when yfinance fails or returns sparse data)

This eliminates the single-point-of-failure problem v1 had on Streamlit Cloud.

## Known limitations (still imperfect)

1. **Survivorship bias.** The universe is current-listed ETFs. Pre-inception backtests aren't
   possible for newer ETFs (HYG: 2007, QQQ: 1999, VUG: 2004).
2. **Single-factor model.** CAPM uses SPY as the market. Multi-factor (Fama-French / Carhart)
   attribution would be more rigorous but adds dependencies on Ken French's data library.
3. **Static bucket priors.** The excess-return assumptions don't adapt to changing yield curves
   beyond what flows through the risk-free rate input.
4. **No tax considerations.** Asset location (bonds in IRA, growth in Roth, broad index in taxable)
   matters in reality and isn't modeled.
5. **Whole-share rounding** creates small cash drag (reported transparently in the UI).

These are documented in the app itself in methodology-note blocks so users (and reviewers)
see them in context.
