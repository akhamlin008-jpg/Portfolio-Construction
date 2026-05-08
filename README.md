# Portfolio Construction Model

A Streamlit app that builds a quarterly contribution portfolio for one of three
investor profiles (Conservative / Hybrid / Growth), with a full backtest and
analysis page.

## File structure

```
portfolio_app/
├── app.py                          ← Page 1: Portfolio Builder
├── portfolio_core.py               ← Shared logic (constants, loaders, optimizer)
├── pages/
│   └── 2_Backtest_and_Analysis.py  ← Page 2: Backtest, metrics, MC, sensitivity
├── requirements.txt
├── README.md
├── test_logic.py                   ← Smoke test for builder logic
└── test_page2_logic.py             ← Smoke test for backtester logic
```

The `pages/` folder is recognized automatically by Streamlit and creates the
multi-page navigation in the sidebar.

## Setup

```bash
cd portfolio_app
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`. Switch between **Portfolio Construction Model** and **Backtest and Analysis** in the sidebar.

## Page 1 — Builder

- Pulls ticker universes from GitHub (`echuvyrov/TrackingETFs`, `Ate329/top-us-stock-tickers`).
- Maps risk score (1–10) to bucket weights (Conservative / Hybrid / Growth).
- Pulls historical prices via yfinance.
- Runs max-Sharpe optimization within each bucket (PyPortfolioOpt, Ledoit-Wolf shrinkage, 40% single-asset cap).
- Sizes whole-share allocations to your quarterly cap (`monthly × 3`).

Outputs persist to `st.session_state` so Page 2 inherits the same risk score, budget, and risk-free rate.

## Page 2 — Backtest & Analysis

Everything below uses the same allocation logic as Page 1, applied historically.

| Section | What it does |
|---------|--------------|
| **Backtest** | Simulates quarterly contributions on the first trading day of each quarter, applied identically to your portfolio and a SPY buy-and-hold. Same dollars, same dates — fair comparison. |
| **Regime breakdown** | Classifies SPY history into Bull / Bear / High-Vol regimes (Bear = 20%+ drawdown; High-Vol = SPY 30-day annualized vol > 25%). Reports your portfolio's annualized return and vol *within each regime*. |
| **Transaction costs** | Configurable basis-points-per-trade plus optional fixed commission per ticker. Applied to both portfolio and SPY benchmark. |
| **Risk metrics** | Sharpe, Sortino, Beta vs SPY, Treynor, Alpha (CAPM-implied), CAGR, max drawdown — for both portfolio and SPY side-by-side. |
| **Kelly criterion** | Multivariate Kelly: f* = Σ⁻¹ · (μ − r_f). Shows full / half / quarter Kelly alongside the optimizer's actual weights so you can see when the optimizer is more conservative than Kelly suggests. |
| **Sensitivity analysis** | Sweeps risk score, transaction cost, and monthly budget independently. Shows how final portfolio value moves with each input. |
| **Monte Carlo forecast** | Forward-looking 1–10 year simulation. Two methods: **Bootstrap** (resamples actual historical returns; preserves fat tails) and **GBM** (parametric lognormal). Reports median / 5th / 95th percentile outcomes and probability of underperforming total contributions. |

## Risk-free rate

Editable from either page. Flows through:
- Sharpe ratio calculation
- Treynor ratio
- Alpha (CAPM)
- Kelly criterion (excess return component)
- Optimizer's max-Sharpe target

## Caveats

- This is a demonstration model. Not investment advice.
- Mean-historical-return is a known weak estimator of expected returns. For production, swap in `expected_returns.capm_return()` or Black-Litterman with custom views.
- Monte Carlo assumes the future return distribution resembles the lookback window. It does not predict regime shifts or structural breaks not present in history.
- Whole-share allocation only — leftover cash sits idle (reported transparently).
- Kelly is shown for reference. Full Kelly is famously aggressive; fractional Kelly (½ or ¼) is what most practitioners actually use.
- The S&P 500 list is loaded but not currently used in allocation. To extend to single-name equities, add a fourth bucket and wire `load_sp500_universe()` into the bucket map.

## Streamlit Cloud deployment

If deploying to Streamlit Cloud:
1. `requirements.txt` must be at the **root** of your repo (same level as `app.py`).
2. Set the main file path in Streamlit Cloud settings to `app.py` (case-sensitive).
3. The `pages/` folder must be at the root too — Streamlit auto-discovers it.
