# Portfolio Construction Model

A Streamlit app that builds a quarterly contribution portfolio for one of three
investor profiles (Conservative / Hybrid / Growth), driven by a risk-tolerance
slider and a monthly budget.

## What it does

1. **Pulls the ticker universe live from GitHub:**
   - [echuvyrov/TrackingETFs](https://github.com/echuvyrov/TrackingETFs) — curated ETF list (bonds + growth)
   - [Ate329/top-us-stock-tickers](https://github.com/Ate329/top-us-stock-tickers) — daily-updated S&P 500 constituents
2. **Maps a risk score (1–10) to bucket weights** (hand-rolled, transparent):
   - Conservative bucket: SHY, AGG, BND, TIP, LQD, CSJ, MBB
   - Hybrid bucket: HYG, JNK, EMB, VIG, SDY, DVY, VNQ, PFF
   - Growth bucket: VUG, IVW, IWF, IWO, VTI, QQQ, XLK, XLV
3. **Pulls historical prices via `yfinance`** for the chosen lookback window.
4. **Runs max-Sharpe optimization within each bucket** using PyPortfolioOpt
   (with weight caps to prevent concentration), then scales each bucket by
   its target weight from step 2.
5. **Converts target dollars → whole shares** at the latest closing price,
   capped at quarterly budget = monthly × 3.

## Setup

```bash
cd portfolio_app
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## How the inputs work

| Input | Effect |
|-------|--------|
| **Monthly budget** | Quarterly cap = `monthly × 3`. All allocation dollar amounts respect this cap. |
| **Risk tolerance (1–10)** | Drives bucket weights via piecewise-linear interpolation. 1 = 90% conservative, 5 = 40/35/25 split, 10 = 75% growth. |
| **Lookback years** | How many years of historical prices to use when estimating expected returns and the covariance matrix. |

## Architecture notes

- **Bucket allocation is hand-rolled** so the risk → weight mapping is fully visible in `bucket_weights_for_risk()`. Edit that function to retune the curve.
- **Within-bucket optimization uses PyPortfolioOpt** (Ledoit-Wolf shrinkage covariance, max-Sharpe with `weight_bounds=(0, 0.40)` to avoid single-asset dominance).
- **Caching:** ticker universes cached for 6h, prices for 30min — adjust the `ttl` values in `@st.cache_data` decorators if you want fresher data.
- **Whole shares only.** The leftover cash is reported as "Cash remaining" so you can see exactly what's unspent each quarter.

## Caveats

- This is a demonstration model. It is not investment advice.
- Mean-historical-return is a known weak estimator of expected returns; for production use, swap in something like Black-Litterman or CAPM expected returns from PyPortfolioOpt.
- The S&P 500 list is loaded but not currently used in allocation — the app is ETF-only by design (cleaner risk buckets). To extend to single-name equities, add a fourth bucket and pull tickers from `load_sp500_universe()`.
