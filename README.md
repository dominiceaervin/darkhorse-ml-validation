# darkhorse-ml-validation

Statistical validation toolkit for time-series binary classification, with focus on financial trade-signal use cases. Extracted from a production algorithmic trading system.

Three components that plug together but also work independently:

- **`walk_forward.py`** — Chronological k-fold cross-validation without lookahead
- **`pipeline.py`** — LightGBM + Random Forest + MLP ensemble with class-imbalance handling
- **`mc_stress.py`** — Block-permutation Monte Carlo for capital sizing and tail-risk quantification

Runs on synthetic data out of the box. Bring your own OHLCV parquet for real use.

## The problem this solves

If you're building a predictive model on time-series data and want to know:
- "Will this AUC hold on out-of-sample data?" → walk-forward
- "How stable are the predictions across model families?" → ensemble
- "How much capital do I need to survive worst-case realizations?" → Monte Carlo

...random k-fold cross-validation lies to you (leaks future info via shuffling), a single model overfits to its own inductive bias, and naive bootstrap of P&L understates tail risk on serially-correlated returns.

This toolkit addresses all three with methodology I've used in production for a live trading system.

## Quick start

```bash
git clone https://github.com/dominiceaervin/darkhorse-overview-ml-validation
cd darkhorse-ml-validation
pip install -r requirements.txt
python example.py
```

That runs the full pipeline on synthetic bar data (~600 sessions, ~10 seconds wall time). Output shows per-fold AUC, precision-at-threshold table, and Monte Carlo capital-sensitivity sweep.

## Module overview

**`synthetic_bars.py`** — Generates realistic-looking 1-min OHLCV bars with intraday volatility profiles, session boundaries at 18:00 ET, and occasional high-vol clusters. Useful for testing without needing real market data.

**`features.py`** — Feature engineering with strict no-lookahead guarantees. Produces 26 features covering recent volatility, momentum, session context, time-of-day cyclic encoding, and volume patterns. Target definition: 40-point upward move within next 60 minutes.

**`walk_forward.py`** — Chronological train/test splits. Each fold trains on all data *before* the test window's start. Returns a DataFrame with per-fold metrics + captured out-of-sample predictions.

**`pipeline.py`** — Three-model ensemble:
- LightGBM: general-purpose gradient booster, handles NaN natively
- Random Forest: class-balanced bagging for robustness
- MLP: nonlinear interactions in scaled feature space, positive class oversampled

Combines via simple probability mean. Design rationale: different model families have different failure modes (LGBM sharp probability distributions, MLP tends toward extremes on imbalanced data). Averaging smooths this out for downstream use.

**`mc_stress.py`** — Block-permutation MC on trade P&L series. Samples contiguous blocks (default 20 trades) to preserve serial correlation in outcomes. Reports:
- Breach probability at a kill-switch threshold
- Ruin probability (equity ≤ 0)
- Worst-1% and worst-5% troughs
- Median and quantile end-of-period equity

## Example output

```
STAGE 3 — 3-fold chronological walk-forward ensemble training
Fold 1 (2023-08-29 → 2023-12-26):  n_train=13,430 (pos=2144)  AUC=0.5619
Fold 2 (2023-12-26 → 2024-04-23):  n_train=20,145 (pos=3747)  AUC=0.5008
Fold 3 (2024-04-23 → 2024-08-20):  n_train=26,860 (pos=5717)  AUC=0.5451

STAGE 4 — Precision @ threshold (ensemble OOS predictions)
 threshold  n_predicted   precision   lift_vs_base
      0.30        12086       0.302           1.07x
      0.50         1000       0.464           1.65x
      0.70            9       0.667           2.37x

STAGE 5 — Block-permutation Monte Carlo (55% WR simulated trades)
 start_capital  breach_pct   worst_1pct_trough   median_end
          2500       6.36%              $1,716      $34,750
          5000       0.00%              $4,216      $37,250
         10000       0.00%              $9,216      $42,250
```

**Note:** AUC hovers around 0.50 on synthetic data because the bars are generated from pure Brownian motion — there's no real signal to learn. This is the correct behavior (validates the framework isn't leaking future data). On real market data with genuine signal, expect AUC 0.65-0.85 depending on the target.

Full example output: [`example_output.txt`](example_output.txt)

## Bring your own data

The pipeline expects a 1-min OHLCV DataFrame with:
- **Index:** tz-aware UTC DatetimeIndex
- **Columns:** `open`, `high`, `low`, `close`, `volume`

Replace `generate_bars()` in `example.py` with `pd.read_parquet("your_bars.parquet")`. Everything downstream is data-source-agnostic.

## Methodology notes and caveats

**Lookahead:** All features use `.shift(1)` or bar-close values. Target uses `shift(-forward_window)` — future data is used only for label generation, never for features.

**Base rate depends on data:** The 40pt-move-in-60-min target has different frequencies in different regimes. Synthetic data uses ~23-31% base rate; real ES futures data typically shows 1-3% depending on volatility regime.

**Ensemble is not always best:** LGBM alone often has higher AUC than the ensemble, especially in regime-favorable folds. The ensemble's value is in producing smoother probability distributions across the [0, 1] range, which matters more for downstream use (e.g., bracket-widening decisions on high-conviction predictions) than for pure classification AUC.

**Block-permutation vs. IID bootstrap:** Block-permutation preserves autocorrelation in the P&L series. If your strategy has meaningful clustering of wins/losses (most do), IID bootstrap will underestimate tail risk by ~30-50%.

**Not financial advice:** This is validation methodology, not a trading strategy. If you use this to size real capital, you own the outcome.

## Related

Extracted from [DarkHorse](https://github.com/dominiceaervin/darkhorse-overview) — a production algorithmic trading system for ES futures. Original methodology used to size kill switches, calibrate walk-forward gates, and evaluate signal deployment.

## License

MIT — do what you want with this. If it saves you time, let me know.

## Contact

Dominic Ervin — dominiceaervin@gmail.com
