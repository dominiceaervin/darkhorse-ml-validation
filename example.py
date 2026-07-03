"""End-to-end example: synthetic data → features → walk-forward ensemble → MC stress.

Run this to see the full pipeline in action on generated synthetic bars.

    python example.py

Expected wall time: ~30-60 seconds on a modern laptop.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from synthetic_bars import generate_bars
from features import resample_5m, build_features, compute_target
from walk_forward import evaluate_walk_forward
from pipeline import train_ensemble, precision_by_threshold, EnsembleConfig
from mc_stress import block_permutation_mc, stress_sweep_capital


def main() -> None:
    t0 = time.time()

    # ─── Stage 1: Generate synthetic bars ───
    print("=" * 70)
    print("STAGE 1 — Generate synthetic bars")
    print("=" * 70)
    bars_1m = generate_bars(n_days=600, seed=42)
    print(f"  1-min bars: {len(bars_1m):,}")
    print(f"  Date range: {bars_1m.index.min().date()} → {bars_1m.index.max().date()}")
    print(f"  Price range: {bars_1m['close'].min():.2f} → {bars_1m['close'].max():.2f}")

    # ─── Stage 2: Feature engineering ───
    print("\n" + "=" * 70)
    print("STAGE 2 — Feature engineering (5-min resample + 30 features)")
    print("=" * 70)
    bars_5m = resample_5m(bars_1m)
    df, features = build_features(bars_5m)
    df["target"] = compute_target(bars_5m, forward_window_bars=12, up_threshold_pts=40.0)
    df = df.dropna(subset=["target"])
    # Restrict to RTH+PREMKT
    df = df[df["is_eth"] == 0]
    print(f"  5-min bars: {len(bars_5m):,}")
    print(f"  Feature count: {len(features)}")
    print(f"  Training samples: {len(df):,}")
    print(f"  Base rate (40pt UP in 60 min): {df['target'].mean() * 100:.2f}%")

    # ─── Stage 3: Walk-forward validation ───
    print("\n" + "=" * 70)
    print("STAGE 3 — 3-fold chronological walk-forward ensemble training")
    print("=" * 70)

    # Track per-fold ensemble AUC + collect OOS predictions
    fold_results = []
    oos_predictions = pd.Series(np.nan, index=df.index)

    def train_and_capture(X_train, y_train, X_test, y_test):
        r = train_ensemble(X_train, y_train, X_test, y_test)
        fold_results.append(
            {
                "auc_lgbm": r["auc_lgbm"],
                "auc_rf": r["auc_rf"],
                "auc_mlp": r["auc_mlp"],
                "auc_ensemble": r["auc_ensemble"],
            }
        )
        oos_predictions.loc[X_test.index] = r["p_ensemble"]
        return r["p_ensemble"]

    # Choose test windows based on data range
    max_date = df.index.max().date()
    min_date = df.index.min().date()
    total_days = (max_date - min_date).days
    # 3 equal test windows across the last 60% of the data
    train_end_day = int(total_days * 0.4)
    fold_span_days = int((total_days - train_end_day) / 3)

    def d(day_offset):
        return (pd.Timestamp(min_date) + pd.Timedelta(days=day_offset)).strftime(
            "%Y-%m-%d"
        )

    test_windows = [
        (d(train_end_day + i * fold_span_days), d(train_end_day + (i + 1) * fold_span_days))
        for i in range(3)
    ]
    print(f"  Test windows: {test_windows}")
    print()

    wf_summary = evaluate_walk_forward(
        df,
        features,
        "target",
        train_and_capture,
        test_windows,
        verbose=True,
    )

    print("\n  Per-model AUC by fold:")
    for i, r in enumerate(fold_results, 1):
        print(
            f"    Fold {i}:  "
            f"LGBM={r['auc_lgbm']:.4f}  "
            f"RF={r['auc_rf']:.4f}  "
            f"MLP={r['auc_mlp']:.4f}  "
            f"ENSEMBLE={r['auc_ensemble']:.4f}"
        )

    # ─── Stage 4: Precision @ threshold ───
    print("\n" + "=" * 70)
    print("STAGE 4 — Precision @ threshold (ensemble OOS predictions)")
    print("=" * 70)
    oos_df = df.dropna(subset=[])
    oos_df = oos_df.loc[oos_predictions.dropna().index]
    y_true = oos_df["target"].values
    p = oos_predictions.dropna().values

    prec_df = precision_by_threshold(y_true, p)
    print(prec_df.to_string(index=False))

    # ─── Stage 5: Monte Carlo stress test ───
    print("\n" + "=" * 70)
    print("STAGE 5 — Block-permutation Monte Carlo stress test")
    print("=" * 70)
    # Simulated per-trade P&L based on 55% WR at 1.5:1 R/R
    rng = np.random.default_rng(42)
    n_trades = 1000
    wins = rng.random(n_trades) < 0.55
    trade_pnl = np.where(wins, 146.0, -104.0)
    print(f"  Simulated trade P&L series: {n_trades} trades")
    print(f"  Realized WR:                {wins.mean() * 100:.1f}%")
    print(f"  Realized $/trade:           ${trade_pnl.mean():+.2f}")

    print("\n  Capital sensitivity sweep:")
    sweep = stress_sweep_capital(
        trade_pnl,
        starting_capitals=[2500, 5000, 10000, 20000],
        kill_threshold=2000,
        n_paths=20_000,
    )
    print(sweep.to_string(index=False))

    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
