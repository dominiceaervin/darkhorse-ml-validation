"""Chronological walk-forward validation.

For time-series prediction tasks, k-fold cross-validation with random shuffling
leaks future info into training. This module implements strict chronological
splits: each test window uses only data prior to it for training.

Typical use:
    - 3 non-overlapping test windows
    - Each fold trains on all data before test_start
    - Reports per-fold + aggregate metrics
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


@dataclass
class Fold:
    """A single train/test split."""

    fold_id: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_idx: pd.Index
    test_idx: pd.Index


def make_chronological_folds(
    df: pd.DataFrame,
    test_windows: list[tuple[str, str]],
) -> list[Fold]:
    """Build train/test index pairs for each specified test window.

    Args:
        df: Feature DataFrame with DatetimeIndex (must be tz-aware UTC).
        test_windows: List of (start, end) date strings for each test window.

    Returns:
        List of Fold objects. train_idx contains all rows before test_start.
    """
    folds = []
    for i, (start_str, end_str) in enumerate(test_windows, 1):
        test_start = pd.Timestamp(start_str, tz="UTC")
        test_end = pd.Timestamp(end_str, tz="UTC")
        train_mask = df.index < test_start
        test_mask = (df.index >= test_start) & (df.index < test_end)
        folds.append(
            Fold(
                fold_id=i,
                test_start=test_start,
                test_end=test_end,
                train_idx=df.index[train_mask],
                test_idx=df.index[test_mask],
            )
        )
    return folds


def evaluate_walk_forward(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    train_fn: Callable[[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], np.ndarray],
    test_windows: list[tuple[str, str]],
    verbose: bool = True,
) -> pd.DataFrame:
    """Run walk-forward evaluation.

    Args:
        df: Feature+target DataFrame with DatetimeIndex.
        features: List of feature column names.
        target_col: Target column name (binary).
        train_fn: Callable (X_train, y_train, X_test, y_test) → array of test predictions.
        test_windows: List of (start_date, end_date) tuples.
        verbose: Print per-fold results as they complete.

    Returns:
        DataFrame with one row per fold showing metrics.
    """
    folds = make_chronological_folds(df, test_windows)
    results = []
    for fold in folds:
        X_train = df.loc[fold.train_idx, features]
        y_train = df.loc[fold.train_idx, target_col]
        X_test = df.loc[fold.test_idx, features]
        y_test = df.loc[fold.test_idx, target_col]

        n_pos_train = int(y_train.sum())
        n_pos_test = int(y_test.sum())

        if n_pos_train < 5:
            if verbose:
                print(
                    f"Fold {fold.fold_id}: skipped (only {n_pos_train} positive"
                    f" training examples)"
                )
            continue

        p_test = train_fn(X_train, y_train, X_test, y_test)
        auc = roc_auc_score(y_test, p_test) if n_pos_test > 0 else np.nan
        base_rate = y_test.mean() * 100

        results.append(
            {
                "fold": fold.fold_id,
                "test_range": f"{fold.test_start.date()} → {fold.test_end.date()}",
                "n_train": len(X_train),
                "n_test": len(X_test),
                "pos_train": n_pos_train,
                "pos_test": n_pos_test,
                "base_rate": base_rate,
                "auc": auc,
            }
        )

        if verbose:
            print(
                f"Fold {fold.fold_id} ({fold.test_start.date()}"
                f" → {fold.test_end.date()}):  "
                f"n_train={len(X_train):,} (pos={n_pos_train}) "
                f"n_test={len(X_test):,} (pos={n_pos_test}, "
                f"base={base_rate:.2f}%)  AUC={auc:.4f}"
            )

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Toy example
    from features import build_features, compute_target, resample_5m
    from synthetic_bars import generate_bars

    bars_1m = generate_bars(n_days=300)
    bars_5m = resample_5m(bars_1m)
    df, features = build_features(bars_5m)
    df["target"] = compute_target(bars_5m)
    df = df.dropna(subset=["target"])

    def dummy_train(X_train, y_train, X_test, y_test):
        # Predict base rate — worst-case AUC baseline
        return np.full(len(X_test), y_train.mean())

    test_windows = [
        ("2023-06-01", "2023-09-01"),
        ("2023-09-01", "2023-12-01"),
        ("2023-12-01", "2024-03-01"),
    ]

    print("Walk-forward evaluation with dummy predictor (AUC ~0.5 expected):")
    results = evaluate_walk_forward(df, features, "target", dummy_train, test_windows)
    print("\nAggregate:")
    print(results)
