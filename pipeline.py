"""Ensemble ML training pipeline: LightGBM + Random Forest + MLP.

The three-model ensemble was developed to address a common problem in
imbalanced binary classification: any single model may over- or under-fit
to the minority class in ways that reduce downstream utility. Averaging
probability outputs across model families with different inductive biases
often produces smoother, more actionable predictions.

Design choices:
    - LightGBM: strong general-purpose gradient booster, handles NaN natively
    - Random Forest: robustness via bagging, class-balanced sampling
    - MLP: nonlinear interactions in scaled feature space, oversampled positives

Ensemble strategy: simple mean of the three predict_proba outputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


@dataclass
class EnsembleConfig:
    """Hyperparameters for the ensemble."""

    # LightGBM
    lgbm_learning_rate: float = 0.03
    lgbm_num_leaves: int = 63
    lgbm_min_child_samples: int = 300
    lgbm_n_estimators: int = 500
    lgbm_early_stopping: int = 30

    # Random Forest
    rf_n_estimators: int = 300
    rf_max_depth: int = 15
    rf_min_samples_leaf: int = 100

    # MLP
    mlp_hidden_layers: tuple = (64, 32)
    mlp_max_iter: int = 200
    mlp_target_pos_frac: float = 0.15  # oversample positives to this fraction

    random_state: int = 42


def train_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: EnsembleConfig | None = None,
) -> dict:
    """Train LGBM + RF + MLP on a single fold.

    Returns a dict containing per-model probabilities on X_test + ensemble mean.
    """
    cfg = cfg or EnsembleConfig()
    result: dict = {}

    # ── LightGBM (handles NaN natively, no scaling needed) ──
    lgbm = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        learning_rate=cfg.lgbm_learning_rate,
        num_leaves=cfg.lgbm_num_leaves,
        min_child_samples=cfg.lgbm_min_child_samples,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        verbose=-1,
        n_estimators=cfg.lgbm_n_estimators,
    )
    lgbm.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(cfg.lgbm_early_stopping, verbose=False)],
    )
    p_lgbm = lgbm.predict_proba(X_test)[:, 1]
    result["p_lgbm"] = p_lgbm
    result["auc_lgbm"] = (
        roc_auc_score(y_test, p_lgbm) if y_test.sum() > 0 else np.nan
    )

    # ── Preprocess for RF + MLP (impute NaNs + scale for MLP) ──
    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train)
    X_test_i = imputer.transform(X_test)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_i)
    X_test_s = scaler.transform(X_test_i)

    # ── Random Forest ──
    rf = RandomForestClassifier(
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth,
        min_samples_leaf=cfg.rf_min_samples_leaf,
        class_weight="balanced",
        n_jobs=-1,
        random_state=cfg.random_state,
    )
    rf.fit(X_train_i, y_train)
    p_rf = rf.predict_proba(X_test_i)[:, 1]
    result["p_rf"] = p_rf
    result["auc_rf"] = roc_auc_score(y_test, p_rf) if y_test.sum() > 0 else np.nan

    # ── MLP with oversampled positives (handles class imbalance) ──
    y_arr = y_train.values
    pos_idx = np.where(y_arr == 1)[0]
    neg_idx = np.where(y_arr == 0)[0]
    n_pos_target = int(
        len(neg_idx) * cfg.mlp_target_pos_frac / (1 - cfg.mlp_target_pos_frac)
    )
    if n_pos_target > len(pos_idx):
        rng = np.random.RandomState(cfg.random_state)
        pos_over = rng.choice(pos_idx, size=n_pos_target, replace=True)
        train_idx = np.concatenate([neg_idx, pos_over])
        rng.shuffle(train_idx)
        X_mlp = X_train_s[train_idx]
        y_mlp = y_arr[train_idx]
    else:
        X_mlp, y_mlp = X_train_s, y_arr

    mlp = MLPClassifier(
        hidden_layer_sizes=cfg.mlp_hidden_layers,
        max_iter=cfg.mlp_max_iter,
        early_stopping=True,
        validation_fraction=0.15,
        learning_rate_init=0.001,
        alpha=1e-4,
        random_state=cfg.random_state,
        verbose=False,
    )
    mlp.fit(X_mlp, y_mlp)
    p_mlp = mlp.predict_proba(X_test_s)[:, 1]
    result["p_mlp"] = p_mlp
    result["auc_mlp"] = roc_auc_score(y_test, p_mlp) if y_test.sum() > 0 else np.nan

    # ── Ensemble: simple mean of probabilities ──
    p_ensemble = (p_lgbm + p_rf + p_mlp) / 3.0
    result["p_ensemble"] = p_ensemble
    result["auc_ensemble"] = (
        roc_auc_score(y_test, p_ensemble) if y_test.sum() > 0 else np.nan
    )

    # ── Feature importance (LGBM-based) ──
    result["feature_importance"] = pd.Series(
        lgbm.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    return result


def precision_by_threshold(
    y_true: pd.Series | np.ndarray,
    p: np.ndarray,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Compute precision at various probability thresholds.

    Useful for high-conviction signal use cases where you only act on
    predictions above some threshold. Reports lift vs base rate.
    """
    y_true = np.asarray(y_true)
    base_rate = y_true.mean()
    if thresholds is None:
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70]

    rows = []
    for thr in thresholds:
        pred_pos = p >= thr
        n = int(pred_pos.sum())
        if n == 0:
            continue
        prec = float(y_true[pred_pos].mean())
        rows.append(
            {
                "threshold": thr,
                "n_predicted": n,
                "n_hit": int(y_true[pred_pos].sum()),
                "precision": prec,
                "lift_vs_base": prec / max(base_rate, 1e-6),
            }
        )
    return pd.DataFrame(rows)
