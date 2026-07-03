"""Feature engineering for intraday move-prediction models.

All features are strictly pre-t (no lookahead): each row's features use only
information available at bar-close time t.

Feature categories:
    - Recent volatility (rolling range on multiple windows)
    - Momentum (close-to-close returns on multiple windows)
    - Session context (bars since session start, distance from running H/L)
    - Time-of-day (cyclic encoding + regime flags)
    - Volume (rolling means, current vs 20-bar avg)

Designed to work on 5-minute bars resampled from 1-minute source.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def resample_5m(bars_1m: pd.DataFrame) -> pd.DataFrame:
    """Right-labeled, right-closed 5-min resample (no lookahead).

    Right-label means the bar timestamp is the END of the interval, so
    features derived from this bar are legitimately known at that time.
    """
    g = bars_1m.resample("5min", label="right", closed="right", origin="epoch")
    out = pd.DataFrame(
        {
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
            "volume": g["volume"].sum(),
        }
    ).dropna()
    return out


def add_session_metadata(bars: pd.DataFrame) -> pd.DataFrame:
    """Attach et_min and session_date. Session boundary is 18:00 ET."""
    df = bars.copy()
    et = df.index.tz_convert("America/New_York")
    df["et_min"] = et.hour * 60 + et.minute
    df["session_date"] = et.date
    boundary = et.hour >= 18
    if boundary.any():
        new_sd = (et + pd.Timedelta(days=1)).date
        df.loc[boundary, "session_date"] = pd.Series(new_sd, index=df.index)[boundary]
    df["session_date"] = pd.to_datetime(df["session_date"])
    return df


def build_features(bars_5m: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Vectorized feature construction.

    Returns:
        (df_with_features, feature_column_names)

    All features are shifted so they use only pre-t bars. Any feature that
    uses bar_t's close/open is safe (known at bar close time).
    """
    df = bars_5m.copy()
    df = add_session_metadata(df)

    # Basic price features (use current bar's OHLC — safe at close time)
    df["bar_range"] = df["high"] - df["low"]
    df["bar_body"] = df["close"] - df["open"]

    # Rolling volatility (previous N bars only — .shift(1) excludes current)
    for win in (3, 6, 12, 24):
        df[f"range_max_{win*5}m"] = df["bar_range"].rolling(win).max().shift(1)
        df[f"range_sum_{win*5}m"] = df["bar_range"].rolling(win).sum().shift(1)

    # Momentum (close vs N bars ago)
    for win in (3, 6, 12, 24):
        df[f"momentum_{win*5}m"] = df["close"] - df["close"].shift(win)

    # Previous bar strength
    df["prev_body_range_ratio"] = (
        df["bar_body"].shift(1).abs()
        / df["bar_range"].shift(1).replace(0, np.nan)
    )

    # Volume features
    for win in (3, 12, 24):
        df[f"volume_mean_{win*5}m"] = df["volume"].rolling(win).mean().shift(1)
    df["curr_bar_volume_vs_20"] = (
        df["volume"] / df["volume"].rolling(20).mean().shift(1)
    )

    # Session context
    sg = df.groupby("session_date")
    df["session_high_so_far"] = sg["high"].cummax()
    df["session_low_so_far"] = sg["low"].cummin()
    df["close_from_session_high"] = df["close"] - df["session_high_so_far"]
    df["close_from_session_low"] = df["close"] - df["session_low_so_far"]
    df["session_range_so_far"] = df["session_high_so_far"] - df["session_low_so_far"]
    df["bars_since_session_start"] = sg.cumcount()

    # Time-of-day (cyclic encoding — models can learn periodic patterns)
    df["hour_sin"] = np.sin(2 * np.pi * (df["et_min"] / 1440))
    df["hour_cos"] = np.cos(2 * np.pi * (df["et_min"] / 1440))
    df["is_rth"] = ((df["et_min"] >= 570) & (df["et_min"] <= 960)).astype(int)
    df["is_premkt"] = ((df["et_min"] >= 240) & (df["et_min"] < 570)).astype(int)
    df["is_eth"] = ((df["et_min"] < 240) | (df["et_min"] > 960)).astype(int)

    # Feature columns for downstream use (exclude raw price/session cols)
    exclude = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "et_min",
        "session_date",
        "bar_range",
        "bar_body",
        "session_high_so_far",
        "session_low_so_far",
    }
    features = [c for c in df.columns if c not in exclude]
    return df, features


def compute_target(
    bars_5m: pd.DataFrame,
    forward_window_bars: int = 12,
    up_threshold_pts: float = 40.0,
) -> pd.Series:
    """Compute target label: 1 if max(high) in next N bars >= close + threshold.

    forward_window_bars=12 at 5-min bars = 60 minutes ahead.
    """
    future_max = (
        bars_5m["high"]
        .rolling(forward_window_bars, min_periods=1)
        .max()
        .shift(-forward_window_bars)
    )
    return (future_max >= bars_5m["close"] + up_threshold_pts).astype(int)


if __name__ == "__main__":
    from synthetic_bars import generate_bars

    print("Generating synthetic bars...")
    bars_1m = generate_bars(n_days=100)
    print(f"1-min bars: {len(bars_1m):,}")

    bars_5m = resample_5m(bars_1m)
    print(f"5-min bars: {len(bars_5m):,}")

    df, features = build_features(bars_5m)
    df["target"] = compute_target(bars_5m)
    print(f"Features: {len(features)}")
    print(f"Target positive rate: {df['target'].mean() * 100:.2f}%")
    print(f"\nSample features (last row):")
    print(df[features + ['target']].tail(1).T)
