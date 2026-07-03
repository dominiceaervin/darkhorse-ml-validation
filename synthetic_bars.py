"""Synthetic 1-minute OHLCV bar generator for demo/testing.

Produces realistic-looking bar data with:
- Intraday session boundaries (18:00 ET rollover)
- RTH (09:30-16:00 ET) vs off-hours volume patterns
- Geometric Brownian motion price paths
- Occasional high-volatility clusters (regime shifts)

Useful for testing feature engineering, walk-forward validation, and
Monte Carlo stress-test pipelines without needing real market data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_bars(
    n_days: int = 500,
    start_date: str = "2023-01-01",
    seed: int = 42,
    base_price: float = 4500.0,
    daily_drift: float = 0.0002,
    intraday_vol_pct: float = 0.0008,
    cluster_prob: float = 0.05,
) -> pd.DataFrame:
    """Generate n_days worth of 1-min OHLCV bars.

    Args:
        n_days: Number of trading sessions to generate.
        start_date: First session date (YYYY-MM-DD).
        seed: RNG seed for reproducibility.
        base_price: Starting price level (roughly ES futures range).
        daily_drift: Expected daily log-return (positive = uptrend).
        intraday_vol_pct: Per-minute return std dev.
        cluster_prob: Probability of a high-volatility cluster per session.

    Returns:
        DataFrame indexed by UTC timestamp, with columns:
            open, high, low, close, volume
    """
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start_date, tz="UTC")

    all_bars = []
    price = base_price

    for day_i in range(n_days):
        # Skip weekends
        session_start = start_ts + pd.Timedelta(days=day_i)
        if session_start.tz_convert("America/New_York").weekday() >= 5:
            continue

        # ES session: 18:00 ET prior day → 17:00 ET (with 16:00-17:00 maintenance break)
        # We simulate 09:30-16:00 ET RTH + light overnight
        et_offset_hours = 4  # ET is UTC-4 in summer, UTC-5 in winter (approximation)
        rth_open_utc = session_start.replace(hour=13, minute=30)  # 09:30 ET
        rth_close_utc = session_start.replace(hour=20, minute=0)  # 16:00 ET

        # RTH bars: 6.5 hours = 390 minutes
        rth_minutes = 390
        rth_idx = pd.date_range(rth_open_utc, periods=rth_minutes, freq="1min", tz="UTC")

        # Volatility for this session
        session_vol = intraday_vol_pct * (
            2.0 if rng.random() < cluster_prob else 1.0
        )

        # Per-minute returns
        returns = rng.normal(daily_drift / rth_minutes, session_vol, size=rth_minutes)
        log_prices = np.log(price) + np.cumsum(returns)
        closes = np.exp(log_prices)

        # OHLC: use return-based generation for realism
        opens = np.concatenate([[price], closes[:-1]])
        # High/low: add intra-bar noise
        wick = np.abs(rng.normal(0, session_vol * 0.4, size=rth_minutes)) * closes
        highs = np.maximum(opens, closes) + wick
        lows = np.minimum(opens, closes) - wick

        # Volume: higher at open, dip at lunch, higher at close
        minute_of_session = np.arange(rth_minutes)
        vol_profile = (
            1.5 * np.exp(-minute_of_session / 30)  # open peak
            + 1.0
            + 1.3 * np.exp(-((minute_of_session - 360) ** 2) / 500)  # close peak
        )
        volumes = (vol_profile * 500 * (1 + rng.normal(0, 0.2, size=rth_minutes))).clip(
            50, None
        ).astype(int)

        df = pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            },
            index=rth_idx,
        )
        all_bars.append(df)
        price = float(closes[-1])

    result = pd.concat(all_bars).sort_index()
    return result


def add_session_metadata(bars: pd.DataFrame) -> pd.DataFrame:
    """Attach et_min and session_date columns.

    Session date rolls at 18:00 ET (matches CME ES convention).
    et_min = minute of day in ET (0-1439).
    """
    df = bars.copy()
    et = df.index.tz_convert("America/New_York")
    df["et_min"] = et.hour * 60 + et.minute
    df["session_date"] = et.date
    # 18:00 ET boundary — bars at or past 18:00 belong to next session
    boundary = et.hour >= 18
    if boundary.any():
        new_sd = (et + pd.Timedelta(days=1)).date
        df.loc[boundary, "session_date"] = pd.Series(new_sd, index=df.index)[boundary]
    df["session_date"] = pd.to_datetime(df["session_date"])
    return df


if __name__ == "__main__":
    print("Generating 500 sessions of synthetic 1-min bars...")
    bars = generate_bars(n_days=500)
    bars = add_session_metadata(bars)
    print(f"Total bars: {len(bars):,}")
    print(f"Date range: {bars.index.min()} → {bars.index.max()}")
    print(f"Price range: {bars['close'].min():.2f} → {bars['close'].max():.2f}")
    print(f"Sample:\n{bars.head()}")
