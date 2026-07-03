"""Block-permutation Monte Carlo stress-testing for trade P&L distributions.

For strategies with serial correlation in trade outcomes (clustered wins,
clustered losses), naive bootstrap resampling of individual trades produces
misleadingly tame tail distributions. Block permutation preserves local
autocorrelation by sampling contiguous blocks of trades.

Common uses:
    - Kill-switch calibration (probability of breaching a floor)
    - Ruin probability (probability of hitting zero)
    - Confidence intervals on end-of-period equity
    - Sensitivity of tail risk to starting capital
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MCResult:
    """Summary statistics from a block-permutation MC run."""

    n_paths: int
    n_trades: int
    n_blocks: int
    breach_pct: float
    ruin_pct: float
    worst_1pct_trough: float
    worst_5pct_trough: float
    median_end: float
    worst_5pct_end: float
    best_5pct_end: float

    def as_dict(self) -> dict:
        return {
            "n_paths": self.n_paths,
            "n_trades": self.n_trades,
            "n_blocks": self.n_blocks,
            "breach_pct": self.breach_pct,
            "ruin_pct": self.ruin_pct,
            "worst_1pct_trough": self.worst_1pct_trough,
            "worst_5pct_trough": self.worst_5pct_trough,
            "median_end": self.median_end,
            "worst_5pct_end": self.worst_5pct_end,
            "best_5pct_end": self.best_5pct_end,
        }


def block_permutation_mc(
    pnl_series: np.ndarray | pd.Series,
    n_paths: int = 20_000,
    block_size: int = 20,
    start_capital: float = 10_000.0,
    kill_threshold: float = 5_000.0,
    seed: int = 42,
) -> MCResult:
    """Run block-permutation MC on a trade P&L series.

    Args:
        pnl_series: Array of per-trade P&L (in currency units).
        n_paths: Number of Monte Carlo paths to simulate.
        block_size: Size of contiguous blocks to sample. Larger = more
            autocorrelation preserved but fewer effectively-independent samples.
        start_capital: Starting equity for each simulated path.
        kill_threshold: Equity floor. Breach = trough < this level.
        seed: RNG seed for reproducibility.

    Returns:
        MCResult with breach probability, ruin probability, and quantile stats.
    """
    pnl = np.asarray(pnl_series, dtype=np.float64)
    n_trades = len(pnl)

    if n_trades < block_size * 4:
        raise ValueError(
            f"Need at least {block_size * 4} trades for meaningful MC; "
            f"got {n_trades}"
        )

    n_blocks = n_trades // block_size
    blocks = pnl[: n_blocks * block_size].reshape(n_blocks, block_size)
    rng = np.random.default_rng(seed)

    breaches = 0
    ruins = 0
    troughs = np.empty(n_paths)
    ends = np.empty(n_paths)

    for i in range(n_paths):
        # Sample block indices with replacement
        perm = rng.integers(0, n_blocks, size=n_blocks)
        # Concatenate sampled blocks into a full-length path
        path = blocks[perm].flatten()
        equity = start_capital + np.cumsum(path)
        trough = equity.min()
        if trough < kill_threshold:
            breaches += 1
        if trough <= 0:
            ruins += 1
        troughs[i] = trough
        ends[i] = equity[-1]

    return MCResult(
        n_paths=n_paths,
        n_trades=n_trades,
        n_blocks=n_blocks,
        breach_pct=breaches / n_paths * 100,
        ruin_pct=ruins / n_paths * 100,
        worst_1pct_trough=float(np.percentile(troughs, 1)),
        worst_5pct_trough=float(np.percentile(troughs, 5)),
        median_end=float(np.median(ends)),
        worst_5pct_end=float(np.percentile(ends, 5)),
        best_5pct_end=float(np.percentile(ends, 95)),
    )


def stress_sweep_capital(
    pnl_series: np.ndarray | pd.Series,
    starting_capitals: list[float],
    kill_threshold: float,
    n_paths: int = 20_000,
    block_size: int = 20,
) -> pd.DataFrame:
    """Sweep MC breach probability across multiple starting capital scenarios.

    Useful for capital-sizing decisions: "how much do I need to deposit to
    stay above the kill switch with 99% confidence?"
    """
    rows = []
    for cap in starting_capitals:
        r = block_permutation_mc(
            pnl_series,
            n_paths=n_paths,
            block_size=block_size,
            start_capital=cap,
            kill_threshold=kill_threshold,
        )
        rows.append(
            {
                "start_capital": cap,
                "breach_pct": r.breach_pct,
                "ruin_pct": r.ruin_pct,
                "worst_1pct_trough": r.worst_1pct_trough,
                "worst_5pct_trough": r.worst_5pct_trough,
                "median_end": r.median_end,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Toy example: simulated trade series with clustered outcomes
    rng = np.random.default_rng(42)
    n_trades = 2000
    # 55% WR at 1.5:1 R/R with slight clustering
    wins = rng.random(n_trades) < 0.55
    pnl = np.where(wins, 146.0, -104.0)
    # Add some clustering: 10% of trades come in bunches of the same outcome
    for i in range(0, n_trades - 4, 20):
        if rng.random() < 0.3:
            pnl[i : i + 4] = pnl[i]  # 4-trade run of same outcome

    print(f"Simulated P&L series: {n_trades} trades")
    print(f"  win rate:   {wins.mean() * 100:.1f}%")
    print(f"  total P&L:  ${pnl.sum():,.0f}")
    print(f"  mean/trade: ${pnl.mean():+.2f}\n")

    r = block_permutation_mc(
        pnl,
        start_capital=5000,
        kill_threshold=2000,
    )
    print(f"MC result @ $5000 start, $2000 kill floor:")
    for k, v in r.as_dict().items():
        print(f"  {k}: {v}")

    print("\nCapital sensitivity:")
    sweep = stress_sweep_capital(
        pnl, [2500, 5000, 10000, 20000], kill_threshold=2000
    )
    print(sweep.to_string(index=False))
