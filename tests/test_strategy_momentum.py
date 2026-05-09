"""Tests for src/baseline/strategies/momentum_breakout.py.

Hand-crafted bars with two ts_codes:
  BREAK.SH    -> rises monotonically; final close = period high; final volume well above avg
  NOBREAK.SH  -> rises but final close is NOT a new lookback high; final volume normal
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.baseline.strategies.momentum_breakout import MomentumBreakout
from src.baseline.strategy_base import StrategyContext


def _make_bars():
    days = pd.bdate_range("2025-01-01", periods=80)
    rows = []

    # BREAK.SH: linearly rising close; vol baseline 1e6, surge to 5e6 on the final bar.
    base = np.linspace(10.0, 20.0, 80)
    for i, d in enumerate(days):
        vol = 5_000_000 if i == 79 else 1_000_000
        rows.append({"trade_date": d, "ts_code": "BREAK.SH",
                     "open": base[i], "high": base[i] * 1.01, "low": base[i] * 0.99,
                     "close": base[i], "vol": vol, "amount": base[i] * vol})

    # NOBREAK.SH: rises until day 60, then flat at lower-than-peak; vol normal throughout.
    nb_close = np.concatenate([np.linspace(10.0, 25.0, 60), np.full(20, 22.0)])
    for i, d in enumerate(days):
        rows.append({"trade_date": d, "ts_code": "NOBREAK.SH",
                     "open": nb_close[i], "high": nb_close[i] * 1.01,
                     "low": nb_close[i] * 0.99, "close": nb_close[i],
                     "vol": 1_000_000, "amount": nb_close[i] * 1_000_000})

    return pd.DataFrame(rows)


def test_picks_breakout_when_volume_gate_satisfied():
    bars = _make_bars()
    ctx = StrategyContext(
        rebalance_date=bars["trade_date"].max(),
        universe=["BREAK.SH", "NOBREAK.SH"],
        bars=bars,
        params={},
    )
    strat = MomentumBreakout(lookback=20, vol_window=20, vol_ratio=2.0,
                             mom_window=60, top_k=5, holding_period=5)
    sigs = strat.generate_signals(ctx)
    assert "BREAK.SH" in sigs.index
    assert "NOBREAK.SH" not in sigs.index


def test_excludes_breakout_when_volume_gate_not_satisfied():
    bars = _make_bars()
    ctx = StrategyContext(
        rebalance_date=bars["trade_date"].max(),
        universe=["BREAK.SH", "NOBREAK.SH"],
        bars=bars,
        params={},
    )
    # BREAK.SH final-day vol = 5e6, avg vol over last 20 days ~ (1e6*19 + 5e6)/20 = 1.2e6
    # so vol_ratio threshold > 5e6 / 1.2e6 ≈ 4.17 should reject the breakout.
    strat = MomentumBreakout(lookback=20, vol_window=20, vol_ratio=10.0,
                             mom_window=60, top_k=5, holding_period=5)
    sigs = strat.generate_signals(ctx)
    assert "BREAK.SH" not in sigs.index


def test_get_positions_top_k_equal_weight():
    sigs = pd.DataFrame({"score": [0.5, 0.3, 0.1, 0.05]},
                        index=["A", "B", "C", "D"])
    ctx = StrategyContext(rebalance_date=pd.Timestamp("2025-01-31"),
                          universe=["A", "B", "C", "D"], bars=pd.DataFrame(), params={})
    strat = MomentumBreakout(top_k=2)
    weights = strat.get_positions(sigs, ctx)
    assert list(weights.index) == ["A", "B"]
    assert all(abs(w - 0.5) < 1e-9 for w in weights.values)
