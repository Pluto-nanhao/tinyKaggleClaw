"""Backtest engine: daily-bar driven, weekly (Friday) rebalance by default.

Engine responsibilities:
- pull bars + calendar via DataFetcher
- on each rebalance day: hand a StrategyContext to the strategy, get target weights
- simulate next-bar execution with commission + slippage
- mark-to-market portfolio NAV daily
- emit clear startup banner + per-rebalance + per-month progress logs
- return strategy NAV, benchmark NAV, trade log → metrics module
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_fetcher import DataFetcher
from .strategy_base import BaseStrategy, StrategyContext

log = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    start_date: str
    end_date: str
    initial_capital: float = 1_000_000.0
    commission_rate: float = 0.0003
    slippage_bps: float = 5.0
    benchmark: str = "000905.SH"
    rebalance: str = "W-FRI"
    warmup_start: str | None = None  # e.g. "2024-07-01" for momentum lookback warmup
    output_dir: str | None = None


@dataclass
class BacktestResult:
    strat_nav: pd.Series
    bench_nav: pd.Series
    trades: pd.DataFrame
    weights_history: pd.DataFrame
    config: EngineConfig
    extra: dict[str, Any] = field(default_factory=dict)


class BacktestEngine:
    def __init__(
        self,
        config: EngineConfig,
        fetcher: DataFetcher,
        strategy: BaseStrategy,
        universe: list[str],
    ) -> None:
        self.cfg = config
        self.fetcher = fetcher
        self.strategy = strategy
        self.universe = list(universe)

    def _startup_banner(self) -> None:
        log.info("=" * 72)
        log.info("[BT-START] backtest engine starting")
        log.info("  strategy   : %s", self.strategy.__class__.__name__)
        log.info("  params     : %s", self.strategy.params)
        log.info("  universe   : %d codes (sample: %s)", len(self.universe), self.universe[:5])
        log.info("  bt period  : %s -> %s", self.cfg.start_date, self.cfg.end_date)
        log.info("  warmup from: %s", self.cfg.warmup_start or self.cfg.start_date)
        log.info("  benchmark  : %s   rebalance: %s", self.cfg.benchmark, self.cfg.rebalance)
        log.info("  capital    : %.0f   commission: %.4f   slippage: %.1f bps",
                 self.cfg.initial_capital, self.cfg.commission_rate, self.cfg.slippage_bps)
        log.info("  output_dir : %s", self.cfg.output_dir)
        log.info("=" * 72)

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
        bar_start = self.cfg.warmup_start or self.cfg.start_date
        bars = self.fetcher.get_daily_bars(self.universe, bar_start, self.cfg.end_date)
        bench = self.fetcher.get_index_data(self.cfg.benchmark, bar_start, self.cfg.end_date)
        cal = self.fetcher.get_trade_calendar(self.cfg.start_date, self.cfg.end_date)
        return bars, bench, pd.DatetimeIndex(cal)

    @staticmethod
    def _rebalance_dates(cal: pd.DatetimeIndex, rule: str) -> pd.DatetimeIndex:
        if rule.upper() == "W-FRI":
            df = pd.Series(1, index=cal)
            return df.resample("W-FRI").last().dropna().index.intersection(cal)
        if rule.upper() == "M":
            df = pd.Series(1, index=cal)
            return df.resample("ME").last().dropna().index.intersection(cal)
        raise ValueError(f"unsupported rebalance rule: {rule!r}")

    def run(self, dry_run: bool = False) -> BacktestResult:
        self._startup_banner()

        bars, bench, cal = self._load_data()
        if bars.empty:
            raise RuntimeError("engine: no bars returned by fetcher")

        rebalance_days = self._rebalance_dates(cal, self.cfg.rebalance)
        log.info("[BT-CAL] trading days=%d, rebalance days=%d, first=%s, last=%s",
                 len(cal), len(rebalance_days),
                 cal[0].date() if len(cal) else None,
                 cal[-1].date() if len(cal) else None)

        if dry_run:
            log.info("[BT-DRYRUN] dry-run flag set; exiting before main backtest loop.")
            log.info("[BT-DRYRUN] bars rows=%d, bench rows=%d, rebalance count=%d",
                     len(bars), len(bench), len(rebalance_days))
            empty_nav = pd.Series([1.0], index=[cal[0]] if len(cal) else [pd.Timestamp(self.cfg.start_date)])
            return BacktestResult(
                strat_nav=empty_nav,
                bench_nav=empty_nav.copy(),
                trades=pd.DataFrame(columns=["ts_code", "side", "qty", "price", "trade_date", "pnl"]),
                weights_history=pd.DataFrame(),
                config=self.cfg,
                extra={"dry_run": True, "rebalance_days": int(len(rebalance_days))},
            )

        bars_pivot_close = bars.pivot(index="trade_date", columns="ts_code", values="close").sort_index()
        bars_pivot_close = bars_pivot_close.reindex(cal).ffill()

        cash = self.cfg.initial_capital
        positions: dict[str, dict[str, float]] = {}  # ts_code -> {qty, cost}
        nav_records: list[tuple[pd.Timestamp, float]] = []
        trade_records: list[dict] = []
        weight_records: list[dict] = []

        last_log_month: tuple[int, int] | None = None
        slip = self.cfg.slippage_bps / 1e4
        comm = self.cfg.commission_rate

        for i, day in enumerate(cal):
            prices_today = bars_pivot_close.loc[day]
            mark = 0.0
            for code, pos in positions.items():
                px = prices_today.get(code, np.nan)
                if not np.isnan(px):
                    mark += pos["qty"] * px
            nav = cash + mark
            nav_records.append((day, nav))

            if day in rebalance_days and i + 1 < len(cal):
                lookback_bars = bars[bars["trade_date"] <= day]
                ctx = StrategyContext(
                    rebalance_date=day,
                    universe=self.universe,
                    bars=lookback_bars,
                    params=self.strategy.params,
                )
                signals = self.strategy.generate_signals(ctx)
                target_weights = self.strategy.get_positions(signals, ctx)

                exec_day = cal[i + 1]
                exec_prices = bars_pivot_close.loc[exec_day]

                target_value = {code: w * nav for code, w in target_weights.items()}
                all_codes = set(positions) | set(target_value)
                for code in all_codes:
                    px = exec_prices.get(code, np.nan)
                    if pd.isna(px):
                        continue
                    cur_pos = positions.get(code, {"qty": 0.0, "cost": 0.0})
                    cur_qty = cur_pos["qty"]
                    cur_cost = cur_pos["cost"]
                    cur_value = cur_qty * px
                    tgt_value = target_value.get(code, 0.0)
                    diff_value = tgt_value - cur_value
                    if abs(diff_value) < 1e-6:
                        continue
                    side = "buy" if diff_value > 0 else "sell"
                    px_eff = px * (1 + slip) if side == "buy" else px * (1 - slip)
                    qty = diff_value / px_eff
                    fee = abs(diff_value) * comm
                    cash -= diff_value + fee

                    if side == "buy":
                        new_qty = cur_qty + qty
                        new_cost = (
                            (cur_qty * cur_cost + qty * px_eff) / new_qty
                            if new_qty > 0 else 0.0
                        )
                        positions[code] = {"qty": new_qty, "cost": new_cost}
                        realized = -fee  # entry leg: only fee contributes to realized pnl
                    else:
                        sell_qty = -qty  # qty is negative on sells
                        realized = (px_eff - cur_cost) * sell_qty - fee
                        new_qty = cur_qty + qty
                        if abs(new_qty) < 1e-9:
                            positions.pop(code, None)
                        else:
                            positions[code] = {"qty": new_qty, "cost": cur_cost}

                    trade_records.append({
                        "ts_code": code, "side": side, "qty": qty, "price": px_eff,
                        "trade_date": exec_day, "pnl": realized,
                    })

                for code, w in target_weights.items():
                    weight_records.append({"trade_date": exec_day, "ts_code": code, "weight": w})

                log.info("[BT-REBAL] %s -> %s | picks=%d | nav=%.2f | cash=%.2f",
                         day.date(), exec_day.date(),
                         int((target_weights > 0).sum()) if len(target_weights) else 0,
                         nav, cash)

            month_key = (day.year, day.month)
            if last_log_month != month_key:
                log.info("[BT-MONTH] %s nav=%.2f positions=%d", day.strftime("%Y-%m"),
                         nav, len(positions))
                last_log_month = month_key

        nav_idx = pd.DatetimeIndex([d for d, _ in nav_records])
        nav_vals = pd.Series([v for _, v in nav_records], index=nav_idx, name="nav")
        strat_nav = nav_vals / nav_vals.iloc[0]
        bench_close = bench["close"].reindex(nav_idx).ffill()
        bench_nav = bench_close / bench_close.iloc[0]

        log.info("[BT-DONE] final NAV=%.4f benchmark NAV=%.4f trades=%d",
                 float(strat_nav.iloc[-1]), float(bench_nav.iloc[-1]), len(trade_records))

        return BacktestResult(
            strat_nav=strat_nav,
            bench_nav=bench_nav,
            trades=pd.DataFrame(trade_records),
            weights_history=pd.DataFrame(weight_records),
            config=self.cfg,
        )

    def write_outputs(self, result: BacktestResult, metrics: dict[str, Any]) -> None:
        if not self.cfg.output_dir:
            return
        out = Path(self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        result.strat_nav.to_csv(out / "strat_nav.csv", header=True)
        result.bench_nav.to_csv(out / "bench_nav.csv", header=True)
        result.trades.to_csv(out / "trades.csv", index=False)
        result.weights_history.to_csv(out / "weights.csv", index=False)
        pd.Series(metrics).to_csv(out / "metrics.csv", header=["value"])
        log.info("[BT-OUT] wrote backtest artifacts to %s", out)
