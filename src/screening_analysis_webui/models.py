from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable

import pandas as pd


ProgressCallback = Callable[[str, float], None]

MARKET_OPTIONS = ("プライム", "スタンダード", "グロース")


@dataclass(frozen=True)
class ScreeningConfig:
    signal_start: date
    signal_end: date
    target_markets: tuple[str, ...] = ("プライム", "スタンダード")
    breakout_lookback_days: int = 252
    min_breakout_pct: float = 0.0
    volume_lookback_days: int = 20
    min_volume_ratio: float = 2.0
    min_turnover_oku: float | None = None
    max_market_cap_oku: float | None = 100.0
    max_pbr: float | None = 1.0
    min_equity_ratio_pct: float | None = 20.0
    min_sales_growth_pct: float | None = None
    min_operating_profit_growth_pct: float | None = None
    min_operating_margin_pct: float | None = None
    cooldown_business_days: int = 20
    return_horizons: tuple[int, ...] = (3, 6, 9, 12)
    exclude_funds: bool = True

    def __post_init__(self) -> None:
        if self.signal_end < self.signal_start:
            raise ValueError("signal_end must be on or after signal_start")
        horizons = tuple(sorted({int(months) for months in self.return_horizons if int(months) > 0}))
        if not horizons:
            raise ValueError("return_horizons must include at least one positive month")
        object.__setattr__(self, "return_horizons", horizons)

    @property
    def max_horizon_months(self) -> int:
        return max(self.return_horizons)


@dataclass
class AnalysisResult:
    signals: pd.DataFrame
    cooldown_signals: pd.DataFrame
    latest_screening: pd.DataFrame
    pipeline: pd.DataFrame
    horizon_summary: pd.DataFrame
    cooldown_horizon_summary: pd.DataFrame
    market_summary: pd.DataFrame
    sector_summary: pd.DataFrame
    year_summary: pd.DataFrame
    top_signals: pd.DataFrame
    bottom_signals: pd.DataFrame
    metadata: dict[str, object]
