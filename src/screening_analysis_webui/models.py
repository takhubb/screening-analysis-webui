from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable

import pandas as pd


ProgressCallback = Callable[[str, float], None]

MARKET_OPTIONS = ("プライム", "スタンダード", "グロース")
SECTOR_33_OPTIONS = (
    "水産・農林業",
    "鉱業",
    "建設業",
    "食料品",
    "繊維製品",
    "パルプ・紙",
    "化学",
    "医薬品",
    "石油・石炭製品",
    "ゴム製品",
    "ガラス・土石製品",
    "鉄鋼",
    "非鉄金属",
    "金属製品",
    "機械",
    "電気機器",
    "輸送用機器",
    "精密機器",
    "その他製品",
    "電気・ガス業",
    "陸運業",
    "海運業",
    "空運業",
    "倉庫・運輸関連業",
    "情報・通信業",
    "卸売業",
    "小売業",
    "銀行業",
    "証券、商品先物取引業",
    "保険業",
    "その他金融業",
    "不動産業",
    "サービス業",
)


@dataclass(frozen=True)
class ScreeningConfig:
    signal_start: date
    signal_end: date
    filter_name: str = "パターン1"
    target_markets: tuple[str, ...] = ("プライム", "スタンダード")
    target_sectors: tuple[str, ...] = ()
    breakout_lookback_days: int = 252
    min_breakout_pct: float = 0.0
    volume_lookback_days: int = 20
    min_volume_ratio: float = 2.0
    min_turnover_oku: float | None = None
    min_market_cap_oku: float | None = None
    max_market_cap_oku: float | None = 100.0
    max_per: float | None = None
    max_pbr: float | None = 1.0
    min_equity_ratio_pct: float | None = 20.0
    min_sales_growth_pct: float | None = None
    min_operating_profit_growth_pct: float | None = None
    min_operating_margin_pct: float | None = None
    cooldown_business_days: int = 20
    exclude_funds: bool = True

    def __post_init__(self) -> None:
        if self.signal_end < self.signal_start:
            raise ValueError("signal_end must be on or after signal_start")


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
