from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .jquants import DEFAULT_CACHE_DIR, create_client, load_calendar, load_financial_summaries, load_master_for_dates, load_prices
from .models import AnalysisResult, ProgressCallback, ScreeningConfig


SIGNAL_CLOSE_TIME = "15:00:00"
QUARTER_ORDER_MAP = {"1Q": 1, "2Q": 2, "3Q": 3, "4Q": 4, "FY": 4}


def _emit_progress(callback: ProgressCallback | None, message: str, progress: float) -> None:
    if callback is None:
        return
    callback(message, min(max(progress, 0.0), 1.0))


def build_adjusted_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return prices.copy()

    adjusted = prices.copy()
    adjusted["AdjFactor"] = pd.to_numeric(adjusted["AdjFactor"], errors="coerce").fillna(1.0).replace(0, 1.0)
    adjusted["O"] = pd.to_numeric(adjusted["O"], errors="coerce")
    adjusted["C"] = pd.to_numeric(adjusted["C"], errors="coerce")
    adjusted["Vo"] = pd.to_numeric(adjusted["Vo"], errors="coerce")
    adjusted["Va"] = pd.to_numeric(adjusted["Va"], errors="coerce")

    # bulk CSV は調整後価格を持たないため、将来側の調整係数を累積して復元する。
    adjusted = adjusted.sort_values(["Code", "Date"], ascending=[True, False]).reset_index(drop=True)
    future_factor = adjusted.groupby("Code", sort=False)["AdjFactor"].shift(1).fillna(1.0)
    adjusted["CumAdj"] = future_factor.groupby(adjusted["Code"], sort=False).cumprod()
    adjusted["AdjO"] = adjusted["O"] * adjusted["CumAdj"]
    adjusted["AdjC"] = adjusted["C"] * adjusted["CumAdj"]
    adjusted["AdjVo"] = adjusted["Vo"] / adjusted["CumAdj"]
    adjusted = adjusted.sort_values(["Code", "Date"]).reset_index(drop=True)
    return adjusted


def build_price_features(
    prices: pd.DataFrame,
    breakout_lookback_days: int,
    volume_lookback_days: int,
) -> pd.DataFrame:
    if prices.empty:
        return prices.copy()

    featured = prices.copy()
    grouped_close = featured.groupby("Code", sort=False)["AdjC"]
    grouped_volume = featured.groupby("Code", sort=False)["AdjVo"]

    shifted_close = grouped_close.shift(1)
    shifted_volume = grouped_volume.shift(1)

    featured["rolling_high_excl_today"] = (
        shifted_close.groupby(featured["Code"], sort=False)
        .rolling(breakout_lookback_days, min_periods=breakout_lookback_days)
        .max()
        .reset_index(level=0, drop=True)
    )
    featured["avg_volume_excl_today"] = (
        shifted_volume.groupby(featured["Code"], sort=False)
        .rolling(volume_lookback_days, min_periods=volume_lookback_days)
        .mean()
        .reset_index(level=0, drop=True)
    )
    featured["volume_ratio"] = featured["AdjVo"] / featured["avg_volume_excl_today"]
    featured["high_break_ratio"] = featured["AdjC"] / featured["rolling_high_excl_today"] - 1.0
    featured["turnover_yen"] = pd.to_numeric(featured["Va"], errors="coerce")
    return featured


def filter_target_markets(master_df: pd.DataFrame, config: ScreeningConfig) -> pd.DataFrame:
    filtered = master_df.copy()
    if filtered.empty:
        return filtered

    filtered["Code"] = filtered["Code"].astype(str)
    filtered = filtered[filtered["MktNm"].isin(config.target_markets)].copy()

    if config.exclude_funds and "CoName" in filtered.columns:
        excluded_pattern = r"ETF|ETN|REIT|投資法人|インフラファンド|ベンチャーファンド"
        filtered = filtered[
            ~filtered["CoName"].fillna("").astype(str).str.contains(excluded_pattern, regex=True)
        ].copy()

    return filtered.reset_index(drop=True)


def financial_statement_priority(doc_type: str) -> int:
    if "FinancialStatements_Consolidated" in doc_type:
        return 0
    if "FinancialStatements" in doc_type and "NonConsolidated" not in doc_type:
        return 1
    if "FinancialStatements_NonConsolidated" in doc_type:
        return 2
    return 9


def pick_first_positive_value(row: pd.Series, columns: tuple[str, ...]) -> float | pd._libs.missing.NAType:
    for column in columns:
        if column not in row.index:
            continue
        value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
        if pd.notna(value) and float(value) > 0:
            return float(value)
    return pd.NA


def prepare_financial_snapshots(financials: pd.DataFrame) -> pd.DataFrame:
    if financials.empty:
        return pd.DataFrame()

    statements = financials.copy()
    statements["Code"] = statements["Code"].astype(str)
    statements["DocType"] = statements["DocType"].fillna("").astype(str)
    statements = statements[statements["DocType"].str.contains("FinancialStatements", na=False)].copy()
    statements["CurPerType"] = statements["CurPerType"].fillna("").astype(str)
    statements = statements[statements["CurPerType"].isin(QUARTER_ORDER_MAP)].copy()
    if statements.empty:
        return pd.DataFrame()

    statements["DiscDate"] = pd.to_datetime(statements["DiscDate"], errors="coerce")
    statements["CurPerEn"] = pd.to_datetime(statements["CurPerEn"], errors="coerce")
    statements["CurFYEn"] = pd.to_datetime(statements["CurFYEn"], errors="coerce")
    statements["DiscTime"] = statements["DiscTime"].fillna("00:00:00").astype(str)
    statements["QuarterOrder"] = statements["CurPerType"].map(QUARTER_ORDER_MAP)
    statements["StatementPriority"] = statements["DocType"].map(financial_statement_priority)

    numeric_columns = [
        "Sales",
        "OP",
        "EqAR",
        "BPS",
        "NCBPS",
        "EPS",
        "NCEPS",
        "FEPS",
        "FNCEPS",
        "ShOutFY",
        "TrShFY",
    ]
    for column in numeric_columns:
        if column in statements.columns:
            statements[column] = pd.to_numeric(statements[column], errors="coerce")

    statements = statements.dropna(subset=["Code", "CurFYEn", "CurPerEn", "QuarterOrder", "DiscDate"])
    if statements.empty:
        return pd.DataFrame()

    statements["disc_ts"] = pd.to_datetime(
        statements["DiscDate"].dt.strftime("%Y-%m-%d") + " " + statements["DiscTime"]
    )
    statements = statements.sort_values(
        ["Code", "disc_ts", "CurFYEn", "QuarterOrder", "StatementPriority", "DiscNo"],
        ascending=[True, True, True, True, True, False],
    )
    statements = statements.drop_duplicates(subset=["Code", "disc_ts", "CurFYEn", "QuarterOrder"], keep="first")
    statements = statements.sort_values(["Code", "disc_ts", "CurFYEn", "QuarterOrder"]).reset_index(drop=True)

    quarterly_rows: list[dict[str, object]] = []
    for code, group in statements.groupby("Code", sort=False):
        group = group.sort_values(["disc_ts", "CurFYEn", "QuarterOrder"]).reset_index(drop=True)
        latest_by_key: dict[tuple[pd.Timestamp, int], dict[str, object]] = {}
        for _, row in group.iterrows():
            quarter_order = int(row["QuarterOrder"])
            current_fy = pd.Timestamp(row["CurFYEn"])
            previous_quarter = latest_by_key.get((current_fy, quarter_order - 1))
            standalone_sales = pd.to_numeric(pd.Series([row["Sales"]]), errors="coerce").iloc[0]
            standalone_op = pd.to_numeric(pd.Series([row["OP"]]), errors="coerce").iloc[0]
            if quarter_order > 1 and previous_quarter is not None:
                standalone_sales = standalone_sales - pd.to_numeric(
                    pd.Series([previous_quarter["raw_sales"]]), errors="coerce"
                ).iloc[0]
                standalone_op = standalone_op - pd.to_numeric(
                    pd.Series([previous_quarter["raw_op"]]), errors="coerce"
                ).iloc[0]

            previous_year_keys = [
                key
                for key in latest_by_key
                if key[1] == quarter_order and key[0] < current_fy
            ]
            previous_year_snapshot = latest_by_key[max(previous_year_keys)] if previous_year_keys else None

            prev_year_sales = (
                pd.to_numeric(pd.Series([previous_year_snapshot["standalone_sales"]]), errors="coerce").iloc[0]
                if previous_year_snapshot is not None
                else np.nan
            )
            prev_year_op = (
                pd.to_numeric(pd.Series([previous_year_snapshot["standalone_op"]]), errors="coerce").iloc[0]
                if previous_year_snapshot is not None
                else np.nan
            )
            sales_growth_yoy = (
                standalone_sales / prev_year_sales - 1.0
                if pd.notna(prev_year_sales) and prev_year_sales != 0 and pd.notna(standalone_sales)
                else np.nan
            )
            operating_profit_growth_yoy = (
                (standalone_op - prev_year_op) / abs(prev_year_op)
                if pd.notna(prev_year_op) and prev_year_op != 0 and pd.notna(standalone_op)
                else np.nan
            )
            operating_margin = (
                standalone_op / standalone_sales
                if pd.notna(standalone_sales) and standalone_sales != 0 and pd.notna(standalone_op)
                else np.nan
            )
            treasury_shares = row.get("TrShFY", np.nan)
            treasury_shares = 0 if pd.isna(treasury_shares) else treasury_shares
            effective_shares = row.get("ShOutFY", np.nan) - treasury_shares

            quarterly_rows.append(
                {
                    "Code": code,
                    "disc_ts": row["disc_ts"],
                    "DiscDate": row["DiscDate"],
                    "QuarterOrder": quarter_order,
                    "CurFYEn": row["CurFYEn"],
                    "CurPerEn": row["CurPerEn"],
                    "StandaloneSales": standalone_sales,
                    "StandaloneOP": standalone_op,
                    "sales_growth_yoy": sales_growth_yoy,
                    "operating_profit_growth_yoy": operating_profit_growth_yoy,
                    "operating_margin": operating_margin,
                    "equity_ratio": row.get("EqAR", pd.NA),
                    "effective_shares": effective_shares,
                    "valuation_eps": pick_first_positive_value(row, ("FEPS", "FNCEPS", "EPS", "NCEPS")),
                    "valuation_bps": pick_first_positive_value(row, ("BPS", "NCBPS")),
                }
            )
            latest_by_key[(current_fy, quarter_order)] = {
                "standalone_sales": standalone_sales,
                "standalone_op": standalone_op,
                "raw_sales": row["Sales"],
                "raw_op": row["OP"],
            }

    quarterly = pd.DataFrame(quarterly_rows)
    if quarterly.empty:
        return quarterly

    quarterly["StandaloneSales"] = pd.to_numeric(quarterly["StandaloneSales"], errors="coerce")
    quarterly["StandaloneOP"] = pd.to_numeric(quarterly["StandaloneOP"], errors="coerce")
    quarterly["equity_ratio"] = pd.to_numeric(quarterly["equity_ratio"], errors="coerce")
    quarterly["effective_shares"] = pd.to_numeric(quarterly["effective_shares"], errors="coerce")
    quarterly.loc[quarterly["effective_shares"] <= 0, "effective_shares"] = np.nan

    return quarterly.sort_values(["Code", "disc_ts"]).reset_index(drop=True)


def attach_master(price_candidates: pd.DataFrame, master: pd.DataFrame, config: ScreeningConfig) -> pd.DataFrame:
    filtered_master = filter_target_markets(master, config)
    if filtered_master.empty:
        return pd.DataFrame(columns=price_candidates.columns)
    enriched = price_candidates.merge(filtered_master, on=["Date", "Code"], how="inner")
    return enriched.sort_values(["Date", "Code"]).reset_index(drop=True)


def attach_financials(candidates: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty or snapshots.empty:
        return pd.DataFrame(columns=candidates.columns)

    left = candidates.copy()
    left["signal_ts"] = pd.to_datetime(left["Date"].dt.strftime("%Y-%m-%d") + f" {SIGNAL_CLOSE_TIME}")
    left = left.sort_values(["signal_ts", "Code"]).reset_index(drop=True)
    right = snapshots.sort_values(["disc_ts", "Code"]).reset_index(drop=True)

    merged = pd.merge_asof(
        left,
        right,
        left_on="signal_ts",
        right_on="disc_ts",
        by="Code",
        direction="backward",
    )

    merged["market_cap"] = merged["AdjC"] * merged["effective_shares"]
    merged["PER"] = np.where(
        pd.to_numeric(merged["valuation_eps"], errors="coerce").gt(0),
        merged["AdjC"] / pd.to_numeric(merged["valuation_eps"], errors="coerce"),
        np.nan,
    )
    merged["PBR"] = np.where(
        pd.to_numeric(merged["valuation_bps"], errors="coerce").gt(0),
        merged["AdjC"] / pd.to_numeric(merged["valuation_bps"], errors="coerce"),
        np.nan,
    )
    return merged.sort_values(["Date", "Code"]).reset_index(drop=True)


def apply_screening_filters(candidates: pd.DataFrame, config: ScreeningConfig) -> pd.DataFrame:
    filtered = candidates.copy()
    if filtered.empty:
        return filtered

    mask = pd.Series(True, index=filtered.index)

    if config.min_turnover_oku is not None:
        mask &= pd.to_numeric(filtered["turnover_yen"], errors="coerce") >= config.min_turnover_oku * 100_000_000
    if config.max_market_cap_oku is not None:
        mask &= pd.to_numeric(filtered["market_cap"], errors="coerce").le(config.max_market_cap_oku * 100_000_000)
    if config.max_pbr is not None:
        pbr = pd.to_numeric(filtered["PBR"], errors="coerce")
        mask &= pbr.notna() & pbr.le(config.max_pbr)
    if config.min_equity_ratio_pct is not None:
        mask &= pd.to_numeric(filtered["equity_ratio"], errors="coerce") >= config.min_equity_ratio_pct / 100.0
    if config.min_sales_growth_pct is not None:
        mask &= pd.to_numeric(filtered["sales_growth_yoy"], errors="coerce") >= config.min_sales_growth_pct / 100.0
    if config.min_operating_profit_growth_pct is not None:
        mask &= (
            pd.to_numeric(filtered["operating_profit_growth_yoy"], errors="coerce")
            >= config.min_operating_profit_growth_pct / 100.0
        )
    if config.min_operating_margin_pct is not None:
        mask &= pd.to_numeric(filtered["operating_margin"], errors="coerce") >= config.min_operating_margin_pct / 100.0

    filtered = filtered.loc[mask].copy()
    return filtered.sort_values(["Date", "volume_ratio", "turnover_yen", "Code"], ascending=[True, False, False, True]).reset_index(drop=True)


def attach_forward_returns(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    business_days: pd.DatetimeIndex,
    exit_date: pd.Timestamp,
    shares: int = 100,
) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()

    result = signals.copy()
    trading_days = business_days.sort_values()

    entry_positions = trading_days.searchsorted(result["Date"].values, side="right")
    entry_dates = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns]")
    valid_entries = entry_positions < len(trading_days)
    if valid_entries.any():
        entry_dates.loc[valid_entries] = trading_days[entry_positions[valid_entries]]

    exit_ts = pd.Timestamp(exit_date).normalize()
    exit_dates = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns]")
    if exit_ts in trading_days:
        exit_dates.loc[entry_dates.notna() & (entry_dates <= exit_ts)] = exit_ts

    entry_lookup = prices[["Code", "Date", "AdjO"]].rename(columns={"AdjO": "entry_price"})
    exit_lookup = prices[["Code", "Date", "AdjC"]].rename(columns={"AdjC": "exit_price"})

    priced = pd.DataFrame(
        {
            "Code": result["Code"].values,
            "entry_date": entry_dates.values,
            "exit_date": exit_dates.values,
        },
        index=result.index,
    )
    priced = priced.merge(
        entry_lookup,
        left_on=["Code", "entry_date"],
        right_on=["Code", "Date"],
        how="left",
    ).drop(columns=["Date"])
    priced = priced.merge(
        exit_lookup,
        left_on=["Code", "exit_date"],
        right_on=["Code", "Date"],
        how="left",
    ).drop(columns=["Date"])

    result["entry_date"] = priced["entry_date"].values
    result["exit_date"] = priced["exit_date"].values
    result["entry_price"] = priced["entry_price"].values
    result["exit_price"] = priced["exit_price"].values
    result["buy_amount_yen"] = result["entry_price"] * shares
    result["sell_amount_yen"] = result["exit_price"] * shares
    result["profit_yen"] = result["sell_amount_yen"] - result["buy_amount_yen"]
    result["ret"] = result["exit_price"] / result["entry_price"] - 1.0

    return result


def apply_cooldown(signals: pd.DataFrame, business_days: pd.DatetimeIndex, cooldown_bdays: int) -> pd.DataFrame:
    if signals.empty or cooldown_bdays <= 0:
        return signals.copy()

    day_rank = {day: rank for rank, day in enumerate(business_days)}
    kept_indices: list[int] = []

    for _, group in signals.sort_values(["Code", "Date"]).groupby("Code", sort=False):
        last_rank = -10**9
        for idx, row in group.iterrows():
            rank = day_rank.get(row["Date"])
            if rank is None:
                continue
            if rank - last_rank > cooldown_bdays:
                kept_indices.append(idx)
                last_rank = rank

    return signals.loc[sorted(kept_indices)].copy()


def summarize_returns(signals: pd.DataFrame, return_col: str = "ret") -> pd.DataFrame:
    sample = signals[return_col].dropna() if return_col in signals.columns else pd.Series(dtype=float)
    return pd.DataFrame(
        [
            {
                "return_type": "翌営業日始値→指定日終値",
                "count": int(sample.shape[0]),
                "mean": float(sample.mean()) if not sample.empty else np.nan,
                "median": float(sample.median()) if not sample.empty else np.nan,
                "std": float(sample.std()) if not sample.empty else np.nan,
                "win_rate": float((sample > 0).mean()) if not sample.empty else np.nan,
                "p10": float(sample.quantile(0.10)) if not sample.empty else np.nan,
                "p25": float(sample.quantile(0.25)) if not sample.empty else np.nan,
                "p75": float(sample.quantile(0.75)) if not sample.empty else np.nan,
                "p90": float(sample.quantile(0.90)) if not sample.empty else np.nan,
            }
        ]
    )


def summarize_by_group(
    signals: pd.DataFrame,
    group_col: str,
    return_col: str,
    min_count: int = 0,
    top_n: int | None = None,
) -> pd.DataFrame:
    if signals.empty or group_col not in signals.columns or return_col not in signals.columns:
        return pd.DataFrame(columns=[group_col, "count", "mean", "median", "win_rate"])

    sample = signals[[group_col, return_col]].dropna()
    if sample.empty:
        return pd.DataFrame(columns=[group_col, "count", "mean", "median", "win_rate"])

    summary = (
        sample.groupby(group_col)
        .agg(
            count=(return_col, "size"),
            mean=(return_col, "mean"),
            median=(return_col, "median"),
            win_rate=(return_col, lambda series: (series > 0).mean()),
        )
        .reset_index()
        .sort_values(["mean", "median"], ascending=False)
        .reset_index(drop=True)
    )
    if min_count > 0:
        summary = summary[summary["count"] >= min_count].copy()
    if top_n is not None:
        summary = summary.head(top_n).reset_index(drop=True)
    return summary


def summarize_by_year(signals: pd.DataFrame, return_col: str) -> pd.DataFrame:
    if signals.empty or return_col not in signals.columns:
        return pd.DataFrame(columns=["year", "count", "mean", "median", "win_rate"])

    sample = signals[["Date", return_col]].dropna().copy()
    if sample.empty:
        return pd.DataFrame(columns=["year", "count", "mean", "median", "win_rate"])

    sample["year"] = sample["Date"].dt.year
    return (
        sample.groupby("year")
        .agg(
            count=(return_col, "size"),
            mean=(return_col, "mean"),
            median=(return_col, "median"),
            win_rate=(return_col, lambda series: (series > 0).mean()),
        )
        .reset_index()
        .sort_values("year")
        .reset_index(drop=True)
    )


def build_pipeline_table(
    price_candidate_count: int,
    market_filtered_count: int,
    final_signal_count: int,
    complete_signal_count: int,
    cooldown_signal_count: int,
) -> pd.DataFrame:
    baseline = max(price_candidate_count, 1)
    return pd.DataFrame(
        [
            {"stage": "価格条件シグナル", "count": price_candidate_count, "share_vs_initial": 1.0 if price_candidate_count else 0.0},
            {
                "stage": "市場フィルタ後",
                "count": market_filtered_count,
                "share_vs_initial": market_filtered_count / baseline,
            },
            {
                "stage": "追加条件適用後",
                "count": final_signal_count,
                "share_vs_initial": final_signal_count / baseline,
            },
            {
                "stage": "リターン観測可能",
                "count": complete_signal_count,
                "share_vs_initial": complete_signal_count / baseline,
            },
            {
                "stage": "クールダウン後",
                "count": cooldown_signal_count,
                "share_vs_initial": cooldown_signal_count / baseline,
            },
        ]
    )


def build_empty_result(
    config: ScreeningConfig,
    latest_price_date: pd.Timestamp | None,
    price_candidate_count: int = 0,
    market_filtered_count: int = 0,
    final_signal_count: int = 0,
    complete_signal_count: int = 0,
    cooldown_signal_count: int = 0,
) -> AnalysisResult:
    pipeline = build_pipeline_table(
        price_candidate_count=price_candidate_count,
        market_filtered_count=market_filtered_count,
        final_signal_count=final_signal_count,
        complete_signal_count=complete_signal_count,
        cooldown_signal_count=cooldown_signal_count,
    )
    metadata = {
        "analysis_start": config.signal_start.isoformat(),
        "analysis_end": config.signal_end.isoformat(),
        "latest_price_date": latest_price_date.date().isoformat() if latest_price_date is not None else None,
        "return_exit_date": config.effective_return_exit_date.isoformat(),
        "signal_count": final_signal_count,
        "cooldown_signal_count": cooldown_signal_count,
    }
    empty = pd.DataFrame()
    return AnalysisResult(
        signals=empty,
        cooldown_signals=empty,
        latest_screening=empty,
        pipeline=pipeline,
        horizon_summary=summarize_returns(empty),
        cooldown_horizon_summary=summarize_returns(empty),
        market_summary=empty,
        sector_summary=empty,
        year_summary=empty,
        top_signals=empty,
        bottom_signals=empty,
        metadata=metadata,
    )


def run_analysis(
    config: ScreeningConfig,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    progress_callback: ProgressCallback | None = None,
) -> AnalysisResult:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    _emit_progress(progress_callback, "J-Quants クライアントを初期化しています", 0.02)
    client = create_client()

    history_buffer_days = max(800, max(config.breakout_lookback_days, config.volume_lookback_days) * 3)
    calendar_start = pd.Timestamp(config.signal_start) - pd.Timedelta(days=history_buffer_days)
    requested_exit_date = pd.Timestamp(config.effective_return_exit_date).normalize()
    calendar_end = (max(pd.Timestamp(config.signal_end), requested_exit_date) + pd.Timedelta(days=14)).normalize()

    _emit_progress(progress_callback, "営業日カレンダーを取得しています", 0.08)
    calendar = load_calendar(client=client, start=calendar_start, end=calendar_end, cache_dir=cache_dir)
    business_days = pd.DatetimeIndex(calendar.loc[calendar["HolDiv"].astype(str) == "1", "Date"]).sort_values()
    if business_days.empty:
        raise RuntimeError("営業日カレンダーが空でした。")
    if requested_exit_date not in business_days:
        raise RuntimeError("売却日は営業日を指定してください。指定日の終値でリターンを計算します。")

    signal_start_ts = pd.Timestamp(config.signal_start)
    signal_end_ts = pd.Timestamp(config.signal_end)
    signal_days = business_days[(business_days >= signal_start_ts) & (business_days <= signal_end_ts)]
    if signal_days.empty:
        raise RuntimeError("指定期間に営業日がありません。")

    first_signal_index = int(business_days.get_indexer([signal_days.min()])[0])
    price_start_index = max(0, first_signal_index - max(config.breakout_lookback_days, config.volume_lookback_days) - 5)
    price_start = business_days[price_start_index]
    price_end = business_days.max()

    _emit_progress(progress_callback, "価格 bulk を取得して調整後価格を計算しています", 0.22)
    prices = load_prices(client=client, start=price_start, end=price_end, cache_dir=cache_dir)
    if prices.empty:
        raise RuntimeError("価格データを取得できませんでした。")

    prices = build_adjusted_prices(prices)
    latest_price_date = pd.Timestamp(prices["Date"].max())
    price_dates = pd.DatetimeIndex(prices["Date"].dropna().drop_duplicates()).normalize()
    if requested_exit_date > latest_price_date or requested_exit_date not in price_dates:
        raise RuntimeError(
            f"売却日 {requested_exit_date.date()} の価格データがまだ取得できません。"
            f"最新価格日 {latest_price_date.date()} 以前の営業日を指定してください。"
        )
    business_days = business_days[business_days <= latest_price_date]
    signal_days = signal_days[signal_days <= latest_price_date]
    if signal_days.empty:
        raise RuntimeError(
            f"指定したシグナル期間に価格データがありません。"
            f"最新価格日 {latest_price_date.date()} 以前の期間を指定してください。"
        )
    prices = prices[prices["Date"] <= latest_price_date].copy()
    prices = build_price_features(
        prices=prices,
        breakout_lookback_days=config.breakout_lookback_days,
        volume_lookback_days=config.volume_lookback_days,
    )

    price_in_window = prices[prices["Date"].isin(signal_days)].copy()
    if math.isclose(config.min_breakout_pct, 0.0):
        breakout_mask = price_in_window["AdjC"] > price_in_window["rolling_high_excl_today"]
    else:
        breakout_mask = price_in_window["high_break_ratio"] >= config.min_breakout_pct / 100.0
    volume_mask = pd.to_numeric(price_in_window["volume_ratio"], errors="coerce") >= config.min_volume_ratio
    price_candidates = price_in_window[breakout_mask & volume_mask].copy()
    price_candidate_count = int(price_candidates.shape[0])

    if price_candidates.empty:
        return build_empty_result(config=config, latest_price_date=latest_price_date, price_candidate_count=0)

    _emit_progress(progress_callback, "シグナル日の銘柄マスタを読み込んで市場フィルタを適用しています", 0.42)
    signal_dates = pd.DatetimeIndex(price_candidates["Date"].drop_duplicates().sort_values())
    master = load_master_for_dates(client=client, dates=signal_dates, cache_dir=cache_dir)
    market_candidates = attach_master(price_candidates, master, config)
    market_filtered_count = int(market_candidates.shape[0])

    if market_candidates.empty:
        return build_empty_result(
            config=config,
            latest_price_date=latest_price_date,
            price_candidate_count=price_candidate_count,
            market_filtered_count=0,
        )

    _emit_progress(progress_callback, "財務データを時点整合つきで結合しています", 0.60)
    financials = load_financial_summaries(
        client=client,
        start=signal_days.min() - pd.Timedelta(days=800),
        end=signal_days.max(),
        cache_dir=cache_dir,
    )
    snapshots = prepare_financial_snapshots(financials)
    candidates = attach_financials(market_candidates, snapshots)
    filtered = apply_screening_filters(candidates, config)
    final_signal_count = int(filtered.shape[0])

    if filtered.empty:
        return build_empty_result(
            config=config,
            latest_price_date=latest_price_date,
            price_candidate_count=price_candidate_count,
            market_filtered_count=market_filtered_count,
            final_signal_count=0,
        )

    _emit_progress(progress_callback, "指定日のリターンを計算しています", 0.78)
    signals = attach_forward_returns(filtered, prices, business_days, requested_exit_date)
    signals = signals.sort_values(["Date", "Code"]).reset_index(drop=True)
    cooldown_signals = apply_cooldown(signals, business_days, config.cooldown_business_days)
    cooldown_signals = cooldown_signals.sort_values(["Date", "Code"]).reset_index(drop=True)

    return_col = "ret"
    complete_signal_count = int(signals[return_col].notna().sum()) if return_col in signals.columns else 0
    cooldown_signal_count = int(cooldown_signals.shape[0])

    pipeline = build_pipeline_table(
        price_candidate_count=price_candidate_count,
        market_filtered_count=market_filtered_count,
        final_signal_count=final_signal_count,
        complete_signal_count=complete_signal_count,
        cooldown_signal_count=cooldown_signal_count,
    )

    _emit_progress(progress_callback, "集計テーブルを作成しています", 0.92)
    horizon_summary = summarize_returns(signals, return_col)
    cooldown_horizon_summary = summarize_returns(cooldown_signals, return_col)
    market_summary = summarize_by_group(signals, "MktNm", return_col, min_count=10)
    sector_summary = summarize_by_group(signals, "S33Nm", return_col, min_count=10, top_n=15)
    year_summary = summarize_by_year(signals, return_col)

    latest_signal_date = pd.Timestamp(signals["Date"].max()) if not signals.empty else None
    latest_screening = (
        signals[signals["Date"] == latest_signal_date]
        .sort_values(["volume_ratio", "turnover_yen", "Code"], ascending=[False, False, True])
        .reset_index(drop=True)
        if latest_signal_date is not None
        else pd.DataFrame()
    )

    observed = signals.dropna(subset=[return_col]).copy()
    top_signals = observed.sort_values(return_col, ascending=False).head(50).reset_index(drop=True)
    bottom_signals = observed.sort_values(return_col, ascending=True).head(50).reset_index(drop=True)

    metadata = {
        "analysis_start": config.signal_start.isoformat(),
        "analysis_end": config.signal_end.isoformat(),
        "latest_price_date": latest_price_date.date().isoformat(),
        "latest_signal_date": latest_signal_date.date().isoformat() if latest_signal_date is not None else None,
        "return_exit_date": requested_exit_date.date().isoformat(),
        "signal_count": int(signals.shape[0]),
        "cooldown_signal_count": int(cooldown_signals.shape[0]),
        "complete_signal_count": complete_signal_count,
    }

    _emit_progress(progress_callback, "分析が完了しました", 1.0)
    return AnalysisResult(
        signals=signals,
        cooldown_signals=cooldown_signals,
        latest_screening=latest_screening,
        pipeline=pipeline,
        horizon_summary=horizon_summary,
        cooldown_horizon_summary=cooldown_horizon_summary,
        market_summary=market_summary,
        sector_summary=sector_summary,
        year_summary=year_summary,
        top_signals=top_signals,
        bottom_signals=bottom_signals,
        metadata=metadata,
    )
