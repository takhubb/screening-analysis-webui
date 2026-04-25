from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from screening_analysis_webui import AnalysisResult, ScreeningConfig, run_analysis
from screening_analysis_webui.analysis import SUMMARY_METRIC_COLUMNS, summarize_returns
from screening_analysis_webui.jquants import DEFAULT_CACHE_DIR, load_dotenv
from screening_analysis_webui.models import MARKET_OPTIONS


st.set_page_config(
    page_title="J-Quants Screening Analysis",
    page_icon="📈",
    layout="wide",
)

load_dotenv()


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "output"
RETURN_HORIZON_LABELS = ("3ヶ月後", "6ヶ月後", "9ヶ月後", "12ヶ月後")


def parse_optional_float(raw_value: str, label: str, filter_name: str, errors: list[str]) -> float | None:
    text = raw_value.strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        errors.append(f"{filter_name} の {label} は数値で入力してください。")
        return None


def optional_float_input(
    label: str,
    *,
    default: str,
    key: str,
    filter_name: str,
    errors: list[str],
) -> float | None:
    raw_value = st.text_input(label, value=default, key=key)
    return parse_optional_float(raw_value, label, filter_name, errors)


def render_filter_patterns() -> tuple[list[dict[str, Any]], list[str]]:
    st.subheader("スクリーニング条件")
    st.caption("数値条件は空欄にすると、そのフィルターでは適用しません。業種はフィルターせず、CSVで全業種と業種別に集計します。")

    defaults = [
        {
            "name": "パターン1",
            "target_markets": ["プライム", "スタンダード"],
            "min_turnover_oku": "",
            "min_market_cap_oku": "",
            "max_market_cap_oku": "100",
            "max_per": "",
            "max_pbr": "1",
            "min_equity_ratio_pct": "20",
            "min_sales_growth_pct": "",
            "min_operating_profit_growth_pct": "",
            "min_operating_margin_pct": "",
        },
        {
            "name": "パターン2",
            "target_markets": ["プライム", "スタンダード", "グロース"],
            "min_turnover_oku": "1",
            "min_market_cap_oku": "",
            "max_market_cap_oku": "300",
            "max_per": "",
            "max_pbr": "2",
            "min_equity_ratio_pct": "20",
            "min_sales_growth_pct": "",
            "min_operating_profit_growth_pct": "",
            "min_operating_margin_pct": "",
        },
        {
            "name": "パターン3",
            "target_markets": ["プライム", "スタンダード", "グロース"],
            "min_turnover_oku": "",
            "min_market_cap_oku": "",
            "max_market_cap_oku": "",
            "max_per": "",
            "max_pbr": "",
            "min_equity_ratio_pct": "",
            "min_sales_growth_pct": "",
            "min_operating_profit_growth_pct": "",
            "min_operating_margin_pct": "",
        },
    ]

    errors: list[str] = []
    patterns: list[dict[str, Any]] = []
    columns = st.columns(3)
    for index, column in enumerate(columns):
        default = defaults[index]
        with column:
            st.markdown(f"**パターン {index + 1}**")
            filter_name = st.text_input("名称", value=str(default["name"]), key=f"filter_name_{index}")
            filter_name = filter_name.strip() or f"パターン{index + 1}"
            target_markets = st.multiselect(
                "対象市場",
                options=list(MARKET_OPTIONS),
                default=list(default["target_markets"]),
                key=f"target_markets_{index}",
                help="空欄の場合は市場で絞り込みません。",
            )
            exclude_funds = st.checkbox("ETF / ETN / REIT を除外", value=True, key=f"exclude_funds_{index}")

            min_turnover_oku = optional_float_input(
                "売買代金下限(億円)",
                default=str(default["min_turnover_oku"]),
                key=f"min_turnover_oku_{index}",
                filter_name=filter_name,
                errors=errors,
            )
            min_market_cap_oku = optional_float_input(
                "時価総額下限(億円)",
                default=str(default["min_market_cap_oku"]),
                key=f"min_market_cap_oku_{index}",
                filter_name=filter_name,
                errors=errors,
            )
            max_market_cap_oku = optional_float_input(
                "時価総額上限(億円)",
                default=str(default["max_market_cap_oku"]),
                key=f"max_market_cap_oku_{index}",
                filter_name=filter_name,
                errors=errors,
            )
            max_per = optional_float_input(
                "PER上限",
                default=str(default["max_per"]),
                key=f"max_per_{index}",
                filter_name=filter_name,
                errors=errors,
            )
            max_pbr = optional_float_input(
                "PBR上限",
                default=str(default["max_pbr"]),
                key=f"max_pbr_{index}",
                filter_name=filter_name,
                errors=errors,
            )
            min_equity_ratio_pct = optional_float_input(
                "自己資本比率下限(%)",
                default=str(default["min_equity_ratio_pct"]),
                key=f"min_equity_ratio_pct_{index}",
                filter_name=filter_name,
                errors=errors,
            )
            min_sales_growth_pct = optional_float_input(
                "売上成長率下限(%)",
                default=str(default["min_sales_growth_pct"]),
                key=f"min_sales_growth_pct_{index}",
                filter_name=filter_name,
                errors=errors,
            )
            min_operating_profit_growth_pct = optional_float_input(
                "営業利益成長率下限(%)",
                default=str(default["min_operating_profit_growth_pct"]),
                key=f"min_operating_profit_growth_pct_{index}",
                filter_name=filter_name,
                errors=errors,
            )
            min_operating_margin_pct = optional_float_input(
                "営業利益率下限(%)",
                default=str(default["min_operating_margin_pct"]),
                key=f"min_operating_margin_pct_{index}",
                filter_name=filter_name,
                errors=errors,
            )

            if (
                min_market_cap_oku is not None
                and max_market_cap_oku is not None
                and min_market_cap_oku > max_market_cap_oku
            ):
                errors.append(f"{filter_name} の時価総額下限は上限以下にしてください。")

            patterns.append(
                {
                    "filter_name": filter_name,
                    "target_markets": tuple(target_markets),
                    "exclude_funds": exclude_funds,
                    "min_turnover_oku": min_turnover_oku,
                    "min_market_cap_oku": min_market_cap_oku,
                    "max_market_cap_oku": max_market_cap_oku,
                    "max_per": max_per,
                    "max_pbr": max_pbr,
                    "min_equity_ratio_pct": min_equity_ratio_pct,
                    "min_sales_growth_pct": min_sales_growth_pct,
                    "min_operating_profit_growth_pct": min_operating_profit_growth_pct,
                    "min_operating_margin_pct": min_operating_margin_pct,
                }
            )

    for error in errors:
        st.error(error)
    return patterns, errors


def join_values(values: tuple[str, ...]) -> str:
    return "全市場" if not values else " / ".join(values)


def describe_optional(label: str, value: float | None, suffix: str = "") -> str:
    if value is None:
        return f"{label}=なし"
    return f"{label}={value:g}{suffix}"


def build_buy_signal_description(config: ScreeningConfig) -> str:
    return "; ".join(
        [
            f"期間={config.signal_start.isoformat()}..{config.signal_end.isoformat()}",
            f"高値更新比較={config.breakout_lookback_days}営業日",
            f"高値更新率下限={config.min_breakout_pct:g}%",
            f"出来高平均比較={config.volume_lookback_days}営業日",
            f"出来高倍率下限={config.min_volume_ratio:g}倍",
            "エントリー=買いシグナル翌営業日始値",
            f"評価={', '.join(RETURN_HORIZON_LABELS)}",
            f"クールダウン={config.cooldown_business_days}営業日",
        ]
    )


def build_screening_description(config: ScreeningConfig) -> str:
    return "; ".join(
        [
            f"対象市場={join_values(config.target_markets)}",
            f"ETF/ETN/REIT除外={'あり' if config.exclude_funds else 'なし'}",
            describe_optional("売買代金下限", config.min_turnover_oku, "億円"),
            describe_optional("時価総額下限", config.min_market_cap_oku, "億円"),
            describe_optional("時価総額上限", config.max_market_cap_oku, "億円"),
            describe_optional("PER上限", config.max_per),
            describe_optional("PBR上限", config.max_pbr),
            describe_optional("自己資本比率下限", config.min_equity_ratio_pct, "%"),
            describe_optional("売上成長率下限", config.min_sales_growth_pct, "%"),
            describe_optional("営業利益成長率下限", config.min_operating_profit_growth_pct, "%"),
            describe_optional("営業利益率下限", config.min_operating_margin_pct, "%"),
            "業種フィルター=なし",
        ]
    )


def build_condition_columns(config: ScreeningConfig, result: AnalysisResult) -> dict[str, object]:
    return {
        "filter_pattern": config.filter_name,
        "buy_signal_conditions": build_buy_signal_description(config),
        "screening_conditions": build_screening_description(config),
        "signal_start": config.signal_start.isoformat(),
        "signal_end": config.signal_end.isoformat(),
        "buy_signal_breakout_lookback_days": config.breakout_lookback_days,
        "buy_signal_min_breakout_pct": config.min_breakout_pct,
        "buy_signal_volume_lookback_days": config.volume_lookback_days,
        "buy_signal_min_volume_ratio": config.min_volume_ratio,
        "cooldown_business_days": config.cooldown_business_days,
        "screening_target_markets": join_values(config.target_markets),
        "screening_exclude_funds": config.exclude_funds,
        "screening_min_turnover_oku": config.min_turnover_oku,
        "screening_min_market_cap_oku": config.min_market_cap_oku,
        "screening_max_market_cap_oku": config.max_market_cap_oku,
        "screening_max_per": config.max_per,
        "screening_max_pbr": config.max_pbr,
        "screening_min_equity_ratio_pct": config.min_equity_ratio_pct,
        "screening_min_sales_growth_pct": config.min_sales_growth_pct,
        "screening_min_operating_profit_growth_pct": config.min_operating_profit_growth_pct,
        "screening_min_operating_margin_pct": config.min_operating_margin_pct,
        "screening_sector_filter": "なし",
        "signal_count": int(result.metadata.get("signal_count", 0)),
        "cooldown_signal_count": int(result.metadata.get("cooldown_signal_count", 0)),
        "complete_signal_count": int(result.metadata.get("complete_signal_count", 0)),
        "latest_signal_date": result.metadata.get("latest_signal_date"),
        "latest_price_date": result.metadata.get("latest_price_date"),
    }


def summarize_cooldown_by_sector(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty or "S33Nm" not in signals.columns:
        return pd.DataFrame(columns=["summary_scope", "sector", "horizon", *SUMMARY_METRIC_COLUMNS])

    frames: list[pd.DataFrame] = []
    sector_source = signals.copy()
    sector_source["S33Nm"] = sector_source["S33Nm"].fillna("未分類").astype(str)
    for sector_name, sector_signals in sector_source.groupby("S33Nm", sort=True):
        summary = summarize_returns(sector_signals)
        summary.insert(0, "summary_scope", "業種別")
        summary.insert(1, "sector", sector_name)
        frames.append(summary)

    if not frames:
        return pd.DataFrame(columns=["summary_scope", "sector", "horizon", *SUMMARY_METRIC_COLUMNS])
    return pd.concat(frames, ignore_index=True)


def build_export_frame(configs: list[ScreeningConfig], results: list[AnalysisResult]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for config, result in zip(configs, results, strict=True):
        condition_columns = build_condition_columns(config, result)

        all_sector_summary = result.cooldown_horizon_summary.copy()
        all_sector_summary.insert(0, "summary_scope", "全業種")
        all_sector_summary.insert(1, "sector", "全業種")

        sector_summary = summarize_cooldown_by_sector(result.cooldown_signals)
        combined = pd.concat([all_sector_summary, sector_summary], ignore_index=True)

        for column_name, value in condition_columns.items():
            combined.insert(0, column_name, value)
        frames.append(combined)

    if not frames:
        return pd.DataFrame()

    condition_columns = list(build_condition_columns(configs[0], results[0]).keys())
    ordered_columns = [
        *condition_columns,
        "summary_scope",
        "sector",
        "horizon",
        *SUMMARY_METRIC_COLUMNS,
    ]
    export_frame = pd.concat(frames, ignore_index=True)
    return export_frame.loc[:, [column for column in ordered_columns if column in export_frame.columns]]


def compact_number(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def sanitize_filename_component(value: str) -> str:
    safe = re.sub(r"[^\w.-]+", "-", value.strip())
    safe = safe.strip("-_.")
    return safe or "conditions"


def compact_optional_number(value: float | None) -> str:
    return "none" if value is None else compact_number(value)


def compact_markets(markets: tuple[str, ...]) -> str:
    if not markets:
        return "allmkt"
    labels = {"プライム": "prime", "スタンダード": "std", "グロース": "growth"}
    return "".join(labels.get(market, sanitize_filename_component(market)) for market in markets)


def build_output_filename(configs: list[ScreeningConfig]) -> str:
    first = configs[0]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    start = first.signal_start.strftime("%Y%m%d")
    end = first.signal_end.strftime("%Y%m%d")
    signal_part = (
        f"high{first.breakout_lookback_days}d{compact_number(first.min_breakout_pct)}pct_"
        f"vol{first.volume_lookback_days}d{compact_number(first.min_volume_ratio)}x_"
        f"cd{first.cooldown_business_days}d"
    )
    market_part = "mkt" + "-".join(compact_markets(config.target_markets) for config in configs)
    cap_part = "cap" + "-".join(compact_optional_number(config.max_market_cap_oku) for config in configs)
    pbr_part = "pbr" + "-".join(compact_optional_number(config.max_pbr) for config in configs)
    pattern_part = sanitize_filename_component("-".join(config.filter_name for config in configs))
    return f"cooldown_returns_{start}-{end}_{signal_part}_{market_part}_{cap_part}_{pbr_part}_{pattern_part}_{timestamp}.csv"


def write_export_csv(configs: list[ScreeningConfig], results: list[AnalysisResult]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / build_output_filename(configs)
    export_frame = build_export_frame(configs, results)
    export_frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def main() -> None:
    st.title("J-Quants スクリーニング & リターン分析 WebUI")
    st.caption("条件を指定して実行すると、クールダウン後の全業種・業種別リターン結果をCSVに出力します。")

    with st.sidebar:
        st.subheader("分析期間")
        signal_start = st.date_input("シグナル開始日", value=pd.Timestamp("2021-04-19").date())
        signal_end = st.date_input("シグナル終了日", value=pd.Timestamp.today().date())

        st.subheader("買いシグナル条件")
        breakout_lookback_days = st.number_input("高値更新の比較期間(営業日)", min_value=20, max_value=520, value=252, step=5)
        min_breakout_pct = st.number_input("高値更新率の下限(%)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
        volume_lookback_days = st.number_input("出来高平均の比較期間(営業日)", min_value=5, max_value=120, value=20, step=1)
        min_volume_ratio = st.number_input("出来高倍率の下限(倍)", min_value=1.0, max_value=20.0, value=2.0, step=0.1)

        st.subheader("リターン分析")
        cooldown_business_days = st.number_input("同一銘柄のクールダウン(営業日)", min_value=0, max_value=120, value=20, step=1)
        run_button = st.button("CSVを出力", use_container_width=True, type="primary")

    filter_patterns, filter_errors = render_filter_patterns()

    if run_button:
        if filter_errors:
            st.stop()

        configs = [
            ScreeningConfig(
                signal_start=signal_start,
                signal_end=signal_end,
                filter_name=str(pattern["filter_name"]),
                target_markets=tuple(pattern["target_markets"]),
                breakout_lookback_days=int(breakout_lookback_days),
                min_breakout_pct=float(min_breakout_pct),
                volume_lookback_days=int(volume_lookback_days),
                min_volume_ratio=float(min_volume_ratio),
                min_turnover_oku=pattern["min_turnover_oku"],
                min_market_cap_oku=pattern["min_market_cap_oku"],
                max_market_cap_oku=pattern["max_market_cap_oku"],
                max_per=pattern["max_per"],
                max_pbr=pattern["max_pbr"],
                min_equity_ratio_pct=pattern["min_equity_ratio_pct"],
                min_sales_growth_pct=pattern["min_sales_growth_pct"],
                min_operating_profit_growth_pct=pattern["min_operating_profit_growth_pct"],
                min_operating_margin_pct=pattern["min_operating_margin_pct"],
                cooldown_business_days=int(cooldown_business_days),
                exclude_funds=bool(pattern["exclude_funds"]),
            )
            for pattern in filter_patterns
        ]

        status = st.status("分析を開始しています", expanded=True)
        progress_bar = st.progress(0)
        results: list[AnalysisResult] = []

        try:
            for index, config in enumerate(configs):
                pattern_offset = index / len(configs)

                def on_progress(message: str, progress: float, *, current_config: ScreeningConfig = config) -> None:
                    total_progress = pattern_offset + progress / len(configs)
                    status.update(label=f"{current_config.filter_name}: {message}", state="running")
                    progress_bar.progress(min(max(int(total_progress * 100), 0), 100))

                results.append(run_analysis(config=config, cache_dir=DEFAULT_CACHE_DIR, progress_callback=on_progress))

            output_path = write_export_csv(configs, results)
        except Exception as exc:  # pragma: no cover
            status.update(label="分析に失敗しました", state="error")
            progress_bar.progress(0)
            st.exception(exc)
            st.stop()

        status.update(label="完了しました", state="complete")
        progress_bar.progress(100)
        st.session_state["last_output_path"] = str(output_path)

    output_path = st.session_state.get("last_output_path")
    if output_path:
        st.success(f"完了しました。CSVを出力しました: {output_path}")
    else:
        st.info("条件を設定して「CSVを出力」を押すと、結果CSVを `output/` に作成します。")


if __name__ == "__main__":
    main()
