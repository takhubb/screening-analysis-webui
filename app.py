from __future__ import annotations

import pandas as pd
import streamlit as st

from screening_analysis_webui import AnalysisResult, ScreeningConfig, run_analysis
from screening_analysis_webui.jquants import DEFAULT_CACHE_DIR, load_dotenv
from screening_analysis_webui.models import MARKET_OPTIONS, SECTOR_33_OPTIONS


st.set_page_config(
    page_title="J-Quants Screening Analysis",
    page_icon="📈",
    layout="wide",
)

load_dotenv()


RETURN_MONTHS = (3, 6, 9, 12)
SUMMARY_METRIC_COLUMNS = [
    "count",
    "mean",
    "median",
    "std",
    "min",
    "p10",
    "p25",
    "p75",
    "p90",
    "max",
    "win_rate",
    "sharpe",
    "max_drawdown",
]
SUMMARY_PERCENT_COLUMNS = [
    "mean",
    "median",
    "std",
    "min",
    "p10",
    "p25",
    "p75",
    "p90",
    "max",
    "win_rate",
    "max_drawdown",
]


def format_percent_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    formatted = df.copy()
    for column in columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "")
    return formatted


def format_signal_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    columns = [
        "Date",
        "Code",
        "CoName",
        "MktNm",
        "S33Nm",
        "AdjC",
        "entry_date",
        "entry_price",
        "buy_amount_yen",
    ]
    for months in RETURN_MONTHS:
        columns.extend([f"exit_date_{months}m", f"ret_{months}m"])
    columns.extend(
        [
            "volume_ratio",
            "high_break_ratio",
            "turnover_yen",
            "market_cap",
            "PER",
            "PBR",
            "equity_ratio",
            "sales_growth_yoy",
            "operating_profit_growth_yoy",
            "operating_margin",
        ]
    )

    available = [column for column in columns if column in df.columns]
    formatted = df.loc[:, available].copy()
    renamed = {
        "Date": "シグナル日",
        "Code": "銘柄コード",
        "CoName": "銘柄名",
        "MktNm": "市場",
        "S33Nm": "業種",
        "AdjC": "シグナル日終値(調整後)",
        "entry_date": "エントリー日",
        "entry_price": "エントリー始値(調整後)",
        "buy_amount_yen": "購入額(100株)",
        "volume_ratio": "出来高倍率",
        "high_break_ratio": "高値更新率",
        "turnover_yen": "売買代金(円)",
        "market_cap": "時価総額(円)",
        "PER": "PER",
        "PBR": "PBR",
        "equity_ratio": "自己資本比率",
        "sales_growth_yoy": "売上成長率",
        "operating_profit_growth_yoy": "営業利益成長率",
        "operating_margin": "営業利益率",
    }
    for months in RETURN_MONTHS:
        renamed[f"exit_date_{months}m"] = f"{months}ヶ月後評価日"
        renamed[f"ret_{months}m"] = f"{months}ヶ月後リターン"
    formatted = formatted.rename(columns=renamed)

    date_columns = ["シグナル日", "エントリー日", *[f"{months}ヶ月後評価日" for months in RETURN_MONTHS]]
    for column in date_columns:
        if column in formatted.columns:
            formatted[column] = pd.to_datetime(formatted[column], errors="coerce").dt.strftime("%Y-%m-%d")

    for column in ("シグナル日終値(調整後)", "エントリー始値(調整後)", "PER", "PBR", "出来高倍率"):
        if column in formatted.columns:
            formatted[column] = pd.to_numeric(formatted[column], errors="coerce").round(2)
    for column in ("売買代金(円)", "時価総額(円)", "購入額(100株)"):
        if column in formatted.columns:
            formatted[column] = pd.to_numeric(formatted[column], errors="coerce").round(0)
    for column in (
        "高値更新率",
        "自己資本比率",
        "売上成長率",
        "営業利益成長率",
        "営業利益率",
        *[f"{months}ヶ月後リターン" for months in RETURN_MONTHS],
    ):
        if column in formatted.columns:
            formatted[column] = pd.to_numeric(formatted[column], errors="coerce").map(
                lambda value: round(value * 100.0, 2) if pd.notna(value) else value
            )

    return formatted


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


def render_filter_patterns() -> tuple[list[dict[str, object]], list[str]]:
    st.subheader("フィルター")
    st.caption("数値条件は空欄にすると、そのフィルターでは適用しません。")

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
    patterns: list[dict[str, object]] = []
    columns = st.columns(3)
    for index, column in enumerate(columns):
        default = defaults[index]
        with column:
            st.markdown(f"**フィルター {index + 1}**")
            filter_name = st.text_input("名称", value=str(default["name"]), key=f"filter_name_{index}")
            filter_name = filter_name.strip() or f"パターン{index + 1}"
            target_markets = st.multiselect(
                "対象市場",
                options=list(MARKET_OPTIONS),
                default=list(default["target_markets"]),
                key=f"target_markets_{index}",
            )
            target_sectors = st.multiselect(
                "業種",
                options=list(SECTOR_33_OPTIONS),
                default=[],
                key=f"target_sectors_{index}",
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
                    "target_sectors": tuple(target_sectors),
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


def render_summary_metrics(results: list[AnalysisResult]) -> None:
    latest_price_dates = [result.metadata.get("latest_price_date") for result in results if result.metadata.get("latest_price_date")]
    latest_signal_dates = [
        result.metadata.get("latest_signal_date") for result in results if result.metadata.get("latest_signal_date")
    ]
    complete_count = sum(int(result.metadata.get("complete_signal_count", 0)) for result in results)

    metrics = st.columns(4)
    metrics[0].metric("フィルター数", f"{len(results):,}")
    metrics[1].metric("リターン観測可能", f"{complete_count:,}")
    metrics[2].metric("最終価格日", str(max(latest_price_dates) if latest_price_dates else "-"))
    metrics[3].metric("最新シグナル日", str(max(latest_signal_dates) if latest_signal_dates else "-"))


def build_combined_summary(results: list[AnalysisResult], attr_name: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for result in results:
        frame = getattr(result, attr_name).copy()
        frame.insert(0, "filter_pattern", result.metadata.get("filter_name", "-"))
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["filter_pattern", "horizon", *SUMMARY_METRIC_COLUMNS])
    combined = pd.concat(frames, ignore_index=True)
    columns = ["filter_pattern", "horizon", *SUMMARY_METRIC_COLUMNS]
    return combined.loc[:, [column for column in columns if column in combined.columns]]


def build_count_summary(results: list[AnalysisResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "filter_pattern": result.metadata.get("filter_name", "-"),
                "signal_count": int(result.metadata.get("signal_count", 0)),
                "complete_signal_count": int(result.metadata.get("complete_signal_count", 0)),
                "cooldown_signal_count": int(result.metadata.get("cooldown_signal_count", 0)),
                "latest_signal_date": result.metadata.get("latest_signal_date"),
            }
        )
    return pd.DataFrame(rows)


def build_pipeline_summary(results: list[AnalysisResult]) -> pd.DataFrame:
    frames = []
    for result in results:
        pipeline = result.pipeline.copy()
        pipeline.insert(0, "filter_pattern", result.metadata.get("filter_name", "-"))
        frames.append(pipeline)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def select_result(results: list[AnalysisResult], key: str) -> AnalysisResult:
    labels = [str(result.metadata.get("filter_name", f"パターン{index + 1}")) for index, result in enumerate(results)]
    selected_label = st.selectbox("表示するフィルター", options=labels, key=key)
    selected_index = labels.index(selected_label)
    return results[selected_index]


def main() -> None:
    st.title("J-Quants スクリーニング & リターン分析 WebUI")
    st.caption("bulk キャッシュを使って買いシグナル抽出と、翌営業日始値エントリー後の固定期間リターン分析を行います。")

    with st.sidebar:
        st.subheader("分析期間")
        signal_start = st.date_input("シグナル開始日", value=pd.Timestamp("2021-04-19").date())
        signal_end = st.date_input("シグナル終了日", value=pd.Timestamp.today().date())

        st.subheader("買いシグナル")
        breakout_lookback_days = st.number_input("高値更新の比較期間(営業日)", min_value=20, max_value=520, value=252, step=5)
        min_breakout_pct = st.number_input("高値更新率の下限(%)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
        volume_lookback_days = st.number_input("出来高平均の比較期間(営業日)", min_value=5, max_value=120, value=20, step=1)
        min_volume_ratio = st.number_input("出来高倍率の下限(倍)", min_value=1.0, max_value=20.0, value=2.0, step=0.1)

        st.subheader("リターン分析")
        cooldown_business_days = st.number_input("同一銘柄のクールダウン(営業日)", min_value=0, max_value=120, value=20, step=1)
        run_button = st.button("スクリーニングと分析を実行", use_container_width=True, type="primary")

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
                target_sectors=tuple(pattern["target_sectors"]),
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
        except Exception as exc:  # pragma: no cover
            status.update(label="分析に失敗しました", state="error")
            progress_bar.progress(0)
            st.exception(exc)
            st.stop()

        status.update(label="分析が完了しました", state="complete")
        progress_bar.progress(100)
        st.session_state["analysis_results"] = results
        st.session_state["analysis_configs"] = configs

    results = st.session_state.get("analysis_results")
    if not results:
        st.info("買いシグナルとフィルターを設定して実行すると、3パターンのリターンサマリーを比較表示します。")
        return

    render_summary_metrics(results)

    if all(result.signals.empty for result in results):
        st.warning("条件に一致するシグナルはありませんでした。条件を少し緩めて再実行してください。")

    tabs = st.tabs(["比較サマリー", "最新スクリーニング", "全シグナル"])

    with tabs[0]:
        st.subheader("パターン別件数")
        st.dataframe(build_count_summary(results), use_container_width=True)

        st.subheader("リターンサマリー")
        summary_display = format_percent_columns(build_combined_summary(results, "horizon_summary"), SUMMARY_PERCENT_COLUMNS)
        st.dataframe(summary_display, use_container_width=True)

        st.subheader("クールダウン後リターンサマリー")
        cooldown_summary_display = format_percent_columns(
            build_combined_summary(results, "cooldown_horizon_summary"),
            SUMMARY_PERCENT_COLUMNS,
        )
        st.dataframe(cooldown_summary_display, use_container_width=True)

        st.subheader("パイプライン")
        st.dataframe(format_percent_columns(build_pipeline_summary(results), ["share_vs_initial"]), use_container_width=True)

    with tabs[1]:
        selected_result = select_result(results, "latest_result_selector")
        latest = format_signal_table(selected_result.latest_screening)
        latest_date = selected_result.metadata.get("latest_signal_date")
        st.subheader(f"最新シグナル日: {latest_date or '-'}")
        st.dataframe(latest, use_container_width=True, height=520)
        st.download_button(
            "最新スクリーニング CSV をダウンロード",
            data=latest.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"latest_screening_{selected_result.metadata.get('filter_name', 'pattern')}_{latest_date}.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_latest",
        )

    with tabs[2]:
        selected_result = select_result(results, "all_result_selector")
        all_signals = format_signal_table(selected_result.signals)
        cooldown_signals = format_signal_table(selected_result.cooldown_signals)

        st.subheader("全シグナル")
        st.dataframe(all_signals, use_container_width=True, height=520)
        st.download_button(
            "全シグナル CSV をダウンロード",
            data=all_signals.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"signals_all_{selected_result.metadata.get('filter_name', 'pattern')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_all",
        )

        st.subheader("クールダウン後シグナル")
        st.dataframe(cooldown_signals, use_container_width=True, height=420)
        st.download_button(
            "クールダウン後 CSV をダウンロード",
            data=cooldown_signals.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"signals_cooldown_{selected_result.metadata.get('filter_name', 'pattern')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_cooldown",
        )


if __name__ == "__main__":
    main()
