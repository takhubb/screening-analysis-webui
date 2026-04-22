from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from screening_analysis_webui import ScreeningConfig, run_analysis
from screening_analysis_webui.jquants import DEFAULT_CACHE_DIR, load_dotenv
from screening_analysis_webui.models import MARKET_OPTIONS


st.set_page_config(
    page_title="J-Quants Screening Analysis",
    page_icon="📈",
    layout="wide",
)

load_dotenv()


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
        "volume_ratio",
        "high_break_ratio",
        "turnover_yen",
        "market_cap",
        "PBR",
        "equity_ratio",
        "sales_growth_yoy",
        "operating_profit_growth_yoy",
        "operating_margin",
    ]
    return_columns = [column for column in df.columns if column.startswith("ret_")]
    columns.extend(return_columns)
    available = [column for column in columns if column in df.columns]
    formatted = df.loc[:, available].copy()
    renamed = {
        "Date": "シグナル日",
        "Code": "銘柄コード",
        "CoName": "銘柄名",
        "MktNm": "市場",
        "S33Nm": "業種",
        "AdjC": "調整後終値",
        "volume_ratio": "出来高倍率",
        "high_break_ratio": "高値更新率",
        "turnover_yen": "売買代金(円)",
        "market_cap": "時価総額(円)",
        "PBR": "PBR",
        "equity_ratio": "自己資本比率",
        "sales_growth_yoy": "売上成長率",
        "operating_profit_growth_yoy": "営業利益成長率",
        "operating_margin": "営業利益率",
    }
    for column in return_columns:
        renamed[column] = column.replace("ret_", "").replace("m", "か月リターン")
    formatted = formatted.rename(columns=renamed)

    if "シグナル日" in formatted.columns:
        formatted["シグナル日"] = pd.to_datetime(formatted["シグナル日"], errors="coerce").dt.strftime("%Y-%m-%d")

    for column in ("調整後終値", "PBR", "出来高倍率"):
        if column in formatted.columns:
            formatted[column] = pd.to_numeric(formatted[column], errors="coerce").round(2)
    for column in ("売買代金(円)", "時価総額(円)"):
        if column in formatted.columns:
            formatted[column] = pd.to_numeric(formatted[column], errors="coerce").round(0)
    for column in (
        "高値更新率",
        "自己資本比率",
        "売上成長率",
        "営業利益成長率",
        "営業利益率",
        *[name for name in formatted.columns if "リターン" in name],
    ):
        if column in formatted.columns:
            formatted[column] = pd.to_numeric(formatted[column], errors="coerce").map(
                lambda value: round(value * 100.0, 2) if pd.notna(value) else value
            )

    return formatted


def render_summary_metrics(metadata: dict[str, object]) -> None:
    metrics = st.columns(4)
    metrics[0].metric("シグナル件数", f"{int(metadata.get('signal_count', 0)):,}")
    metrics[1].metric("クールダウン後件数", f"{int(metadata.get('cooldown_signal_count', 0)):,}")
    metrics[2].metric("最終価格日", str(metadata.get("latest_price_date") or "-"))
    metrics[3].metric("最新シグナル日", str(metadata.get("latest_signal_date") or "-"))


def main() -> None:
    st.title("J-Quants スクリーニング & リターン分析 WebUI")
    st.caption("`references` のロジックを参考にしつつ、bulk キャッシュを使って条件スクリーニングと将来リターン分析を行います。")

    with st.sidebar:
        st.subheader("分析期間")
        signal_start = st.date_input("シグナル開始日", value=pd.Timestamp("2021-04-19").date())
        signal_end = st.date_input("シグナル終了日", value=pd.Timestamp.today().date())

        st.subheader("基本条件")
        target_markets = st.multiselect(
            "対象市場",
            options=list(MARKET_OPTIONS),
            default=["プライム", "スタンダード"],
            help="References の基本条件に合わせてプライム / スタンダードを初期値にしています。",
        )
        breakout_lookback_days = st.number_input("高値更新の比較期間(営業日)", min_value=20, max_value=520, value=252, step=5)
        min_breakout_pct = st.number_input("高値更新率の下限(%)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
        volume_lookback_days = st.number_input("出来高平均の比較期間(営業日)", min_value=5, max_value=120, value=20, step=1)
        min_volume_ratio = st.number_input("出来高倍率の下限(倍)", min_value=1.0, max_value=20.0, value=2.0, step=0.1)
        exclude_funds = st.checkbox("ETF / ETN / REIT を除外する", value=True)

        st.subheader("追加条件")
        use_market_cap = st.checkbox("時価総額上限を使う", value=True)
        max_market_cap_oku = st.number_input(
            "時価総額上限(億円)",
            min_value=1.0,
            max_value=10000.0,
            value=100.0,
            step=10.0,
            disabled=not use_market_cap,
        )
        use_pbr = st.checkbox("PBR 上限を使う", value=True)
        max_pbr = st.number_input("PBR 上限", min_value=0.1, max_value=20.0, value=1.0, step=0.1, disabled=not use_pbr)
        use_equity_ratio = st.checkbox("自己資本比率下限を使う", value=True)
        min_equity_ratio_pct = st.number_input(
            "自己資本比率下限(%)",
            min_value=0.0,
            max_value=100.0,
            value=20.0,
            step=1.0,
            disabled=not use_equity_ratio,
        )
        use_sales_growth = st.checkbox("売上成長率下限を使う", value=False)
        min_sales_growth_pct = st.number_input(
            "売上成長率下限(%)",
            min_value=-100.0,
            max_value=500.0,
            value=0.0,
            step=5.0,
            disabled=not use_sales_growth,
        )
        use_op_growth = st.checkbox("営業利益成長率下限を使う", value=False)
        min_operating_profit_growth_pct = st.number_input(
            "営業利益成長率下限(%)",
            min_value=-500.0,
            max_value=1000.0,
            value=0.0,
            step=10.0,
            disabled=not use_op_growth,
        )
        use_op_margin = st.checkbox("営業利益率下限を使う", value=False)
        min_operating_margin_pct = st.number_input(
            "営業利益率下限(%)",
            min_value=-100.0,
            max_value=100.0,
            value=5.0,
            step=1.0,
            disabled=not use_op_margin,
        )
        use_turnover = st.checkbox("売買代金下限を使う", value=False)
        min_turnover_oku = st.number_input(
            "売買代金下限(億円)",
            min_value=0.0,
            max_value=1000.0,
            value=1.0,
            step=1.0,
            disabled=not use_turnover,
        )

        st.subheader("リターン分析")
        return_horizons = st.multiselect(
            "分析ホライズン(月)",
            options=[1, 3, 6, 9, 12, 18, 24],
            default=[3, 6, 9, 12],
        )
        cooldown_business_days = st.number_input("同一銘柄のクールダウン(営業日)", min_value=0, max_value=120, value=20, step=1)

        run_button = st.button("スクリーニングと分析を実行", use_container_width=True, type="primary")

    if run_button:
        if not target_markets:
            st.error("対象市場を 1 つ以上選択してください。")
            st.stop()
        if not return_horizons:
            st.error("分析ホライズンを 1 つ以上選択してください。")
            st.stop()

        config = ScreeningConfig(
            signal_start=signal_start,
            signal_end=signal_end,
            target_markets=tuple(target_markets),
            breakout_lookback_days=int(breakout_lookback_days),
            min_breakout_pct=float(min_breakout_pct),
            volume_lookback_days=int(volume_lookback_days),
            min_volume_ratio=float(min_volume_ratio),
            min_turnover_oku=float(min_turnover_oku) if use_turnover else None,
            max_market_cap_oku=float(max_market_cap_oku) if use_market_cap else None,
            max_pbr=float(max_pbr) if use_pbr else None,
            min_equity_ratio_pct=float(min_equity_ratio_pct) if use_equity_ratio else None,
            min_sales_growth_pct=float(min_sales_growth_pct) if use_sales_growth else None,
            min_operating_profit_growth_pct=float(min_operating_profit_growth_pct) if use_op_growth else None,
            min_operating_margin_pct=float(min_operating_margin_pct) if use_op_margin else None,
            cooldown_business_days=int(cooldown_business_days),
            return_horizons=tuple(int(months) for months in return_horizons),
            exclude_funds=exclude_funds,
        )

        status = st.status("分析を開始しています", expanded=True)
        progress_bar = st.progress(0)

        def on_progress(message: str, progress: float) -> None:
            status.update(label=message, state="running")
            progress_bar.progress(min(max(int(progress * 100), 0), 100))

        try:
            result = run_analysis(config=config, cache_dir=DEFAULT_CACHE_DIR, progress_callback=on_progress)
        except Exception as exc:  # pragma: no cover
            status.update(label="分析に失敗しました", state="error")
            progress_bar.progress(0)
            st.exception(exc)
            st.stop()

        status.update(label="分析が完了しました", state="complete")
        progress_bar.progress(100)
        st.session_state["analysis_result"] = result
        st.session_state["analysis_config"] = config

    result = st.session_state.get("analysis_result")
    if result is None:
        st.info("左の条件を設定して実行すると、スクリーニング結果と将来リターン分析を表示します。")
        return

    render_summary_metrics(result.metadata)

    if result.signals.empty:
        st.warning("条件に一致するシグナルはありませんでした。条件を少し緩めて再実行してください。")
        st.dataframe(format_percent_columns(result.pipeline, ["share_vs_initial"]), use_container_width=True)
        return

    longest_horizon = max(st.session_state["analysis_config"].return_horizons)
    longest_return_col = f"ret_{longest_horizon}m"

    tabs = st.tabs(["概要", "集計", "最新スクリーニング", "全シグナル"])

    with tabs[0]:
        st.subheader("パイプライン")
        st.dataframe(format_percent_columns(result.pipeline, ["share_vs_initial"]), use_container_width=True)

        st.subheader("ホライズン別サマリー")
        summary_display = format_percent_columns(
            result.horizon_summary,
            ["mean", "median", "std", "win_rate", "p10", "p25", "p75", "p90"],
        )
        st.dataframe(summary_display, use_container_width=True)

        chart_df = result.horizon_summary.copy()
        if not chart_df.empty:
            melted = chart_df.melt(
                id_vars="horizon",
                value_vars=["mean", "median", "win_rate"],
                var_name="metric",
                value_name="value",
            )
            fig = px.bar(melted, x="horizon", y="value", color="metric", barmode="group", title="平均・中央値・勝率")
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        left, right = st.columns(2)
        with left:
            st.subheader("市場別")
            st.dataframe(format_percent_columns(result.market_summary, ["mean", "median", "win_rate"]), use_container_width=True)
            st.subheader("年別")
            st.dataframe(format_percent_columns(result.year_summary, ["mean", "median", "win_rate"]), use_container_width=True)
        with right:
            st.subheader("業種別")
            st.dataframe(format_percent_columns(result.sector_summary, ["mean", "median", "win_rate"]), use_container_width=True)
            st.subheader(f"{longest_horizon}か月リターン分布")
            returns = result.signals[longest_return_col].dropna()
            if returns.empty:
                st.info("分布を描けるだけの将来リターンがまだありません。")
            else:
                fig = px.histogram(
                    x=returns * 100.0,
                    nbins=40,
                    labels={"x": f"{longest_horizon}か月リターン(%)"},
                    title=f"{longest_horizon}か月リターンの分布",
                )
                st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        latest = format_signal_table(result.latest_screening)
        latest_date = result.metadata.get("latest_signal_date")
        st.subheader(f"最新シグナル日: {latest_date}")
        st.dataframe(latest, use_container_width=True, height=520)
        st.download_button(
            "最新スクリーニング CSV をダウンロード",
            data=latest.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"latest_screening_{latest_date}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with tabs[3]:
        all_signals = format_signal_table(result.signals)
        cooldown_signals = format_signal_table(result.cooldown_signals)

        st.subheader("全シグナル")
        st.dataframe(all_signals, use_container_width=True, height=520)
        st.download_button(
            "全シグナル CSV をダウンロード",
            data=all_signals.to_csv(index=False).encode("utf-8-sig"),
            file_name="signals_all.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.subheader("クールダウン後シグナル")
        st.dataframe(cooldown_signals, use_container_width=True, height=420)
        st.download_button(
            "クールダウン後 CSV をダウンロード",
            data=cooldown_signals.to_csv(index=False).encode("utf-8-sig"),
            file_name="signals_cooldown.csv",
            mime="text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
