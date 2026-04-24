from __future__ import annotations

import os
import re
import shlex
import time
from pathlib import Path

import pandas as pd
from requests.exceptions import HTTPError, RequestException, RetryError

try:
    import jquantsapi
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "jquants-api-client が見つかりません。requirements.txt をインストールしてください。"
    ) from exc


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = ROOT_DIR / ".cache" / "jquants"
BULK_LIST_CACHE_SECONDS = 6 * 60 * 60
LIVE_BULK_REFRESH_SECONDS = 6 * 60 * 60
RATE_LIMIT_INITIAL_WAIT_SECONDS = 15.0
RATE_LIMIT_MAX_WAIT_SECONDS = 180.0
RATE_LIMIT_MAX_RETRIES = 5

PRICE_COLUMNS = ["Date", "Code", "O", "H", "L", "C", "Vo", "Va", "AdjFactor"]
MASTER_COLUMNS = ["Date", "Code", "CoName", "S17Nm", "S33Nm", "ScaleCat", "MktNm"]
FIN_COLUMNS = [
    "DiscDate",
    "DiscTime",
    "Code",
    "DiscNo",
    "DocType",
    "CurPerType",
    "CurPerEn",
    "CurFYEn",
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


def load_dotenv(dotenv_path: Path | None = None) -> None:
    target = dotenv_path or ROOT_DIR / ".env"
    if not target.exists():
        return

    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if value:
            value = shlex.split(value)[0] if value[:1] in {"'", '"'} else value
        os.environ[key] = value.strip("'\"")


def create_client() -> jquantsapi.ClientV2:
    load_dotenv()
    api_key = os.getenv("JQUANTS_API_KEY")
    if not api_key:
        raise RuntimeError("JQUANTS_API_KEY が設定されていません。.env を確認してください。")
    return jquantsapi.ClientV2()


def _sanitize_endpoint(endpoint: str) -> str:
    return endpoint.strip("/").replace("/", "_")


def _parse_bulk_key_coverage(key: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    filename = Path(key).name
    match = re.search(r"(20\d{6}|20\d{4})", filename)
    if not match:
        raise RuntimeError(f"bulk キーから日付を解釈できませんでした: {key}")

    token = match.group(1)
    if len(token) == 8:
        current = pd.Timestamp(token)
        return current.normalize(), current.normalize()

    month_start = pd.Timestamp(f"{token}01")
    month_end = month_start + pd.offsets.MonthEnd(0)
    return month_start.normalize(), month_end.normalize()


def _is_rate_limited_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "too many requests" in text or "rate limit" in text


def _parse_subscription_start_date(exc: Exception) -> pd.Timestamp | None:
    match = re.search(r"covers the following dates:\s*(\d{4}-\d{2}-\d{2})\s*~", str(exc))
    if match is None:
        return None
    return pd.Timestamp(match.group(1)).normalize()


def _call_with_rate_limit_retry(
    operation,
    *,
    description: str,
    max_retries: int = RATE_LIMIT_MAX_RETRIES,
    initial_wait_seconds: float = RATE_LIMIT_INITIAL_WAIT_SECONDS,
):
    for attempt in range(max_retries + 1):
        try:
            return operation()
        except (RequestException, RetryError) as exc:
            if not _is_rate_limited_error(exc) or attempt >= max_retries:
                raise

            wait_seconds = min(initial_wait_seconds * (2**attempt), RATE_LIMIT_MAX_WAIT_SECONDS)
            print(
                f"[warn] {description} でレート制限に到達したため {wait_seconds:.0f} 秒待機して再試行します。",
                flush=True,
            )
            time.sleep(wait_seconds)


def _load_bulk_list(
    client: jquantsapi.ClientV2,
    endpoint: str,
    cache_dir: Path,
) -> pd.DataFrame:
    listing_dir = cache_dir / "meta"
    listing_dir.mkdir(parents=True, exist_ok=True)
    cache_path = listing_dir / f"bulk_list_{_sanitize_endpoint(endpoint)}.parquet"

    if cache_path.exists() and time.time() - cache_path.stat().st_mtime <= BULK_LIST_CACHE_SECONDS:
        cached = pd.read_parquet(cache_path)
        if not cached.empty:
            return cached

    listing = _call_with_rate_limit_retry(
        lambda: client.get_bulk_list(endpoint),
        description=f"{endpoint} の bulk list 取得",
    )
    if not listing.empty:
        listing.to_parquet(cache_path, index=False)
    return listing


def _iter_selected_bulk_files(
    client: jquantsapi.ClientV2,
    endpoint: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path,
) -> list[tuple[str, Path]]:
    listing = _load_bulk_list(client, endpoint, cache_dir)
    if listing.empty:
        raise RuntimeError(f"{endpoint} の bulk 一覧を取得できませんでした。")

    matched_rows: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for row in listing.itertuples(index=False):
        key = str(row.Key)
        covered_start, covered_end = _parse_bulk_key_coverage(key)
        if covered_end < start or covered_start > end:
            continue
        matched_rows.append((key, covered_start, covered_end))

    if not matched_rows:
        raise RuntimeError(
            f"{endpoint} で {start.date()} から {end.date()} に該当する bulk ファイルがありません。"
        )

    latest_live_end = max(
        (covered_end for key, _, covered_end in matched_rows if "/live/" in key),
        default=None,
    )
    selected: list[tuple[str, Path]] = []
    target_dir = cache_dir / "raw" / _sanitize_endpoint(endpoint)
    target_dir.mkdir(parents=True, exist_ok=True)

    now = time.time()

    for key, _, covered_end in matched_rows:
        local_path = target_dir / Path(key).name
        should_refresh_live = (
            "/live/" in key
            and latest_live_end is not None
            and covered_end == latest_live_end
            and local_path.exists()
            and now - local_path.stat().st_mtime > LIVE_BULK_REFRESH_SECONDS
        )
        if not local_path.exists() or should_refresh_live:
            try:
                _call_with_rate_limit_retry(
                    lambda: client.download_bulk(key=key, output_path=str(local_path)),
                    description=f"{key} の bulk ダウンロード",
                )
            except Exception:
                if local_path.exists():
                    print(
                        f"[warn] {key} の更新取得に失敗したため、既存のキャッシュを使用します。",
                        flush=True,
                    )
                else:
                    raise
        selected.append((key, local_path))

    selected.sort(key=lambda item: item[0])
    return selected


def _load_cached_bulk_frame(
    raw_path: Path,
    parquet_dir: Path,
    usecols: list[str],
    date_columns: tuple[str, ...],
) -> pd.DataFrame:
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = parquet_dir / f"{raw_path.name}.parquet"
    if parquet_path.exists() and parquet_path.stat().st_mtime >= raw_path.stat().st_mtime:
        return pd.read_parquet(parquet_path)

    df = pd.read_csv(
        raw_path,
        usecols=usecols,
        dtype={"Code": str, "DiscNo": str, "DocType": str, "CurPerType": str},
    )
    for column in date_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    if "DiscTime" in df.columns:
        df["DiscTime"] = df["DiscTime"].fillna("00:00:00").astype(str)
    df.to_parquet(parquet_path, index=False)
    return df


def load_calendar(
    client: jquantsapi.ClientV2,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "calendar.parquet"

    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached["Date"] = pd.to_datetime(cached["Date"], errors="coerce")
        if not cached.empty and cached["Date"].min() <= start and cached["Date"].max() >= end:
            mask = (cached["Date"] >= start) & (cached["Date"] <= end)
            return cached.loc[mask].sort_values("Date").reset_index(drop=True)
    else:
        cached = pd.DataFrame()

    fetch_start = start
    try:
        fetched = client.get_mkt_calendar(
            from_yyyymmdd=fetch_start.strftime("%Y-%m-%d"),
            to_yyyymmdd=end.strftime("%Y-%m-%d"),
        )
    except HTTPError as exc:
        subscription_start = _parse_subscription_start_date(exc)
        if subscription_start is None or subscription_start > end or fetch_start >= subscription_start:
            raise

        fetch_start = subscription_start
        print(
            f"[warn] 契約開始日より前のカレンダーは取得できないため、"
            f"{fetch_start.date()} から取得し直します。",
            flush=True,
        )
        fetched = client.get_mkt_calendar(
            from_yyyymmdd=fetch_start.strftime("%Y-%m-%d"),
            to_yyyymmdd=end.strftime("%Y-%m-%d"),
        )
    if fetched.empty:
        raise RuntimeError("取引カレンダーを取得できませんでした。")

    fetched["Date"] = pd.to_datetime(fetched["Date"], errors="coerce")
    combined = fetched if cached.empty else pd.concat([cached, fetched], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    combined.to_parquet(cache_path, index=False)
    mask = (combined["Date"] >= start) & (combined["Date"] <= end)
    return combined.loc[mask].sort_values("Date").reset_index(drop=True)


def load_prices(
    client: jquantsapi.ClientV2,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    selected = _iter_selected_bulk_files(client, "/equities/bars/daily", start, end, cache_dir)
    parquet_dir = cache_dir / "parquet" / "prices"
    frames: list[pd.DataFrame] = []

    for _, raw_path in selected:
        frame = _load_cached_bulk_frame(
            raw_path=raw_path,
            parquet_dir=parquet_dir,
            usecols=PRICE_COLUMNS,
            date_columns=("Date",),
        )
        filtered = frame[(frame["Date"] >= start) & (frame["Date"] <= end)].copy()
        if not filtered.empty:
            filtered["Code"] = filtered["Code"].astype(str)
            frames.append(filtered)

    if not frames:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    return pd.concat(frames, ignore_index=True).sort_values(["Code", "Date"]).reset_index(drop=True)


def load_master_for_dates(
    client: jquantsapi.ClientV2,
    dates: pd.DatetimeIndex,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    if dates.empty:
        return pd.DataFrame(columns=MASTER_COLUMNS)

    start = pd.Timestamp(dates.min()).normalize()
    end = pd.Timestamp(dates.max()).normalize()
    selected = _iter_selected_bulk_files(client, "/equities/master", start, end, cache_dir)
    parquet_dir = cache_dir / "parquet" / "master"
    wanted_dates = {pd.Timestamp(value).strftime("%Y-%m-%d") for value in dates}
    frames: list[pd.DataFrame] = []

    for _, raw_path in selected:
        frame = _load_cached_bulk_frame(
            raw_path=raw_path,
            parquet_dir=parquet_dir,
            usecols=MASTER_COLUMNS,
            date_columns=("Date",),
        )
        filtered = frame[frame["Date"].dt.strftime("%Y-%m-%d").isin(wanted_dates)].copy()
        if not filtered.empty:
            filtered["Code"] = filtered["Code"].astype(str)
            frames.append(filtered)

    if not frames:
        return pd.DataFrame(columns=MASTER_COLUMNS)

    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["Date", "Code"])
        .sort_values(["Date", "Code"])
        .reset_index(drop=True)
    )


def load_financial_summaries(
    client: jquantsapi.ClientV2,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    selected = _iter_selected_bulk_files(client, "/fins/summary", start, end, cache_dir)
    parquet_dir = cache_dir / "parquet" / "fins"
    frames: list[pd.DataFrame] = []

    for _, raw_path in selected:
        frame = _load_cached_bulk_frame(
            raw_path=raw_path,
            parquet_dir=parquet_dir,
            usecols=FIN_COLUMNS,
            date_columns=("DiscDate", "CurPerEn", "CurFYEn"),
        )
        filtered = frame[(frame["DiscDate"] >= start) & (frame["DiscDate"] <= end)].copy()
        if not filtered.empty:
            filtered["Code"] = filtered["Code"].astype(str)
            frames.append(filtered)

    if not frames:
        return pd.DataFrame(columns=FIN_COLUMNS)

    return pd.concat(frames, ignore_index=True).sort_values(["DiscDate", "DiscTime", "Code"]).reset_index(drop=True)
