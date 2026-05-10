from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from investment_backtest_lab.html_ui import render_html_head
from investment_backtest_lab.models import BacktestConfig

EXTERNAL_SIGNAL_FEATURE_COLUMNS = (
    "fear_greed_score",
    "vix_percentile_252",
    "usdtwd_return_63d_percentile_252",
    "tw_margin_balance_percentile_252",
    "tw_institutional_net_buy_21d_percentile_252",
)
CORE_EXTERNAL_SIGNAL_FEATURE_COLUMNS = (
    "vix_percentile_252",
    "usdtwd_return_63d_percentile_252",
    "tw_margin_balance_percentile_252",
    "tw_institutional_net_buy_21d_percentile_252",
)
SUPPORTED_EXTERNAL_SIGNAL_FEATURE_SETS = ("all", "core")
EXTERNAL_SIGNAL_FEATURE_METADATA = {
    "fear_greed_score": {"risk_tail": "both", "label": "CNN Fear & Greed"},
    "vix_percentile_252": {"risk_tail": "high", "label": "VIX percentile"},
    "usdtwd_return_63d_percentile_252": {
        "risk_tail": "high",
        "label": "USD/TWD 63D return percentile",
    },
    "tw_margin_balance_percentile_252": {
        "risk_tail": "high",
        "label": "TW margin balance percentile",
    },
    "tw_institutional_net_buy_21d_percentile_252": {
        "risk_tail": "low",
        "label": "TW institutional flow percentile",
    },
}
FEATURES_FILE_TEMPLATE = "market_regime_features_{family}.csv"
MANIFEST_FILE_TEMPLATE = "market_regime_manifest_{family}.json"


class ExternalSignalError(RuntimeError):
    pass


def external_signal_columns_for_feature_set(feature_set: str) -> tuple[str, ...]:
    normalized = str(feature_set or "all").lower().replace("-", "_")
    if normalized == "all":
        return EXTERNAL_SIGNAL_FEATURE_COLUMNS
    if normalized == "core":
        return CORE_EXTERNAL_SIGNAL_FEATURE_COLUMNS
    supported = ", ".join(SUPPORTED_EXTERNAL_SIGNAL_FEATURE_SETS)
    raise ValueError(f"Unsupported external signal feature set: {feature_set}. Use: {supported}.")


@dataclass(frozen=True)
class ExternalSignalFetchResult:
    family: str
    features_path: Path
    manifest_path: Path
    raw_paths: dict[str, Path]
    features: pd.DataFrame
    manifest: dict[str, Any]


def fetch_external_signals(
    *,
    config: BacktestConfig,
    family: str,
    external_dir: Path,
    end_date: str,
    require_fear_greed: bool = True,
) -> ExternalSignalFetchResult:
    family = family.lower()
    external_dir.mkdir(parents=True, exist_ok=True)
    start_date = config.start_date.isoformat()
    end_exclusive = pd.Timestamp(end_date).date().isoformat()
    raw_paths: dict[str, Path] = {}
    sources: list[dict[str, Any]] = []

    try:
        fear_greed = _fetch_fear_greed(start_date=start_date, end_date=end_exclusive)
    except ExternalSignalError:
        if require_fear_greed:
            raise
        fear_greed = pd.DataFrame(columns=["date", "score", "rating"])
    if fear_greed.empty and require_fear_greed:
        raise ExternalSignalError(
            "CNN Fear & Greed fetch failed or returned no rows. "
            "Create data/external/fear_greed.csv with columns date,score,rating, "
            "or rerun with --allow-missing-fear-greed for non-F&G smoke tests."
        )
    if not fear_greed.empty:
        path = external_dir / "fear_greed.csv"
        _write_csv(fear_greed, path)
        raw_paths["fear_greed"] = path
        sources.append(
            _source_record("fear_greed", path, fear_greed, "CNN internal API/fear-greed")
        )

    vix = _fetch_fred_vix(start_date=start_date, end_date=end_exclusive)
    path = external_dir / "fred_vixcls.csv"
    _write_csv(vix, path)
    raw_paths["vixcls"] = path
    sources.append(_source_record("vixcls", path, vix, "FRED:VIXCLS"))

    usdtwd = _fetch_yfinance_close(
        ticker="USDTWD=X",
        start_date=start_date,
        end_date=end_exclusive,
        value_column="usdtwd",
    )
    path = external_dir / "yfinance_usdtwd.csv"
    _write_csv(usdtwd, path)
    raw_paths["usdtwd"] = path
    sources.append(_source_record("usdtwd", path, usdtwd, "yfinance:USDTWD=X"))

    margin = _fetch_finmind_margin_total(start_date=start_date, end_date=end_exclusive)
    path = external_dir / "finmind_tw_margin_total.csv"
    _write_csv(margin, path)
    raw_paths["tw_margin_total"] = path
    sources.append(
        _source_record(
            "tw_margin_total",
            path,
            margin,
            "FinMind:TaiwanStockTotalMarginPurchaseShortSale",
        )
    )

    institutional = _fetch_finmind_institutional_total(
        start_date=start_date,
        end_date=end_exclusive,
    )
    path = external_dir / "finmind_tw_institutional_total.csv"
    _write_csv(institutional, path)
    raw_paths["tw_institutional_total"] = path
    sources.append(
        _source_record(
            "tw_institutional_total",
            path,
            institutional,
            "FinMind:TaiwanStockTotalInstitutionalInvestors",
        )
    )

    features = build_market_regime_features(
        fear_greed=fear_greed,
        vix=vix,
        usdtwd=usdtwd,
        margin=margin,
        institutional=institutional,
    )
    features_path = external_dir / FEATURES_FILE_TEMPLATE.format(family=family)
    _write_csv(features, features_path)
    raw_paths["features"] = features_path
    manifest = {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "family": family,
        "start_date": start_date,
        "end_date_exclusive": end_exclusive,
        "feature_file": str(features_path),
        "feature_columns": list(EXTERNAL_SIGNAL_FEATURE_COLUMNS),
        "sources": sources,
        "notes": [
            "External signal files are frozen local artifacts and are not committed to git.",
            "Strategy search aligns features to trading dates and shifts them to t-1.",
            "Taiwan VIX is intentionally excluded from V1 optimization due shorter coverage.",
        ],
    }
    manifest_path = external_dir / MANIFEST_FILE_TEMPLATE.format(family=family)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return ExternalSignalFetchResult(
        family=family,
        features_path=features_path,
        manifest_path=manifest_path,
        raw_paths=raw_paths,
        features=features,
        manifest=manifest,
    )


def build_market_regime_features(
    *,
    fear_greed: pd.DataFrame,
    vix: pd.DataFrame,
    usdtwd: pd.DataFrame,
    margin: pd.DataFrame,
    institutional: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    if not fear_greed.empty:
        fg = fear_greed[["date", "score"]].rename(columns={"score": "fear_greed_score"})
        frames.append(_normalize_date_column(fg))
    if not vix.empty:
        vx = vix[["date", "vixcls"]].copy()
        vx["vix_percentile_252"] = _rolling_percentile(vx["vixcls"])
        frames.append(_normalize_date_column(vx))
    if not usdtwd.empty:
        fx = usdtwd[["date", "usdtwd"]].copy()
        fx["usdtwd_return_63d"] = fx["usdtwd"].pct_change(63)
        fx["usdtwd_return_63d_percentile_252"] = _rolling_percentile(
            fx["usdtwd_return_63d"]
        )
        frames.append(_normalize_date_column(fx))
    if not margin.empty:
        mg = _margin_features(margin)
        frames.append(_normalize_date_column(mg))
    if not institutional.empty:
        inst = _institutional_features(institutional)
        frames.append(_normalize_date_column(inst))
    if not frames:
        raise ExternalSignalError("No external signal frames are available.")
    merged = frames[0].copy()
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="outer")
    merged = _normalize_date_column(merged)
    merged = merged.sort_values("date").drop_duplicates("date")
    for column in EXTERNAL_SIGNAL_FEATURE_COLUMNS:
        if column not in merged.columns:
            merged[column] = np.nan
    ordered = ["date", *EXTERNAL_SIGNAL_FEATURE_COLUMNS]
    extras = [column for column in merged.columns if column not in ordered]
    return merged[ordered + extras].reset_index(drop=True)


def load_external_signal_features(
    path: Path,
    *,
    trading_index: pd.Index,
    feature_columns: tuple[str, ...] | list[str] | None = None,
    require_complete: bool = False,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing external signal feature CSV: {path}. "
            "Run scripts\\fetch_external_signals.py or use --no-external-signals."
        )
    data = pd.read_csv(path)
    validate_external_signal_features(data)
    columns = (
        tuple(feature_columns)
        if feature_columns is not None
        else tuple(column for column in data.columns if column != "date")
    )
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"External signal features missing columns: {missing}")
    dates = pd.DatetimeIndex(pd.to_datetime(data["date"], errors="raise"))
    numeric = data[list(columns)].apply(pd.to_numeric, errors="coerce")
    raw = pd.DataFrame(numeric.to_numpy(), index=dates, columns=numeric.columns).sort_index()
    aligned = raw.reindex(pd.DatetimeIndex(trading_index).sort_values()).ffill().shift(1)
    if require_complete:
        aligned = trim_to_common_feature_coverage(aligned, columns)
        missing_after_trim = aligned[list(columns)].isna().any()
        if bool(missing_after_trim.any()):
            bad = ", ".join(missing_after_trim[missing_after_trim].index.astype(str))
            raise ValueError(f"External signal common coverage still has missing values: {bad}")
    return aligned[list(columns)]


def trim_to_common_feature_coverage(
    features: pd.DataFrame,
    feature_columns: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    bounds = feature_coverage_bounds(features, feature_columns)
    missing = [item["feature"] for item in bounds if item["usable_start"] == ""]
    if missing:
        raise ValueError(f"External signal features have no usable values: {missing}")
    common_start = max(pd.Timestamp(item["usable_start"]) for item in bounds)
    common_end = min(pd.Timestamp(item["usable_end"]) for item in bounds)
    if common_start > common_end:
        raise ValueError(
            "External signal features have no common usable coverage: "
            f"{common_start.date().isoformat()} > {common_end.date().isoformat()}"
        )
    trimmed = features.loc[
        (features.index >= common_start) & (features.index <= common_end),
        list(feature_columns),
    ].copy()
    if trimmed.empty:
        raise ValueError("External signal common coverage produced an empty frame.")
    return trimmed


def feature_coverage_bounds(
    features: pd.DataFrame,
    feature_columns: tuple[str, ...] | list[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for column in feature_columns:
        if column not in features.columns:
            rows.append({"feature": column, "usable_start": "", "usable_end": ""})
            continue
        values = pd.to_numeric(features[column], errors="coerce")
        valid = values[values.notna()]
        if valid.empty:
            rows.append({"feature": column, "usable_start": "", "usable_end": ""})
            continue
        rows.append(
            {
                "feature": column,
                "usable_start": pd.Timestamp(valid.index.min()).date().isoformat(),
                "usable_end": pd.Timestamp(valid.index.max()).date().isoformat(),
            }
        )
    return rows


def validate_external_signal_features(data: pd.DataFrame) -> None:
    if "date" not in data.columns:
        raise ValueError("External signal features CSV missing date column.")
    dates = pd.to_datetime(data["date"], errors="raise")
    if dates.duplicated().any():
        raise ValueError("External signal features contain duplicate dates.")
    if not dates.is_monotonic_increasing:
        raise ValueError("External signal features must be sorted by date ascending.")
    numeric_columns = [column for column in data.columns if column != "date"]
    if not numeric_columns:
        raise ValueError("External signal features require at least one numeric column.")
    for column in numeric_columns:
        pd.to_numeric(data[column], errors="raise")
    if dates.max() > pd.Timestamp.now().normalize() + pd.Timedelta(days=1):
        raise ValueError("External signal features contain future dates.")


def write_external_signal_audit(
    *,
    family: str,
    external_dir: Path,
    output_dir: Path,
    feature_set: str = "all",
) -> tuple[pd.DataFrame, Path, Path, Path]:
    family = family.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    checks = build_external_signal_audit_checks(
        family=family,
        external_dir=external_dir,
        feature_set=feature_set,
    )
    prefix = f"external_signal_audit_{family}"
    csv_path = output_dir / f"{prefix}.csv"
    md_path = output_dir / f"{prefix}.md"
    html_path = output_dir / f"{prefix}.html"
    checks.to_csv(csv_path, index=False)
    md_path.write_text(_render_audit_markdown(family, checks), encoding="utf-8")
    html_path.write_text(_render_audit_html(family, checks), encoding="utf-8")
    return checks, csv_path, md_path, html_path


def build_external_signal_audit_checks(
    *,
    family: str,
    external_dir: Path,
    feature_set: str = "all",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    feature_set = str(feature_set or "all").lower().replace("-", "_")
    feature_columns = external_signal_columns_for_feature_set(feature_set)
    expected = {
        "vixcls": ("fred_vixcls.csv", ("date", "vixcls")),
        "usdtwd": ("yfinance_usdtwd.csv", ("date", "usdtwd")),
        "tw_margin_total": ("finmind_tw_margin_total.csv", ("date",)),
        "tw_institutional_total": ("finmind_tw_institutional_total.csv", ("date",)),
        "features": (FEATURES_FILE_TEMPLATE.format(family=family), ("date",)),
        "manifest": (MANIFEST_FILE_TEMPLATE.format(family=family), ()),
    }
    if feature_set == "all":
        expected = {
            "fear_greed": ("fear_greed.csv", ("date", "score", "rating")),
            **expected,
        }
    else:
        rows.append(
            _audit_row(
                "scope",
                "fear_greed_excluded_from_core",
                "pass",
                "CNN Fear & Greed is excluded from the formal core external feature set.",
                "",
            )
        )
    for check_id, (filename, required_columns) in expected.items():
        path = external_dir / filename
        if not path.exists():
            rows.append(_audit_row("artifact", check_id, "fail", f"Missing {filename}", path))
            continue
        rows.append(_audit_row("artifact", check_id, "pass", f"Found {filename}", path))
        if path.suffix == ".csv":
            rows.append(_audit_schema_row(check_id, path, required_columns))
            rows.append(_audit_date_row(check_id, path))
            if check_id == "fear_greed":
                rows.append(_audit_fear_greed_coverage_row(path))
    features_path = external_dir / FEATURES_FILE_TEMPLATE.format(family=family)
    if features_path.exists():
        try:
            features = pd.read_csv(features_path)
            validate_external_signal_features(features)
            usable = [
                column
                for column in EXTERNAL_SIGNAL_FEATURE_COLUMNS
                if column in features.columns
                and pd.to_numeric(features[column], errors="coerce").notna().any()
            ]
            rows.append(
                _audit_row(
                    "features",
                    "usable_feature_columns",
                    "pass" if usable else "fail",
                    f"Usable feature columns: {', '.join(usable) if usable else 'none'}",
                    "",
                )
            )
            rows.extend(_feature_coverage_audit_rows(features, feature_columns))
        except Exception as exc:
            rows.append(_audit_row("features", "feature_validation", "fail", str(exc), ""))
    rows.append(
        _audit_row(
            "scope",
            "taiwan_vix_v1_scope",
            "warn",
            "Taiwan VIX is documented but excluded from V1 optimizer due shorter coverage.",
            "",
        )
    )
    return pd.DataFrame(rows, columns=["category", "check_id", "status", "summary", "details"])


def _feature_coverage_audit_rows(
    features: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    coverage: list[dict[str, str]] = []
    for column in feature_columns:
        if column not in features.columns:
            rows.append(
                _audit_row(
                    "coverage",
                    f"{column}_coverage",
                    "fail",
                    f"{column} is missing from market regime features.",
                    "",
                )
            )
            coverage.append({"feature": column, "usable_start": "", "usable_end": ""})
            continue
        values = pd.to_numeric(features[column], errors="coerce")
        valid = features.loc[values.notna(), ["date"]].copy()
        if valid.empty:
            rows.append(
                _audit_row(
                    "coverage",
                    f"{column}_coverage",
                    "fail",
                    f"{column} has no usable values.",
                    "",
                )
            )
            coverage.append({"feature": column, "usable_start": "", "usable_end": ""})
            continue
        dates = pd.to_datetime(valid["date"], errors="raise")
        usable_start = dates.iloc[1] if len(dates) > 1 else dates.iloc[0]
        usable_end = dates.max()
        coverage.append(
            {
                "feature": column,
                "usable_start": usable_start.date().isoformat(),
                "usable_end": usable_end.date().isoformat(),
            }
        )
        rows.append(
            _audit_row(
                "coverage",
                f"{column}_coverage",
                "pass",
                f"raw {dates.min().date().isoformat()} to {dates.max().date().isoformat()}; "
                f"usable t-1 {usable_start.date().isoformat()} to "
                f"{usable_end.date().isoformat()}",
                "",
            )
        )
    if coverage and all(item["usable_start"] and item["usable_end"] for item in coverage):
        common_start = max(pd.Timestamp(item["usable_start"]) for item in coverage)
        common_end = min(pd.Timestamp(item["usable_end"]) for item in coverage)
        status = "pass" if common_start <= common_end else "fail"
        coverage_check_id = (
            "core_common_coverage"
            if tuple(feature_columns) == CORE_EXTERNAL_SIGNAL_FEATURE_COLUMNS
            else "common_coverage"
        )
        rows.append(
            _audit_row(
                "coverage",
                coverage_check_id,
                status,
                f"Common usable coverage: {common_start.date().isoformat()} to "
                f"{common_end.date().isoformat()}",
                ", ".join(item["feature"] for item in coverage),
            )
        )
    return rows


def external_signal_audit_status(checks: pd.DataFrame) -> str:
    statuses = set(checks["status"].astype(str)) if not checks.empty else {"fail"}
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"


def _fetch_fear_greed(*, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        import fear_greed

        rows = fear_greed.get_history(start=start_date, end=end_date)
    except Exception as exc:
        raise ExternalSignalError(f"CNN Fear & Greed fetch failed: {exc}") from exc
    data = pd.DataFrame(rows)
    if data.empty:
        return pd.DataFrame(columns=["date", "score", "rating"])
    data = data.rename(columns={column: column.lower() for column in data.columns})
    missing = {"date", "score", "rating"} - set(data.columns)
    if missing:
        raise ExternalSignalError(f"Fear & Greed response missing columns: {sorted(missing)}")
    result = data[["date", "score", "rating"]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result["score"] = pd.to_numeric(result["score"], errors="raise").astype(float)
    result["rating"] = result["rating"].astype(str).str.lower()
    if ((result["score"] < 0.0) | (result["score"] > 100.0)).any():
        raise ExternalSignalError("Fear & Greed scores must be between 0 and 100.")
    return result.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def _fetch_fred_vix(*, start_date: str, end_date: str) -> pd.DataFrame:
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id=VIXCLS&cosd={start_date}&coed={_inclusive_end_date(end_date)}"
    )
    data = pd.read_csv(url)
    column = "VIXCLS"
    if column not in data.columns:
        raise ExternalSignalError("FRED VIXCLS response missing VIXCLS column.")
    result = data.rename(columns={"observation_date": "date", column: "vixcls"})
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result["vixcls"] = pd.to_numeric(result["vixcls"].replace(".", np.nan), errors="coerce")
    return result[["date", "vixcls"]].dropna().sort_values("date").reset_index(drop=True)


def _fetch_yfinance_close(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    value_column: str,
) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
    if raw.empty:
        raise ExternalSignalError(f"yfinance returned no rows for {ticker}.")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return pd.DataFrame(
        {
            "date": pd.to_datetime(close.index),
            value_column: pd.to_numeric(close.to_numpy(), errors="coerce"),
        }
    ).dropna().sort_values("date").reset_index(drop=True)


def _fetch_finmind_margin_total(*, start_date: str, end_date: str) -> pd.DataFrame:
    from FinMind.data import DataLoader

    loader = DataLoader()
    raw = loader.taiwan_stock_margin_purchase_short_sale_total(
        start_date=start_date,
        end_date=end_date,
    )
    if raw.empty:
        raise ExternalSignalError("FinMind returned no TW margin total rows.")
    raw["date"] = pd.to_datetime(raw["date"], errors="raise")
    return raw.sort_values("date").reset_index(drop=True)


def _fetch_finmind_institutional_total(*, start_date: str, end_date: str) -> pd.DataFrame:
    from FinMind.data import DataLoader

    loader = DataLoader()
    raw = loader.taiwan_stock_institutional_investors_total(
        start_date=start_date,
        end_date=end_date,
    )
    if raw.empty:
        raise ExternalSignalError("FinMind returned no TW institutional total rows.")
    raw["date"] = pd.to_datetime(raw["date"], errors="raise")
    return raw.sort_values("date").reset_index(drop=True)


def _margin_features(margin: pd.DataFrame) -> pd.DataFrame:
    data = margin.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    if "TodayBalance" in data.columns:
        value = pd.to_numeric(data["TodayBalance"], errors="coerce")
    elif "MarginPurchaseTodayBalance" in data.columns:
        value = pd.to_numeric(data["MarginPurchaseTodayBalance"], errors="coerce")
    else:
        numeric_columns = [column for column in data.columns if column != "date"]
        value = data[numeric_columns].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    data["_value"] = value
    if "name" in data.columns:
        mask = data["name"].astype(str).str.contains("融資|margin", case=False, regex=True)
        if mask.any():
            data = data[mask]
    grouped = data.groupby("date", as_index=False)["_value"].sum()
    grouped = grouped.rename(columns={"_value": "tw_margin_balance"})
    grouped["tw_margin_balance_percentile_252"] = _rolling_percentile(
        grouped["tw_margin_balance"]
    )
    return grouped


def _institutional_features(institutional: pd.DataFrame) -> pd.DataFrame:
    data = institutional.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    buy = pd.to_numeric(data.get("buy", 0.0), errors="coerce")
    sell = pd.to_numeric(data.get("sell", 0.0), errors="coerce")
    data["_net"] = buy - sell
    grouped = data.groupby("date", as_index=False)["_net"].sum()
    grouped = grouped.rename(columns={"_net": "tw_institutional_net_buy_sell"})
    grouped["tw_institutional_net_buy_21d"] = (
        grouped["tw_institutional_net_buy_sell"].rolling(21, min_periods=1).sum()
    )
    grouped["tw_institutional_net_buy_21d_percentile_252"] = _rolling_percentile(
        grouped["tw_institutional_net_buy_21d"]
    )
    return grouped


def _rolling_percentile(series: pd.Series, window: int = 252) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")

    def percentile(window_values: pd.Series) -> float:
        last = window_values.iloc[-1]
        if pd.isna(last):
            return np.nan
        valid = window_values.dropna()
        if valid.empty:
            return np.nan
        return float((valid <= last).mean() * 100.0)

    return values.rolling(window, min_periods=20).apply(percentile, raw=False)


def _normalize_date_column(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = (
        pd.to_datetime(result["date"], errors="raise", utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )
    return result


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    output = frame.copy()
    if "date" in output.columns:
        output["date"] = pd.to_datetime(output["date"]).dt.date.astype(str)
    output.to_csv(path, index=False)


def _source_record(name: str, path: Path, frame: pd.DataFrame, source: str) -> dict[str, Any]:
    return {
        "name": name,
        "source": source,
        "path": str(path),
        "rows": int(len(frame)),
        "start": _date_bound(frame, "min"),
        "end": _date_bound(frame, "max"),
    }


def _date_bound(frame: pd.DataFrame, op: str) -> str:
    if frame.empty or "date" not in frame.columns:
        return ""
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if dates.empty:
        return ""
    return (dates.min() if op == "min" else dates.max()).date().isoformat()


def _inclusive_end_date(end_date: str) -> str:
    return (pd.Timestamp(end_date) - pd.Timedelta(days=1)).date().isoformat()


def _audit_schema_row(
    check_id: str,
    path: Path,
    required_columns: tuple[str, ...],
) -> dict[str, Any]:
    try:
        columns = set(pd.read_csv(path, nrows=0).columns)
    except Exception as exc:
        return _audit_row("schema", f"{check_id}_schema", "fail", f"Cannot read CSV: {exc}", path)
    missing = sorted(set(required_columns) - columns)
    return _audit_row(
        "schema",
        f"{check_id}_schema",
        "fail" if missing else "pass",
        f"Missing columns: {missing}" if missing else "Schema contains required columns",
        path,
    )


def _audit_date_row(check_id: str, path: Path) -> dict[str, Any]:
    try:
        frame = pd.read_csv(path)
        if "date" not in frame.columns:
            return _audit_row("date", f"{check_id}_dates", "warn", "No date column", path)
        dates = pd.to_datetime(frame["date"], errors="raise")
        if dates.duplicated().any():
            if "name" in frame.columns:
                unique_names = frame["name"].astype(str).nunique()
                return _audit_row(
                    "date",
                    f"{check_id}_dates",
                    "pass",
                    f"{dates.min().date().isoformat()} to {dates.max().date().isoformat()} "
                    f"with {unique_names} raw categories per date",
                    path,
                )
            return _audit_row("date", f"{check_id}_dates", "fail", "Duplicate dates", path)
        if not dates.is_monotonic_increasing:
            return _audit_row("date", f"{check_id}_dates", "fail", "Dates not sorted", path)
        return _audit_row(
            "date",
            f"{check_id}_dates",
            "pass",
            f"{dates.min().date().isoformat()} to {dates.max().date().isoformat()}",
            path,
        )
    except Exception as exc:
        return _audit_row("date", f"{check_id}_dates", "fail", str(exc), path)


def _audit_fear_greed_coverage_row(path: Path) -> dict[str, Any]:
    try:
        frame = pd.read_csv(path)
        dates = pd.to_datetime(frame["date"], errors="raise")
        rows = len(frame)
        if rows < 1_000:
            return _audit_row(
                "coverage",
                "fear_greed_history_depth",
                "warn",
                f"Fear & Greed coverage is limited to {rows:,} rows from "
                f"{dates.min().date().isoformat()} to {dates.max().date().isoformat()}",
                "The current fear-greed package exposes about one year of history.",
            )
        return _audit_row(
            "coverage",
            "fear_greed_history_depth",
            "pass",
            f"Fear & Greed coverage has {rows:,} rows",
            path,
        )
    except Exception as exc:
        return _audit_row("coverage", "fear_greed_history_depth", "fail", str(exc), path)


def _audit_row(
    category: str,
    check_id: str,
    status: str,
    summary: str,
    details: Any,
) -> dict[str, Any]:
    return {
        "category": category,
        "check_id": check_id,
        "status": status,
        "summary": summary,
        "details": "" if details is None else str(details),
    }


def _render_audit_markdown(family: str, checks: pd.DataFrame) -> str:
    status = external_signal_audit_status(checks)
    lines = [
        f"# External Signal Audit: {family}",
        "",
        f"- Overall status: `{status}`",
        "",
        "| Category | Check | Status | Summary |",
        "|---|---|---:|---|",
    ]
    for row in checks.itertuples(index=False):
        lines.append(f"| {row.category} | `{row.check_id}` | `{row.status}` | {row.summary} |")
    lines.append("")
    return "\n".join(lines)


def _render_audit_html(family: str, checks: pd.DataFrame) -> str:
    status = external_signal_audit_status(checks)
    rows = "\n".join(
        "<tr>"
        f"<td>{escape(str(row.category))}</td>"
        f"<td>{escape(str(row.check_id))}</td>"
        f"<td>{escape(str(row.status))}</td>"
        f"<td>{escape(str(row.summary))}</td>"
        f"<td>{escape(str(row.details))}</td>"
        "</tr>"
        for row in checks.itertuples(index=False)
    )
    return f"""<!doctype html>
<html lang="zh-Hant">
{render_html_head(title=f"External Signal Audit {family}")}
<body>
<main>
  <section class="hero">
    <div class="panel">
      <p class="eyebrow">External Signal Audit</p>
      <h1>{escape(family.upper())} 外部 regime 指標</h1>
      <p class="lede">
        檢查 Fear & Greed、VIX、USD/TWD、台股融資與法人買賣超 frozen data 是否可用。
        Optuna 只讀 frozen CSV，並在交易日對齊後使用 t-1。
      </p>
      <span class="badge {'ok' if status == 'pass' else 'danger'}">overall {escape(status)}</span>
    </div>
  </section>
  <section class="panel">
    <h2>Checks</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Category</th><th>Check</th><th>Status</th><th>Summary</th><th>Details</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
  </section>
</main>
</body>
</html>
"""


__all__ = [
    "EXTERNAL_SIGNAL_FEATURE_COLUMNS",
    "EXTERNAL_SIGNAL_FEATURE_METADATA",
    "CORE_EXTERNAL_SIGNAL_FEATURE_COLUMNS",
    "SUPPORTED_EXTERNAL_SIGNAL_FEATURE_SETS",
    "ExternalSignalError",
    "ExternalSignalFetchResult",
    "build_external_signal_audit_checks",
    "build_market_regime_features",
    "external_signal_audit_status",
    "external_signal_columns_for_feature_set",
    "feature_coverage_bounds",
    "fetch_external_signals",
    "load_external_signal_features",
    "trim_to_common_feature_coverage",
    "validate_external_signal_features",
    "write_external_signal_audit",
]
