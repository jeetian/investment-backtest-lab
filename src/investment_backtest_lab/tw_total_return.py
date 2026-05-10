from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from investment_backtest_lab.leveraged_etf_lab import ProductSpec

TW50_FAMILY = "tw50"
TW50_BASE_TICKER = "0050"
TW50_LEVERAGED_TICKER = "00631L"
TW50_PRICE_PROXY_SOURCE = "yfinance:^TWII"
TW50_TOTAL_RETURN_SOURCE = "FinMind:TaiwanStockTotalReturnIndex:TAIEX"
TW50_SPLIT_DATE = "2025-06-18"
TW50_SPLIT_RATIO = 4.0


@dataclass(frozen=True)
class SplitEvent:
    date: str
    ratio: float


@dataclass(frozen=True)
class TW50TotalReturnInputs:
    actual_prices: pd.DataFrame
    hybrid_prices: pd.DataFrame
    source_coverage: pd.DataFrame
    dividend_audit: pd.DataFrame


def build_tw50_total_return_inputs(
    *,
    start_date: str,
    end_date: str,
    products: list[ProductSpec],
) -> TW50TotalReturnInputs:
    """Build auditable TW50 total-return prices for optimizer and replay workflows."""
    import yfinance as yf
    from FinMind.data import DataLoader

    loader = DataLoader()
    base_product = _product(products, TW50_BASE_TICKER)
    leveraged_product = _product(products, TW50_LEVERAGED_TICKER)

    twii_price = _download_yfinance_close(
        yf,
        "^TWII",
        start_date=start_date,
        end_date=end_date,
    )
    taiex_tr = _download_taiex_total_return(
        loader,
        start_date=max(str(start_date), "2003-01-01"),
        end_date=end_date,
    )
    base_raw = _download_finmind_price(
        loader,
        TW50_BASE_TICKER,
        start_date=start_date,
        end_date=end_date,
    )
    leveraged_raw = _download_finmind_price(
        loader,
        TW50_LEVERAGED_TICKER,
        start_date=start_date,
        end_date=end_date,
    )
    base_dividends = _download_finmind_dividends(
        loader,
        TW50_BASE_TICKER,
        start_date=start_date,
        end_date=end_date,
    )
    leveraged_dividends = _download_finmind_dividends(
        loader,
        TW50_LEVERAGED_TICKER,
        start_date=start_date,
        end_date=end_date,
    )

    splits = [SplitEvent(TW50_SPLIT_DATE, TW50_SPLIT_RATIO)]
    base_actual, base_audit = build_total_return_price_from_raw(
        raw_prices=base_raw,
        dividends=base_dividends,
        ticker=TW50_BASE_TICKER,
        splits=splits,
        source="FinMind:TaiwanStockPrice+TaiwanStockDividend:0050",
    )
    leveraged_actual, leveraged_audit = build_total_return_price_from_raw(
        raw_prices=leveraged_raw,
        dividends=leveraged_dividends,
        ticker=TW50_LEVERAGED_TICKER,
        splits=[],
        source="FinMind:TaiwanStockPrice+TaiwanStockDividend:00631L",
    )

    base_proxy = splice_price_series(
        earlier=twii_price,
        later=taiex_tr,
        later_source_start=pd.Timestamp("2003-01-02"),
    )
    base_hybrid = splice_price_series(
        earlier=base_proxy,
        later=base_actual,
        later_source_start=base_actual.index.min(),
    ).rename(base_product.ticker)

    leveraged_synthetic = synthetic_leveraged_from_base(
        base_hybrid,
        leverage=leveraged_product.leverage,
        name=leveraged_product.ticker,
    )
    leveraged_hybrid = splice_price_series(
        earlier=leveraged_synthetic,
        later=leveraged_actual,
        later_source_start=leveraged_actual.index.min(),
    ).rename(leveraged_product.ticker)

    actual_prices = pd.concat(
        [
            base_actual.rename(base_product.ticker),
            leveraged_actual.rename(leveraged_product.ticker),
        ],
        axis=1,
    ).dropna(how="any")
    hybrid_prices = pd.concat([base_hybrid, leveraged_hybrid], axis=1).dropna(how="any")
    source_coverage = build_tw50_source_coverage(
        twii_price=twii_price,
        taiex_tr=taiex_tr,
        base_actual=base_actual,
        leveraged_actual=leveraged_actual,
        hybrid_prices=hybrid_prices,
    )
    audit_frames = [frame for frame in [base_audit, leveraged_audit] if not frame.empty]
    dividend_audit = (
        pd.concat(audit_frames, ignore_index=True)
        if audit_frames
        else pd.DataFrame(columns=base_audit.columns)
    )
    return TW50TotalReturnInputs(
        actual_prices=actual_prices,
        hybrid_prices=hybrid_prices,
        source_coverage=source_coverage,
        dividend_audit=dividend_audit,
    )


def build_total_return_price_from_raw(
    *,
    raw_prices: pd.DataFrame,
    dividends: pd.DataFrame,
    ticker: str,
    splits: list[SplitEvent],
    source: str,
) -> tuple[pd.Series, pd.DataFrame]:
    prices = _close_series(raw_prices, ticker=ticker)
    split_adjusted_close = apply_split_adjustment(prices, splits=splits)
    dividend_audit = build_dividend_audit(
        dividends=dividends,
        ticker=ticker,
        splits=splits,
        price_index=split_adjusted_close.index,
    )
    dividend_by_ex_date = (
        dividend_audit.groupby("ex_dividend_date")["split_adjusted_cash_dividend"].sum()
        if not dividend_audit.empty
        else pd.Series(dtype="float64")
    )
    dividend_series = pd.Series(0.0, index=split_adjusted_close.index, dtype="float64")
    for date, value in dividend_by_ex_date.items():
        timestamp = pd.Timestamp(date)
        if timestamp in dividend_series.index:
            dividend_series.loc[timestamp] += float(value)

    prior_close = split_adjusted_close.shift(1)
    total_return = (split_adjusted_close + dividend_series) / prior_close - 1.0
    total_return.iloc[0] = 0.0
    total_return = total_return.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    total_return_price = split_adjusted_close.iloc[0] * (1.0 + total_return).cumprod()
    total_return_price.name = ticker
    dividend_audit["price_source"] = source
    return total_return_price, dividend_audit


def apply_split_adjustment(series: pd.Series, *, splits: list[SplitEvent]) -> pd.Series:
    adjusted = series.dropna().astype(float).sort_index().copy()
    for split in splits:
        split_date = pd.Timestamp(split.date)
        adjusted.loc[adjusted.index < split_date] = adjusted.loc[
            adjusted.index < split_date
        ] / float(split.ratio)
    return adjusted


def build_dividend_audit(
    *,
    dividends: pd.DataFrame,
    ticker: str,
    splits: list[SplitEvent],
    price_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    columns = [
        "ticker",
        "announcement_date",
        "ex_dividend_date",
        "payment_date",
        "raw_cash_dividend",
        "split_factor",
        "split_adjusted_cash_dividend",
    ]
    if dividends.empty:
        return pd.DataFrame(columns=columns)
    data = dividends.copy()
    data.columns = [str(column).strip() for column in data.columns]
    valid_price_dates = set(price_index)
    rows: list[dict[str, Any]] = []
    for row in data.to_dict("records"):
        ex_date = _first_valid_date(
            row.get("CashExDividendTradingDate"),
            row.get("cash_ex_dividend_trading_date"),
        )
        amount = _safe_float(
            row.get("CashEarningsDistribution", row.get("cash_earnings_distribution"))
        )
        if ex_date is None or amount <= 0.0:
            continue
        if ex_date not in valid_price_dates:
            continue
        factor = _split_factor_for_date(ex_date, splits)
        rows.append(
            {
                "ticker": ticker,
                "announcement_date": _date_or_blank(
                    row.get("AnnouncementDate", row.get("date", ""))
                ),
                "ex_dividend_date": ex_date.date().isoformat(),
                "payment_date": _date_or_blank(
                    row.get("CashDividendPaymentDate", row.get("payment_date", ""))
                ),
                "raw_cash_dividend": amount,
                "split_factor": factor,
                "split_adjusted_cash_dividend": amount / factor,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def splice_price_series(
    *,
    earlier: pd.Series,
    later: pd.Series,
    later_source_start: pd.Timestamp,
) -> pd.Series:
    earlier = earlier.dropna().astype(float).sort_index()
    later = later.dropna().astype(float).sort_index()
    if earlier.empty:
        return later
    if later.empty:
        return earlier
    start = pd.Timestamp(later_source_start)
    later_window = later[later.index >= start]
    if later_window.empty:
        raise ValueError(f"Cannot splice price series: no later rows at or after {start.date()}.")
    later_first = later_window.iloc[0]
    first_date = later_window.index[0]
    earlier_part = earlier[earlier.index < first_date]
    if earlier_part.empty:
        return later[later.index >= first_date]
    scale_base = earlier_part.iloc[-1]
    scaled_earlier = earlier_part * (float(later_first) / float(scale_base))
    return pd.concat([scaled_earlier, later[later.index >= first_date]]).sort_index()


def synthetic_leveraged_from_base(
    base: pd.Series,
    *,
    leverage: float,
    name: str,
    initial_price: float = 100.0,
) -> pd.Series:
    returns = base.dropna().astype(float).sort_index().pct_change().fillna(0.0)
    synthetic_returns = (returns * float(leverage)).clip(lower=-0.99)
    result = initial_price * (1.0 + synthetic_returns).cumprod()
    result.name = name
    return result


def build_tw50_source_coverage(
    *,
    twii_price: pd.Series,
    taiex_tr: pd.Series,
    base_actual: pd.Series,
    leveraged_actual: pd.Series,
    hybrid_prices: pd.DataFrame,
) -> pd.DataFrame:
    hybrid_start = _date_or_blank(hybrid_prices.index.min())
    hybrid_end = _date_or_blank(hybrid_prices.index.max())
    return pd.DataFrame(
        [
            {
                "ticker": TW50_BASE_TICKER,
                "replay_start": hybrid_start,
                "replay_end": hybrid_end,
                "actual_start": _date_or_blank(base_actual.index.min()),
                "actual_end": _date_or_blank(base_actual.index.max()),
                "synthetic_backfill_start": "",
                "synthetic_backfill_end": "",
                "proxy_start": _date_or_blank(twii_price.index.min()),
                "proxy_end": "2002-12-31",
                "total_return_proxy_start": _date_or_blank(taiex_tr.index.min()),
                "total_return_proxy_end": _date_or_blank(
                    taiex_tr[taiex_tr.index < base_actual.index.min()].index.max()
                ),
                "splice_date": _date_or_blank(base_actual.index.min()),
                "source_notes": (
                    "price_proxy_not_total_return: 1999-2002 uses ^TWII price proxy; "
                    "2003 onward uses TAIEX total return until 0050 actual."
                ),
            },
            {
                "ticker": TW50_LEVERAGED_TICKER,
                "replay_start": hybrid_start,
                "replay_end": hybrid_end,
                "actual_start": _date_or_blank(leveraged_actual.index.min()),
                "actual_end": _date_or_blank(leveraged_actual.index.max()),
                "synthetic_backfill_start": hybrid_start,
                "synthetic_backfill_end": _date_or_blank(
                    hybrid_prices[hybrid_prices.index < leveraged_actual.index.min()].index.max()
                ),
                "proxy_start": "",
                "proxy_end": "",
                "total_return_proxy_start": "",
                "total_return_proxy_end": "",
                "splice_date": _date_or_blank(leveraged_actual.index.min()),
                "source_notes": (
                    "Pre-listing 00631L uses daily-reset 2x synthetic from 0050 "
                    "hybrid total return."
                ),
            },
        ]
    )


def write_tw50_audit_files(
    *,
    output_dir: Any,
    result: TW50TotalReturnInputs,
) -> None:
    from pathlib import Path

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    result.source_coverage.to_csv(path / "tw50_total_return_sources.csv", index=False)
    result.dividend_audit.to_csv(path / "tw50_dividend_audit.csv", index=False)


def _download_yfinance_close(
    yf: Any,
    ticker: str,
    *,
    start_date: str,
    end_date: str,
) -> pd.Series:
    raw = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise ValueError(f"yfinance returned no rows for {ticker}.")
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw[("Close", ticker)]
    else:
        close = raw["Close"]
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    close = close.astype(float).rename(ticker).sort_index()
    return close[close.index < pd.Timestamp(end_date)]


def _download_taiex_total_return(
    loader: Any,
    *,
    start_date: str,
    end_date: str,
) -> pd.Series:
    raw = loader.taiwan_stock_total_return_index(
        index_id="TAIEX",
        start_date=start_date,
        end_date=end_date,
    )
    if raw.empty:
        raise ValueError("FinMind returned no TAIEX total return rows.")
    data = raw.copy()
    data["date"] = pd.to_datetime(data["date"])
    series = pd.Series(
        pd.to_numeric(data["price"], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(data["date"]).tz_localize(None),
        name="TAIEX_TR",
    )
    series = series.dropna().sort_index()
    return series[series.index < pd.Timestamp(end_date)]


def _download_finmind_price(
    loader: Any,
    ticker: str,
    *,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    raw = loader.taiwan_stock_daily(
        stock_id=ticker,
        start_date=start_date,
        end_date=end_date,
    )
    if raw.empty:
        raise ValueError(f"FinMind returned no TaiwanStockPrice rows for {ticker}.")
    data = raw.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data[data["date"] < pd.Timestamp(end_date)]
    if data.empty:
        raise ValueError(
            f"FinMind returned no TaiwanStockPrice rows before {end_date} for {ticker}."
        )
    return data.sort_values("date").reset_index(drop=True)


def _download_finmind_dividends(
    loader: Any,
    ticker: str,
    *,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    raw = loader.taiwan_stock_dividend(
        stock_id=ticker,
        start_date=start_date,
        end_date=end_date,
    )
    return raw.copy() if raw is not None else pd.DataFrame()


def _close_series(frame: pd.DataFrame, *, ticker: str) -> pd.Series:
    if frame.empty:
        raise ValueError(f"Cannot build total-return series for {ticker}: empty prices.")
    data = frame.copy()
    data.columns = [str(column).strip().lower() for column in data.columns]
    data["date"] = pd.to_datetime(data["date"])
    close = pd.Series(
        pd.to_numeric(data["close"], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(data["date"]).tz_localize(None),
        name=ticker,
    )
    return close.dropna().sort_index()


def _product(products: list[ProductSpec], ticker: str) -> ProductSpec:
    for product in products:
        if product.ticker.upper() == ticker.upper():
            return product
    raise ValueError(f"TW50 family requires product {ticker}.")


def _split_factor_for_date(date: pd.Timestamp, splits: list[SplitEvent]) -> float:
    factor = 1.0
    for split in splits:
        if date < pd.Timestamp(split.date):
            factor *= float(split.ratio)
    return factor


def _first_valid_date(*values: Any) -> pd.Timestamp | None:
    for value in values:
        text = str(value).strip().lower()
        if value is None or pd.isna(value) or text in {"", "0", "nat", "nan", "none"}:
            continue
        return pd.Timestamp(value)
    return None


def _date_or_blank(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if text in {"", "nat", "nan", "none"}:
        return ""
    return pd.Timestamp(value).date().isoformat()


def _safe_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


__all__ = [
    "TW50_FAMILY",
    "TW50TotalReturnInputs",
    "SplitEvent",
    "apply_split_adjustment",
    "build_dividend_audit",
    "build_total_return_price_from_raw",
    "build_tw50_source_coverage",
    "build_tw50_total_return_inputs",
    "splice_price_series",
    "synthetic_leveraged_from_base",
    "write_tw50_audit_files",
]
