from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any

import pandas as pd


class Market(StrEnum):
    TW = "TW"
    US = "US"
    FX = "FX"


class AssetType(StrEnum):
    STOCK = "stock"
    ETF = "etf"
    MUTUAL_FUND = "mutual_fund"
    FX = "fx"


class DataSource(StrEnum):
    YFINANCE = "yfinance"
    FINMIND = "finmind"
    CSV = "csv"
    PARQUET = "parquet"


class DividendMode(StrEnum):
    CASH = "cash"
    REINVEST = "reinvest"


class LeverageKind(StrEnum):
    MARGIN_LOAN = "margin_loan"
    LEVERAGED_ETF_PRODUCT = "leveraged_etf_product"


@dataclass(frozen=True)
class AssetSpec:
    ticker: str
    market: Market
    asset_type: AssetType
    currency: str
    data_source: DataSource
    name: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssetSpec:
        return cls(
            ticker=str(data["ticker"]),
            name=data.get("name"),
            market=Market(str(data["market"]).upper()),
            asset_type=AssetType(str(data["asset_type"]).lower()),
            currency=str(data["currency"]).upper(),
            data_source=DataSource(str(data["data_source"]).lower()),
        )

    @property
    def cache_key(self) -> str:
        safe_ticker = self.ticker.replace("/", "_").replace("=", "_").replace(".", "_")
        return f"{self.market.value}_{safe_ticker}_{self.currency}_{self.data_source.value}"


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StrategyConfig:
        return cls(name=str(data["name"]), params=dict(data.get("params", {})))


@dataclass(frozen=True)
class RebalanceConfig:
    frequency: str = "monthly"
    target_weights: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RebalanceConfig:
        data = data or {}
        return cls(
            frequency=str(data.get("frequency", "monthly")).lower(),
            target_weights={str(k): float(v) for k, v in data.get("target_weights", {}).items()},
        )


@dataclass(frozen=True)
class DCAConfig:
    contribution: float = 1_000.0
    frequency: str = "MS"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DCAConfig:
        data = data or {}
        return cls(
            contribution=float(data.get("contribution", 1_000.0)),
            frequency=str(data.get("frequency", "MS")),
        )


@dataclass(frozen=True)
class USTaxConfig:
    dividend_withholding_rate: float = 0.30

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> USTaxConfig:
        data = data or {}
        return cls(
            dividend_withholding_rate=float(data.get("dividend_withholding_rate", 0.30)),
        )


@dataclass(frozen=True)
class TaxConfig:
    us: USTaxConfig = field(default_factory=USTaxConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TaxConfig:
        data = data or {}
        return cls(us=USTaxConfig.from_dict(data.get("us")))


@dataclass(frozen=True)
class DividendConfig:
    mode: DividendMode = DividendMode.CASH

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DividendConfig:
        data = data or {}
        return cls(mode=DividendMode(str(data.get("mode", DividendMode.CASH.value)).lower()))


@dataclass(frozen=True)
class LedgerConfig:
    base_currency: str = "TWD"
    account_currency: str = "USD"
    initial_cash: float = 10_000.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> LedgerConfig:
        data = data or {}
        return cls(
            base_currency=str(data.get("base_currency", "TWD")).upper(),
            account_currency=str(data.get("account_currency", "USD")).upper(),
            initial_cash=float(data.get("initial_cash", 10_000.0)),
        )


@dataclass(frozen=True)
class LeverageConfig:
    enabled: bool = False
    kind: LeverageKind = LeverageKind.MARGIN_LOAN
    target_leverage: float = 1.30
    max_leverage: float = 1.30
    annual_borrow_rate: float = 0.065
    maintenance_requirement: float = 0.35
    min_safety_buffer: float = 0.25
    deleverage_to: float = 1.10

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> LeverageConfig:
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", False)),
            kind=LeverageKind(str(data.get("kind", LeverageKind.MARGIN_LOAN.value)).lower()),
            target_leverage=float(data.get("target_leverage", 1.30)),
            max_leverage=float(data.get("max_leverage", 1.30)),
            annual_borrow_rate=float(data.get("annual_borrow_rate", 0.065)),
            maintenance_requirement=float(data.get("maintenance_requirement", 0.35)),
            min_safety_buffer=float(data.get("min_safety_buffer", 0.25)),
            deleverage_to=float(data.get("deleverage_to", 1.10)),
        )


@dataclass(frozen=True)
class BacktestConfig:
    universe: list[AssetSpec]
    start_date: date
    end_date: date
    base_currency: str
    strategy: StrategyConfig
    benchmark: str | None = None
    rebalance: RebalanceConfig = field(default_factory=RebalanceConfig)
    dca: DCAConfig = field(default_factory=DCAConfig)
    cost_model: dict[str, Any] = field(default_factory=dict)
    tax: TaxConfig = field(default_factory=TaxConfig)
    dividend: DividendConfig = field(default_factory=DividendConfig)
    ledger: LedgerConfig = field(default_factory=LedgerConfig)
    leverage: LeverageConfig = field(default_factory=LeverageConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BacktestConfig:
        return cls(
            universe=[AssetSpec.from_dict(item) for item in data["universe"]],
            start_date=date.fromisoformat(str(data["start_date"])),
            end_date=date.fromisoformat(str(data["end_date"])),
            base_currency=str(data.get("base_currency", "TWD")).upper(),
            strategy=StrategyConfig.from_dict(data["strategy"]),
            benchmark=data.get("benchmark"),
            rebalance=RebalanceConfig.from_dict(data.get("rebalance")),
            dca=DCAConfig.from_dict(data.get("dca")),
            cost_model=dict(data.get("cost_model", {})),
            tax=TaxConfig.from_dict(data.get("tax")),
            dividend=DividendConfig.from_dict(data.get("dividend")),
            ledger=LedgerConfig.from_dict(data.get("ledger")),
            leverage=LeverageConfig.from_dict(data.get("leverage")),
        )


@dataclass(frozen=True)
class PriceFrame:
    asset: AssetSpec
    data: pd.DataFrame
    adjusted: bool
    source: str

    def close(self) -> pd.Series:
        return self.data["close"].rename(self.asset.ticker)


@dataclass(frozen=True)
class DividendFrame:
    asset: AssetSpec
    data: pd.DataFrame
    currency: str
    source: str

    def dividend_per_share(self) -> pd.Series:
        return self.data["dividend_per_share"].rename(self.asset.ticker)


def normalize_ohlcv(
    frame: pd.DataFrame,
    *,
    date_column: str = "date",
    column_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return a date-indexed OHLCV frame with lower-case canonical columns."""
    if frame.empty:
        raise ValueError("Cannot normalize an empty price frame.")

    data = frame.copy()
    if column_map:
        data = data.rename(columns=column_map)
    data.columns = [str(col).strip().lower().replace(" ", "_") for col in data.columns]
    normalized_date_column = str(date_column).strip().lower().replace(" ", "_")

    if normalized_date_column in data.columns:
        data[normalized_date_column] = pd.to_datetime(data[normalized_date_column])
        data = data.set_index(normalized_date_column)
    elif not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Price data must include a date column or DatetimeIndex.")

    if "adj_close" in data.columns and "close" not in data.columns:
        data["close"] = data["adj_close"]

    required = ["open", "high", "low", "close"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required price columns: {missing}")

    if "volume" not in data.columns:
        data["volume"] = 0

    data = data[["open", "high", "low", "close", "volume"]].sort_index()
    data.index = data.index.tz_localize(None)
    return data.apply(pd.to_numeric, errors="coerce").dropna(subset=["close"])


def normalize_dividends(
    frame: pd.Series | pd.DataFrame,
    *,
    date_column: str = "date",
    value_column: str = "dividend_per_share",
) -> pd.DataFrame:
    """Return a date-indexed dividend frame with one canonical value column."""
    if isinstance(frame, pd.Series):
        data = frame.rename(value_column).to_frame()
    else:
        data = frame.copy()
        data.columns = [str(col).strip().lower().replace(" ", "_") for col in data.columns]
        normalized_value_column = str(value_column).strip().lower().replace(" ", "_")
        if normalized_value_column not in data.columns:
            candidates = ["dividends", "dividend", "cash_dividend", "value", "amount"]
            match = next((column for column in candidates if column in data.columns), None)
            if match is not None:
                data = data.rename(columns={match: normalized_value_column})

    if data.empty:
        return pd.DataFrame(
            {"dividend_per_share": pd.Series(dtype="float64")},
            index=pd.DatetimeIndex([], name="date"),
        )

    normalized_date_column = str(date_column).strip().lower().replace(" ", "_")
    if normalized_date_column in data.columns:
        data[normalized_date_column] = pd.to_datetime(data[normalized_date_column])
        data = data.set_index(normalized_date_column)
    elif not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Dividend data must include a date column or DatetimeIndex.")

    if value_column not in data.columns:
        raise ValueError(f"Missing dividend value column: {value_column}")

    data.index = data.index.tz_localize(None)
    data = data[[value_column]].sort_index()
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
    data = data.dropna(subset=[value_column])
    data = data[data[value_column] > 0]
    return data.rename_axis("date")
