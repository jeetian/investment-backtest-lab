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
class DynamicLeverageConfig:
    enabled: bool = True
    trend_window: int = 200
    volatility_window: int = 63
    high_volatility: float = 0.25
    drawdown_guard: float = -0.10
    crash_guard: float = -0.20
    risk_on_leverage: float = 1.30
    neutral_leverage: float = 1.10
    risk_off_leverage: float = 1.00
    safety_buffer_guard: float = 0.30

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DynamicLeverageConfig:
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", True)),
            trend_window=int(data.get("trend_window", 200)),
            volatility_window=int(data.get("volatility_window", 63)),
            high_volatility=float(data.get("high_volatility", 0.25)),
            drawdown_guard=float(data.get("drawdown_guard", -0.10)),
            crash_guard=float(data.get("crash_guard", -0.20)),
            risk_on_leverage=float(data.get("risk_on_leverage", 1.30)),
            neutral_leverage=float(data.get("neutral_leverage", 1.10)),
            risk_off_leverage=float(data.get("risk_off_leverage", 1.00)),
            safety_buffer_guard=float(data.get("safety_buffer_guard", 0.30)),
        )


@dataclass(frozen=True)
class LeveragedETFProductConfig:
    ticker: str
    leverage: float
    label: str

    @classmethod
    def from_dict(cls, ticker: str, data: dict[str, Any] | None) -> LeveragedETFProductConfig:
        data = data or {}
        return cls(
            ticker=str(data.get("ticker", ticker)).upper(),
            leverage=float(data.get("leverage", 1.0)),
            label=str(data.get("label", ticker.upper())),
        )


def _default_leveraged_etf_products() -> dict[str, LeveragedETFProductConfig]:
    return {
        "QQQ": LeveragedETFProductConfig("QQQ", 1.0, "QQQ 1x"),
        "QLD": LeveragedETFProductConfig("QLD", 2.0, "QLD 2x"),
        "TQQQ": LeveragedETFProductConfig("TQQQ", 3.0, "TQQQ 3x"),
    }


@dataclass(frozen=True)
class LeveragedETFLabConfig:
    family: str = "qqq"
    initial_cash: float = 10_000.0
    actual_start_date: str | None = "2011-01-01"
    synthetic_start_date: str | None = "1999-03-10"
    grid_step: float = 0.10
    top_n: int = 24
    high_risk_drawdown: float = -0.65
    synthetic_failure_drawdown: float = -0.85
    trend_windows: tuple[int, ...] = (100, 200)
    drawdown_guards: tuple[float, float] = (-0.10, -0.20)
    products: dict[str, LeveragedETFProductConfig] = field(
        default_factory=_default_leveraged_etf_products
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> LeveragedETFLabConfig:
        data = data or {}
        product_items = data.get("products") or {}
        products = (
            {
                str(ticker).upper(): LeveragedETFProductConfig.from_dict(str(ticker), product_data)
                for ticker, product_data in product_items.items()
            }
            if product_items
            else _default_leveraged_etf_products()
        )
        drawdown_guards = tuple(
            float(value) for value in data.get("drawdown_guards", [-0.10, -0.20])
        )
        if len(drawdown_guards) != 2:
            raise ValueError("leveraged_etf_lab.drawdown_guards must contain two values.")
        return cls(
            family=str(data.get("family", "qqq")).lower(),
            initial_cash=float(data.get("initial_cash", 10_000.0)),
            actual_start_date=(
                None
                if data.get("actual_start_date") is None
                else str(data.get("actual_start_date"))
            ),
            synthetic_start_date=(
                None
                if data.get("synthetic_start_date") is None
                else str(data.get("synthetic_start_date"))
            ),
            grid_step=float(data.get("grid_step", 0.10)),
            top_n=int(data.get("top_n", 24)),
            high_risk_drawdown=float(data.get("high_risk_drawdown", -0.65)),
            synthetic_failure_drawdown=float(data.get("synthetic_failure_drawdown", -0.85)),
            trend_windows=tuple(int(value) for value in data.get("trend_windows", [100, 200])),
            drawdown_guards=(drawdown_guards[0], drawdown_guards[1]),
            products=products,
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
    dynamic_leverage: DynamicLeverageConfig = field(default_factory=DynamicLeverageConfig)
    leveraged_etf_lab: LeveragedETFLabConfig = field(default_factory=LeveragedETFLabConfig)

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
            dynamic_leverage=DynamicLeverageConfig.from_dict(data.get("dynamic_leverage")),
            leveraged_etf_lab=LeveragedETFLabConfig.from_dict(data.get("leveraged_etf_lab")),
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
