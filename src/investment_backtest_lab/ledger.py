from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any

import pandas as pd

from investment_backtest_lab.costs import CostModel, TradeCostBreakdown, TradeSide
from investment_backtest_lab.models import AssetSpec, Market

DateLike = str | date | datetime | pd.Timestamp


@dataclass(frozen=True)
class TradeEvent:
    date: pd.Timestamp
    asset: str
    side: TradeSide
    quantity: float
    price: float
    gross_amount: float
    fees: float
    taxes: float
    net_cash_flow: float
    currency: str
    note: str = ""


@dataclass(frozen=True)
class DividendEvent:
    date: pd.Timestamp
    asset: str
    shares: float
    dividend_per_share: float
    gross_amount: float
    withholding_tax: float
    net_amount: float
    cash_amount: float
    currency: str
    reinvested_quantity: float = 0.0
    reinvest_price: float | None = None
    note: str = ""


@dataclass(frozen=True)
class FeeEvent:
    date: pd.Timestamp
    asset: str
    amount: float
    currency: str
    description: str


@dataclass(frozen=True)
class TaxEvent:
    date: pd.Timestamp
    asset: str
    amount: float
    currency: str
    description: str


@dataclass(frozen=True)
class InterestEvent:
    date: pd.Timestamp
    asset: str
    amount: float
    currency: str
    description: str


@dataclass(frozen=True)
class PositionSnapshot:
    date: pd.Timestamp
    cash: float
    quantity: float
    price: float
    market_value: float
    total_equity: float
    exposure: float
    currency: str


@dataclass
class AccountLedger:
    """Auditable cash-flow ledger for one USD asset.

    Phase 1 deliberately supports a single long-only US asset. More markets,
    leverage, settlement timing, and multi-asset accounting should be added
    after the golden cases stay stable.
    """

    asset: AssetSpec
    starting_cash: float
    cost_model: CostModel = field(default_factory=CostModel)
    dividend_withholding_rate: float = 0.30
    account_currency: str = "USD"
    base_currency: str = "TWD"

    def __post_init__(self) -> None:
        if self.asset.market != Market.US or self.asset.currency.upper() != "USD":
            raise ValueError("Phase 1 AccountLedger supports only USD-denominated US assets.")
        self.account_currency = self.account_currency.upper()
        self.base_currency = self.base_currency.upper()
        if self.account_currency != "USD":
            raise ValueError("Phase 1 AccountLedger account currency must be USD.")
        self.cash = float(self.starting_cash)
        self.quantity = 0.0
        self._trades: list[TradeEvent] = []
        self._dividends: list[DividendEvent] = []
        self._fees: list[FeeEvent] = []
        self._taxes: list[TaxEvent] = []
        self._interest: list[InterestEvent] = []
        self._snapshots: list[PositionSnapshot] = []

    @property
    def positions(self) -> dict[str, float]:
        return {self.asset.ticker: self.quantity}

    @property
    def trades(self) -> pd.DataFrame:
        return _events_to_frame(self._trades)

    @property
    def dividends(self) -> pd.DataFrame:
        return _events_to_frame(self._dividends)

    @property
    def fees(self) -> pd.DataFrame:
        return _events_to_frame(self._fees)

    @property
    def taxes(self) -> pd.DataFrame:
        return _events_to_frame(self._taxes)

    @property
    def interest(self) -> pd.DataFrame:
        return _events_to_frame(self._interest)

    @property
    def equity_curve(self) -> pd.DataFrame:
        return _events_to_frame(self._snapshots)

    @property
    def total_fees_paid(self) -> float:
        return float(sum(event.amount for event in self._fees))

    @property
    def total_taxes_paid(self) -> float:
        return float(sum(event.amount for event in self._taxes))

    def buy(
        self,
        trade_date: DateLike,
        *,
        quantity: float,
        price: float,
        note: str = "",
    ) -> TradeEvent:
        quantity = _positive_float(quantity, "quantity")
        price = _positive_float(price, "price")
        timestamp = _to_timestamp(trade_date)
        costs = self.cost_model.estimate_trade(
            self.asset,
            side=TradeSide.BUY,
            quantity=quantity,
            price=price,
        )
        gross_amount = quantity * price
        cash_required = gross_amount + costs.total
        _ensure_cash_available(self.cash, cash_required)

        self.cash -= cash_required
        self.quantity += quantity
        event = TradeEvent(
            date=timestamp,
            asset=self.asset.ticker,
            side=TradeSide.BUY,
            quantity=quantity,
            price=price,
            gross_amount=gross_amount,
            fees=_fee_amount(costs),
            taxes=costs.transaction_tax,
            net_cash_flow=-cash_required,
            currency=self.account_currency,
            note=note,
        )
        self._trades.append(event)
        self._record_cost_events(timestamp, costs, note=f"buy {self.asset.ticker}")
        return event

    def sell(
        self,
        trade_date: DateLike,
        *,
        quantity: float,
        price: float,
        note: str = "",
    ) -> TradeEvent:
        quantity = _positive_float(quantity, "quantity")
        price = _positive_float(price, "price")
        if quantity > self.quantity + 1e-9:
            raise ValueError("Cannot sell more shares than the ledger holds.")

        timestamp = _to_timestamp(trade_date)
        costs = self.cost_model.estimate_trade(
            self.asset,
            side=TradeSide.SELL,
            quantity=quantity,
            price=price,
        )
        gross_amount = quantity * price
        cash_received = gross_amount - costs.total

        self.cash += cash_received
        self.quantity -= quantity
        if abs(self.quantity) < 1e-12:
            self.quantity = 0.0

        event = TradeEvent(
            date=timestamp,
            asset=self.asset.ticker,
            side=TradeSide.SELL,
            quantity=quantity,
            price=price,
            gross_amount=gross_amount,
            fees=_fee_amount(costs),
            taxes=costs.transaction_tax,
            net_cash_flow=cash_received,
            currency=self.account_currency,
            note=note,
        )
        self._trades.append(event)
        self._record_cost_events(timestamp, costs, note=f"sell {self.asset.ticker}")
        return event

    def buy_with_cash(
        self,
        trade_date: DateLike,
        *,
        cash_amount: float,
        price: float,
        note: str = "",
    ) -> TradeEvent | None:
        cash_amount = _non_negative_float(cash_amount, "cash_amount")
        price = _positive_float(price, "price")
        cash_budget = min(cash_amount, self.cash)
        quantity = self._max_affordable_buy_quantity(cash_budget, price)
        if quantity <= 0:
            return None
        return self.buy(trade_date, quantity=quantity, price=price, note=note)

    def cash_dividend(
        self,
        payment_date: DateLike,
        *,
        dividend_per_share: float,
        withholding_rate: float | None = None,
        reinvest: bool = False,
        price: float | None = None,
        note: str = "",
    ) -> DividendEvent:
        dividend_per_share = _non_negative_float(dividend_per_share, "dividend_per_share")
        withholding_rate = (
            self.dividend_withholding_rate if withholding_rate is None else float(withholding_rate)
        )
        if not 0.0 <= withholding_rate <= 1.0:
            raise ValueError("withholding_rate must be between 0 and 1.")

        timestamp = _to_timestamp(payment_date)
        shares = self.quantity
        gross_amount = shares * dividend_per_share
        withholding_tax = gross_amount * withholding_rate
        net_amount = gross_amount - withholding_tax
        cash_before = self.cash

        self.cash += net_amount
        if withholding_tax > 0:
            self._taxes.append(
                TaxEvent(
                    date=timestamp,
                    asset=self.asset.ticker,
                    amount=withholding_tax,
                    currency=self.account_currency,
                    description="US dividend withholding tax",
                )
            )

        reinvested_quantity = 0.0
        reinvest_price = None
        if reinvest and net_amount > 0:
            if price is None:
                raise ValueError("price is required when reinvest=True.")
            reinvest_price = _positive_float(price, "price")
            trade = self.buy_with_cash(
                timestamp,
                cash_amount=net_amount,
                price=reinvest_price,
                note="dividend reinvestment",
            )
            reinvested_quantity = trade.quantity if trade is not None else 0.0

        cash_amount = self.cash - cash_before
        if abs(cash_amount) < 1e-9:
            cash_amount = 0.0

        event = DividendEvent(
            date=timestamp,
            asset=self.asset.ticker,
            shares=shares,
            dividend_per_share=dividend_per_share,
            gross_amount=gross_amount,
            withholding_tax=withholding_tax,
            net_amount=net_amount,
            cash_amount=cash_amount,
            currency=self.account_currency,
            reinvested_quantity=reinvested_quantity,
            reinvest_price=reinvest_price,
            note=note,
        )
        self._dividends.append(event)
        return event

    def snapshot(self, snapshot_date: DateLike, *, price: float) -> PositionSnapshot:
        price = _non_negative_float(price, "price")
        market_value = self.quantity * price
        total_equity = self.cash + market_value
        exposure = market_value / total_equity if total_equity else 0.0
        event = PositionSnapshot(
            date=_to_timestamp(snapshot_date),
            cash=self.cash,
            quantity=self.quantity,
            price=price,
            market_value=market_value,
            total_equity=total_equity,
            exposure=exposure,
            currency=self.account_currency,
        )
        self._snapshots.append(event)
        return event

    def build_equity_curve(self, prices: pd.Series) -> pd.DataFrame:
        for snapshot_date, price in prices.dropna().sort_index().items():
            self.snapshot(snapshot_date, price=float(price))
        return self.equity_curve

    def _record_cost_events(
        self,
        event_date: pd.Timestamp,
        costs: TradeCostBreakdown,
        *,
        note: str,
    ) -> None:
        fee_amount = _fee_amount(costs)
        if fee_amount > 0:
            self._fees.append(
                FeeEvent(
                    date=event_date,
                    asset=self.asset.ticker,
                    amount=fee_amount,
                    currency=self.account_currency,
                    description=note,
                )
            )
        if costs.transaction_tax > 0:
            self._taxes.append(
                TaxEvent(
                    date=event_date,
                    asset=self.asset.ticker,
                    amount=costs.transaction_tax,
                    currency=self.account_currency,
                    description=note,
                )
            )

    def _max_affordable_buy_quantity(self, cash_budget: float, price: float) -> float:
        cash_budget = _non_negative_float(cash_budget, "cash_budget")
        price = _positive_float(price, "price")
        if cash_budget == 0:
            return 0.0

        high = cash_budget / price
        if self._buy_cash_required(high, price) <= cash_budget:
            return high

        low = 0.0
        for _ in range(80):
            mid = (low + high) / 2.0
            if self._buy_cash_required(mid, price) <= cash_budget:
                low = mid
            else:
                high = mid
        return low

    def _buy_cash_required(self, quantity: float, price: float) -> float:
        costs = self.cost_model.estimate_trade(
            self.asset,
            side=TradeSide.BUY,
            quantity=quantity,
            price=price,
        )
        return quantity * price + costs.total


def _events_to_frame(events: list[Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for event in events:
        record = asdict(event)
        record["date"] = pd.Timestamp(record["date"]).date().isoformat()
        if "side" in record:
            record["side"] = str(record["side"])
        records.append(record)
    return pd.DataFrame(records)


def _to_timestamp(value: DateLike) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp.normalize()


def _fee_amount(costs: TradeCostBreakdown) -> float:
    return (
        costs.commission
        + costs.sec_fee
        + costs.finra_taf
        + costs.slippage
        + costs.fx_spread
    )


def _positive_float(value: float, name: str) -> float:
    value = float(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _non_negative_float(value: float, name: str) -> float:
    value = float(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _ensure_cash_available(cash: float, required: float) -> None:
    if required > cash + 1e-9:
        raise ValueError("Insufficient cash. Phase 1 ledger does not support leverage.")


__all__ = [
    "AccountLedger",
    "DividendEvent",
    "FeeEvent",
    "InterestEvent",
    "PositionSnapshot",
    "TaxEvent",
    "TradeEvent",
]
