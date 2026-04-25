from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any

import pandas as pd

from investment_backtest_lab.costs import CostModel, TradeCostBreakdown, TradeSide
from investment_backtest_lab.models import AssetSpec, Market

DateLike = str | date | datetime | pd.Timestamp


@dataclass(frozen=True)
class PortfolioTradeEvent:
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
class PortfolioDividendEvent:
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
class PortfolioFeeEvent:
    date: pd.Timestamp
    asset: str
    amount: float
    currency: str
    description: str


@dataclass(frozen=True)
class PortfolioTaxEvent:
    date: pd.Timestamp
    asset: str
    amount: float
    currency: str
    description: str


@dataclass(frozen=True)
class PortfolioEquitySnapshot:
    date: pd.Timestamp
    cash: float
    market_value: float
    total_equity: float
    exposure: float
    currency: str


@dataclass(frozen=True)
class PortfolioPositionSnapshot:
    date: pd.Timestamp
    asset: str
    quantity: float
    price: float
    market_value: float
    weight: float
    currency: str


@dataclass
class PortfolioLedger:
    """Auditable long-only cash ledger for a US/USD multi-asset portfolio."""

    assets: list[AssetSpec]
    starting_cash: float
    cost_model: CostModel = field(default_factory=CostModel)
    dividend_withholding_rate: float = 0.30
    account_currency: str = "USD"
    base_currency: str = "TWD"

    def __post_init__(self) -> None:
        self.account_currency = self.account_currency.upper()
        self.base_currency = self.base_currency.upper()
        if self.account_currency != "USD":
            raise ValueError("PortfolioLedger v1 account currency must be USD.")
        tickers = [asset.ticker for asset in self.assets]
        if len(set(tickers)) != len(tickers):
            raise ValueError("PortfolioLedger assets must have unique tickers.")
        for asset in self.assets:
            if asset.market != Market.US or asset.currency.upper() != "USD":
                raise ValueError("PortfolioLedger v1 supports only USD-denominated US assets.")

        self._assets_by_ticker = {asset.ticker: asset for asset in self.assets}
        self.cash = _non_negative_float(self.starting_cash, "starting_cash")
        self.quantities = {asset.ticker: 0.0 for asset in self.assets}
        self._trades: list[PortfolioTradeEvent] = []
        self._dividends: list[PortfolioDividendEvent] = []
        self._fees: list[PortfolioFeeEvent] = []
        self._taxes: list[PortfolioTaxEvent] = []
        self._equity_snapshots: list[PortfolioEquitySnapshot] = []
        self._position_snapshots: list[PortfolioPositionSnapshot] = []

    @property
    def positions(self) -> dict[str, float]:
        return dict(self.quantities)

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
    def cash_flows(self) -> pd.DataFrame:
        return pd.DataFrame()

    @property
    def equity_curve(self) -> pd.DataFrame:
        return _events_to_frame(self._equity_snapshots)

    @property
    def positions_history(self) -> pd.DataFrame:
        return _events_to_frame(self._position_snapshots)

    @property
    def total_fees_paid(self) -> float:
        return float(sum(event.amount for event in self._fees))

    @property
    def total_taxes_paid(self) -> float:
        return float(sum(event.amount for event in self._taxes))

    def buy(
        self,
        ticker: str,
        trade_date: DateLike,
        *,
        quantity: float,
        price: float,
        note: str = "",
    ) -> PortfolioTradeEvent:
        asset = self._asset(ticker)
        quantity = _positive_float(quantity, "quantity")
        price = _positive_float(price, "price")
        timestamp = _to_timestamp(trade_date)
        costs = self.cost_model.estimate_trade(
            asset,
            side=TradeSide.BUY,
            quantity=quantity,
            price=price,
        )
        gross_amount = quantity * price
        cash_required = gross_amount + costs.total
        _ensure_cash_available(self.cash, cash_required)

        self.cash -= cash_required
        self.quantities[ticker] += quantity
        event = PortfolioTradeEvent(
            date=timestamp,
            asset=ticker,
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
        self._record_cost_events(timestamp, ticker, costs, note=f"buy {ticker}")
        return event

    def sell(
        self,
        ticker: str,
        trade_date: DateLike,
        *,
        quantity: float,
        price: float,
        note: str = "",
    ) -> PortfolioTradeEvent:
        asset = self._asset(ticker)
        quantity = _positive_float(quantity, "quantity")
        price = _positive_float(price, "price")
        if quantity > self.quantities[ticker] + 1e-9:
            raise ValueError("Cannot sell more shares than the portfolio holds.")

        timestamp = _to_timestamp(trade_date)
        costs = self.cost_model.estimate_trade(
            asset,
            side=TradeSide.SELL,
            quantity=quantity,
            price=price,
        )
        gross_amount = quantity * price
        cash_received = gross_amount - costs.total

        self.cash += cash_received
        self.quantities[ticker] -= quantity
        if abs(self.quantities[ticker]) < 1e-12:
            self.quantities[ticker] = 0.0

        event = PortfolioTradeEvent(
            date=timestamp,
            asset=ticker,
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
        self._record_cost_events(timestamp, ticker, costs, note=f"sell {ticker}")
        return event

    def buy_with_cash(
        self,
        ticker: str,
        trade_date: DateLike,
        *,
        cash_amount: float,
        price: float,
        note: str = "",
    ) -> PortfolioTradeEvent | None:
        cash_amount = _non_negative_float(cash_amount, "cash_amount")
        price = _positive_float(price, "price")
        cash_budget = min(cash_amount, self.cash)
        quantity = self._max_affordable_buy_quantity(ticker, cash_budget, price)
        if quantity <= 0:
            return None
        return self.buy(ticker, trade_date, quantity=quantity, price=price, note=note)

    def cash_dividend(
        self,
        ticker: str,
        payment_date: DateLike,
        *,
        dividend_per_share: float,
        withholding_rate: float | None = None,
        reinvest: bool = False,
        price: float | None = None,
        note: str = "",
    ) -> PortfolioDividendEvent:
        self._asset(ticker)
        dividend_per_share = _non_negative_float(dividend_per_share, "dividend_per_share")
        withholding_rate = (
            self.dividend_withholding_rate if withholding_rate is None else float(withholding_rate)
        )
        if not 0.0 <= withholding_rate <= 1.0:
            raise ValueError("withholding_rate must be between 0 and 1.")

        timestamp = _to_timestamp(payment_date)
        shares = self.quantities[ticker]
        gross_amount = shares * dividend_per_share
        withholding_tax = gross_amount * withholding_rate
        net_amount = gross_amount - withholding_tax
        cash_before = self.cash

        self.cash += net_amount
        if withholding_tax > 0:
            self._taxes.append(
                PortfolioTaxEvent(
                    date=timestamp,
                    asset=ticker,
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
                ticker,
                timestamp,
                cash_amount=net_amount,
                price=reinvest_price,
                note="dividend reinvestment",
            )
            reinvested_quantity = trade.quantity if trade is not None else 0.0

        cash_amount = self.cash - cash_before
        if abs(cash_amount) < 1e-9:
            cash_amount = 0.0

        event = PortfolioDividendEvent(
            date=timestamp,
            asset=ticker,
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

    def rebalance_to_weights(
        self,
        rebalance_date: DateLike,
        *,
        prices: Mapping[str, float],
        target_weights: Mapping[str, float],
        note: str = "rebalance",
    ) -> list[PortfolioTradeEvent]:
        timestamp = _to_timestamp(rebalance_date)
        clean_prices = self._validated_prices(prices)
        weights = self._validated_target_weights(target_weights)
        trades: list[PortfolioTradeEvent] = []

        total_equity = self._total_equity(clean_prices)
        if total_equity <= 0:
            return trades

        current_values = self._market_values(clean_prices)
        for ticker, current_value in sorted(
            current_values.items(),
            key=lambda item: item[1] - weights.get(item[0], 0.0) * total_equity,
            reverse=True,
        ):
            target_value = weights.get(ticker, 0.0) * total_equity
            excess_value = current_value - target_value
            if excess_value <= 1e-8:
                continue
            quantity = min(self.quantities[ticker], excess_value / clean_prices[ticker])
            if quantity > 1e-12:
                trades.append(
                    self.sell(
                        ticker,
                        timestamp,
                        quantity=quantity,
                        price=clean_prices[ticker],
                        note=note,
                    )
                )

        post_sell_equity = self._total_equity(clean_prices)
        current_values = self._market_values(clean_prices)
        gaps = {
            ticker: weights.get(ticker, 0.0) * post_sell_equity - current_value
            for ticker, current_value in current_values.items()
        }
        for ticker, gap in sorted(gaps.items(), key=lambda item: item[1], reverse=True):
            if gap <= 1e-8 or self.cash <= 1e-8:
                continue
            trade = self.buy_with_cash(
                ticker,
                timestamp,
                cash_amount=min(gap, self.cash),
                price=clean_prices[ticker],
                note=note,
            )
            if trade is not None:
                trades.append(trade)
        return trades

    def snapshot(
        self,
        snapshot_date: DateLike,
        *,
        prices: Mapping[str, float],
    ) -> PortfolioEquitySnapshot:
        clean_prices = self._validated_prices(prices)
        timestamp = _to_timestamp(snapshot_date)
        market_values = self._market_values(clean_prices)
        market_value = float(sum(market_values.values()))
        total_equity = self.cash + market_value
        exposure = market_value / total_equity if total_equity else 0.0

        equity = PortfolioEquitySnapshot(
            date=timestamp,
            cash=self.cash,
            market_value=market_value,
            total_equity=total_equity,
            exposure=exposure,
            currency=self.account_currency,
        )
        self._equity_snapshots.append(equity)

        for ticker, asset in self._assets_by_ticker.items():
            asset_market_value = market_values[ticker]
            weight = asset_market_value / total_equity if total_equity else 0.0
            self._position_snapshots.append(
                PortfolioPositionSnapshot(
                    date=timestamp,
                    asset=ticker,
                    quantity=self.quantities[ticker],
                    price=clean_prices[ticker],
                    market_value=asset_market_value,
                    weight=weight,
                    currency=asset.currency,
                )
            )
        return equity

    def final_weights(self, prices: Mapping[str, float]) -> dict[str, float]:
        clean_prices = self._validated_prices(prices)
        total_equity = self._total_equity(clean_prices)
        if total_equity <= 0:
            return {ticker: 0.0 for ticker in self._assets_by_ticker}
        return {
            ticker: value / total_equity
            for ticker, value in self._market_values(clean_prices).items()
        }

    def _asset(self, ticker: str) -> AssetSpec:
        try:
            return self._assets_by_ticker[ticker]
        except KeyError as exc:
            raise ValueError(f"Unknown portfolio asset: {ticker}") from exc

    def _validated_prices(self, prices: Mapping[str, float]) -> dict[str, float]:
        clean: dict[str, float] = {}
        missing = set(self._assets_by_ticker) - set(prices)
        if missing:
            raise ValueError(f"Missing prices for portfolio assets: {sorted(missing)}")
        for ticker in self._assets_by_ticker:
            clean[ticker] = _positive_float(float(prices[ticker]), f"{ticker} price")
        return clean

    def _validated_target_weights(self, target_weights: Mapping[str, float]) -> dict[str, float]:
        weights = {str(ticker): float(weight) for ticker, weight in target_weights.items()}
        unknown = set(weights) - set(self._assets_by_ticker)
        if unknown:
            raise ValueError(f"Target weights include unknown assets: {sorted(unknown)}")
        if any(weight < 0 for weight in weights.values()):
            raise ValueError("Target weights must be non-negative.")
        total_weight = sum(weights.values())
        if not 0.999 <= total_weight <= 1.001:
            raise ValueError(f"Target weights must sum to 1.0, got {total_weight:.4f}.")
        return {ticker: weights.get(ticker, 0.0) for ticker in self._assets_by_ticker}

    def _market_values(self, prices: Mapping[str, float]) -> dict[str, float]:
        return {
            ticker: self.quantities[ticker] * float(prices[ticker])
            for ticker in self._assets_by_ticker
        }

    def _total_equity(self, prices: Mapping[str, float]) -> float:
        return self.cash + sum(self._market_values(prices).values())

    def _record_cost_events(
        self,
        event_date: pd.Timestamp,
        ticker: str,
        costs: TradeCostBreakdown,
        *,
        note: str,
    ) -> None:
        fee_amount = _fee_amount(costs)
        if fee_amount > 0:
            self._fees.append(
                PortfolioFeeEvent(
                    date=event_date,
                    asset=ticker,
                    amount=fee_amount,
                    currency=self.account_currency,
                    description=note,
                )
            )
        if costs.transaction_tax > 0:
            self._taxes.append(
                PortfolioTaxEvent(
                    date=event_date,
                    asset=ticker,
                    amount=costs.transaction_tax,
                    currency=self.account_currency,
                    description=note,
                )
            )

    def _max_affordable_buy_quantity(self, ticker: str, cash_budget: float, price: float) -> float:
        cash_budget = _non_negative_float(cash_budget, "cash_budget")
        price = _positive_float(price, "price")
        if cash_budget == 0:
            return 0.0

        high = cash_budget / price
        if self._buy_cash_required(ticker, high, price) <= cash_budget:
            return high

        low = 0.0
        for _ in range(80):
            mid = (low + high) / 2.0
            if self._buy_cash_required(ticker, mid, price) <= cash_budget:
                low = mid
            else:
                high = mid
        return low

    def _buy_cash_required(self, ticker: str, quantity: float, price: float) -> float:
        costs = self.cost_model.estimate_trade(
            self._asset(ticker),
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
        raise ValueError("Insufficient cash. PortfolioLedger v1 does not support leverage.")


__all__ = [
    "PortfolioDividendEvent",
    "PortfolioEquitySnapshot",
    "PortfolioFeeEvent",
    "PortfolioLedger",
    "PortfolioPositionSnapshot",
    "PortfolioTaxEvent",
    "PortfolioTradeEvent",
]
