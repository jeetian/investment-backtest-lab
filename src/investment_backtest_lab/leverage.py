from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from math import inf
from typing import Any

import pandas as pd

from investment_backtest_lab.costs import CostModel, TradeCostBreakdown, TradeSide
from investment_backtest_lab.models import (
    AssetSpec,
    AssetType,
    LeverageConfig,
    LeverageKind,
    Market,
)

DateLike = str | date | datetime | pd.Timestamp


@dataclass(frozen=True)
class LeverageTradeEvent:
    date: pd.Timestamp
    asset: str
    side: TradeSide
    quantity: float
    price: float
    gross_amount: float
    fees: float
    taxes: float
    net_cash_flow: float
    debt_change: float
    debt_after: float
    forced: bool
    currency: str
    note: str = ""


@dataclass(frozen=True)
class LeverageInterestEvent:
    date: pd.Timestamp
    asset: str
    amount: float
    debt_after: float
    annual_borrow_rate: float
    currency: str
    note: str = ""


@dataclass(frozen=True)
class LeverageCashFlowEvent:
    date: pd.Timestamp
    asset: str
    amount: float
    currency: str
    kind: str
    note: str = ""


@dataclass(frozen=True)
class LeverageDividendEvent:
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
class LeverageEvent:
    date: pd.Timestamp
    asset: str
    event_type: str
    target_leverage: float
    actual_leverage: float
    equity_ratio: float
    safety_buffer: float
    debt: float
    currency: str
    note: str = ""


@dataclass(frozen=True)
class LeverageSnapshot:
    date: pd.Timestamp
    asset: str
    cash: float
    quantity: float
    price: float
    market_value: float
    debt: float
    total_equity: float
    actual_leverage: float
    equity_ratio: float
    maintenance_requirement: float
    safety_buffer: float
    currency: str


@dataclass(frozen=True)
class PortfolioLeveragePositionSnapshot:
    date: pd.Timestamp
    portfolio: str
    asset: str
    quantity: float
    price: float
    market_value: float
    weight: float
    currency: str


@dataclass
class MarginLoanLedger:
    """Research ledger for one USD ETF financed with a margin loan.

    This is intentionally separate from AccountLedger. Margin debt, daily
    interest, margin-call flags, and forced deleveraging are audit concerns
    that should stay explicit instead of being hidden inside cash balances.
    """

    asset: AssetSpec
    starting_cash: float
    leverage: LeverageConfig
    cost_model: CostModel = field(default_factory=CostModel)
    account_currency: str = "USD"

    def __post_init__(self) -> None:
        _validate_us_etf(self.asset)
        if self.leverage.kind != LeverageKind.MARGIN_LOAN:
            raise ValueError("MarginLoanLedger only supports leverage.kind=margin_loan.")
        _validate_leverage_config(self.leverage)
        self.account_currency = self.account_currency.upper()
        if self.account_currency != "USD":
            raise ValueError("MarginLoanLedger v1 account currency must be USD.")
        self.cash = float(self.starting_cash)
        self.quantity = 0.0
        self.debt = 0.0
        self._trades: list[LeverageTradeEvent] = []
        self._interest: list[LeverageInterestEvent] = []
        self._cash_flows: list[LeverageCashFlowEvent] = []
        self._dividends: list[LeverageDividendEvent] = []
        self._events: list[LeverageEvent] = []
        self._snapshots: list[LeverageSnapshot] = []

    @property
    def trades(self) -> pd.DataFrame:
        return _events_to_frame(self._trades)

    @property
    def interest_events(self) -> pd.DataFrame:
        return _events_to_frame(self._interest)

    @property
    def cash_flows(self) -> pd.DataFrame:
        return _events_to_frame(self._cash_flows)

    @property
    def dividends(self) -> pd.DataFrame:
        return _events_to_frame(self._dividends)

    @property
    def leverage_events(self) -> pd.DataFrame:
        return _events_to_frame(self._events)

    @property
    def margin_calls(self) -> pd.DataFrame:
        frame = self.leverage_events
        if frame.empty:
            return frame
        return frame[frame["event_type"] == "margin_call"].reset_index(drop=True)

    @property
    def forced_deleveraging_trades(self) -> pd.DataFrame:
        frame = self.trades
        return frame[frame["forced"]].reset_index(drop=True) if not frame.empty else frame

    @property
    def equity_curve(self) -> pd.DataFrame:
        return _events_to_frame(self._snapshots)

    @property
    def leverage_curve(self) -> pd.DataFrame:
        frame = self.equity_curve
        if frame.empty:
            return frame
        return frame[
            [
                "date",
                "asset",
                "actual_leverage",
                "debt",
                "total_equity",
                "market_value",
                "currency",
            ]
        ].copy()

    @property
    def safety_buffer_curve(self) -> pd.DataFrame:
        frame = self.equity_curve
        if frame.empty:
            return frame
        return frame[
            [
                "date",
                "asset",
                "equity_ratio",
                "maintenance_requirement",
                "safety_buffer",
                "currency",
            ]
        ].copy()

    @property
    def total_interest_paid(self) -> float:
        return float(sum(event.amount for event in self._interest))

    @property
    def total_fees_paid(self) -> float:
        return float(sum(event.fees for event in self._trades))

    @property
    def total_gross_dividends(self) -> float:
        return float(sum(event.gross_amount for event in self._dividends))

    @property
    def total_withholding_tax(self) -> float:
        return float(sum(event.withholding_tax for event in self._dividends))

    def deposit(self, deposit_date: DateLike, *, amount: float, note: str = "") -> None:
        amount = _positive_float(amount, "amount")
        timestamp = _to_timestamp(deposit_date)
        self.cash += amount
        self._cash_flows.append(
            LeverageCashFlowEvent(
                date=timestamp,
                asset=self.asset.ticker,
                amount=amount,
                currency=self.account_currency,
                kind="deposit",
                note=note,
            )
        )

    def initialize_to_target_leverage(self, trade_date: DateLike, *, price: float) -> None:
        self.rebalance_to_target_leverage(
            trade_date,
            price=price,
            target_leverage=self.leverage.target_leverage,
            note="initial target leverage allocation",
        )

    def rebalance_to_target_leverage(
        self,
        trade_date: DateLike,
        *,
        price: float,
        target_leverage: float,
        note: str = "",
    ) -> None:
        target_leverage = _target_leverage(target_leverage, self.leverage)
        price = _positive_float(price, "price")
        timestamp = _to_timestamp(trade_date)
        equity = self.current_equity(price)
        if equity <= 0:
            self._record_event(
                timestamp,
                price=price,
                event_type="target_leverage_skipped",
                target_leverage=target_leverage,
                note="equity is non-positive",
            )
            return

        current_market_value = self.market_value(price)
        desired_market_value = equity * target_leverage
        difference = desired_market_value - current_market_value
        if abs(difference) <= 1e-8:
            return
        if difference > 0:
            self._buy_notional(timestamp, price=price, notional=difference, forced=False, note=note)
        else:
            self._sell_notional(
                timestamp,
                price=price,
                notional=abs(difference),
                forced=False,
                note=note,
            )
            self._repay_debt_from_cash()
        self._record_event(
            timestamp,
            price=price,
            event_type="target_leverage_rebalance",
            target_leverage=target_leverage,
            note=note,
        )

    def accrue_interest(self, interest_date: DateLike, *, days: int = 1) -> float:
        days = int(days)
        if days <= 0 or self.debt <= 0:
            return 0.0
        amount = self.debt * self.leverage.annual_borrow_rate * days / 365.0
        if amount <= 0:
            return 0.0
        self.debt += amount
        event = LeverageInterestEvent(
            date=_to_timestamp(interest_date),
            asset=self.asset.ticker,
            amount=amount,
            debt_after=self.debt,
            annual_borrow_rate=self.leverage.annual_borrow_rate,
            currency=self.account_currency,
            note=f"{days} day(s) margin interest",
        )
        self._interest.append(event)
        return amount

    def cash_dividend(
        self,
        dividend_date: DateLike,
        *,
        dividend_per_share: float,
        withholding_rate: float,
        reinvest: bool = False,
        price: float | None = None,
        note: str = "",
    ) -> LeverageDividendEvent | None:
        dividend_per_share = _positive_float(dividend_per_share, "dividend_per_share")
        withholding_rate = _withholding_rate(withholding_rate)
        timestamp = _to_timestamp(dividend_date)
        shares = self.quantity
        if shares <= 0:
            return None

        gross_amount = shares * dividend_per_share
        withholding_tax = gross_amount * withholding_rate
        net_amount = gross_amount - withholding_tax
        self.cash += net_amount
        cash_amount = net_amount
        reinvested_quantity = 0.0
        reinvest_price: float | None = None

        if reinvest:
            if price is None:
                raise ValueError("reinvest=True requires a reinvestment price.")
            reinvest_price = _positive_float(price, "price")
            reinvested_quantity = self._buy_with_cash_budget(
                timestamp,
                price=reinvest_price,
                cash_budget=net_amount,
                note="dividend reinvestment",
            )
            cash_amount = 0.0 if reinvested_quantity > 0 else net_amount

        event = LeverageDividendEvent(
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

    def check_margin_risk(self, check_date: DateLike, *, price: float) -> None:
        timestamp = _to_timestamp(check_date)
        price = _positive_float(price, "price")
        metrics = self._metrics(price)
        if metrics["market_value"] <= 0:
            return

        if metrics["equity_ratio"] <= self.leverage.maintenance_requirement:
            self._record_event(
                timestamp,
                price=price,
                event_type="margin_call",
                target_leverage=self.leverage.deleverage_to,
                note="equity ratio is at or below maintenance requirement",
            )
            self._force_deleverage(timestamp, price=price, note="margin call forced deleverage")
            return

        if metrics["safety_buffer"] < self.leverage.min_safety_buffer:
            self._record_event(
                timestamp,
                price=price,
                event_type="safety_deleverage",
                target_leverage=self.leverage.deleverage_to,
                note="safety buffer is below configured minimum",
            )
            self.deleverage_to(
                timestamp,
                price=price,
                target_leverage=self.leverage.deleverage_to,
                forced=False,
                note="automatic safety deleverage",
            )

    def deleverage_to(
        self,
        trade_date: DateLike,
        *,
        price: float,
        target_leverage: float,
        forced: bool,
        note: str = "",
    ) -> None:
        target_leverage = _target_leverage(target_leverage, self.leverage, allow_below_target=True)
        timestamp = _to_timestamp(trade_date)
        price = _positive_float(price, "price")
        equity = self.current_equity(price)
        if equity <= 0:
            self._force_deleverage(timestamp, price=price, note=note or "non-positive equity")
            return

        desired_market_value = equity * target_leverage
        current_market_value = self.market_value(price)
        sale_notional = current_market_value - desired_market_value
        if sale_notional <= 1e-8:
            return
        self._sell_notional(
            timestamp,
            price=price,
            notional=sale_notional,
            forced=forced,
            note=note,
        )
        self._repay_debt_from_cash()
        self._record_event(
            timestamp,
            price=price,
            event_type="forced_deleverage" if forced else "deleverage",
            target_leverage=target_leverage,
            note=note,
        )

    def snapshot(self, snapshot_date: DateLike, *, price: float) -> LeverageSnapshot:
        timestamp = _to_timestamp(snapshot_date)
        price = _positive_float(price, "price")
        metrics = self._metrics(price)
        event = LeverageSnapshot(
            date=timestamp,
            asset=self.asset.ticker,
            cash=self.cash,
            quantity=self.quantity,
            price=price,
            market_value=metrics["market_value"],
            debt=self.debt,
            total_equity=metrics["total_equity"],
            actual_leverage=metrics["actual_leverage"],
            equity_ratio=metrics["equity_ratio"],
            maintenance_requirement=self.leverage.maintenance_requirement,
            safety_buffer=metrics["safety_buffer"],
            currency=self.account_currency,
        )
        self._snapshots.append(event)
        return event

    def market_value(self, price: float) -> float:
        return self.quantity * float(price)

    def current_equity(self, price: float) -> float:
        return self.cash + self.market_value(price) - self.debt

    def _metrics(self, price: float) -> dict[str, float]:
        market_value = self.market_value(price)
        total_equity = self.current_equity(price)
        actual_leverage = market_value / total_equity if total_equity > 0 else inf
        equity_ratio = total_equity / market_value if market_value > 0 else inf
        safety_buffer = equity_ratio - self.leverage.maintenance_requirement
        return {
            "market_value": market_value,
            "total_equity": total_equity,
            "actual_leverage": actual_leverage,
            "equity_ratio": equity_ratio,
            "safety_buffer": safety_buffer,
        }

    def _buy_notional(
        self,
        trade_date: pd.Timestamp,
        *,
        price: float,
        notional: float,
        forced: bool,
        note: str,
    ) -> None:
        notional = _positive_float(notional, "notional")
        quantity = notional / price
        costs = self.cost_model.estimate_trade(
            self.asset,
            side=TradeSide.BUY,
            quantity=quantity,
            price=price,
        )
        cash_required = notional + costs.total
        debt_drawn = max(0.0, cash_required - self.cash)
        if debt_drawn > 0:
            self.debt += debt_drawn
            self.cash += debt_drawn
        self.cash -= cash_required
        self.quantity += quantity
        self._trades.append(
            LeverageTradeEvent(
                date=trade_date,
                asset=self.asset.ticker,
                side=TradeSide.BUY,
                quantity=quantity,
                price=price,
                gross_amount=notional,
                fees=_fee_amount(costs),
                taxes=costs.transaction_tax,
                net_cash_flow=-cash_required,
                debt_change=debt_drawn,
                debt_after=self.debt,
                forced=forced,
                currency=self.account_currency,
                note=note,
            )
        )

    def _buy_with_cash_budget(
        self,
        trade_date: pd.Timestamp,
        *,
        price: float,
        cash_budget: float,
        note: str,
    ) -> float:
        cash_budget = min(_non_negative_float(cash_budget, "cash_budget"), max(0.0, self.cash))
        if cash_budget <= 0:
            return 0.0
        high = cash_budget
        low = 0.0
        for _ in range(80):
            mid = (low + high) / 2.0
            if self._buy_cash_required(mid, price) <= cash_budget:
                low = mid
            else:
                high = mid
        notional = low
        if notional <= 1e-12:
            return 0.0
        before_debt = self.debt
        self._buy_notional(
            trade_date,
            price=price,
            notional=notional,
            forced=False,
            note=note,
        )
        if self.debt > before_debt + 1e-9:
            raise RuntimeError("Cash-budgeted buy unexpectedly drew margin debt.")
        return notional / price

    def _buy_cash_required(self, notional: float, price: float) -> float:
        quantity = float(notional) / float(price)
        costs = self.cost_model.estimate_trade(
            self.asset,
            side=TradeSide.BUY,
            quantity=quantity,
            price=price,
        )
        return float(notional) + costs.total

    def _sell_notional(
        self,
        trade_date: pd.Timestamp,
        *,
        price: float,
        notional: float,
        forced: bool,
        note: str,
    ) -> None:
        notional = min(_positive_float(notional, "notional"), self.market_value(price))
        quantity = min(notional / price, self.quantity)
        if quantity <= 0:
            return
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
        self._trades.append(
            LeverageTradeEvent(
                date=trade_date,
                asset=self.asset.ticker,
                side=TradeSide.SELL,
                quantity=quantity,
                price=price,
                gross_amount=gross_amount,
                fees=_fee_amount(costs),
                taxes=costs.transaction_tax,
                net_cash_flow=cash_received,
                debt_change=0.0,
                debt_after=self.debt,
                forced=forced,
                currency=self.account_currency,
                note=note,
            )
        )

    def _force_deleverage(self, trade_date: pd.Timestamp, *, price: float, note: str) -> None:
        if self.current_equity(price) > 0 and self.market_value(price) > 0:
            self.deleverage_to(
                trade_date,
                price=price,
                target_leverage=self.leverage.deleverage_to,
                forced=True,
                note=note,
            )
            return
        if self.quantity > 0:
            self._sell_notional(
                trade_date,
                price=price,
                notional=self.market_value(price),
                forced=True,
                note=note,
            )
        self._repay_debt_from_cash()
        self._record_event(
            trade_date,
            price=price,
            event_type="forced_deleverage",
            target_leverage=0.0,
            note=note,
        )

    def _repay_debt_from_cash(self) -> float:
        if self.debt <= 0 or self.cash <= 0:
            return 0.0
        repayment = min(self.cash, self.debt)
        self.cash -= repayment
        self.debt -= repayment
        if abs(self.debt) < 1e-12:
            self.debt = 0.0
        return repayment

    def _record_event(
        self,
        event_date: pd.Timestamp,
        *,
        price: float,
        event_type: str,
        target_leverage: float | None = None,
        note: str = "",
    ) -> None:
        metrics = self._metrics(price)
        self._events.append(
            LeverageEvent(
                date=event_date,
                asset=self.asset.ticker,
                event_type=event_type,
                target_leverage=(
                    self.leverage.target_leverage if target_leverage is None else target_leverage
                ),
                actual_leverage=metrics["actual_leverage"],
                equity_ratio=metrics["equity_ratio"],
                safety_buffer=metrics["safety_buffer"],
                debt=self.debt,
                currency=self.account_currency,
                note=note,
            )
        )


@dataclass
class PortfolioMarginLedger:
    """Research margin-loan ledger for a long-only USD ETF portfolio."""

    assets: list[AssetSpec]
    starting_cash: float
    leverage: LeverageConfig
    cost_model: CostModel = field(default_factory=CostModel)
    account_currency: str = "USD"
    portfolio_label: str = "PORTFOLIO"

    def __post_init__(self) -> None:
        if not self.assets:
            raise ValueError("PortfolioMarginLedger requires at least one asset.")
        for asset in self.assets:
            _validate_us_etf(asset)
        if self.leverage.kind != LeverageKind.MARGIN_LOAN:
            raise ValueError("PortfolioMarginLedger only supports leverage.kind=margin_loan.")
        _validate_leverage_config(self.leverage)
        self.account_currency = self.account_currency.upper()
        if self.account_currency != "USD":
            raise ValueError("PortfolioMarginLedger v1 account currency must be USD.")
        self.cash = float(self.starting_cash)
        self.debt = 0.0
        self.positions = {asset.ticker: 0.0 for asset in self.assets}
        self._asset_map = {asset.ticker: asset for asset in self.assets}
        self._trades: list[LeverageTradeEvent] = []
        self._interest: list[LeverageInterestEvent] = []
        self._cash_flows: list[LeverageCashFlowEvent] = []
        self._dividends: list[LeverageDividendEvent] = []
        self._events: list[LeverageEvent] = []
        self._snapshots: list[LeverageSnapshot] = []
        self._position_snapshots: list[PortfolioLeveragePositionSnapshot] = []

    @property
    def trades(self) -> pd.DataFrame:
        return _events_to_frame(self._trades)

    @property
    def interest_events(self) -> pd.DataFrame:
        return _events_to_frame(self._interest)

    @property
    def cash_flows(self) -> pd.DataFrame:
        return _events_to_frame(self._cash_flows)

    @property
    def dividends(self) -> pd.DataFrame:
        return _events_to_frame(self._dividends)

    @property
    def leverage_events(self) -> pd.DataFrame:
        return _events_to_frame(self._events)

    @property
    def margin_calls(self) -> pd.DataFrame:
        frame = self.leverage_events
        if frame.empty:
            return frame
        return frame[frame["event_type"] == "margin_call"].reset_index(drop=True)

    @property
    def forced_deleveraging_trades(self) -> pd.DataFrame:
        frame = self.trades
        return frame[frame["forced"]].reset_index(drop=True) if not frame.empty else frame

    @property
    def equity_curve(self) -> pd.DataFrame:
        return _events_to_frame(self._snapshots)

    @property
    def position_curve(self) -> pd.DataFrame:
        return _events_to_frame(self._position_snapshots)

    @property
    def total_interest_paid(self) -> float:
        return float(sum(event.amount for event in self._interest))

    @property
    def total_fees_paid(self) -> float:
        return float(sum(event.fees for event in self._trades))

    @property
    def total_gross_dividends(self) -> float:
        return float(sum(event.gross_amount for event in self._dividends))

    @property
    def total_withholding_tax(self) -> float:
        return float(sum(event.withholding_tax for event in self._dividends))

    def rebalance_to_weights(
        self,
        rebalance_date: DateLike,
        *,
        prices: dict[str, float],
        target_weights: dict[str, float],
        target_leverage: float,
        note: str = "",
    ) -> None:
        timestamp = _to_timestamp(rebalance_date)
        prices = _clean_prices(prices, self.positions.keys())
        target_weights = _validate_target_weights(target_weights, self.positions.keys())
        target_leverage = _target_leverage(target_leverage, self.leverage)
        equity = self.current_equity(prices)
        if equity <= 0:
            self._record_event(
                timestamp,
                prices=prices,
                event_type="target_leverage_skipped",
                target_leverage=target_leverage,
                note="equity is non-positive",
            )
            return

        self._sell_overweights(
            timestamp,
            prices=prices,
            target_weights=target_weights,
            target_leverage=target_leverage,
            forced=False,
            note=note,
        )
        self._repay_debt_from_cash()
        self._buy_underweights(
            timestamp,
            prices=prices,
            target_weights=target_weights,
            target_leverage=target_leverage,
            note=note,
        )
        self._record_event(
            timestamp,
            prices=prices,
            event_type="target_leverage_rebalance",
            target_leverage=target_leverage,
            note=note,
        )

    def accrue_interest(self, interest_date: DateLike, *, days: int = 1) -> float:
        days = int(days)
        if days <= 0 or self.debt <= 0:
            return 0.0
        amount = self.debt * self.leverage.annual_borrow_rate * days / 365.0
        if amount <= 0:
            return 0.0
        self.debt += amount
        self._interest.append(
            LeverageInterestEvent(
                date=_to_timestamp(interest_date),
                asset=self.portfolio_label,
                amount=amount,
                debt_after=self.debt,
                annual_borrow_rate=self.leverage.annual_borrow_rate,
                currency=self.account_currency,
                note=f"{days} day(s) portfolio margin interest",
            )
        )
        return amount

    def cash_dividend(
        self,
        dividend_date: DateLike,
        *,
        ticker: str,
        dividend_per_share: float,
        withholding_rate: float,
        reinvest: bool = False,
        price: float | None = None,
        note: str = "",
    ) -> LeverageDividendEvent | None:
        if ticker not in self.positions:
            raise ValueError(f"Unknown portfolio asset: {ticker}")
        dividend_per_share = _positive_float(dividend_per_share, "dividend_per_share")
        withholding_rate = _withholding_rate(withholding_rate)
        timestamp = _to_timestamp(dividend_date)
        shares = self.positions[ticker]
        if shares <= 0:
            return None

        gross_amount = shares * dividend_per_share
        withholding_tax = gross_amount * withholding_rate
        net_amount = gross_amount - withholding_tax
        self.cash += net_amount
        cash_amount = net_amount
        reinvested_quantity = 0.0
        reinvest_price: float | None = None

        if reinvest:
            if price is None:
                raise ValueError("reinvest=True requires a reinvestment price.")
            reinvest_price = _positive_float(price, "price")
            reinvested_quantity = self._buy_asset_with_cash_budget(
                timestamp,
                ticker=ticker,
                price=reinvest_price,
                cash_budget=net_amount,
                note="portfolio dividend reinvestment",
            )
            cash_amount = 0.0 if reinvested_quantity > 0 else net_amount

        event = LeverageDividendEvent(
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

    def check_margin_risk(self, check_date: DateLike, *, prices: dict[str, float]) -> None:
        timestamp = _to_timestamp(check_date)
        prices = _clean_prices(prices, self.positions.keys())
        metrics = self._metrics(prices)
        if metrics["market_value"] <= 0:
            return
        if metrics["equity_ratio"] <= self.leverage.maintenance_requirement:
            self._record_event(
                timestamp,
                prices=prices,
                event_type="margin_call",
                target_leverage=self.leverage.deleverage_to,
                note="equity ratio is at or below maintenance requirement",
            )
            self._force_deleverage(timestamp, prices=prices, note="portfolio margin call")
            return
        if metrics["safety_buffer"] < self.leverage.min_safety_buffer:
            self._record_event(
                timestamp,
                prices=prices,
                event_type="safety_deleverage",
                target_leverage=self.leverage.deleverage_to,
                note="safety buffer is below configured minimum",
            )
            self.deleverage_to(
                timestamp,
                prices=prices,
                target_leverage=self.leverage.deleverage_to,
                forced=False,
                note="portfolio automatic safety deleverage",
            )

    def deleverage_to(
        self,
        trade_date: DateLike,
        *,
        prices: dict[str, float],
        target_leverage: float,
        forced: bool,
        note: str = "",
    ) -> None:
        timestamp = _to_timestamp(trade_date)
        prices = _clean_prices(prices, self.positions.keys())
        target_leverage = _target_leverage(target_leverage, self.leverage, allow_below_target=True)
        equity = self.current_equity(prices)
        if equity <= 0:
            self._force_deleverage(timestamp, prices=prices, note=note or "non-positive equity")
            return
        current_market_value = self.market_value(prices)
        sale_notional = current_market_value - equity * target_leverage
        if sale_notional <= 1e-8:
            return
        market_values = self.market_values(prices)
        for ticker, market_value in market_values.items():
            if market_value <= 0:
                continue
            notional = sale_notional * market_value / current_market_value
            self._sell_asset_notional(
                timestamp,
                ticker=ticker,
                price=prices[ticker],
                notional=notional,
                forced=forced,
                note=note,
            )
        self._repay_debt_from_cash()
        self._record_event(
            timestamp,
            prices=prices,
            event_type="forced_deleverage" if forced else "deleverage",
            target_leverage=target_leverage,
            note=note,
        )

    def snapshot(
        self,
        snapshot_date: DateLike,
        *,
        prices: dict[str, float],
    ) -> LeverageSnapshot:
        timestamp = _to_timestamp(snapshot_date)
        prices = _clean_prices(prices, self.positions.keys())
        metrics = self._metrics(prices)
        event = LeverageSnapshot(
            date=timestamp,
            asset=self.portfolio_label,
            cash=self.cash,
            quantity=float("nan"),
            price=float("nan"),
            market_value=metrics["market_value"],
            debt=self.debt,
            total_equity=metrics["total_equity"],
            actual_leverage=metrics["actual_leverage"],
            equity_ratio=metrics["equity_ratio"],
            maintenance_requirement=self.leverage.maintenance_requirement,
            safety_buffer=metrics["safety_buffer"],
            currency=self.account_currency,
        )
        self._snapshots.append(event)
        for ticker, market_value in self.market_values(prices).items():
            weight = market_value / metrics["market_value"] if metrics["market_value"] > 0 else 0.0
            self._position_snapshots.append(
                PortfolioLeveragePositionSnapshot(
                    date=timestamp,
                    portfolio=self.portfolio_label,
                    asset=ticker,
                    quantity=self.positions[ticker],
                    price=prices[ticker],
                    market_value=market_value,
                    weight=weight,
                    currency=self.account_currency,
                )
            )
        return event

    def market_values(self, prices: dict[str, float]) -> dict[str, float]:
        return {ticker: self.positions[ticker] * float(prices[ticker]) for ticker in self.positions}

    def market_value(self, prices: dict[str, float]) -> float:
        return float(sum(self.market_values(prices).values()))

    def current_equity(self, prices: dict[str, float]) -> float:
        return self.cash + self.market_value(prices) - self.debt

    def _metrics(self, prices: dict[str, float]) -> dict[str, float]:
        market_value = self.market_value(prices)
        total_equity = self.current_equity(prices)
        actual_leverage = market_value / total_equity if total_equity > 0 else inf
        equity_ratio = total_equity / market_value if market_value > 0 else inf
        safety_buffer = equity_ratio - self.leverage.maintenance_requirement
        return {
            "market_value": market_value,
            "total_equity": total_equity,
            "actual_leverage": actual_leverage,
            "equity_ratio": equity_ratio,
            "safety_buffer": safety_buffer,
        }

    def _sell_overweights(
        self,
        trade_date: pd.Timestamp,
        *,
        prices: dict[str, float],
        target_weights: dict[str, float],
        target_leverage: float,
        forced: bool,
        note: str,
    ) -> None:
        target_values = self._target_values(prices, target_weights, target_leverage)
        current_values = self.market_values(prices)
        for ticker in sorted(self.positions):
            excess = current_values[ticker] - target_values[ticker]
            if excess > 1e-8:
                self._sell_asset_notional(
                    trade_date,
                    ticker=ticker,
                    price=prices[ticker],
                    notional=excess,
                    forced=forced,
                    note=note,
                )

    def _buy_underweights(
        self,
        trade_date: pd.Timestamp,
        *,
        prices: dict[str, float],
        target_weights: dict[str, float],
        target_leverage: float,
        note: str,
    ) -> None:
        target_values = self._target_values(prices, target_weights, target_leverage)
        current_values = self.market_values(prices)
        for ticker in sorted(self.positions):
            shortfall = target_values[ticker] - current_values[ticker]
            if shortfall > 1e-8:
                self._buy_asset_notional(
                    trade_date,
                    ticker=ticker,
                    price=prices[ticker],
                    notional=shortfall,
                    forced=False,
                    note=note,
                )

    def _target_values(
        self,
        prices: dict[str, float],
        target_weights: dict[str, float],
        target_leverage: float,
    ) -> dict[str, float]:
        equity = self.current_equity(prices)
        target_market_value = max(0.0, equity * target_leverage)
        return {ticker: target_market_value * target_weights[ticker] for ticker in self.positions}

    def _buy_asset_notional(
        self,
        trade_date: pd.Timestamp,
        *,
        ticker: str,
        price: float,
        notional: float,
        forced: bool,
        note: str,
    ) -> None:
        notional = _positive_float(notional, "notional")
        quantity = notional / price
        costs = self.cost_model.estimate_trade(
            self._asset_map[ticker],
            side=TradeSide.BUY,
            quantity=quantity,
            price=price,
        )
        cash_required = notional + costs.total
        debt_drawn = max(0.0, cash_required - self.cash)
        if debt_drawn > 0:
            self.debt += debt_drawn
            self.cash += debt_drawn
        self.cash -= cash_required
        self.positions[ticker] += quantity
        self._trades.append(
            LeverageTradeEvent(
                date=trade_date,
                asset=ticker,
                side=TradeSide.BUY,
                quantity=quantity,
                price=price,
                gross_amount=notional,
                fees=_fee_amount(costs),
                taxes=costs.transaction_tax,
                net_cash_flow=-cash_required,
                debt_change=debt_drawn,
                debt_after=self.debt,
                forced=forced,
                currency=self.account_currency,
                note=note,
            )
        )

    def _buy_asset_with_cash_budget(
        self,
        trade_date: pd.Timestamp,
        *,
        ticker: str,
        price: float,
        cash_budget: float,
        note: str,
    ) -> float:
        cash_budget = min(_non_negative_float(cash_budget, "cash_budget"), max(0.0, self.cash))
        if cash_budget <= 0:
            return 0.0
        high = cash_budget
        low = 0.0
        for _ in range(80):
            mid = (low + high) / 2.0
            cash_required = self._buy_asset_cash_required(
                ticker=ticker,
                notional=mid,
                price=price,
            )
            if cash_required <= cash_budget:
                low = mid
            else:
                high = mid
        notional = low
        if notional <= 1e-12:
            return 0.0
        before_debt = self.debt
        self._buy_asset_notional(
            trade_date,
            ticker=ticker,
            price=price,
            notional=notional,
            forced=False,
            note=note,
        )
        if self.debt > before_debt + 1e-9:
            raise RuntimeError("Cash-budgeted portfolio buy unexpectedly drew margin debt.")
        return notional / price

    def _buy_asset_cash_required(self, *, ticker: str, notional: float, price: float) -> float:
        quantity = float(notional) / float(price)
        costs = self.cost_model.estimate_trade(
            self._asset_map[ticker],
            side=TradeSide.BUY,
            quantity=quantity,
            price=price,
        )
        return float(notional) + costs.total

    def _sell_asset_notional(
        self,
        trade_date: pd.Timestamp,
        *,
        ticker: str,
        price: float,
        notional: float,
        forced: bool,
        note: str,
    ) -> None:
        notional = min(_positive_float(notional, "notional"), self.positions[ticker] * price)
        quantity = min(notional / price, self.positions[ticker])
        if quantity <= 0:
            return
        costs = self.cost_model.estimate_trade(
            self._asset_map[ticker],
            side=TradeSide.SELL,
            quantity=quantity,
            price=price,
        )
        gross_amount = quantity * price
        cash_received = gross_amount - costs.total
        self.cash += cash_received
        self.positions[ticker] -= quantity
        if abs(self.positions[ticker]) < 1e-12:
            self.positions[ticker] = 0.0
        self._trades.append(
            LeverageTradeEvent(
                date=trade_date,
                asset=ticker,
                side=TradeSide.SELL,
                quantity=quantity,
                price=price,
                gross_amount=gross_amount,
                fees=_fee_amount(costs),
                taxes=costs.transaction_tax,
                net_cash_flow=cash_received,
                debt_change=0.0,
                debt_after=self.debt,
                forced=forced,
                currency=self.account_currency,
                note=note,
            )
        )

    def _force_deleverage(
        self,
        trade_date: pd.Timestamp,
        *,
        prices: dict[str, float],
        note: str,
    ) -> None:
        if self.current_equity(prices) > 0 and self.market_value(prices) > 0:
            self.deleverage_to(
                trade_date,
                prices=prices,
                target_leverage=self.leverage.deleverage_to,
                forced=True,
                note=note,
            )
            return
        for ticker, market_value in self.market_values(prices).items():
            if market_value > 0:
                self._sell_asset_notional(
                    trade_date,
                    ticker=ticker,
                    price=prices[ticker],
                    notional=market_value,
                    forced=True,
                    note=note,
                )
        self._repay_debt_from_cash()
        self._record_event(
            trade_date,
            prices=prices,
            event_type="forced_deleverage",
            target_leverage=0.0,
            note=note,
        )

    def _repay_debt_from_cash(self) -> float:
        if self.debt <= 0 or self.cash <= 0:
            return 0.0
        repayment = min(self.cash, self.debt)
        self.cash -= repayment
        self.debt -= repayment
        if abs(self.debt) < 1e-12:
            self.debt = 0.0
        return repayment

    def _record_event(
        self,
        event_date: pd.Timestamp,
        *,
        prices: dict[str, float],
        event_type: str,
        target_leverage: float | None = None,
        note: str = "",
    ) -> None:
        metrics = self._metrics(prices)
        self._events.append(
            LeverageEvent(
                date=event_date,
                asset=self.portfolio_label,
                event_type=event_type,
                target_leverage=(
                    self.leverage.target_leverage if target_leverage is None else target_leverage
                ),
                actual_leverage=metrics["actual_leverage"],
                equity_ratio=metrics["equity_ratio"],
                safety_buffer=metrics["safety_buffer"],
                debt=self.debt,
                currency=self.account_currency,
                note=note,
            )
        )


def _events_to_frame(events: list[Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for event in events:
        record = asdict(event)
        record["date"] = pd.Timestamp(record["date"]).date().isoformat()
        if "side" in record:
            record["side"] = str(record["side"])
        records.append(record)
    return pd.DataFrame(records)


def _validate_us_etf(asset: AssetSpec) -> None:
    if asset.market != Market.US or asset.currency.upper() != "USD":
        raise ValueError("Margin loan v1 supports only USD-denominated US ETFs.")
    if asset.asset_type != AssetType.ETF:
        raise ValueError("Margin loan v1 supports only ETF assets.")


def _validate_leverage_config(config: LeverageConfig) -> None:
    if config.target_leverage < 1.0:
        raise ValueError("target_leverage must be at least 1.0.")
    if config.max_leverage < config.target_leverage:
        raise ValueError("max_leverage must be greater than or equal to target_leverage.")
    if config.annual_borrow_rate < 0:
        raise ValueError("annual_borrow_rate must be non-negative.")
    if not 0 < config.maintenance_requirement < 1:
        raise ValueError("maintenance_requirement must be between 0 and 1.")
    if config.min_safety_buffer < 0:
        raise ValueError("min_safety_buffer must be non-negative.")
    if config.deleverage_to < 1.0:
        raise ValueError("deleverage_to must be at least 1.0.")
    if config.deleverage_to > config.target_leverage:
        raise ValueError("deleverage_to should not exceed target_leverage.")


def _clean_prices(prices: dict[str, float], tickers: Any) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    missing: list[str] = []
    for ticker in tickers:
        if ticker not in prices:
            missing.append(str(ticker))
            continue
        cleaned[str(ticker)] = _positive_float(prices[str(ticker)], f"price[{ticker}]")
    if missing:
        raise ValueError(f"Missing prices for: {missing}")
    return cleaned


def _validate_target_weights(
    target_weights: dict[str, float],
    tickers: Any,
) -> dict[str, float]:
    expected = [str(ticker) for ticker in tickers]
    missing = [ticker for ticker in expected if ticker not in target_weights]
    extra = [ticker for ticker in target_weights if ticker not in expected]
    if missing or extra:
        raise ValueError(f"Target weights mismatch. Missing={missing}, extra={extra}")
    weights = {ticker: float(target_weights[ticker]) for ticker in expected}
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("Target weights must be non-negative.")
    total = sum(weights.values())
    if not 0.999 <= total <= 1.001:
        raise ValueError(f"Target weights must sum to 1.0, got {total:.6f}.")
    return weights


def _target_leverage(
    value: float,
    config: LeverageConfig,
    *,
    allow_below_target: bool = False,
) -> float:
    value = float(value)
    if value < 1.0:
        raise ValueError("target leverage must be at least 1.0.")
    if value > config.max_leverage + 1e-12:
        raise ValueError("target leverage exceeds configured max_leverage.")
    if not allow_below_target and value > config.target_leverage + 1e-12:
        raise ValueError("requested target leverage exceeds configured target_leverage.")
    return value


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


def _withholding_rate(value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("withholding_rate must be between 0 and 1.")
    return value


__all__ = [
    "LeverageCashFlowEvent",
    "LeverageDividendEvent",
    "LeverageEvent",
    "LeverageInterestEvent",
    "LeverageSnapshot",
    "LeverageTradeEvent",
    "MarginLoanLedger",
    "PortfolioLeveragePositionSnapshot",
    "PortfolioMarginLedger",
]
