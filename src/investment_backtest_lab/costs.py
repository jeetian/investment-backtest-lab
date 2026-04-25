from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from investment_backtest_lab.models import AssetSpec, AssetType, Market


class TradeSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class TradeCostBreakdown:
    commission: float = 0.0
    transaction_tax: float = 0.0
    sec_fee: float = 0.0
    finra_taf: float = 0.0
    slippage: float = 0.0
    fx_spread: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.commission
            + self.transaction_tax
            + self.sec_fee
            + self.finra_taf
            + self.slippage
            + self.fx_spread
        )


@dataclass(frozen=True)
class TaiwanCostConfig:
    commission_rate: float = 0.001425
    commission_discount: float = 0.28
    min_commission: float = 20.0
    stock_transaction_tax_rate: float = 0.003
    etf_transaction_tax_rate: float = 0.001
    slippage_bps: float = 1.0


@dataclass(frozen=True)
class USCostConfig:
    commission_per_share: float = 0.0
    min_commission: float = 0.0
    sec_fee_rate: float = 0.0000278
    finra_taf_per_share: float = 0.000166
    finra_taf_cap: float = 8.30
    slippage_bps: float = 1.0


@dataclass(frozen=True)
class FXCostConfig:
    spread_bps: float = 10.0


@dataclass(frozen=True)
class CostModel:
    tw: TaiwanCostConfig = field(default_factory=TaiwanCostConfig)
    us: USCostConfig = field(default_factory=USCostConfig)
    fx: FXCostConfig = field(default_factory=FXCostConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CostModel:
        data = data or {}
        return cls(
            tw=TaiwanCostConfig(**data.get("tw", {})),
            us=USCostConfig(**data.get("us", {})),
            fx=FXCostConfig(**data.get("fx", {})),
        )

    def estimate_trade(
        self,
        asset: AssetSpec,
        *,
        side: TradeSide | str,
        quantity: float,
        price: float,
    ) -> TradeCostBreakdown:
        side = TradeSide(side)
        quantity = abs(float(quantity))
        price = float(price)

        if asset.market == Market.TW:
            return self._estimate_tw(asset, side=side, quantity=quantity, price=price)
        if asset.market == Market.US:
            return self._estimate_us(side=side, quantity=quantity, price=price)

        raise NotImplementedError(f"No cost model for market {asset.market}.")

    def estimate_fx_conversion(self, amount: float) -> TradeCostBreakdown:
        return TradeCostBreakdown(fx_spread=abs(float(amount)) * self.fx.spread_bps / 10_000.0)

    def vectorbt_fee_rate(self, asset: AssetSpec) -> float:
        """Return an approximate proportional fee for vectorbt strategy prototypes."""
        if asset.market == Market.TW:
            tax_rate = (
                self.tw.etf_transaction_tax_rate
                if asset.asset_type == AssetType.ETF
                else self.tw.stock_transaction_tax_rate
            )
            buy_rate = self.tw.commission_rate * self.tw.commission_discount
            sell_rate = buy_rate + tax_rate
            return (buy_rate + sell_rate) / 2.0 + self.tw.slippage_bps / 10_000.0

        if asset.market == Market.US:
            return self.us.slippage_bps / 10_000.0

        return 0.0

    def _estimate_tw(
        self,
        asset: AssetSpec,
        *,
        side: TradeSide,
        quantity: float,
        price: float,
    ) -> TradeCostBreakdown:
        notional = quantity * price
        commission = max(
            self.tw.min_commission,
            notional * self.tw.commission_rate * self.tw.commission_discount,
        )
        tax_rate = (
            self.tw.etf_transaction_tax_rate
            if asset.asset_type == AssetType.ETF
            else self.tw.stock_transaction_tax_rate
        )
        transaction_tax = notional * tax_rate if side == TradeSide.SELL else 0.0
        slippage = notional * self.tw.slippage_bps / 10_000.0
        return TradeCostBreakdown(
            commission=commission,
            transaction_tax=transaction_tax,
            slippage=slippage,
        )

    def _estimate_us(
        self,
        *,
        side: TradeSide,
        quantity: float,
        price: float,
    ) -> TradeCostBreakdown:
        notional = quantity * price
        commission = max(self.us.min_commission, quantity * self.us.commission_per_share)
        sec_fee = notional * self.us.sec_fee_rate if side == TradeSide.SELL else 0.0
        finra_taf = (
            min(quantity * self.us.finra_taf_per_share, self.us.finra_taf_cap)
            if side == TradeSide.SELL
            else 0.0
        )
        slippage = notional * self.us.slippage_bps / 10_000.0
        return TradeCostBreakdown(
            commission=commission,
            sec_fee=sec_fee,
            finra_taf=finra_taf,
            slippage=slippage,
        )
