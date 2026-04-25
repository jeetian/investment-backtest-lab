from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from investment_backtest_lab.costs import CostModel, TradeSide
from investment_backtest_lab.models import AssetSpec


@dataclass(frozen=True)
class DCAResult:
    orders: pd.DataFrame
    equity: pd.Series
    returns: pd.Series
    total_contributed: float
    total_fees_paid: float


def run_dca_cash_flow(
    close: pd.Series,
    *,
    asset: AssetSpec,
    contribution: float,
    frequency: str = "MS",
    cost_model: CostModel | None = None,
) -> DCAResult:
    """Simple cash-flow DCA accounting for periodic contributions.

    This is not an order-matching engine. It exists because periodic external
    cash contributions are awkward to model in vectorbt without distorting
    idle-cash returns.
    """
    prices = close.dropna().sort_index()
    if prices.empty:
        raise ValueError("DCA requires a non-empty close series.")

    schedule_start = prices.index.min()
    if frequency.upper() == "MS":
        schedule_start = schedule_start.to_period("M").to_timestamp()
    schedule = pd.date_range(schedule_start, prices.index.max(), freq=frequency)
    positions = prices.index.searchsorted(schedule, side="left")
    positions = positions[positions < len(prices.index)]
    contribution_dates = prices.index[positions]
    contribution_dates = pd.Index(contribution_dates).drop_duplicates()

    units = 0.0
    rows: list[dict[str, float | pd.Timestamp]] = []
    equity = pd.Series(index=prices.index, dtype="float64")
    total_fees = 0.0

    for dt, price in prices.items():
        if dt in contribution_dates:
            gross_cash = float(contribution)
            provisional_units = gross_cash / float(price)
            costs = (
                cost_model.estimate_trade(
                    asset,
                    side=TradeSide.BUY,
                    quantity=provisional_units,
                    price=float(price),
                )
                if cost_model
                else None
            )
            fees = costs.total if costs else 0.0
            net_cash = max(gross_cash - fees, 0.0)
            bought_units = net_cash / float(price)
            units += bought_units
            total_fees += fees
            rows.append(
                {
                    "date": dt,
                    "contribution": gross_cash,
                    "price": float(price),
                    "units": bought_units,
                    "fees": fees,
                }
            )
        equity.loc[dt] = units * float(price)

    orders = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    returns = equity.pct_change().fillna(0.0).rename(f"{asset.ticker}_dca")
    return DCAResult(
        orders=orders,
        equity=equity.rename(f"{asset.ticker}_dca_equity"),
        returns=returns,
        total_contributed=float(contribution) * len(orders),
        total_fees_paid=total_fees,
    )
