from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from investment_backtest_lab.costs import CostModel
from investment_backtest_lab.ledger_reports import run_buy_and_hold_ledger, run_rebalance_ledger
from investment_backtest_lab.models import (
    AssetSpec,
    AssetType,
    DataSource,
    DividendFrame,
    DividendMode,
    Market,
    PriceFrame,
)
from investment_backtest_lab.strategies.rebalance import run_periodic_rebalance_bt
from investment_backtest_lab.strategies.vectorbt_wrappers import run_buy_and_hold


@dataclass(frozen=True)
class CrossValidationCheck:
    case: str
    primary: str
    reference: str
    metric: str
    primary_value: float
    reference_value: float
    abs_diff: float
    tolerance: float
    passed: bool
    note: str


def run_cross_validation(*, tolerance: float = 1e-4) -> pd.DataFrame:
    checks = [
        buy_hold_ledger_vs_vectorbt(tolerance=tolerance),
        rebalance_ledger_vs_bt(tolerance=tolerance),
    ]
    return pd.DataFrame([asdict(check) for check in checks])


def buy_hold_ledger_vs_vectorbt(*, tolerance: float = 1e-8) -> CrossValidationCheck:
    close = pd.Series(
        [100.0, 110.0, 121.0, 115.0, 130.0],
        index=pd.bdate_range("2024-01-02", periods=5),
        name="SPY",
    )
    initial_cash = 10_000.0
    price_frame = _price_frame("SPY", close)
    ledger = run_buy_and_hold_ledger(
        price_frame=price_frame,
        dividend_frame=_empty_dividend_frame("SPY"),
        cost_model=_zero_cost_model(),
        initial_cash=initial_cash,
        dividend_mode=DividendMode.CASH,
        withholding_rate=0.0,
    )
    ledger_final = float(ledger.equity_curve["total_equity"].iloc[-1])

    vectorbt_portfolio = run_buy_and_hold(close, init_cash=initial_cash, fees=0.0)
    vectorbt_final = float(vectorbt_portfolio.value().iloc[-1])
    return _check(
        case="buy_hold_price_only",
        primary="AccountLedger",
        reference="vectorbt",
        metric="ending_equity_usd",
        primary_value=ledger_final,
        reference_value=vectorbt_final,
        tolerance=tolerance,
        note="Synthetic SPY, no fees, no dividends, fractional shares.",
    )


def rebalance_ledger_vs_bt(*, tolerance: float = 1e-4) -> CrossValidationCheck:
    prices = _synthetic_rebalance_prices()
    initial_cash = 10_000.0
    price_frames = [_price_frame(ticker, prices[ticker]) for ticker in prices.columns]
    dividend_frames = [_empty_dividend_frame(ticker) for ticker in prices.columns]
    target_weights = {"SPY": 0.60, "QQQ": 0.40}
    ledger = run_rebalance_ledger(
        price_frames=price_frames,
        dividend_frames=dividend_frames,
        cost_model=_zero_cost_model(),
        initial_cash=initial_cash,
        target_weights=target_weights,
        frequency="monthly",
        dividend_mode=DividendMode.CASH,
        withholding_rate=0.0,
    )
    ledger_return = float(ledger.equity_curve["total_equity"].iloc[-1]) / initial_cash - 1.0

    bt_result = run_periodic_rebalance_bt(
        prices,
        target_weights=target_weights,
        frequency="monthly",
        name="ledger_rebalance_reference",
    )
    bt_curve = bt_result.prices["ledger_rebalance_reference"]
    bt_return = float(bt_curve.iloc[-1] / bt_curve.iloc[0] - 1.0)
    return _check(
        case="monthly_rebalance_price_only",
        primary="PortfolioLedger",
        reference="bt",
        metric="total_return",
        primary_value=ledger_return,
        reference_value=bt_return,
        tolerance=tolerance,
        note="Synthetic SPY/QQQ 60/40, monthly, no fees, no dividends.",
    )


def _check(
    *,
    case: str,
    primary: str,
    reference: str,
    metric: str,
    primary_value: float,
    reference_value: float,
    tolerance: float,
    note: str,
) -> CrossValidationCheck:
    abs_diff = abs(primary_value - reference_value)
    return CrossValidationCheck(
        case=case,
        primary=primary,
        reference=reference,
        metric=metric,
        primary_value=primary_value,
        reference_value=reference_value,
        abs_diff=abs_diff,
        tolerance=tolerance,
        passed=abs_diff <= tolerance,
        note=note,
    )


def _synthetic_rebalance_prices() -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", "2024-03-08")
    return pd.DataFrame(
        {
            "SPY": [100.0 + index_number * 0.8 for index_number in range(len(index))],
            "QQQ": [100.0 + ((index_number % 7) - 3) * 1.5 for index_number in range(len(index))],
        },
        index=index,
    )


def _price_frame(ticker: str, close: pd.Series) -> PriceFrame:
    asset = _asset(ticker)
    data = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 0,
        }
    )
    return PriceFrame(asset=asset, data=data, adjusted=False, source="synthetic")


def _empty_dividend_frame(ticker: str) -> DividendFrame:
    data = pd.DataFrame(
        {"dividend_per_share": pd.Series(dtype="float64")},
        index=pd.DatetimeIndex([], name="date"),
    )
    return DividendFrame(asset=_asset(ticker), data=data, currency="USD", source="synthetic")


def _asset(ticker: str) -> AssetSpec:
    return AssetSpec(
        ticker=ticker,
        market=Market.US,
        asset_type=AssetType.ETF,
        currency="USD",
        data_source=DataSource.CSV,
    )


def _zero_cost_model() -> CostModel:
    return CostModel.from_dict(
        {
            "us": {
                "commission_per_share": 0.0,
                "min_commission": 0.0,
                "sec_fee_rate": 0.0,
                "finra_taf_per_share": 0.0,
                "finra_taf_cap": 0.0,
                "slippage_bps": 0.0,
            }
        }
    )


__all__ = [
    "CrossValidationCheck",
    "buy_hold_ledger_vs_vectorbt",
    "rebalance_ledger_vs_bt",
    "run_cross_validation",
]
