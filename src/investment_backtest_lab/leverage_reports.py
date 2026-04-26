from __future__ import annotations

from dataclasses import dataclass
from html import escape
from math import inf
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from investment_backtest_lab.costs import CostModel
from investment_backtest_lab.data.fx import align_fx_rate
from investment_backtest_lab.ledger_reports import (
    align_dividends_to_trading_dates,
    dca_contribution_dates,
    rebalance_schedule_dates,
)
from investment_backtest_lab.leverage import MarginLoanLedger, PortfolioMarginLedger
from investment_backtest_lab.leverage_policy import (
    DynamicLeverageDecision,
    decide_dynamic_leverage,
    decisions_to_frame,
    weighted_signal_series,
)
from investment_backtest_lab.models import (
    DividendFrame,
    DividendMode,
    DynamicLeverageConfig,
    LeverageConfig,
    PriceFrame,
)
from investment_backtest_lab.reports import max_drawdown, performance_summary


@dataclass(frozen=True)
class LeveragedRunResult:
    ticker: str
    strategy: str
    dividend_mode: DividendMode
    ledger: MarginLoanLedger | PortfolioMarginLedger
    equity_curve: pd.DataFrame
    aligned_dividends: pd.DataFrame
    policy_decisions: pd.DataFrame
    price_source: str
    dividend_source: str
    total_contributed: float
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class LeverageReportResult:
    metrics: pd.DataFrame
    trades: pd.DataFrame
    interest: pd.DataFrame
    dividends: pd.DataFrame
    leverage_events: pd.DataFrame
    cash_flows: pd.DataFrame
    curves: pd.DataFrame
    positions: pd.DataFrame
    policy: pd.DataFrame
    warnings: list[str]
    markdown_path: Path
    metrics_path: Path
    trades_path: Path
    interest_path: Path
    dividends_path: Path
    events_path: Path
    cash_flows_path: Path
    curves_path: Path
    positions_path: Path
    policy_path: Path
    html_path: Path


def run_buy_hold_leveraged(
    *,
    price_frame: PriceFrame,
    dividend_frame: DividendFrame,
    cost_model: CostModel,
    initial_cash: float,
    leverage: LeverageConfig,
    dividend_mode: DividendMode | str,
    withholding_rate: float,
) -> LeveragedRunResult:
    dividend_mode = DividendMode(dividend_mode)
    prices = price_frame.close().dropna().sort_index()
    if prices.empty:
        raise ValueError("Leveraged buy-and-hold requires non-empty price data.")

    warnings = _price_warnings(price_frame)
    ledger = MarginLoanLedger(
        price_frame.asset,
        starting_cash=initial_cash,
        leverage=leverage,
        cost_model=cost_model,
        account_currency=price_frame.asset.currency,
    )
    aligned_dividends, align_warnings = align_dividends_to_trading_dates(
        dividend_frame.data,
        prices.index,
    )
    warnings.extend(f"{price_frame.asset.ticker}: {warning}" for warning in align_warnings)
    dividends_by_date = _dividends_by_effective_date(aligned_dividends)
    _run_fixed_target_leverage_path(
        ledger,
        prices,
        dividends_by_date=dividends_by_date,
        dividend_mode=dividend_mode,
        withholding_rate=withholding_rate,
    )
    policy_decisions = _static_policy_frame(
        prices.index,
        target_leverage=leverage.target_leverage,
        strategy="buy_hold_leveraged",
    )
    return LeveragedRunResult(
        ticker=price_frame.asset.ticker,
        strategy="buy_hold_leveraged",
        dividend_mode=dividend_mode,
        ledger=ledger,
        equity_curve=ledger.equity_curve,
        aligned_dividends=aligned_dividends,
        policy_decisions=policy_decisions,
        price_source=price_frame.source,
        dividend_source=dividend_frame.source,
        total_contributed=float(initial_cash),
        warnings=tuple(warnings),
    )


def run_dca_leveraged(
    *,
    price_frame: PriceFrame,
    dividend_frame: DividendFrame,
    cost_model: CostModel,
    contribution: float,
    frequency: str,
    leverage: LeverageConfig,
    dividend_mode: DividendMode | str,
    withholding_rate: float,
) -> LeveragedRunResult:
    dividend_mode = DividendMode(dividend_mode)
    prices = price_frame.close().dropna().sort_index()
    if prices.empty:
        raise ValueError("Leveraged DCA requires non-empty price data.")

    warnings = _price_warnings(price_frame)
    ledger = MarginLoanLedger(
        price_frame.asset,
        starting_cash=0.0,
        leverage=leverage,
        cost_model=cost_model,
        account_currency=price_frame.asset.currency,
    )
    aligned_dividends, align_warnings = align_dividends_to_trading_dates(
        dividend_frame.data,
        prices.index,
    )
    warnings.extend(f"{price_frame.asset.ticker}: {warning}" for warning in align_warnings)
    dividends_by_date = _dividends_by_effective_date(aligned_dividends)
    contribution_dates = dca_contribution_dates(prices.index, frequency=frequency)
    previous_date: pd.Timestamp | None = None
    total_contributed = 0.0
    for current_date, price in prices.items():
        current_date = pd.Timestamp(current_date)
        current_price = float(price)
        if previous_date is not None:
            days = max(1, (current_date - previous_date).days)
            ledger.accrue_interest(current_date, days=days)
        _apply_single_asset_dividends(
            ledger,
            current_date=current_date,
            current_price=current_price,
            rows=dividends_by_date.get(current_date, pd.DataFrame()),
            dividend_mode=dividend_mode,
            withholding_rate=withholding_rate,
        )
        if current_date in contribution_dates:
            ledger.deposit(
                current_date,
                amount=contribution,
                note=f"leveraged DCA {frequency} contribution",
            )
            total_contributed += contribution
            ledger.rebalance_to_target_leverage(
                current_date,
                price=current_price,
                target_leverage=leverage.target_leverage,
                note="leveraged DCA target leverage buy",
            )
        ledger.check_margin_risk(current_date, price=current_price)
        ledger.snapshot(current_date, price=current_price)
        previous_date = current_date
    policy_frame = _static_policy_frame(
        prices.index,
        target_leverage=leverage.target_leverage,
        strategy="dca_leveraged",
    )

    return LeveragedRunResult(
        ticker=price_frame.asset.ticker,
        strategy="dca_leveraged",
        dividend_mode=dividend_mode,
        ledger=ledger,
        equity_curve=ledger.equity_curve,
        aligned_dividends=aligned_dividends,
        policy_decisions=policy_frame,
        price_source=price_frame.source,
        dividend_source=dividend_frame.source,
        total_contributed=float(total_contributed),
        warnings=tuple(warnings),
    )


def run_rebalance_leveraged(
    *,
    price_frames: list[PriceFrame],
    dividend_frames: list[DividendFrame],
    cost_model: CostModel,
    initial_cash: float,
    target_weights: dict[str, float],
    frequency: str,
    leverage: LeverageConfig,
    dividend_mode: DividendMode | str,
    withholding_rate: float,
) -> LeveragedRunResult:
    dividend_mode = DividendMode(dividend_mode)
    if len(price_frames) < 2:
        raise ValueError("Leveraged rebalance requires at least two price frames.")
    if len(price_frames) != len(dividend_frames):
        raise ValueError("price_frames and dividend_frames must have the same length.")
    tickers = [frame.asset.ticker for frame in price_frames]
    prices = pd.concat([frame.close() for frame in price_frames], axis=1).dropna().sort_index()
    if prices.empty:
        raise ValueError("Leveraged rebalance requires overlapping non-empty price data.")

    warnings: list[str] = []
    for frame in price_frames:
        warnings.extend(_price_warnings(frame))
    aligned_dividends = []
    for frame in dividend_frames:
        aligned, align_warnings = align_dividends_to_trading_dates(frame.data, prices.index)
        if not aligned.empty:
            aligned = aligned.copy()
            aligned.insert(0, "asset", frame.asset.ticker)
            aligned_dividends.append(aligned)
        warnings.extend(f"{frame.asset.ticker}: {warning}" for warning in align_warnings)
    aligned_dividend_frame = (
        pd.concat(aligned_dividends, ignore_index=True)
        if aligned_dividends
        else pd.DataFrame()
    )
    dividends_by_date = _portfolio_dividends_by_effective_date(aligned_dividend_frame)
    ledger = PortfolioMarginLedger(
        [frame.asset for frame in price_frames],
        starting_cash=initial_cash,
        leverage=leverage,
        cost_model=cost_model,
        account_currency=price_frames[0].asset.currency,
        portfolio_label="_".join(tickers),
    )
    schedule_dates = rebalance_schedule_dates(prices.index, frequency=frequency)
    previous_date: pd.Timestamp | None = None
    for current_date, row in prices.iterrows():
        current_date = pd.Timestamp(current_date)
        current_prices = {ticker: float(row[ticker]) for ticker in tickers}
        if previous_date is not None:
            days = max(1, (current_date - previous_date).days)
            ledger.accrue_interest(current_date, days=days)
        _apply_portfolio_dividends(
            ledger,
            current_date=current_date,
            current_prices=current_prices,
            rows=dividends_by_date.get(current_date, pd.DataFrame()),
            dividend_mode=dividend_mode,
            withholding_rate=withholding_rate,
        )
        if previous_date is None or current_date in schedule_dates:
            ledger.rebalance_to_weights(
                current_date,
                prices=current_prices,
                target_weights=target_weights,
                target_leverage=leverage.target_leverage,
                note=f"leveraged {frequency} rebalance",
            )
        ledger.check_margin_risk(current_date, prices=current_prices)
        ledger.snapshot(current_date, prices=current_prices)
        previous_date = current_date
    policy_frame = _static_policy_frame(
        prices.index,
        target_leverage=leverage.target_leverage,
        strategy="rebalance_leveraged",
    )

    return LeveragedRunResult(
        ticker="_".join(tickers),
        strategy="rebalance_leveraged",
        dividend_mode=dividend_mode,
        ledger=ledger,
        equity_curve=ledger.equity_curve,
        aligned_dividends=aligned_dividend_frame,
        policy_decisions=policy_frame,
        price_source="; ".join(f"{frame.asset.ticker}:{frame.source}" for frame in price_frames),
        dividend_source="; ".join(
            f"{frame.asset.ticker}:{frame.source}" for frame in dividend_frames
        ),
        total_contributed=float(initial_cash),
        warnings=tuple(warnings),
    )


def run_dynamic_buy_hold_leveraged(
    *,
    price_frame: PriceFrame,
    dividend_frame: DividendFrame,
    cost_model: CostModel,
    initial_cash: float,
    leverage: LeverageConfig,
    dynamic: DynamicLeverageConfig,
    dividend_mode: DividendMode | str,
    withholding_rate: float,
) -> LeveragedRunResult:
    dividend_mode = DividendMode(dividend_mode)
    prices = price_frame.close().dropna().sort_index()
    if prices.empty:
        raise ValueError("Dynamic leveraged buy-and-hold requires non-empty price data.")

    warnings = _price_warnings(price_frame)
    ledger = MarginLoanLedger(
        price_frame.asset,
        starting_cash=initial_cash,
        leverage=leverage,
        cost_model=cost_model,
        account_currency=price_frame.asset.currency,
    )
    aligned_dividends, align_warnings = align_dividends_to_trading_dates(
        dividend_frame.data,
        prices.index,
    )
    warnings.extend(f"{price_frame.asset.ticker}: {warning}" for warning in align_warnings)
    dividends_by_date = _dividends_by_effective_date(aligned_dividends)
    policy_decisions = _run_dynamic_single_asset_path(
        ledger,
        prices,
        signal_prices=prices,
        dynamic=dynamic,
        dividends_by_date=dividends_by_date,
        dividend_mode=dividend_mode,
        withholding_rate=withholding_rate,
        contribution_dates=set(),
        contribution=0.0,
    )
    return LeveragedRunResult(
        ticker=price_frame.asset.ticker,
        strategy="dynamic_buy_hold_leveraged",
        dividend_mode=dividend_mode,
        ledger=ledger,
        equity_curve=ledger.equity_curve,
        aligned_dividends=aligned_dividends,
        policy_decisions=decisions_to_frame(policy_decisions),
        price_source=price_frame.source,
        dividend_source=dividend_frame.source,
        total_contributed=float(initial_cash),
        warnings=tuple(warnings),
    )


def run_dynamic_dca_leveraged(
    *,
    price_frame: PriceFrame,
    dividend_frame: DividendFrame,
    cost_model: CostModel,
    contribution: float,
    frequency: str,
    leverage: LeverageConfig,
    dynamic: DynamicLeverageConfig,
    dividend_mode: DividendMode | str,
    withholding_rate: float,
) -> LeveragedRunResult:
    dividend_mode = DividendMode(dividend_mode)
    prices = price_frame.close().dropna().sort_index()
    if prices.empty:
        raise ValueError("Dynamic leveraged DCA requires non-empty price data.")

    warnings = _price_warnings(price_frame)
    ledger = MarginLoanLedger(
        price_frame.asset,
        starting_cash=0.0,
        leverage=leverage,
        cost_model=cost_model,
        account_currency=price_frame.asset.currency,
    )
    aligned_dividends, align_warnings = align_dividends_to_trading_dates(
        dividend_frame.data,
        prices.index,
    )
    warnings.extend(f"{price_frame.asset.ticker}: {warning}" for warning in align_warnings)
    dividends_by_date = _dividends_by_effective_date(aligned_dividends)
    contribution_dates = dca_contribution_dates(prices.index, frequency=frequency)
    policy_decisions = _run_dynamic_single_asset_path(
        ledger,
        prices,
        signal_prices=prices,
        dynamic=dynamic,
        dividends_by_date=dividends_by_date,
        dividend_mode=dividend_mode,
        withholding_rate=withholding_rate,
        contribution_dates=contribution_dates,
        contribution=contribution,
    )
    total_contributed = len(contribution_dates) * float(contribution)
    return LeveragedRunResult(
        ticker=price_frame.asset.ticker,
        strategy="dynamic_dca_leveraged",
        dividend_mode=dividend_mode,
        ledger=ledger,
        equity_curve=ledger.equity_curve,
        aligned_dividends=aligned_dividends,
        policy_decisions=decisions_to_frame(policy_decisions),
        price_source=price_frame.source,
        dividend_source=dividend_frame.source,
        total_contributed=float(total_contributed),
        warnings=tuple(warnings),
    )


def run_dynamic_rebalance_leveraged(
    *,
    price_frames: list[PriceFrame],
    dividend_frames: list[DividendFrame],
    cost_model: CostModel,
    initial_cash: float,
    target_weights: dict[str, float],
    frequency: str,
    leverage: LeverageConfig,
    dynamic: DynamicLeverageConfig,
    dividend_mode: DividendMode | str,
    withholding_rate: float,
) -> LeveragedRunResult:
    dividend_mode = DividendMode(dividend_mode)
    if len(price_frames) < 2:
        raise ValueError("Dynamic leveraged rebalance requires at least two price frames.")
    if len(price_frames) != len(dividend_frames):
        raise ValueError("price_frames and dividend_frames must have the same length.")
    tickers = [frame.asset.ticker for frame in price_frames]
    prices = pd.concat([frame.close() for frame in price_frames], axis=1).dropna().sort_index()
    if prices.empty:
        raise ValueError("Dynamic leveraged rebalance requires overlapping price data.")

    warnings: list[str] = []
    for frame in price_frames:
        warnings.extend(_price_warnings(frame))
    aligned_dividends = []
    for frame in dividend_frames:
        aligned, align_warnings = align_dividends_to_trading_dates(frame.data, prices.index)
        if not aligned.empty:
            aligned = aligned.copy()
            aligned.insert(0, "asset", frame.asset.ticker)
            aligned_dividends.append(aligned)
        warnings.extend(f"{frame.asset.ticker}: {warning}" for warning in align_warnings)
    aligned_dividend_frame = (
        pd.concat(aligned_dividends, ignore_index=True)
        if aligned_dividends
        else pd.DataFrame()
    )
    dividends_by_date = _portfolio_dividends_by_effective_date(aligned_dividend_frame)
    ledger = PortfolioMarginLedger(
        [frame.asset for frame in price_frames],
        starting_cash=initial_cash,
        leverage=leverage,
        cost_model=cost_model,
        account_currency=price_frames[0].asset.currency,
        portfolio_label="_".join(tickers),
    )
    policy_signal = weighted_signal_series(prices, target_weights)
    schedule_dates = rebalance_schedule_dates(prices.index, frequency=frequency)
    policy_decisions: list[DynamicLeverageDecision] = []
    previous_date: pd.Timestamp | None = None
    previous_target: float | None = None
    for current_date, row in prices.iterrows():
        current_date = pd.Timestamp(current_date)
        current_prices = {ticker: float(row[ticker]) for ticker in tickers}
        if previous_date is not None:
            days = max(1, (current_date - previous_date).days)
            ledger.accrue_interest(current_date, days=days)
        _apply_portfolio_dividends(
            ledger,
            current_date=current_date,
            current_prices=current_prices,
            rows=dividends_by_date.get(current_date, pd.DataFrame()),
            dividend_mode=dividend_mode,
            withholding_rate=withholding_rate,
        )
        safety_buffer = _portfolio_safety_buffer(ledger, current_prices)
        decision = decide_dynamic_leverage(
            decision_date=current_date,
            signal_prices=policy_signal,
            dynamic=dynamic,
            leverage=leverage,
            safety_buffer=safety_buffer,
        )
        policy_decisions.append(decision)
        should_rebalance = (
            previous_date is None
            or current_date in schedule_dates
            or previous_target is None
            or abs(decision.target_leverage - previous_target) > 1e-9
        )
        if should_rebalance:
            ledger.rebalance_to_weights(
                current_date,
                prices=current_prices,
                target_weights=target_weights,
                target_leverage=decision.target_leverage,
                note=f"dynamic policy: {decision.regime}; {decision.reason}",
            )
        ledger.check_margin_risk(current_date, prices=current_prices)
        ledger.snapshot(current_date, prices=current_prices)
        previous_date = current_date
        previous_target = decision.target_leverage

    return LeveragedRunResult(
        ticker="_".join(tickers),
        strategy="dynamic_rebalance_leveraged",
        dividend_mode=dividend_mode,
        ledger=ledger,
        equity_curve=ledger.equity_curve,
        aligned_dividends=aligned_dividend_frame,
        policy_decisions=decisions_to_frame(policy_decisions),
        price_source="; ".join(f"{frame.asset.ticker}:{frame.source}" for frame in price_frames),
        dividend_source="; ".join(
            f"{frame.asset.ticker}:{frame.source}" for frame in dividend_frames
        ),
        total_contributed=float(initial_cash),
        warnings=tuple(warnings),
    )


def write_leverage_report(
    *,
    results: list[LeveragedRunResult],
    output_dir: Path,
    slug: str,
    base_currency: str,
    usd_twd: pd.Series | None,
    config_path: Path,
    report_context: dict[str, Any],
) -> LeverageReportResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings = sorted({warning for result in results for warning in result.warnings})
    metrics = _build_metrics(results, usd_twd=usd_twd, base_currency=base_currency)
    trades = _collect_frames(results, "trades")
    interest = _collect_frames(results, "interest_events")
    dividends = _collect_frames(results, "dividends")
    leverage_events = _collect_frames(results, "leverage_events")
    cash_flows = _collect_frames(results, "cash_flows")
    curves = _collect_curves(results, usd_twd=usd_twd, base_currency=base_currency)
    positions = _collect_positions(results)
    policy = _collect_policy(results)

    prefix = f"leverage_{slug}"
    metrics_path = output_dir / f"{prefix}_metrics.csv"
    trades_path = output_dir / f"{prefix}_trades.csv"
    interest_path = output_dir / f"{prefix}_interest.csv"
    dividends_path = output_dir / f"{prefix}_dividends.csv"
    events_path = output_dir / f"{prefix}_events.csv"
    cash_flows_path = output_dir / f"{prefix}_cash_flows.csv"
    curves_path = output_dir / f"{prefix}_curve.csv"
    positions_path = output_dir / f"{prefix}_positions.csv"
    policy_path = output_dir / f"{prefix}_policy.csv"
    markdown_path = output_dir / f"{prefix}.md"
    html_path = output_dir / f"{prefix}.html"

    metrics.to_csv(metrics_path, index=False)
    trades.to_csv(trades_path, index=False)
    interest.to_csv(interest_path, index=False)
    dividends.to_csv(dividends_path, index=False)
    leverage_events.to_csv(events_path, index=False)
    cash_flows.to_csv(cash_flows_path, index=False)
    curves.to_csv(curves_path, index=False)
    positions.to_csv(positions_path, index=False)
    policy.to_csv(policy_path, index=False)

    markdown_path.write_text(
        _render_markdown(
            metrics=metrics,
            warnings=warnings,
            config_path=config_path,
            report_context=report_context,
            paths={
                "metrics": metrics_path,
                "trades": trades_path,
                "interest": interest_path,
                "dividends": dividends_path,
                "events": events_path,
                "cash_flows": cash_flows_path,
                "curve": curves_path,
                "positions": positions_path,
                "policy": policy_path,
                "html": html_path,
            },
        ),
        encoding="utf-8",
    )
    html_path.write_text(
        _render_html(
            metrics=metrics,
            curves=curves,
            policy=policy,
            warnings=warnings,
            report_context=report_context,
            paths={
                "metrics": metrics_path,
                "trades": trades_path,
                "interest": interest_path,
                "dividends": dividends_path,
                "events": events_path,
                "cash_flows": cash_flows_path,
                "curve": curves_path,
                "positions": positions_path,
                "policy": policy_path,
            },
        ),
        encoding="utf-8",
    )

    return LeverageReportResult(
        metrics=metrics,
        trades=trades,
        interest=interest,
        dividends=dividends,
        leverage_events=leverage_events,
        cash_flows=cash_flows,
        curves=curves,
        positions=positions,
        policy=policy,
        warnings=warnings,
        markdown_path=markdown_path,
        metrics_path=metrics_path,
        trades_path=trades_path,
        interest_path=interest_path,
        dividends_path=dividends_path,
        events_path=events_path,
        cash_flows_path=cash_flows_path,
        curves_path=curves_path,
        positions_path=positions_path,
        policy_path=policy_path,
        html_path=html_path,
    )


def _run_fixed_target_leverage_path(
    ledger: MarginLoanLedger,
    prices: pd.Series,
    *,
    dividends_by_date: dict[pd.Timestamp, pd.DataFrame],
    dividend_mode: DividendMode,
    withholding_rate: float,
) -> None:
    previous_date: pd.Timestamp | None = None
    for current_date, price in prices.items():
        current_date = pd.Timestamp(current_date)
        current_price = float(price)
        if previous_date is None:
            ledger.initialize_to_target_leverage(current_date, price=current_price)
        else:
            days = max(1, (current_date - previous_date).days)
            ledger.accrue_interest(current_date, days=days)
        _apply_single_asset_dividends(
            ledger,
            current_date=current_date,
            current_price=current_price,
            rows=dividends_by_date.get(current_date, pd.DataFrame()),
            dividend_mode=dividend_mode,
            withholding_rate=withholding_rate,
        )
        ledger.check_margin_risk(current_date, price=current_price)
        ledger.snapshot(current_date, price=current_price)
        previous_date = current_date


def _run_dynamic_single_asset_path(
    ledger: MarginLoanLedger,
    prices: pd.Series,
    *,
    signal_prices: pd.Series,
    dynamic: DynamicLeverageConfig,
    dividends_by_date: dict[pd.Timestamp, pd.DataFrame],
    dividend_mode: DividendMode,
    withholding_rate: float,
    contribution_dates: set[pd.Timestamp],
    contribution: float,
) -> list[DynamicLeverageDecision]:
    previous_date: pd.Timestamp | None = None
    policy_decisions: list[DynamicLeverageDecision] = []
    for current_date, price in prices.items():
        current_date = pd.Timestamp(current_date)
        current_price = float(price)
        if previous_date is not None:
            days = max(1, (current_date - previous_date).days)
            ledger.accrue_interest(current_date, days=days)
        _apply_single_asset_dividends(
            ledger,
            current_date=current_date,
            current_price=current_price,
            rows=dividends_by_date.get(current_date, pd.DataFrame()),
            dividend_mode=dividend_mode,
            withholding_rate=withholding_rate,
        )
        if current_date in contribution_dates:
            ledger.deposit(
                current_date,
                amount=contribution,
                note="dynamic leveraged DCA contribution",
            )
        decision = decide_dynamic_leverage(
            decision_date=current_date,
            signal_prices=signal_prices,
            dynamic=dynamic,
            leverage=ledger.leverage,
            safety_buffer=_single_asset_safety_buffer(ledger, current_price),
        )
        policy_decisions.append(decision)
        ledger.rebalance_to_target_leverage(
            current_date,
            price=current_price,
            target_leverage=decision.target_leverage,
            note=f"dynamic policy: {decision.regime}; {decision.reason}",
        )
        ledger.check_margin_risk(current_date, price=current_price)
        ledger.snapshot(current_date, price=current_price)
        previous_date = current_date
    return policy_decisions


def _single_asset_safety_buffer(ledger: MarginLoanLedger, price: float) -> float | None:
    market_value = ledger.market_value(price)
    if market_value <= 0:
        return None
    equity = ledger.current_equity(price)
    return equity / market_value - ledger.leverage.maintenance_requirement


def _portfolio_safety_buffer(
    ledger: PortfolioMarginLedger,
    prices: dict[str, float],
) -> float | None:
    market_value = ledger.market_value(prices)
    if market_value <= 0:
        return None
    equity = ledger.current_equity(prices)
    return equity / market_value - ledger.leverage.maintenance_requirement


def _static_policy_frame(
    index: pd.Index,
    *,
    target_leverage: float,
    strategy: str,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(index).sort_values()
    return pd.DataFrame(
        {
            "date": [pd.Timestamp(date).date().isoformat() for date in dates],
            "target_leverage": target_leverage,
            "regime": "static",
            "reason": strategy,
            "price": np.nan,
            "trend_average": np.nan,
            "drawdown": np.nan,
            "annualized_volatility": np.nan,
            "safety_buffer": np.nan,
        }
    )


def _dividends_by_effective_date(
    aligned_dividends: pd.DataFrame,
) -> dict[pd.Timestamp, pd.DataFrame]:
    if aligned_dividends.empty:
        return {}
    effective = aligned_dividends.dropna(subset=["effective_date"]).copy()
    if effective.empty:
        return {}
    effective["effective_date"] = pd.to_datetime(effective["effective_date"])
    return {
        pd.Timestamp(effective_date): group
        for effective_date, group in effective.groupby("effective_date")
    }


def _portfolio_dividends_by_effective_date(
    aligned_dividends: pd.DataFrame,
) -> dict[pd.Timestamp, pd.DataFrame]:
    if aligned_dividends.empty:
        return {}
    effective = aligned_dividends.dropna(subset=["effective_date"]).copy()
    if effective.empty:
        return {}
    effective["effective_date"] = pd.to_datetime(effective["effective_date"])
    return {
        pd.Timestamp(effective_date): group
        for effective_date, group in effective.groupby("effective_date")
    }


def _apply_single_asset_dividends(
    ledger: MarginLoanLedger,
    *,
    current_date: pd.Timestamp,
    current_price: float,
    rows: pd.DataFrame,
    dividend_mode: DividendMode,
    withholding_rate: float,
) -> None:
    if rows.empty:
        return
    for _, row in rows.iterrows():
        reinvest = dividend_mode == DividendMode.REINVEST
        ledger.cash_dividend(
            current_date,
            dividend_per_share=float(row["dividend_per_share"]),
            withholding_rate=withholding_rate,
            reinvest=reinvest,
            price=current_price if reinvest else None,
            note=f"source_date={pd.Timestamp(row['dividend_date']).date().isoformat()}",
        )


def _apply_portfolio_dividends(
    ledger: PortfolioMarginLedger,
    *,
    current_date: pd.Timestamp,
    current_prices: dict[str, float],
    rows: pd.DataFrame,
    dividend_mode: DividendMode,
    withholding_rate: float,
) -> None:
    if rows.empty:
        return
    for _, row in rows.iterrows():
        ticker = str(row["asset"])
        reinvest = dividend_mode == DividendMode.REINVEST
        ledger.cash_dividend(
            current_date,
            ticker=ticker,
            dividend_per_share=float(row["dividend_per_share"]),
            withholding_rate=withholding_rate,
            reinvest=reinvest,
            price=current_prices[ticker] if reinvest else None,
            note=f"source_date={pd.Timestamp(row['dividend_date']).date().isoformat()}",
        )


def _price_warnings(price_frame: PriceFrame) -> list[str]:
    warnings: list[str] = []
    if price_frame.adjusted:
        warnings.append(
            f"{price_frame.asset.ticker}: leveraged margin report should use raw prices; "
            "adjusted prices can hide tradable price path details."
        )
    return warnings


def _build_metrics(
    results: list[LeveragedRunResult],
    *,
    usd_twd: pd.Series | None,
    base_currency: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        curve = _curve_with_datetime(result.equity_curve)
        if curve.empty:
            continue
        equity = curve["total_equity"].astype(float)
        returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        has_external_cash_flows = not result.ledger.cash_flows.empty
        summary = (
            performance_summary(returns)
            if not has_external_cash_flows and not returns.empty and equity.iloc[0] > 0
            else pd.Series(dtype="float64")
        )
        total_contributed = float(result.total_contributed)
        ending_equity = float(equity.iloc[-1])
        simple_cash_return = (
            ending_equity / total_contributed - 1.0 if total_contributed > 0 else np.nan
        )
        twd_values = _twd_values(curve, result, usd_twd, base_currency)
        policy_targets = result.policy_decisions["target_leverage"].astype(float)
        rows.append(
            {
                "ticker": result.ticker,
                "strategy": result.strategy,
                "dividend_mode": result.dividend_mode.value,
                "target_leverage": result.ledger.leverage.target_leverage,
                "avg_target_leverage": float(policy_targets.mean()),
                "min_target_leverage": float(policy_targets.min()),
                "max_target_leverage": float(policy_targets.max()),
                "max_leverage_allowed": result.ledger.leverage.max_leverage,
                "annual_borrow_rate": result.ledger.leverage.annual_borrow_rate,
                "maintenance_requirement": result.ledger.leverage.maintenance_requirement,
                "min_safety_buffer": result.ledger.leverage.min_safety_buffer,
                "deleverage_to": result.ledger.leverage.deleverage_to,
                "total_contributed_usd": total_contributed,
                "ending_equity_usd": ending_equity,
                "ending_equity_twd": twd_values["ending_equity_twd"],
                "total_contributed_twd": twd_values["total_contributed_twd"],
                "simple_cash_return": simple_cash_return,
                "total_return": float(summary.get("total_return", np.nan)),
                "cagr": float(summary.get("cagr", np.nan)),
                "volatility": float(summary.get("volatility", np.nan)),
                "sharpe": float(summary.get("sharpe", np.nan)),
                "sortino": float(summary.get("sortino", np.nan)),
                "calmar": float(summary.get("calmar", np.nan)),
                "max_drawdown": float(max_drawdown(returns)) if not returns.empty else np.nan,
                "interest_paid": result.ledger.total_interest_paid,
                "fees_paid": result.ledger.total_fees_paid,
                "gross_dividends": result.ledger.total_gross_dividends,
                "withholding_tax": result.ledger.total_withholding_tax,
                "final_debt": float(curve["debt"].iloc[-1]),
                "final_cash": float(curve["cash"].iloc[-1]),
                "final_shares": float(curve["quantity"].iloc[-1]),
                "max_actual_leverage": float(curve["actual_leverage"].replace(inf, np.nan).max()),
                "worst_safety_buffer": float(curve["safety_buffer"].min()),
                "margin_call_count": len(result.ledger.margin_calls),
                "forced_deleverage_count": len(result.ledger.forced_deleveraging_trades),
                "time_above_target_leverage": int(
                    (
                        curve["actual_leverage"]
                        > result.ledger.leverage.target_leverage + 1e-9
                    ).sum()
                ),
                "price_source": result.price_source,
                "dividend_source": result.dividend_source,
            }
        )
    return pd.DataFrame(rows)


def _twd_values(
    curve: pd.DataFrame,
    result: LeveragedRunResult,
    usd_twd: pd.Series | None,
    base_currency: str,
) -> dict[str, float]:
    if usd_twd is None or base_currency.upper() != "TWD":
        return {"ending_equity_twd": np.nan, "total_contributed_twd": np.nan}
    fx = align_fx_rate(usd_twd, pd.DatetimeIndex(curve.index))
    ending_equity_twd = float(curve["total_equity"].iloc[-1] * fx.iloc[-1])
    cash_flows = result.ledger.cash_flows
    if cash_flows.empty:
        first_fx = float(fx.iloc[0])
        total_contributed_twd = result.total_contributed * first_fx
    else:
        flow_dates = pd.to_datetime(cash_flows["date"])
        flow_fx = align_fx_rate(usd_twd, flow_dates)
        amounts = cash_flows["amount"].astype(float).to_numpy()
        total_contributed_twd = float((amounts * flow_fx.to_numpy()).sum())
    return {
        "ending_equity_twd": ending_equity_twd,
        "total_contributed_twd": total_contributed_twd,
    }


def _collect_frames(results: list[LeveragedRunResult], attr: str) -> pd.DataFrame:
    frames = []
    for result in results:
        frame = getattr(result.ledger, attr)
        if frame.empty:
            continue
        frame = frame.copy()
        frame = frame.dropna(axis=1, how="all")
        frame.insert(0, "ticker", result.ticker)
        frame.insert(1, "strategy", result.strategy)
        frame.insert(2, "dividend_mode", result.dividend_mode.value)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _collect_curves(
    results: list[LeveragedRunResult],
    *,
    usd_twd: pd.Series | None,
    base_currency: str,
) -> pd.DataFrame:
    frames = []
    for result in results:
        curve = result.equity_curve.copy()
        if curve.empty:
            continue
        curve.insert(0, "ticker", result.ticker)
        curve.insert(1, "strategy", result.strategy)
        curve.insert(2, "dividend_mode", result.dividend_mode.value)
        curve_dt = _curve_with_datetime(curve)
        if usd_twd is not None and base_currency.upper() == "TWD":
            fx = align_fx_rate(usd_twd, pd.DatetimeIndex(curve_dt.index))
            curve["usd_twd"] = fx.to_numpy()
            curve["total_equity_twd"] = curve_dt["total_equity"].to_numpy() * fx.to_numpy()
        frames.append(curve)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _collect_positions(results: list[LeveragedRunResult]) -> pd.DataFrame:
    frames = []
    for result in results:
        position_curve = getattr(result.ledger, "position_curve", pd.DataFrame())
        if position_curve.empty:
            continue
        frame = position_curve.copy()
        frame.insert(0, "ticker", result.ticker)
        frame.insert(1, "strategy", result.strategy)
        frame.insert(2, "dividend_mode", result.dividend_mode.value)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _collect_policy(results: list[LeveragedRunResult]) -> pd.DataFrame:
    frames = []
    for result in results:
        frame = result.policy_decisions
        if frame.empty:
            continue
        frame = frame.copy()
        frame.insert(0, "ticker", result.ticker)
        frame.insert(1, "strategy", result.strategy)
        frame.insert(2, "dividend_mode", result.dividend_mode.value)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _curve_with_datetime(frame: pd.DataFrame) -> pd.DataFrame:
    curve = frame.copy()
    if curve.empty:
        return curve
    curve["date"] = pd.to_datetime(curve["date"])
    return curve.set_index("date").sort_index()


def _render_markdown(
    *,
    metrics: pd.DataFrame,
    warnings: list[str],
    config_path: Path,
    report_context: dict[str, Any],
    paths: dict[str, Path],
) -> str:
    lines = [
        "# 輕槓桿風險模型報表",
        "",
        "這份報表用 margin loan 研究模型檢查借款、利息、維持率、安全緩衝與自動降槓桿。",
        "",
        "## 設定",
        "",
        f"- Config: `{config_path}`",
        f"- 期間: {report_context.get('start_date')} 到 {report_context.get('end_date')}",
        f"- 標的: {', '.join(report_context.get('tickers', []))}",
        f"- 策略: {', '.join(report_context.get('strategies', []))}",
        f"- 股息模式: {', '.join(report_context.get('dividend_modes', []))}",
        f"- 目標槓桿: {report_context.get('target_leverage')}",
        f"- 借款年利率: {report_context.get('annual_borrow_rate')}",
        f"- 維持率: {report_context.get('maintenance_requirement')}",
        f"- 安全緩衝門檻: {report_context.get('min_safety_buffer')}",
        f"- 動態趨勢窗口: {report_context.get('dynamic_trend_window')}",
        f"- 動態波動窗口: {report_context.get('dynamic_volatility_window')}",
        f"- 動態 crash guard: {report_context.get('dynamic_crash_guard')}",
        "",
        "## 關鍵指標",
        "",
        f"```csv\n{metrics.to_csv(index=False).strip()}\n```"
        if not metrics.empty
        else "_No metrics._",
        "",
        "## 輸出檔案",
        "",
    ]
    for label, path in paths.items():
        lines.append(f"- {label}: `{path}`")
    if warnings:
        lines.extend(["", "## 限制與警告", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines) + "\n"


def _render_html(
    *,
    metrics: pd.DataFrame,
    curves: pd.DataFrame,
    policy: pd.DataFrame,
    warnings: list[str],
    report_context: dict[str, Any],
    paths: dict[str, Path],
) -> str:
    font_href = (
        "https://fonts.googleapis.com/css2?"
        "family=Noto+Sans+JP:wght@400;500;600;700&"
        "family=Noto+Sans+TC:wght@400;500;600;700&display=swap"
    )
    period_value = f"{report_context.get('start_date')} 到 {report_context.get('end_date')}"
    tickers_value = ", ".join(report_context.get("tickers", []))
    strategies_value = ", ".join(report_context.get("strategies", []))
    dividend_modes_value = ", ".join(report_context.get("dividend_modes", []))
    settings = [
        ("期間", period_value),
        ("標的", tickers_value),
        ("策略", strategies_value),
        ("股息模式", dividend_modes_value),
        ("目標槓桿", str(report_context.get("target_leverage"))),
        ("借款年利率", str(report_context.get("annual_borrow_rate"))),
        ("維持率", str(report_context.get("maintenance_requirement"))),
        ("安全緩衝門檻", str(report_context.get("min_safety_buffer"))),
        ("降槓桿目標", str(report_context.get("deleverage_to"))),
        ("動態趨勢窗口", str(report_context.get("dynamic_trend_window", ""))),
        ("動態波動窗口", str(report_context.get("dynamic_volatility_window", ""))),
        ("高波動門檻", str(report_context.get("dynamic_high_volatility", ""))),
        ("Crash Guard", str(report_context.get("dynamic_crash_guard", ""))),
    ]
    settings_html = "\n".join(_setting_tile(label, value) for label, value in settings)
    navigation_html = _render_leverage_navigation()
    family_sections = _render_leverage_family_sections(metrics, curves)
    normalized_section = _render_leverage_normalized_section(curves)
    risk_section = _render_leverage_risk_section(metrics, curves, policy)
    audit_section = _render_leverage_audit_section(paths)
    warnings_html = _render_leverage_warnings(warnings)
    style = _leverage_dashboard_css()

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>輕槓桿 Dashboard | Margin Loan Risk Report</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="{font_href}" rel="stylesheet">
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>{style}</style>
</head>
<body>
<main class="dashboard-shell">
  <header class="dashboard-header">
    <div>
      <p class="eyebrow">Margin Loan Risk Report</p>
      <h1>輕槓桿 Dashboard</h1>
      <p class="header-copy">
        先看是否安全，再談報酬最佳化。這份報表檢查 margin loan 借款、
        每日利息、維持率、安全緩衝與自動降槓桿。
      </p>
    </div>
    <div class="header-meta">
      <span>閱讀原則</span>
      <strong>B&H、DCA、再平衡分開比較</strong>
    </div>
  </header>

  <section class="section-block section-tight">
    <h2>設定總覽</h2>
    <p class="section-copy">這些假設會直接影響利息、可用安全緩衝與強制降槓桿機率。</p>
    <div class="settings-grid">{settings_html}</div>
  </section>

  {navigation_html}
  {family_sections}
  {normalized_section}
  {risk_section}
  {audit_section}
  {warnings_html}
  {_scenario_switcher_script()}
</main>
</body>
</html>
"""


def _render_leverage_navigation() -> str:
    items = [
        ("#view-buy-hold", "B&H 一次投入", "固定/動態槓桿的一次投入情境"),
        ("#view-dca", "DCA 定期投入", "有外部現金流，獨立閱讀"),
        ("#view-rebalance", "再平衡", "多資產組合與目標權重"),
        ("#view-leverage-risk", "槓桿風險", "負債、利息、安全緩衝與 margin call"),
        ("#view-audit", "Audit 明細", "CSV 與 policy 決策表"),
    ]
    links = "\n".join(
        f"""<a class="nav-card" href="{href}">
  <strong>{escape(title)}</strong>
  <span>{escape(description)}</span>
</a>"""
        for href, title, description in items
    )
    return f"""<nav class="section-block dashboard-nav" aria-label="Dashboard views">
  <div class="section-heading">
    <div>
      <p class="eyebrow">View Switcher</p>
      <h2>雙層導覽</h2>
    </div>
    <p>先選策略族群，再看固定/動態槓桿、標的與股息模式；不同本金口徑不放在同一個排名。</p>
  </div>
  <div class="nav-grid">{links}</div>
</nav>"""


def _render_leverage_family_sections(metrics: pd.DataFrame, curves: pd.DataFrame) -> str:
    families = [
        (
            "view-buy-hold",
            "Lump Sum",
            "B&H 一次投入",
            (
                "比較 buy_hold_leveraged 與 dynamic_buy_hold_leveraged。"
                "這一區和 DCA 分開，避免本金口徑混淆。"
            ),
            ["buy_hold_leveraged", "dynamic_buy_hold_leveraged"],
            "buy_hold",
        ),
        (
            "view-dca",
            "Cash Flow",
            "DCA 定期投入",
            "DCA 有多次外部投入，主要看累計投入、期末資產、Simple Cash Return 與安全緩衝。",
            ["dca_leveraged", "dynamic_dca_leveraged"],
            "dca",
        ),
        (
            "view-rebalance",
            "Portfolio",
            "再平衡",
            "比較固定槓桿與動態槓桿在 SPY/QQQ 再平衡組合上的風險路徑。",
            ["rebalance_leveraged", "dynamic_rebalance_leveraged"],
            "rebalance",
        ),
    ]
    return "\n".join(
        _render_leverage_family_section(
            section_id=section_id,
            eyebrow=eyebrow,
            title=title,
            description=description,
            strategies=strategies,
            focus=focus,
            metrics=metrics,
            curves=curves,
        )
        for section_id, eyebrow, title, description, strategies, focus in families
    )


def _render_leverage_family_section(
    *,
    section_id: str,
    eyebrow: str,
    title: str,
    description: str,
    strategies: list[str],
    focus: str,
    metrics: pd.DataFrame,
    curves: pd.DataFrame,
) -> str:
    family_metrics = _filter_strategies(metrics, strategies)
    family_curves = _filter_strategies(curves, strategies)
    chips = _render_leverage_chips(family_metrics)
    default_key = _default_scenario_key(family_metrics, focus=focus)
    selector = _render_scenario_selector(
        family_metrics,
        group_id=focus,
        default_key=default_key,
    )
    scenario_panels = _render_leverage_scenario_panels(
        metrics=family_metrics,
        curves=family_curves,
        group_id=focus,
        default_key=default_key,
        focus=focus,
    )
    table = _render_leverage_metrics_table(family_metrics, focus=focus)
    return f"""<section class="section-block family-section" id="{section_id}">
  <div class="section-heading">
    <div>
      <p class="eyebrow">{escape(eyebrow)}</p>
      <h2>{escape(title)}</h2>
    </div>
    <p>{escape(description)}</p>
  </div>
  {chips}
  {selector}
  {scenario_panels}
  {table}
</section>"""


def _render_leverage_chips(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        return '<div class="filter-row"><span>目前沒有這個策略族群的資料</span></div>'
    tickers = sorted(metrics["ticker"].dropna().unique()) if "ticker" in metrics else []
    modes = (
        sorted(metrics["dividend_mode"].dropna().unique())
        if "dividend_mode" in metrics
        else []
    )
    strategies = sorted(metrics["strategy"].dropna().unique()) if "strategy" in metrics else []
    chips = [
        f"標的: {', '.join(map(str, tickers))}",
        f"策略: {', '.join(_strategy_label(strategy) for strategy in strategies)}",
        f"股息: {', '.join(map(str, modes))}",
    ]
    return "<div class=\"filter-row\">" + "".join(
        f"<span>{escape(chip)}</span>" for chip in chips
    ) + "</div>"


def _render_scenario_selector(
    metrics: pd.DataFrame,
    *,
    group_id: str,
    default_key: str,
) -> str:
    if metrics.empty:
        return ""
    buttons: list[str] = []
    for row in metrics.itertuples():
        key = _scenario_key(row.ticker, row.strategy, row.dividend_mode)
        active = " is-active" if key == default_key else ""
        buttons.append(
            f"""<button class="scenario-button{active}" type="button"
    data-scenario-group="{escape(group_id)}"
    data-scenario-button="{escape(key)}">
  <span>{escape(_scenario_button_title(row))}</span>
  <small>{escape(_scenario_button_subtitle(row))}</small>
</button>"""
        )
    return f"""<div class="scenario-selector" aria-label="情境選擇">
  {"".join(buttons)}
</div>"""


def _render_leverage_scenario_panels(
    *,
    metrics: pd.DataFrame,
    curves: pd.DataFrame,
    group_id: str,
    default_key: str,
    focus: str,
) -> str:
    if metrics.empty:
        return '<p class="empty-state">沒有可顯示的情境。</p>'
    panels: list[str] = []
    for row in metrics.itertuples():
        key = _scenario_key(row.ticker, row.strategy, row.dividend_mode)
        active = " is-active" if key == default_key else ""
        scenario_curve = _curve_for_scenario(
            curves,
            ticker=str(row.ticker),
            strategy=str(row.strategy),
            dividend_mode=str(row.dividend_mode),
        )
        summary = _render_scenario_summary(row)
        kpis = _render_single_scenario_kpis(row, focus=focus)
        charts = _render_single_scenario_charts(
            scenario_curve,
            focus=focus,
            scenario_label=_scenario_full(row.ticker, row.strategy, row.dividend_mode),
        )
        panels.append(
            f"""<div class="scenario-panel{active}"
    data-scenario-group="{escape(group_id)}"
    data-scenario-panel="{escape(key)}">
  {summary}
  {kpis}
  {charts}
</div>"""
        )
    return "\n".join(panels)


def _render_scenario_summary(row: Any) -> str:
    items = [
        ("情境", _scenario_full(row.ticker, row.strategy, row.dividend_mode)),
        ("投入本金", f"{_format_money(row.total_contributed_usd)} USD"),
        ("期末資產", f"{_format_money(row.ending_equity_usd)} USD"),
        ("期末負債", f"{_format_money(row.final_debt)} USD"),
        ("最差安全緩衝", _format_percent(row.worst_safety_buffer)),
        ("Margin Call", str(row.margin_call_count)),
    ]
    tiles = "\n".join(
        f"""<div class="mini-stat">
  <span>{escape(label)}</span>
  <strong>{escape(value)}</strong>
</div>"""
        for label, value in items
    )
    return f"""<article class="scenario-summary">
  <div>
    <p class="eyebrow">Selected Scenario</p>
    <h3>{escape(_scenario_button_title(row))}</h3>
  </div>
  <div class="mini-stat-grid">{tiles}</div>
</article>"""


def _render_single_scenario_kpis(row: Any, *, focus: str) -> str:
    if focus == "dca":
        cards = [
            ("累計投入", _format_money(row.total_contributed_usd), "USD 外部現金流"),
            ("期末資產", _format_money(row.ending_equity_usd), "USD"),
            ("Simple Cash Return", _format_percent(row.simple_cash_return), "DCA 主要口徑"),
            ("最高實際槓桿", _format_leverage(row.max_actual_leverage), "路徑最高值"),
        ]
    else:
        cards = [
            ("CAGR", _format_percent(row.cagr), "無外部現金流時較適用"),
            ("Sharpe", _format_number(row.sharpe), "風險調整報酬"),
            ("最大回撤", _format_percent(row.max_drawdown), "槓桿後回撤"),
            ("利息成本", _format_money(row.interest_paid), "USD"),
        ]
    cards_html = "\n".join(
        f"""<article class="kpi-card compact-kpi">
  <span>{escape(label)}</span>
  <strong>{escape(value)}</strong>
  <small>{escape(note)}</small>
</article>"""
        for label, value, note in cards
    )
    return f'<div class="kpi-grid compact-kpi-grid">{cards_html}</div>'


def _render_single_scenario_charts(
    curve: pd.DataFrame,
    *,
    focus: str,
    scenario_label: str,
) -> str:
    chart_specs = [
        (
            f"{focus}-equity-single",
            "權益曲線",
            "只顯示目前選取情境，不再把所有策略塞在同一張圖。",
            _build_single_curve_figure(
                curve,
                value_column="total_equity",
                yaxis_title="USD",
                scenario_label=scenario_label,
            ),
        ),
        (
            f"{focus}-safety-single",
            "安全緩衝",
            "安全緩衝越接近 0 越危險；低於門檻會觸發降槓桿。",
            _build_single_curve_figure(
                curve,
                value_column="safety_buffer",
                yaxis_title="Safety Buffer",
                scenario_label=scenario_label,
                percent=True,
            ),
        ),
    ]
    return '<div class="chart-grid two-up">' + "\n".join(
        _render_chart_card(chart_id, title, description, figure)
        for chart_id, title, description, figure in chart_specs
    ) + "</div>"


def _render_leverage_family_kpis(metrics: pd.DataFrame, *, focus: str) -> str:
    if metrics.empty:
        return '<p class="empty-state">沒有可顯示的策略資料。</p>'
    focus_row = metrics.loc[metrics["ending_equity_usd"].astype(float).idxmax()]
    if focus == "dca":
        cards = [
            (
                "累計投入",
                _format_money(focus_row.total_contributed_usd),
                "USD 外部現金流",
            ),
            ("期末資產", _format_money(focus_row.ending_equity_usd), "USD"),
            (
                "Simple Cash Return",
                _format_percent(focus_row.simple_cash_return),
                "DCA 主要報酬口徑",
            ),
            ("最大實際槓桿", _format_leverage(focus_row.max_actual_leverage), "路徑最高值"),
            ("最差安全緩衝", _format_percent(focus_row.worst_safety_buffer), "越高越安全"),
            ("利息成本", _format_money(focus_row.interest_paid), "USD"),
        ]
    else:
        cards = [
            ("期末資產", _format_money(focus_row.ending_equity_usd), "USD"),
            ("投入本金", _format_money(focus_row.total_contributed_usd), "USD"),
            ("CAGR", _format_percent(focus_row.cagr), "無外部現金流時較適用"),
            ("Sharpe", _format_number(focus_row.sharpe), "風險調整報酬"),
            ("最大回撤", _format_percent(focus_row.max_drawdown), "槓桿後回撤"),
            ("最差安全緩衝", _format_percent(focus_row.worst_safety_buffer), "margin safety"),
        ]
    cards_html = "\n".join(
        f"""<article class="kpi-card">
  <span>{escape(label)}</span>
  <strong>{escape(value)}</strong>
  <small>{escape(note)}</small>
</article>"""
        for label, value, note in cards
    )
    return f'<div class="kpi-grid">{cards_html}</div>'


def _render_leverage_metrics_table(metrics: pd.DataFrame, *, focus: str) -> str:
    if metrics.empty:
        return '<p class="empty-state">沒有情境指標。</p>'
    display = metrics.copy()
    display.insert(
        0,
        "scenario",
        [
            _scenario_full(row.ticker, row.strategy, row.dividend_mode)
            for row in display.itertuples()
        ],
    )
    money_columns = [
        "total_contributed_usd",
        "ending_equity_usd",
        "ending_equity_twd",
        "total_contributed_twd",
        "interest_paid",
        "fees_paid",
        "gross_dividends",
        "withholding_tax",
        "final_debt",
        "final_cash",
    ]
    percent_columns = [
        "simple_cash_return",
        "cagr",
        "max_drawdown",
        "worst_safety_buffer",
        "avg_target_leverage",
        "min_target_leverage",
        "max_target_leverage",
        "max_actual_leverage",
    ]
    for column in money_columns:
        if column in display.columns:
            display[column] = display[column].map(_format_money)
    for column in percent_columns:
        if column in display.columns:
            if column.endswith("leverage"):
                display[column] = display[column].map(_format_leverage)
            else:
                display[column] = display[column].map(_format_percent)
    for column in ["sharpe", "sortino", "calmar"]:
        if column in display.columns:
            display[column] = display[column].map(_format_number)
    if focus == "dca":
        columns = [
            ("scenario", "情境"),
            ("total_contributed_usd", "累計投入 USD"),
            ("ending_equity_usd", "期末資產 USD"),
            ("simple_cash_return", "Simple Cash Return"),
            ("interest_paid", "利息"),
            ("final_debt", "期末負債"),
            ("max_actual_leverage", "最高實際槓桿"),
            ("worst_safety_buffer", "最差安全緩衝"),
            ("margin_call_count", "Margin Call"),
        ]
    else:
        columns = [
            ("scenario", "情境"),
            ("total_contributed_usd", "投入本金 USD"),
            ("ending_equity_usd", "期末資產 USD"),
            ("cagr", "CAGR"),
            ("sharpe", "Sharpe"),
            ("calmar", "Calmar"),
            ("max_drawdown", "最大回撤"),
            ("interest_paid", "利息"),
            ("final_debt", "期末負債"),
            ("worst_safety_buffer", "最差安全緩衝"),
            ("margin_call_count", "Margin Call"),
        ]
    return _html_table(display, columns)


def _render_leverage_family_charts(
    curves: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    focus: str,
) -> str:
    chart_specs = [
        (
            f"{focus}-equity",
            "USD 權益曲線",
            "同一策略族群內比較，不和不同投入本金的情境混排。",
            _build_leverage_equity_figure(curves),
        ),
        (
            f"{focus}-debt",
            "負債與安全緩衝",
            "實線為 debt，虛線為 safety buffer；安全緩衝越接近 0 越危險。",
            _build_debt_safety_figure(curves),
        ),
        (
            f"{focus}-contribution",
            "投入本金 vs 期末資產",
            "DCA 用累計投入；B&H / 再平衡用初始投入。",
            _build_leverage_contribution_figure(metrics),
        ),
    ]
    return "\n".join(
        _render_chart_card(chart_id, title, description, figure)
        for chart_id, title, description, figure in chart_specs
    )


def _render_leverage_normalized_section(curves: pd.DataFrame) -> str:
    chart_specs = [
        (
            "normalized-buy-hold",
            "B&H 標準化路徑",
            ["buy_hold_leveraged", "dynamic_buy_hold_leveraged"],
            "SPY",
        ),
        (
            "normalized-dca",
            "DCA 標準化路徑",
            ["dca_leveraged", "dynamic_dca_leveraged"],
            "SPY",
        ),
        (
            "normalized-rebalance",
            "再平衡標準化路徑",
            ["rebalance_leveraged", "dynamic_rebalance_leveraged"],
            "SPY_QQQ",
        ),
    ]
    charts = "\n".join(
        _render_chart_card(
            chart_id,
            title,
            "最多四條線：固定/動態 × 現金/再投。用來看路徑形狀，不看實際本金。",
            _build_normalized_leverage_figure(
                curves,
                strategies=strategies,
                preferred_ticker=preferred_ticker,
            ),
        )
        for chart_id, title, strategies, preferred_ticker in chart_specs
    )
    return f"""<section class="section-block" id="view-normalized">
  <div class="section-heading">
    <div>
      <p class="eyebrow">Shape Only</p>
      <h2>標準化比較</h2>
    </div>
    <p>B&H、DCA、再平衡本金口徑不同；這裡全部重標成 10,000 USD 起點，只看風險路徑。</p>
  </div>
  <div class="notice-card warning-notice">
    <strong>非實際投入結果，不可當作本金報酬排名。</strong>
    <p>真正的投入與收益請回各策略族群看累計投入、期末資產和現金流。</p>
  </div>
  <div class="chart-grid three-up">{charts}</div>
</section>"""


def _render_leverage_risk_section(
    metrics: pd.DataFrame,
    curves: pd.DataFrame,
    policy: pd.DataFrame,
) -> str:
    risk_table = _render_leverage_risk_table(metrics)
    default_key = _default_scenario_key(metrics, focus="risk")
    selector = _render_scenario_selector(metrics, group_id="risk", default_key=default_key)
    risk_panels = _render_risk_scenario_panels(
        metrics=metrics,
        curves=curves,
        policy=policy,
        default_key=default_key,
    )
    return f"""<section class="section-block" id="view-leverage-risk">
  <div class="section-heading">
    <div>
      <p class="eyebrow">Margin Safety</p>
      <h2>槓桿風險</h2>
    </div>
    <p>這區不追求最高 CAGR，而是先看借款、利息、安全緩衝與 margin call 是否可接受。</p>
  </div>
  {selector}
  {risk_panels}
  {risk_table}
</section>"""


def _render_risk_scenario_panels(
    *,
    metrics: pd.DataFrame,
    curves: pd.DataFrame,
    policy: pd.DataFrame,
    default_key: str,
) -> str:
    if metrics.empty:
        return '<p class="empty-state">沒有槓桿風險資料。</p>'
    panels: list[str] = []
    for row in metrics.itertuples():
        key = _scenario_key(row.ticker, row.strategy, row.dividend_mode)
        active = " is-active" if key == default_key else ""
        curve = _curve_for_scenario(
            curves,
            ticker=str(row.ticker),
            strategy=str(row.strategy),
            dividend_mode=str(row.dividend_mode),
        )
        policy_preview = _render_policy_preview(
            _policy_for_scenario(
                policy,
                ticker=str(row.ticker),
                strategy=str(row.strategy),
                dividend_mode=str(row.dividend_mode),
            )
        )
        charts = "\n".join(
            [
                _render_chart_card(
                    "risk-debt",
                    "Debt 負債",
                    "只看目前選取情境的借款餘額，避免和 safety buffer 混用雙軸。",
                    _build_single_curve_figure(
                        curve,
                        value_column="debt",
                        yaxis_title="Debt USD",
                        scenario_label=_scenario_full(row.ticker, row.strategy, row.dividend_mode),
                    ),
                ),
                _render_chart_card(
                    "risk-safety",
                    "Safety Buffer 安全緩衝",
                    "安全緩衝越接近 0 越危險。",
                    _build_single_curve_figure(
                        curve,
                        value_column="safety_buffer",
                        yaxis_title="Safety Buffer",
                        scenario_label=_scenario_full(row.ticker, row.strategy, row.dividend_mode),
                        percent=True,
                    ),
                ),
                _render_chart_card(
                    "risk-target-actual",
                    "目標槓桿 vs 實際槓桿",
                    "實線為 actual，虛線為 target；只顯示目前選取情境。",
                    _build_single_actual_target_figure(
                        curve,
                        scenario_label=_scenario_full(row.ticker, row.strategy, row.dividend_mode),
                    ),
                ),
            ]
        )
        panels.append(
            f"""<div class="scenario-panel{active}"
    data-scenario-group="risk"
    data-scenario-panel="{escape(key)}">
  {_render_scenario_summary(row)}
  <div class="chart-grid three-up">{charts}</div>
  {policy_preview}
</div>"""
        )
    return "\n".join(panels)


def _render_leverage_risk_table(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        return '<p class="empty-state">沒有槓桿風險指標。</p>'
    display = metrics.copy()
    display.insert(
        0,
        "scenario",
        [
            _scenario_full(row.ticker, row.strategy, row.dividend_mode)
            for row in display.itertuples()
        ],
    )
    for column in ["interest_paid", "final_debt", "ending_equity_usd"]:
        if column in display.columns:
            display[column] = display[column].map(_format_money)
    for column in ["worst_safety_buffer", "max_actual_leverage", "avg_target_leverage"]:
        if column in display.columns:
            display[column] = display[column].map(
                _format_leverage if column.endswith("leverage") else _format_percent
            )
    columns = [
        ("scenario", "情境"),
        ("ending_equity_usd", "期末資產"),
        ("interest_paid", "利息"),
        ("final_debt", "期末負債"),
        ("avg_target_leverage", "平均目標槓桿"),
        ("max_actual_leverage", "最高實際槓桿"),
        ("worst_safety_buffer", "最差安全緩衝"),
        ("margin_call_count", "Margin Call"),
        ("forced_deleverage_count", "強制降槓桿"),
        ("time_above_target_leverage", "高於目標天數"),
    ]
    return _html_table(display, columns)


def _render_policy_preview(policy: pd.DataFrame) -> str:
    if policy.empty:
        return ""
    preview = policy.tail(10).copy()
    for column in ["target_leverage", "drawdown", "annualized_volatility", "safety_buffer"]:
        if column not in preview.columns:
            continue
        if column == "target_leverage":
            preview[column] = preview[column].map(_format_leverage)
        else:
            preview[column] = preview[column].map(_format_percent)
    table = _html_table(
        preview,
        [
            ("date", "日期"),
            ("ticker", "標的"),
            ("strategy", "策略"),
            ("target_leverage", "目標槓桿"),
            ("regime", "狀態"),
            ("reason", "原因"),
            ("drawdown", "回撤"),
            ("annualized_volatility", "波動"),
            ("safety_buffer", "安全緩衝"),
        ],
    )
    return f"""<article class="detail-card">
  <h3>動態槓桿決策預覽</h3>
  <p>最近幾筆 policy decision；完整版本請下載 policy CSV。</p>
  {table}
</article>"""


def _render_leverage_audit_section(paths: dict[str, Path]) -> str:
    links = "\n".join(
        f'<a class="download-link" href="{escape(path.name)}">{escape(label)} CSV</a>'
        for label, path in paths.items()
    )
    return f"""<section class="section-block" id="view-audit">
  <div class="section-heading">
    <div>
      <p class="eyebrow">Audit Trail</p>
      <h2>Audit 明細與 CSV 下載</h2>
    </div>
    <p>HTML 用來閱讀，CSV 用來追查每筆交易、利息、股息、槓桿事件與 policy 決策。</p>
  </div>
  <div class="download-row">{links}</div>
</section>"""


def _render_leverage_warnings(warnings: list[str]) -> str:
    warning_items = "".join(
        f"<li>{escape(warning)}</li>" for warning in warnings
    ) or "<li>目前沒有資料對齊警告。</li>"
    return f"""<section class="section-block limitations">
  <div class="section-heading">
    <div>
      <p class="eyebrow">Limitations</p>
      <h2>限制與警告</h2>
    </div>
    <p>這是研究模型，不是券商保證金規則或投資建議；實際券商維持率可能更嚴格。</p>
  </div>
  <ul>
    {warning_items}
    <li>槓桿 ETF 產品內部期貨、swap 與每日重設機制沒有在 margin loan ledger 內建模。</li>
    <li>DCA 有外部現金流，CAGR / Sharpe 不作主要排序依據。</li>
  </ul>
</section>"""


def _setting_tile(label: str, value: str) -> str:
    return (
        '<div class="tile">'
        f'<div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(value)}</div>'
        "</div>"
    )


def _build_single_curve_figure(
    curve: pd.DataFrame,
    *,
    value_column: str,
    yaxis_title: str,
    scenario_label: str,
    percent: bool = False,
) -> Any:
    figure = go.Figure()
    if curve.empty or value_column not in curve.columns:
        _add_empty_annotation(figure, f"沒有 {value_column} 資料")
        return _style_leverage_figure(figure, yaxis_title=yaxis_title, showlegend=False)
    figure.add_trace(
        go.Scatter(
            x=pd.to_datetime(curve["date"]),
            y=curve[value_column],
            mode="lines",
            name=yaxis_title,
            line={"color": "#3f5f73", "width": 2.2},
            hovertemplate=(
                f"{escape(scenario_label)}<br>%{{x|%Y-%m-%d}}"
                f"<br>{escape(yaxis_title)}: %{{y:{'.2%' if percent else ',.2f'}}}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )
    figure = _style_leverage_figure(figure, yaxis_title=yaxis_title, showlegend=False)
    if percent:
        figure.update_yaxes(tickformat=".0%")
    return figure


def _build_single_actual_target_figure(
    curve: pd.DataFrame,
    *,
    scenario_label: str,
) -> Any:
    figure = go.Figure()
    if curve.empty or "actual_leverage" not in curve.columns:
        _add_empty_annotation(figure, "沒有槓桿資料")
        return _style_leverage_figure(figure, yaxis_title="Leverage", showlegend=False)
    dates = pd.to_datetime(curve["date"])
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=curve["actual_leverage"],
            mode="lines",
            name="Actual",
            line={"color": "#3f5f73", "width": 2.3},
            hovertemplate=(
                f"{escape(scenario_label)}<br>%{{x|%Y-%m-%d}}"
                "<br>actual: %{y:.2f}x<extra></extra>"
            ),
            showlegend=False,
        )
    )
    if "target_leverage" in curve.columns:
        figure.add_trace(
            go.Scatter(
                x=dates,
                y=curve["target_leverage"],
                mode="lines",
                name="Target",
                line={"color": "#b66f52", "width": 1.8, "dash": "dash"},
                hovertemplate=(
                    f"{escape(scenario_label)}<br>%{{x|%Y-%m-%d}}"
                    "<br>target: %{y:.2f}x<extra></extra>"
                ),
                showlegend=False,
            )
        )
    return _style_leverage_figure(figure, yaxis_title="Leverage", showlegend=False)


def _build_leverage_equity_figure(curves: pd.DataFrame) -> Any:
    figure = go.Figure()
    has_trace = False
    color_map = _leverage_color_map(curves)
    for key, group in _iter_curve_groups(curves):
        if "total_equity" not in group.columns:
            continue
        has_trace = True
        figure.add_trace(
            go.Scatter(
                x=pd.to_datetime(group["date"]),
                y=group["total_equity"],
                mode="lines",
                name=_scenario_short(*key),
                legendgroup=_scenario_id(*key),
                line={"color": color_map[key], "width": 2.2},
                hovertemplate=(
                    f"{_scenario_full(*key)}<br>%{{x|%Y-%m-%d}}"
                    "<br>USD equity: %{y:,.2f}<extra></extra>"
                ),
            )
        )
    if not has_trace:
        _add_empty_annotation(figure, "沒有可繪製的權益曲線")
    return _style_leverage_figure(figure, yaxis_title="USD")


def _build_debt_safety_figure(curves: pd.DataFrame) -> Any:
    figure = go.Figure()
    has_trace = False
    color_map = _leverage_color_map(curves)
    for key, group in _iter_curve_groups(curves):
        dates = pd.to_datetime(group["date"])
        color = color_map[key]
        if "debt" in group.columns:
            has_trace = True
            figure.add_trace(
                go.Scatter(
                    x=dates,
                    y=group["debt"],
                    mode="lines",
                    name=f"{_scenario_short(*key)} debt",
                    legendgroup=_scenario_id(*key),
                    line={"color": color, "width": 2},
                    hovertemplate=(
                        f"{_scenario_full(*key)}<br>%{{x|%Y-%m-%d}}"
                        "<br>debt: %{y:,.2f}<extra></extra>"
                    ),
                )
            )
        if "safety_buffer" in group.columns:
            has_trace = True
            figure.add_trace(
                go.Scatter(
                    x=dates,
                    y=group["safety_buffer"],
                    mode="lines",
                    name=f"{_scenario_short(*key)} safety",
                    legendgroup=_scenario_id(*key),
                    yaxis="y2",
                    line={"color": color, "width": 1.8, "dash": "dash"},
                    hovertemplate=(
                        f"{_scenario_full(*key)}<br>%{{x|%Y-%m-%d}}"
                        "<br>safety buffer: %{y:.2%}<extra></extra>"
                    ),
                )
            )
    if not has_trace:
        _add_empty_annotation(figure, "沒有可繪製的負債或安全緩衝資料")
    figure = _style_leverage_figure(figure, yaxis_title="Debt USD")
    figure.update_layout(
        yaxis2={
            "title": "Safety Buffer",
            "overlaying": "y",
            "side": "right",
            "tickformat": ".0%",
            "gridcolor": "rgba(0,0,0,0)",
        }
    )
    return figure


def _build_actual_leverage_figure(curves: pd.DataFrame) -> Any:
    figure = go.Figure()
    has_trace = False
    color_map = _leverage_color_map(curves)
    for key, group in _iter_curve_groups(curves):
        if "actual_leverage" not in group.columns:
            continue
        has_trace = True
        figure.add_trace(
            go.Scatter(
                x=pd.to_datetime(group["date"]),
                y=group["actual_leverage"],
                mode="lines",
                name=f"{_scenario_short(*key)} actual",
                legendgroup=_scenario_id(*key),
                line={"color": color_map[key], "width": 2},
                hovertemplate=(
                    f"{_scenario_full(*key)}<br>%{{x|%Y-%m-%d}}"
                    "<br>actual leverage: %{y:.2f}x<extra></extra>"
                ),
            )
        )
        if "target_leverage" in group.columns:
            figure.add_trace(
                go.Scatter(
                    x=pd.to_datetime(group["date"]),
                    y=group["target_leverage"],
                    mode="lines",
                    name=f"{_scenario_short(*key)} target",
                    legendgroup=_scenario_id(*key),
                    line={"color": color_map[key], "width": 1.6, "dash": "dash"},
                    hovertemplate=(
                        f"{_scenario_full(*key)}<br>%{{x|%Y-%m-%d}}"
                        "<br>target leverage: %{y:.2f}x<extra></extra>"
                    ),
                )
            )
    if not has_trace:
        _add_empty_annotation(figure, "沒有可繪製的槓桿資料")
    return _style_leverage_figure(figure, yaxis_title="Leverage")


def _build_safety_buffer_figure(curves: pd.DataFrame) -> Any:
    figure = go.Figure()
    has_trace = False
    color_map = _leverage_color_map(curves)
    for key, group in _iter_curve_groups(curves):
        if "safety_buffer" not in group.columns:
            continue
        has_trace = True
        figure.add_trace(
            go.Scatter(
                x=pd.to_datetime(group["date"]),
                y=group["safety_buffer"],
                mode="lines",
                name=_scenario_short(*key),
                legendgroup=_scenario_id(*key),
                line={"color": color_map[key], "width": 2},
                hovertemplate=(
                    f"{_scenario_full(*key)}<br>%{{x|%Y-%m-%d}}"
                    "<br>safety buffer: %{y:.2%}<extra></extra>"
                ),
            )
        )
    if not has_trace:
        _add_empty_annotation(figure, "沒有可繪製的安全緩衝資料")
    figure = _style_leverage_figure(figure, yaxis_title="Safety Buffer")
    figure.update_yaxes(tickformat=".0%")
    return figure


def _build_leverage_contribution_figure(metrics: pd.DataFrame) -> Any:
    figure = go.Figure()
    if metrics.empty:
        _add_empty_annotation(figure, "沒有可比較的投入與期末資產資料")
        return _style_leverage_figure(figure, yaxis_title="USD")
    x = [
        _scenario_short(row.ticker, row.strategy, row.dividend_mode)
        for row in metrics.itertuples()
    ]
    figure.add_trace(
        go.Bar(
            x=x,
            y=metrics["total_contributed_usd"],
            name="投入本金",
            marker_color="#6f8375",
        )
    )
    figure.add_trace(
        go.Bar(
            x=x,
            y=metrics["ending_equity_usd"],
            name="期末資產",
            marker_color="#3f5f73",
        )
    )
    figure.update_layout(barmode="group")
    return _style_leverage_figure(figure, yaxis_title="USD")


def _build_normalized_leverage_figure(
    curves: pd.DataFrame,
    *,
    strategies: list[str],
    preferred_ticker: str,
) -> Any:
    figure = go.Figure()
    has_trace = False
    family = _filter_strategies(curves, strategies)
    if not family.empty and preferred_ticker in set(family["ticker"].astype(str)):
        family = family[family["ticker"].astype(str) == preferred_ticker].copy()
    color_map = _leverage_color_map(family)
    for key, group in _iter_curve_groups(family):
        if "total_equity" not in group.columns:
            continue
        values = group["total_equity"].astype(float).replace([np.inf, -np.inf], np.nan)
        values = values.dropna()
        if values.empty or values.iloc[0] == 0:
            continue
        has_trace = True
        dates = pd.to_datetime(group.loc[values.index, "date"])
        normalized = values / values.iloc[0] * 10_000.0
        figure.add_trace(
            go.Scatter(
                x=dates,
                y=normalized,
                mode="lines",
                name=_scenario_short(*key),
                legendgroup=_scenario_id(*key),
                line={"color": color_map[key], "width": 2},
                hovertemplate=(
                    f"{_scenario_full(*key)}<br>%{{x|%Y-%m-%d}}"
                    "<br>normalized equity: %{y:,.2f}<extra></extra>"
                ),
            )
        )
    if not has_trace:
        _add_empty_annotation(figure, "沒有可標準化的權益資料")
    return _style_leverage_figure(
        figure,
        yaxis_title="Normalized USD",
        showlegend=True,
        height=320,
    )


def _render_chart_card(
    chart_id: str,
    title: str,
    description: str,
    figure: Any,
) -> str:
    chart_html = figure.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"displaylogo": False, "responsive": True},
    )
    return f"""<article class="chart-card" id="{escape(chart_id)}">
  <div class="chart-heading">
    <h3>{escape(title)}</h3>
    <p>{escape(description)}</p>
  </div>
  {chart_html}
</article>"""


def _html_table(df: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    available = [(column, label) for column, label in columns if column in df.columns]
    headers = "".join(f"<th>{escape(label)}</th>" for _, label in available)
    rows: list[str] = []
    for _, row in df.iterrows():
        cells = "".join(
            f"<td>{escape(_format_html_cell(row.get(column)))}</td>"
            for column, _ in available
        )
        rows.append(f"<tr>{cells}</tr>")
    body = "\n".join(rows) if rows else '<tr><td colspan="99">沒有資料。</td></tr>'
    return f"""<div class="table-wrap">
  <table class="data-table">
    <thead><tr>{headers}</tr></thead>
    <tbody>{body}</tbody>
  </table>
</div>"""


def _style_leverage_figure(
    figure: Any,
    *,
    yaxis_title: str,
    showlegend: bool = False,
    height: int = 340,
) -> Any:
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={
            "family": "Noto Sans TC, Noto Sans JP, Segoe UI, sans-serif",
            "color": "#202521",
            "size": 12,
        },
        height=height,
        margin={"l": 60, "r": 36, "t": 28, "b": 56},
        hovermode="x unified",
        showlegend=showlegend,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 11},
        },
    )
    figure.update_xaxes(showgrid=False, zeroline=False)
    figure.update_yaxes(title=yaxis_title, gridcolor="#e6e8e1", zeroline=False)
    return figure


def _add_empty_annotation(figure: Any, message: str) -> None:
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"color": "#66706a", "size": 14},
    )


def _iter_curve_groups(curves: pd.DataFrame) -> list[tuple[tuple[str, str, str], pd.DataFrame]]:
    if curves.empty:
        return []
    group_columns = ["ticker", "strategy", "dividend_mode"]
    return [
        ((str(ticker), str(strategy), str(mode)), group.sort_values("date"))
        for (ticker, strategy, mode), group in curves.groupby(group_columns, sort=True)
    ]


def _leverage_color_map(curves: pd.DataFrame) -> dict[tuple[str, str, str], str]:
    palette = [
        "#3f5f73",
        "#6f8375",
        "#b66f52",
        "#7f6d9d",
        "#c09a4c",
        "#52665b",
        "#8a6f56",
        "#5c6f92",
        "#9b7a8f",
        "#557d86",
    ]
    return {
        key: palette[index % len(palette)]
        for index, (key, _) in enumerate(_iter_curve_groups(curves))
    }


def _filter_strategies(frame: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    if frame.empty or "strategy" not in frame.columns:
        return frame.copy()
    return frame[frame["strategy"].isin(strategies)].copy()


def _curve_for_scenario(
    curves: pd.DataFrame,
    *,
    ticker: str,
    strategy: str,
    dividend_mode: str,
) -> pd.DataFrame:
    if curves.empty:
        return curves.copy()
    mask = (
        (curves["ticker"].astype(str) == ticker)
        & (curves["strategy"].astype(str) == strategy)
        & (curves["dividend_mode"].astype(str) == dividend_mode)
    )
    return curves[mask].copy()


def _policy_for_scenario(
    policy: pd.DataFrame,
    *,
    ticker: str,
    strategy: str,
    dividend_mode: str,
) -> pd.DataFrame:
    if policy.empty:
        return policy.copy()
    mask = (
        (policy["ticker"].astype(str) == ticker)
        & (policy["strategy"].astype(str) == strategy)
        & (policy["dividend_mode"].astype(str) == dividend_mode)
    )
    return policy[mask].copy()


def _default_scenario_key(metrics: pd.DataFrame, *, focus: str) -> str:
    if metrics.empty:
        return ""
    priorities = {
        "risk": [
            ("SPY_QQQ", "dynamic_rebalance_leveraged", "cash"),
            ("SPY_QQQ", "rebalance_leveraged", "cash"),
        ],
        "rebalance": [
            ("SPY_QQQ", "dynamic_rebalance_leveraged", "cash"),
            ("SPY_QQQ", "rebalance_leveraged", "cash"),
        ],
        "dca": [
            ("SPY", "dynamic_dca_leveraged", "cash"),
            ("SPY", "dca_leveraged", "cash"),
        ],
        "buy_hold": [
            ("SPY", "dynamic_buy_hold_leveraged", "cash"),
            ("SPY", "buy_hold_leveraged", "cash"),
        ],
    }
    for ticker, strategy, mode in priorities.get(focus, []):
        match = metrics[
            (metrics["ticker"].astype(str) == ticker)
            & (metrics["strategy"].astype(str) == strategy)
            & (metrics["dividend_mode"].astype(str) == mode)
        ]
        if not match.empty:
            return _scenario_key(ticker, strategy, mode)
    first = metrics.iloc[0]
    return _scenario_key(first["ticker"], first["strategy"], first["dividend_mode"])


def _scenario_key(ticker: Any, strategy: Any, mode: Any) -> str:
    raw = f"{ticker}__{strategy}__{mode}"
    return "".join(ch if ch.isalnum() else "-" for ch in raw).strip("-")


def _scenario_button_title(row: Any) -> str:
    return (
        f"{row.ticker} · {_strategy_short(str(row.strategy))} · "
        f"{_mode_short(str(row.dividend_mode))}"
    )


def _scenario_button_subtitle(row: Any) -> str:
    return (
        f"ending {_format_money(row.ending_equity_usd)} USD · "
        f"safety {_format_percent(row.worst_safety_buffer)}"
    )


def _scenario_id(ticker: str, strategy: str, mode: str) -> str:
    return f"{ticker}-{strategy}-{mode}"


def _scenario_short(ticker: str, strategy: str, mode: str) -> str:
    return f"{ticker} {_strategy_short(strategy)} {_mode_short(mode)}"


def _scenario_full(ticker: str, strategy: str, mode: str) -> str:
    return f"{ticker} · {_strategy_label(strategy)} ({strategy}) · {_mode_label(mode)} ({mode})"


def _strategy_short(strategy: str) -> str:
    return {
        "buy_hold_leveraged": "B&H",
        "dynamic_buy_hold_leveraged": "Dyn B&H",
        "dca_leveraged": "DCA",
        "dynamic_dca_leveraged": "Dyn DCA",
        "rebalance_leveraged": "Rebal",
        "dynamic_rebalance_leveraged": "Dyn Rebal",
    }.get(strategy, strategy)


def _strategy_label(strategy: str) -> str:
    return {
        "buy_hold_leveraged": "固定槓桿 B&H",
        "dynamic_buy_hold_leveraged": "動態槓桿 B&H",
        "dca_leveraged": "固定槓桿 DCA",
        "dynamic_dca_leveraged": "動態槓桿 DCA",
        "rebalance_leveraged": "固定槓桿再平衡",
        "dynamic_rebalance_leveraged": "動態槓桿再平衡",
    }.get(strategy, strategy)


def _mode_short(mode: str) -> str:
    return {"cash": "現金", "reinvest": "再投"}.get(mode, mode)


def _mode_label(mode: str) -> str:
    return {"cash": "現金股息", "reinvest": "股息再投入"}.get(mode, mode)


def _format_html_cell(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.4f}".rstrip("0").rstrip(".")
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return str(value)


def _format_money(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):,.2f}"


def _format_percent(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def _format_number(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def _format_leverage(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2f}x"


def _scenario_switcher_script() -> str:
    return """<script>
document.querySelectorAll("[data-scenario-button]").forEach((button) => {
  button.addEventListener("click", () => {
    const group = button.dataset.scenarioGroup;
    const scenario = button.dataset.scenarioButton;
    document
      .querySelectorAll(`[data-scenario-button][data-scenario-group="${group}"]`)
      .forEach((item) => item.classList.toggle("is-active", item === button));
    document
      .querySelectorAll(`[data-scenario-panel][data-scenario-group="${group}"]`)
      .forEach((panel) => {
        const active = panel.dataset.scenarioPanel === scenario;
        panel.classList.toggle("is-active", active);
        if (active && window.Plotly) {
          panel.querySelectorAll(".js-plotly-plot").forEach((plot) => {
            window.Plotly.Plots.resize(plot);
          });
        }
      });
  });
});
</script>"""


def _leverage_dashboard_css() -> str:
    return """
:root {
  --bg: #f7f6f1;
  --surface: #ffffff;
  --surface-soft: #eeefe8;
  --ink: #202521;
  --muted: #66706a;
  --line: #d8ddd3;
  --indigo: #3f5f73;
  --sage: #6f8375;
  --copper: #b66f52;
  --shadow: 0 14px 36px rgba(32, 37, 33, 0.07);
}

* {
  box-sizing: border-box;
}

html {
  background: var(--bg);
  color: var(--ink);
  font-family: "Noto Sans TC", "Noto Sans JP", "Segoe UI", "Microsoft JhengHei", sans-serif;
  letter-spacing: 0;
}

body {
  margin: 0;
  background:
    linear-gradient(180deg, rgba(63, 95, 115, 0.08), rgba(247, 246, 241, 0) 420px),
    var(--bg);
}

.dashboard-shell {
  width: min(1440px, calc(100% - 32px));
  margin: 0 auto;
  padding: 28px 0 56px;
}

.dashboard-header {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 24px;
  align-items: end;
  min-height: 150px;
  padding: 28px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.84);
  box-shadow: var(--shadow);
}

.dashboard-header h1 {
  margin: 6px 0 10px;
  font-size: clamp(2rem, 4vw, 3.2rem);
  line-height: 1.05;
}

.header-copy,
.section-copy,
.section-heading p,
.notice-card p,
.detail-card p {
  color: var(--muted);
  line-height: 1.65;
}

.header-meta {
  min-width: 230px;
  padding: 16px;
  border-left: 3px solid var(--indigo);
  background: var(--surface-soft);
  border-radius: 8px;
}

.header-meta span,
.eyebrow,
.kpi-card span,
.label {
  display: block;
  color: var(--muted);
  font-size: 0.78rem;
  font-weight: 700;
}

.header-meta strong {
  display: block;
  margin-top: 8px;
}

.section-block {
  margin-top: 18px;
  padding: 24px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface);
  box-shadow: var(--shadow);
}

.section-tight {
  margin-top: 14px;
}

.section-heading {
  display: flex;
  justify-content: space-between;
  gap: 24px;
  align-items: end;
  margin-bottom: 18px;
}

.section-heading h2,
.section-block h2 {
  margin: 4px 0 0;
  font-size: 1.25rem;
}

.section-heading p {
  max-width: 700px;
  margin: 0;
}

.eyebrow {
  margin: 0;
  color: var(--indigo);
}

.dashboard-nav {
  position: sticky;
  top: 0;
  z-index: 10;
  background: rgba(255, 255, 255, 0.94);
  backdrop-filter: blur(10px);
}

.nav-grid,
.settings-grid,
.kpi-grid {
  display: grid;
  gap: 10px;
}

.nav-grid {
  grid-template-columns: repeat(5, minmax(0, 1fr));
}

.settings-grid {
  grid-template-columns: repeat(4, minmax(0, 1fr));
}

.kpi-grid {
  grid-template-columns: repeat(3, minmax(0, 1fr));
  margin: 14px 0;
}

.nav-card,
.tile,
.kpi-card,
.chart-card,
.notice-card,
.detail-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfbf7;
}

.nav-card {
  display: block;
  min-height: 88px;
  padding: 13px;
  color: var(--ink);
  text-decoration: none;
}

.nav-card:hover {
  border-color: var(--indigo);
}

.nav-card strong,
.nav-card span {
  display: block;
}

.nav-card span {
  margin-top: 7px;
  color: var(--muted);
  font-size: 0.82rem;
  line-height: 1.45;
}

.tile,
.kpi-card,
.notice-card,
.detail-card,
.chart-card {
  padding: 16px;
}

.value {
  margin-top: 6px;
  font-weight: 700;
  word-break: break-word;
}

.kpi-card {
  background: linear-gradient(180deg, #ffffff, #fafaf6);
}

.kpi-card strong {
  display: block;
  margin-top: 8px;
  font-size: clamp(1.15rem, 2vw, 1.65rem);
  line-height: 1.15;
  overflow-wrap: anywhere;
}

.kpi-card small {
  display: block;
  margin-top: 8px;
  color: var(--muted);
}

.filter-row,
.download-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 14px;
}

.scenario-selector {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 8px;
  margin: 12px 0 16px;
}

.scenario-button {
  min-height: 68px;
  padding: 10px 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfbf7;
  color: var(--ink);
  text-align: left;
  font: inherit;
  cursor: pointer;
}

.scenario-button:hover,
.scenario-button.is-active {
  border-color: var(--indigo);
  background: #f1f5f2;
}

.scenario-button span,
.scenario-button small {
  display: block;
}

.scenario-button span {
  font-weight: 700;
}

.scenario-button small {
  margin-top: 4px;
  color: var(--muted);
  line-height: 1.35;
}

.scenario-panel {
  display: none;
}

.scenario-panel.is-active {
  display: block;
}

.scenario-summary {
  display: grid;
  grid-template-columns: minmax(220px, 0.7fr) minmax(0, 1.3fr);
  gap: 14px;
  align-items: start;
  margin: 12px 0 14px;
  padding: 16px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfbf7;
}

.scenario-summary h3 {
  margin: 4px 0 0;
  font-size: 1.1rem;
}

.mini-stat-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
}

.mini-stat {
  padding: 10px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface);
}

.mini-stat span {
  display: block;
  color: var(--muted);
  font-size: 0.76rem;
  font-weight: 700;
}

.mini-stat strong {
  display: block;
  margin-top: 5px;
  overflow-wrap: anywhere;
}

.compact-kpi-grid {
  grid-template-columns: repeat(4, minmax(0, 1fr));
}

.compact-kpi strong {
  font-size: clamp(1.05rem, 1.7vw, 1.35rem);
}

.chart-grid {
  display: grid;
  gap: 12px;
}

.chart-grid.two-up {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.chart-grid.three-up {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.filter-row span,
.download-link {
  display: inline-flex;
  align-items: center;
  min-height: 34px;
  padding: 7px 11px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface-soft);
  color: var(--ink);
  font-size: 0.9rem;
  text-decoration: none;
}

.download-link:hover {
  border-color: var(--indigo);
  color: var(--indigo);
}

.warning-notice {
  border-color: rgba(182, 111, 82, 0.45);
  background: #fff8f3;
}

.chart-card {
  margin-top: 14px;
  background: var(--surface);
}

.chart-heading {
  display: flex;
  justify-content: space-between;
  gap: 20px;
  margin-bottom: 8px;
}

.chart-heading h3,
.detail-card h3 {
  margin: 0;
  font-size: 1rem;
}

.chart-heading p,
.detail-card p,
.notice-card p {
  max-width: 680px;
  margin: 0 0 10px;
}

.table-wrap {
  width: 100%;
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88rem;
}

th,
td {
  padding: 10px 11px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  vertical-align: top;
  white-space: nowrap;
}

th {
  color: var(--muted);
  font-weight: 700;
  background: #f4f5ee;
}

td:first-child {
  min-width: 280px;
  white-space: normal;
}

.empty-state {
  margin: 0;
  color: var(--muted);
}

.limitations ul {
  margin: 0;
  padding-left: 1.2rem;
  color: var(--muted);
  line-height: 1.8;
}

@media (max-width: 980px) {
  .dashboard-header,
  .section-heading,
  .chart-heading {
    display: block;
  }

  .header-meta,
  .section-heading p,
  .chart-heading p {
    margin-top: 14px;
  }

  .nav-grid,
  .settings-grid,
  .kpi-grid,
  .mini-stat-grid,
  .compact-kpi-grid,
  .chart-grid.two-up,
  .chart-grid.three-up,
  .scenario-summary {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 640px) {
  .dashboard-shell {
    width: min(100% - 20px, 1440px);
    padding-top: 10px;
  }

  .dashboard-header,
  .section-block {
    padding: 18px;
  }

  .nav-grid,
  .settings-grid,
  .kpi-grid,
  .mini-stat-grid,
  .compact-kpi-grid,
  .chart-grid.two-up,
  .chart-grid.three-up,
  .scenario-summary {
    grid-template-columns: 1fr;
  }

  th,
  td {
    padding: 9px;
  }
}
"""


def _render_charts(curves: pd.DataFrame) -> str:
    if curves.empty:
        return "<p>No curve data.</p>"
    curve = curves.copy()
    curve["date"] = pd.to_datetime(curve["date"])
    figure_equity = go.Figure()
    figure_safety = go.Figure()
    for (ticker, strategy, dividend_mode), group in curve.groupby(
        ["ticker", "strategy", "dividend_mode"]
    ):
        name = f"{ticker} {strategy} {dividend_mode}"
        figure_equity.add_trace(
            go.Scatter(
                x=group["date"],
                y=group["total_equity"],
                mode="lines",
                name=name,
            )
        )
        figure_safety.add_trace(
            go.Scatter(
                x=group["date"],
                y=group["safety_buffer"],
                mode="lines",
                name=name,
            )
        )
    figure_equity.update_layout(
        title="USD Equity Curve",
        template="plotly_white",
        height=360,
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    figure_safety.update_layout(
        title="Margin Safety Buffer",
        template="plotly_white",
        height=360,
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    return (
        figure_equity.to_html(full_html=False, include_plotlyjs="cdn")
        + figure_safety.to_html(full_html=False, include_plotlyjs=False)
    )


__all__ = [
    "LeverageReportResult",
    "LeveragedRunResult",
    "run_dynamic_buy_hold_leveraged",
    "run_dynamic_dca_leveraged",
    "run_dynamic_rebalance_leveraged",
    "run_buy_hold_leveraged",
    "run_dca_leveraged",
    "run_rebalance_leveraged",
    "write_leverage_report",
]
