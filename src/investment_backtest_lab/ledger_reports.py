from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from investment_backtest_lab.costs import CostModel
from investment_backtest_lab.data.fx import align_fx_rate
from investment_backtest_lab.ledger import AccountLedger
from investment_backtest_lab.models import DividendFrame, DividendMode, PriceFrame
from investment_backtest_lab.portfolio_ledger import PortfolioLedger
from investment_backtest_lab.reports import max_drawdown, performance_summary


@dataclass(frozen=True)
class LedgerRunResult:
    ticker: str
    strategy: str
    dividend_mode: DividendMode
    ledger: AccountLedger | PortfolioLedger
    equity_curve: pd.DataFrame
    aligned_dividends: pd.DataFrame
    price_source: str
    dividend_source: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class LedgerReportResult:
    metrics: pd.DataFrame
    trades: pd.DataFrame
    dividends: pd.DataFrame
    cash_flows: pd.DataFrame
    equity: pd.DataFrame
    positions: pd.DataFrame
    rebalance: pd.DataFrame
    warnings: list[str]
    markdown_path: Path
    metrics_path: Path
    trades_path: Path
    dividends_path: Path
    cash_flows_path: Path
    equity_path: Path
    positions_path: Path
    rebalance_path: Path
    html_path: Path


def run_buy_and_hold_ledger(
    *,
    price_frame: PriceFrame,
    dividend_frame: DividendFrame,
    cost_model: CostModel,
    initial_cash: float,
    dividend_mode: DividendMode | str,
    withholding_rate: float,
) -> LedgerRunResult:
    dividend_mode = DividendMode(dividend_mode)
    prices = price_frame.close().dropna().sort_index()
    if prices.empty:
        raise ValueError("Ledger buy-and-hold requires non-empty price data.")

    warnings: list[str] = []
    if price_frame.adjusted:
        warnings.append(
            f"{price_frame.asset.ticker}: price data is adjusted; "
            "ledger dividend report expects raw prices to avoid double counting."
        )

    ledger = AccountLedger(
        price_frame.asset,
        starting_cash=initial_cash,
        cost_model=cost_model,
        dividend_withholding_rate=withholding_rate,
        account_currency=price_frame.asset.currency,
    )
    first_trade_date = pd.Timestamp(prices.index.min())
    first_trade_price = float(prices.loc[first_trade_date])
    ledger.buy_with_cash(
        first_trade_date,
        cash_amount=ledger.cash,
        price=first_trade_price,
        note="initial buy-and-hold allocation",
    )

    dividends = dividend_frame.data.copy()
    if not dividends.empty:
        dividends = dividends[dividends.index > first_trade_date]
    aligned_dividends, align_warnings = align_dividends_to_trading_dates(dividends, prices.index)
    warnings.extend(f"{price_frame.asset.ticker}: {warning}" for warning in align_warnings)

    dividends_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    if not aligned_dividends.empty:
        effective = aligned_dividends.dropna(subset=["effective_date"]).copy()
        if not effective.empty:
            effective["effective_date"] = pd.to_datetime(effective["effective_date"])
            for effective_date, group in effective.groupby("effective_date"):
                dividends_by_date[pd.Timestamp(effective_date)] = group

    for current_date, price in prices.items():
        current_date = pd.Timestamp(current_date)
        current_price = float(price)
        for _, row in dividends_by_date.get(current_date, pd.DataFrame()).iterrows():
            reinvest = dividend_mode == DividendMode.REINVEST
            ledger.cash_dividend(
                current_date,
                dividend_per_share=float(row["dividend_per_share"]),
                reinvest=reinvest,
                price=current_price if reinvest else None,
                note=f"source_date={pd.Timestamp(row['dividend_date']).date().isoformat()}",
            )
        ledger.snapshot(current_date, price=current_price)

    late_dividends = (
        aligned_dividends[aligned_dividends["effective_date"].isna()]
        if not aligned_dividends.empty
        else pd.DataFrame()
    )
    for _, row in late_dividends.iterrows():
        ledger.cash_dividend(
            row["dividend_date"],
            dividend_per_share=float(row["dividend_per_share"]),
            reinvest=False,
            note="no later trading price; kept as cash",
        )

    equity_curve = ledger.equity_curve
    return LedgerRunResult(
        ticker=price_frame.asset.ticker,
        strategy="ledger_buy_and_hold",
        dividend_mode=dividend_mode,
        ledger=ledger,
        equity_curve=equity_curve,
        aligned_dividends=aligned_dividends,
        price_source=price_frame.source,
        dividend_source=dividend_frame.source,
        warnings=tuple(warnings),
    )


def run_dca_ledger(
    *,
    price_frame: PriceFrame,
    dividend_frame: DividendFrame,
    cost_model: CostModel,
    contribution: float,
    frequency: str,
    dividend_mode: DividendMode | str,
    withholding_rate: float,
) -> LedgerRunResult:
    dividend_mode = DividendMode(dividend_mode)
    contribution = float(contribution)
    prices = price_frame.close().dropna().sort_index()
    if prices.empty:
        raise ValueError("Ledger DCA requires non-empty price data.")

    warnings: list[str] = []
    if price_frame.adjusted:
        warnings.append(
            f"{price_frame.asset.ticker}: price data is adjusted; "
            "ledger dividend report expects raw prices to avoid double counting."
        )

    ledger = AccountLedger(
        price_frame.asset,
        starting_cash=0.0,
        cost_model=cost_model,
        dividend_withholding_rate=withholding_rate,
        account_currency=price_frame.asset.currency,
    )
    contribution_dates = dca_contribution_dates(prices.index, frequency=frequency)
    dividends = dividend_frame.data.copy()
    aligned_dividends, align_warnings = align_dividends_to_trading_dates(dividends, prices.index)
    warnings.extend(f"{price_frame.asset.ticker}: {warning}" for warning in align_warnings)
    dividends_by_date = _dividends_by_effective_date(aligned_dividends)

    for current_date, price in prices.items():
        current_date = pd.Timestamp(current_date)
        current_price = float(price)
        _apply_dividends_for_date(
            ledger,
            current_date=current_date,
            current_price=current_price,
            rows=dividends_by_date.get(current_date, pd.DataFrame()),
            dividend_mode=dividend_mode,
        )
        if current_date in contribution_dates:
            ledger.deposit(
                current_date,
                amount=contribution,
                note=f"DCA {frequency} contribution",
            )
            ledger.buy_with_cash(
                current_date,
                cash_amount=contribution,
                price=current_price,
                note="DCA buy",
            )
        ledger.snapshot(current_date, price=current_price)

    late_dividends = (
        aligned_dividends[aligned_dividends["effective_date"].isna()]
        if not aligned_dividends.empty
        else pd.DataFrame()
    )
    for _, row in late_dividends.iterrows():
        if ledger.quantity <= 0:
            continue
        ledger.cash_dividend(
            row["dividend_date"],
            dividend_per_share=float(row["dividend_per_share"]),
            reinvest=False,
            note="no later trading price; kept as cash",
        )

    equity_curve = ledger.equity_curve
    return LedgerRunResult(
        ticker=price_frame.asset.ticker,
        strategy="ledger_dca",
        dividend_mode=dividend_mode,
        ledger=ledger,
        equity_curve=equity_curve,
        aligned_dividends=aligned_dividends,
        price_source=price_frame.source,
        dividend_source=dividend_frame.source,
        warnings=tuple(warnings),
    )


def run_rebalance_ledger(
    *,
    price_frames: list[PriceFrame],
    dividend_frames: list[DividendFrame],
    cost_model: CostModel,
    initial_cash: float,
    target_weights: dict[str, float],
    frequency: str,
    dividend_mode: DividendMode | str,
    withholding_rate: float,
) -> LedgerRunResult:
    dividend_mode = DividendMode(dividend_mode)
    if not price_frames:
        raise ValueError("Ledger rebalance requires at least one price frame.")

    assets = [price_frame.asset for price_frame in price_frames]
    tickers = [asset.ticker for asset in assets]
    if len(set(tickers)) != len(tickers):
        raise ValueError("Ledger rebalance requires unique tickers.")
    if set(target_weights) != set(tickers):
        raise ValueError(
            "Ledger rebalance v1 requires target weights for exactly the selected tickers."
        )

    warnings: list[str] = []
    for price_frame in price_frames:
        if price_frame.adjusted:
            warnings.append(
                f"{price_frame.asset.ticker}: price data is adjusted; "
                "ledger dividend report expects raw prices to avoid double counting."
            )

    prices = _combined_close_prices(price_frames)
    if prices.empty:
        raise ValueError("Ledger rebalance requires overlapping non-empty price data.")

    ledger = PortfolioLedger(
        assets=assets,
        starting_cash=initial_cash,
        cost_model=cost_model,
        dividend_withholding_rate=withholding_rate,
        account_currency="USD",
    )
    rebalance_dates = rebalance_schedule_dates(prices.index, frequency=frequency)

    aligned_frames: list[pd.DataFrame] = []
    dividends_by_date: dict[pd.Timestamp, list[tuple[str, pd.Series]]] = {}
    dividends_by_ticker = {frame.asset.ticker: frame for frame in dividend_frames}
    for ticker in tickers:
        dividend_frame = dividends_by_ticker.get(ticker)
        raw_dividends = (
            dividend_frame.data.copy()
            if dividend_frame is not None
            else pd.DataFrame(columns=["dividend_per_share"])
        )
        aligned, align_warnings = align_dividends_to_trading_dates(raw_dividends, prices.index)
        warnings.extend(f"{ticker}: {warning}" for warning in align_warnings)
        if aligned.empty:
            continue
        aligned = aligned.copy()
        aligned.insert(0, "asset", ticker)
        aligned_frames.append(aligned)
        effective = aligned.dropna(subset=["effective_date"]).copy()
        if effective.empty:
            continue
        effective["effective_date"] = pd.to_datetime(effective["effective_date"])
        for effective_date, group in effective.groupby("effective_date"):
            date_key = pd.Timestamp(effective_date)
            dividends_by_date.setdefault(date_key, [])
            for _, row in group.iterrows():
                dividends_by_date[date_key].append((ticker, row))

    for current_date, price_row in prices.iterrows():
        current_date = pd.Timestamp(current_date)
        current_prices = {ticker: float(price_row[ticker]) for ticker in tickers}
        for ticker, row in dividends_by_date.get(current_date, []):
            reinvest = dividend_mode == DividendMode.REINVEST
            ledger.cash_dividend(
                ticker,
                current_date,
                dividend_per_share=float(row["dividend_per_share"]),
                reinvest=reinvest,
                price=current_prices[ticker] if reinvest else None,
                note=f"source_date={pd.Timestamp(row['dividend_date']).date().isoformat()}",
            )
        if current_date in rebalance_dates:
            ledger.rebalance_to_weights(
                current_date,
                prices=current_prices,
                target_weights=target_weights,
                note=f"{frequency} rebalance",
            )
        ledger.snapshot(current_date, prices=current_prices)

    aligned_dividends = (
        pd.concat(aligned_frames, ignore_index=True) if aligned_frames else pd.DataFrame()
    )
    late_dividends = (
        aligned_dividends[aligned_dividends["effective_date"].isna()]
        if not aligned_dividends.empty
        else pd.DataFrame()
    )
    for _, row in late_dividends.iterrows():
        ticker = str(row["asset"])
        if ledger.positions.get(ticker, 0.0) <= 0:
            continue
        ledger.cash_dividend(
            ticker,
            row["dividend_date"],
            dividend_per_share=float(row["dividend_per_share"]),
            reinvest=False,
            note="no later trading price; kept as cash",
        )

    return LedgerRunResult(
        ticker="_".join(tickers),
        strategy="ledger_rebalance",
        dividend_mode=dividend_mode,
        ledger=ledger,
        equity_curve=ledger.equity_curve,
        aligned_dividends=aligned_dividends,
        price_source="; ".join(f"{frame.asset.ticker}:{frame.source}" for frame in price_frames),
        dividend_source="; ".join(
            f"{frame.asset.ticker}:{frame.source}" for frame in dividend_frames
        ),
        warnings=tuple(warnings),
    )


def dca_contribution_dates(trading_index: pd.Index, *, frequency: str) -> set[pd.Timestamp]:
    trading_dates = pd.DatetimeIndex(trading_index).sort_values()
    if trading_dates.empty:
        return set()

    schedule_start = trading_dates.min()
    if frequency.upper() == "MS":
        schedule_start = schedule_start.to_period("M").to_timestamp()
    schedule = pd.date_range(schedule_start, trading_dates.max(), freq=frequency)
    positions = trading_dates.searchsorted(schedule, side="left")
    positions = positions[positions < len(trading_dates)]
    return {pd.Timestamp(date) for date in pd.Index(trading_dates[positions]).drop_duplicates()}


def rebalance_schedule_dates(trading_index: pd.Index, *, frequency: str) -> set[pd.Timestamp]:
    normalized = frequency.lower()
    if normalized in {"monthly", "month", "m"}:
        return dca_contribution_dates(trading_index, frequency="MS")
    if normalized in {"quarterly", "quarter", "q"}:
        return dca_contribution_dates(trading_index, frequency="QS")
    return dca_contribution_dates(trading_index, frequency=frequency)


def align_dividends_to_trading_dates(
    dividends: pd.DataFrame,
    trading_index: pd.Index,
) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    trading_dates = pd.DatetimeIndex(trading_index).sort_values()

    for dividend_date, row in dividends.sort_index().iterrows():
        dividend_date = pd.Timestamp(dividend_date).normalize()
        position = trading_dates.searchsorted(dividend_date, side="left")
        if position >= len(trading_dates):
            effective_date = pd.NaT
            warnings.append(
                f"dividend on {dividend_date.date().isoformat()} has no later trading date; "
                "kept as cash outside the plotted equity curve."
            )
        else:
            effective_date = pd.Timestamp(trading_dates[position])
        rows.append(
            {
                "dividend_date": dividend_date,
                "effective_date": effective_date,
                "dividend_per_share": float(row["dividend_per_share"]),
            }
        )

    return pd.DataFrame(rows), warnings


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


def _apply_dividends_for_date(
    ledger: AccountLedger,
    *,
    current_date: pd.Timestamp,
    current_price: float,
    rows: pd.DataFrame,
    dividend_mode: DividendMode,
) -> None:
    if rows.empty or ledger.quantity <= 0:
        return
    for _, row in rows.iterrows():
        reinvest = dividend_mode == DividendMode.REINVEST
        ledger.cash_dividend(
            current_date,
            dividend_per_share=float(row["dividend_per_share"]),
            reinvest=reinvest,
            price=current_price if reinvest else None,
            note=f"source_date={pd.Timestamp(row['dividend_date']).date().isoformat()}",
        )


def ledger_metrics_records(
    result: LedgerRunResult,
    *,
    base_currency: str,
    usd_twd: pd.Series | None,
) -> list[dict[str, Any]]:
    equity_dates = pd.to_datetime(result.equity_curve["date"])
    equity_usd = pd.Series(
        result.equity_curve["total_equity"].to_numpy(),
        index=equity_dates,
        name="total_equity",
    )
    records = [
        _metric_record(
            result,
            basis="USD",
            equity=equity_usd,
            fx_rate=None,
        )
    ]
    if base_currency.upper() == "TWD":
        if usd_twd is None:
            raise ValueError("USD/TWD FX series is required for TWD ledger metrics.")
        fx_rate = align_fx_rate(usd_twd, equity_dates)
        equity_twd = equity_usd.to_numpy() * fx_rate.to_numpy()
        records.append(
            _metric_record(
                result,
                basis="TWD",
                equity=pd.Series(equity_twd, index=fx_rate.index),
                fx_rate=fx_rate,
            )
        )
    return records


def build_equity_export(
    results: list[LedgerRunResult],
    *,
    base_currency: str,
    usd_twd: pd.Series | None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for result in results:
        equity = result.equity_curve.copy()
        equity.insert(0, "ticker", result.ticker)
        equity.insert(1, "strategy", result.strategy)
        equity.insert(2, "dividend_mode", result.dividend_mode.value)
        equity["fx_rate"] = np.nan
        equity["total_equity_twd"] = np.nan
        if base_currency.upper() == "TWD":
            if usd_twd is None:
                raise ValueError("USD/TWD FX series is required for TWD equity export.")
            fx_rate = align_fx_rate(usd_twd, pd.DatetimeIndex(equity["date"]))
            equity["fx_rate"] = fx_rate.to_numpy()
            equity["total_equity_twd"] = equity["total_equity"].to_numpy() * fx_rate.to_numpy()
        frames.append(equity)
    return pd.concat(frames, ignore_index=True)


def build_positions_export(results: list[LedgerRunResult]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for result in results:
        positions = _positions_frame_for_result(result)
        if positions.empty:
            continue
        positions.insert(0, "ticker", result.ticker)
        positions.insert(1, "strategy", result.strategy)
        positions.insert(2, "dividend_mode", result.dividend_mode.value)
        frames.append(positions)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_rebalance_export(
    *,
    trades: pd.DataFrame,
    positions: pd.DataFrame,
    target_weights: Any,
) -> pd.DataFrame:
    if trades.empty or positions.empty:
        return pd.DataFrame()
    if "strategy" not in trades.columns or "strategy" not in positions.columns:
        return pd.DataFrame()

    rebalance_trades = trades[trades["strategy"] == "ledger_rebalance"].copy()
    rebalance_positions = positions[positions["strategy"] == "ledger_rebalance"].copy()
    if "note" in rebalance_trades.columns:
        rebalance_trades = rebalance_trades[
            rebalance_trades["note"].astype(str).str.contains("rebalance", case=False)
        ]
    if rebalance_trades.empty or rebalance_positions.empty:
        return pd.DataFrame()

    clean_targets = _clean_target_weights(target_weights)
    rebalance_trades["date"] = pd.to_datetime(rebalance_trades["date"])
    rebalance_positions["date"] = pd.to_datetime(rebalance_positions["date"])
    rows: list[dict[str, Any]] = []
    group_columns = ["ticker", "dividend_mode", "date"]
    for (ticker, mode, event_date), group in rebalance_trades.groupby(group_columns, sort=True):
        position_group = rebalance_positions[
            (rebalance_positions["ticker"] == ticker)
            & (rebalance_positions["dividend_mode"] == mode)
            & (rebalance_positions["date"] == event_date)
        ].sort_values("asset")
        post_weights = _format_rebalance_weights(position_group, clean_targets)
        drift_values = _post_rebalance_drift_values(position_group, clean_targets)
        sides = group["side"].astype(str).str.lower()
        rows.append(
            {
                "date": pd.Timestamp(event_date).date().isoformat(),
                "ticker": ticker,
                "strategy": "ledger_rebalance",
                "dividend_mode": mode,
                "trade_count": int(len(group)),
                "buy_count": int(sides.str.contains("buy").sum()),
                "sell_count": int(sides.str.contains("sell").sum()),
                "traded_value": float(group.get("gross_amount", pd.Series(dtype=float)).sum()),
                "fees": float(group.get("fees", pd.Series(dtype=float)).sum()),
                "post_rebalance_max_abs_drift": (
                    float(max(drift_values)) if drift_values else np.nan
                ),
                "post_rebalance_weights": post_weights,
                "trade_reasons": _format_trade_reasons(group),
            }
        )
    return pd.DataFrame(rows)


def _combined_close_prices(price_frames: list[PriceFrame]) -> pd.DataFrame:
    close_series = [frame.close().dropna().sort_index() for frame in price_frames]
    prices = pd.concat(close_series, axis=1, join="inner").sort_index()
    prices = prices.dropna(how="any")
    prices.columns = [frame.asset.ticker for frame in price_frames]
    return prices


def _positions_frame_for_result(result: LedgerRunResult) -> pd.DataFrame:
    positions_history = getattr(result.ledger, "positions_history", None)
    if positions_history is not None:
        frame = positions_history.copy()
        if not frame.empty:
            return frame

    equity = result.equity_curve.copy()
    if equity.empty:
        return pd.DataFrame()
    frame = pd.DataFrame(
        {
            "date": equity["date"],
            "asset": result.ticker,
            "quantity": equity["quantity"],
            "price": equity["price"],
            "market_value": equity["market_value"],
            "weight": equity["exposure"],
            "currency": equity["currency"],
        }
    )
    return frame


def write_ledger_report(
    *,
    results: list[LedgerRunResult],
    base_currency: str,
    usd_twd: pd.Series | None,
    output_dir: Path,
    slug: str,
    config_path: Path,
    report_context: dict[str, Any] | None = None,
) -> LedgerReportResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings = [warning for result in results for warning in result.warnings]
    metrics = pd.DataFrame(
        [
            record
            for result in results
            for record in ledger_metrics_records(
                result,
                base_currency=base_currency,
                usd_twd=usd_twd,
            )
        ]
    )
    trades = _combine_event_frames(results, "trades")
    dividends = _combine_event_frames(results, "dividends")
    cash_flows = _combine_event_frames(results, "cash_flows")
    equity = build_equity_export(results, base_currency=base_currency, usd_twd=usd_twd)
    positions = build_positions_export(results)
    target_weights = (
        report_context.get("target_weights", {}) if report_context is not None else {}
    )
    rebalance = build_rebalance_export(
        trades=trades,
        positions=positions,
        target_weights=target_weights,
    )

    markdown_path = output_dir / f"ledger_{slug}.md"
    metrics_path = output_dir / f"ledger_{slug}_metrics.csv"
    trades_path = output_dir / f"ledger_{slug}_trades.csv"
    dividends_path = output_dir / f"ledger_{slug}_dividends.csv"
    cash_flows_path = output_dir / f"ledger_{slug}_cash_flows.csv"
    equity_path = output_dir / f"ledger_{slug}_equity.csv"
    positions_path = output_dir / f"ledger_{slug}_positions.csv"
    rebalance_path = output_dir / f"ledger_{slug}_rebalance.csv"
    html_path = output_dir / f"ledger_{slug}.html"

    metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    trades.to_csv(trades_path, index=False, encoding="utf-8")
    dividends.to_csv(dividends_path, index=False, encoding="utf-8")
    cash_flows.to_csv(cash_flows_path, index=False, encoding="utf-8")
    equity.to_csv(equity_path, index=False, encoding="utf-8")
    positions.to_csv(positions_path, index=False, encoding="utf-8")
    rebalance.to_csv(rebalance_path, index=False, encoding="utf-8")
    markdown_path.write_text(
        render_ledger_markdown(
            config_path=config_path,
            metrics=metrics,
            rebalance=rebalance,
            warnings=warnings,
            metrics_path=metrics_path,
            trades_path=trades_path,
            dividends_path=dividends_path,
            cash_flows_path=cash_flows_path,
            equity_path=equity_path,
            positions_path=positions_path,
            rebalance_path=rebalance_path,
            html_path=html_path,
        ),
        encoding="utf-8",
    )
    write_ledger_html(
        equity=equity,
        positions=positions,
        metrics=metrics,
        trades=trades,
        dividends=dividends,
        cash_flows=cash_flows,
        rebalance=rebalance,
        warnings=warnings,
        output_path=html_path,
        config_path=config_path,
        metrics_path=metrics_path,
        trades_path=trades_path,
        dividends_path=dividends_path,
        cash_flows_path=cash_flows_path,
        equity_path=equity_path,
        positions_path=positions_path,
        rebalance_path=rebalance_path,
        report_context=report_context,
    )

    return LedgerReportResult(
        metrics=metrics,
        trades=trades,
        dividends=dividends,
        cash_flows=cash_flows,
        equity=equity,
        positions=positions,
        rebalance=rebalance,
        warnings=warnings,
        markdown_path=markdown_path,
        metrics_path=metrics_path,
        trades_path=trades_path,
        dividends_path=dividends_path,
        cash_flows_path=cash_flows_path,
        equity_path=equity_path,
        positions_path=positions_path,
        rebalance_path=rebalance_path,
        html_path=html_path,
    )


def render_ledger_markdown(
    *,
    config_path: Path,
    metrics: pd.DataFrame,
    rebalance: pd.DataFrame,
    warnings: list[str],
    metrics_path: Path,
    trades_path: Path,
    dividends_path: Path,
    cash_flows_path: Path,
    equity_path: Path,
    positions_path: Path,
    rebalance_path: Path,
    html_path: Path,
) -> str:
    metrics_md = _format_metrics_for_markdown(metrics).to_markdown(
        index=False,
        disable_numparse=True,
    )
    rebalance_md = (
        rebalance.tail(12).to_markdown(index=False, disable_numparse=True)
        if not rebalance.empty
        else "目前沒有再平衡摘要。"
    )
    warning_lines = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- 無"
    return f"""# 美股 Ledger 報表

## 摘要

- 設定檔：`{config_path}`
- 指標 CSV：`{metrics_path}`
- 交易明細 CSV：`{trades_path}`
- 股息明細 CSV：`{dividends_path}`
- 外部現金流 CSV：`{cash_flows_path}`
- 權益曲線 CSV：`{equity_path}`
- 部位權重 CSV：`{positions_path}`
- 再平衡摘要 CSV：`{rebalance_path}`
- 圖表 HTML：`{html_path}`

## 怎麼讀

- 本報表使用 raw price 加上明確股息現金流，避免 adjusted price 與股息重複計算。
- `dividend_mode=cash`：股息扣除預扣稅後留在現金。
- `dividend_mode=reinvest`：股息扣除預扣稅後，用對齊後交易日收盤價再投入。
- `strategy=ledger_dca`：每期外部投入會記錄在 cash flows，不用 CAGR/Sharpe 當主要結論。
- `strategy=ledger_rebalance`：多資產共用現金池，按目標權重定期買賣。
- `basis=USD`：原幣結果；`basis=TWD`：用 USD/TWD 匯率換算後結果。

## 指標摘要

{metrics_md}

## 再平衡摘要

{rebalance_md}

## 警告與限制

{warning_lines}
- yfinance dividend date 在 v1 先視為可入帳日期；精確 ex-date/payment-date 差異留到後續強化。
- v1 支援 fractional shares，不模擬券商是否允許碎股。
- 這是研究報表，不是投資建議。
"""


def write_ledger_html(
    *,
    equity: pd.DataFrame,
    positions: pd.DataFrame,
    metrics: pd.DataFrame,
    trades: pd.DataFrame,
    dividends: pd.DataFrame,
    cash_flows: pd.DataFrame,
    rebalance: pd.DataFrame,
    warnings: list[str],
    output_path: Path,
    config_path: Path,
    metrics_path: Path,
    trades_path: Path,
    dividends_path: Path,
    cash_flows_path: Path,
    equity_path: Path,
    positions_path: Path,
    rebalance_path: Path,
    report_context: dict[str, Any] | None = None,
) -> Path:
    preferred_metrics = _preferred_basis_metrics(metrics)
    context = report_context or {}
    dashboard_html = _render_ledger_dashboard_html(
        equity=equity,
        positions=positions,
        metrics=metrics,
        preferred_metrics=preferred_metrics,
        trades=trades,
        dividends=dividends,
        cash_flows=cash_flows,
        rebalance=rebalance,
        warnings=warnings,
        output_path=output_path,
        config_path=config_path,
        metrics_path=metrics_path,
        trades_path=trades_path,
        dividends_path=dividends_path,
        cash_flows_path=cash_flows_path,
        equity_path=equity_path,
        positions_path=positions_path,
        rebalance_path=rebalance_path,
        report_context=context,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dashboard_html, encoding="utf-8")
    return output_path


def _render_ledger_dashboard_html(
    *,
    equity: pd.DataFrame,
    positions: pd.DataFrame,
    metrics: pd.DataFrame,
    preferred_metrics: pd.DataFrame,
    trades: pd.DataFrame,
    dividends: pd.DataFrame,
    cash_flows: pd.DataFrame,
    rebalance: pd.DataFrame,
    warnings: list[str],
    output_path: Path,
    config_path: Path,
    metrics_path: Path,
    trades_path: Path,
    dividends_path: Path,
    cash_flows_path: Path,
    equity_path: Path,
    positions_path: Path,
    rebalance_path: Path,
    report_context: dict[str, Any],
) -> str:
    settings_html = _render_settings_overview(metrics, report_context, config_path)
    navigation_html = _render_ledger_navigation()
    family_sections = _render_ledger_family_sections(
        equity,
        positions,
        preferred_metrics,
        cash_flows,
        rebalance,
        report_context.get("target_weights", {}),
    )
    normalized_section = _render_ledger_normalized_section(equity)
    audit_html = _render_audit_section(
        output_path=output_path,
        metrics_path=metrics_path,
        trades_path=trades_path,
        dividends_path=dividends_path,
        cash_flows_path=cash_flows_path,
        equity_path=equity_path,
        positions_path=positions_path,
        rebalance_path=rebalance_path,
        trades=trades,
        dividends=dividends,
        cash_flows=cash_flows,
        positions=positions,
        rebalance=rebalance,
    )
    warning_html = _render_warning_section(warnings)
    style = _ledger_dashboard_css()
    generated_at = escape(str(report_context.get("generated_at", "依目前資料產生")))
    font_url = (
        "https://fonts.googleapis.com/css2?"
        "family=Noto+Sans+JP:wght@400;500;600;700&"
        "family=Noto+Sans+TC:wght@400;500;600;700&display=swap"
    )

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>投資回測 Dashboard | US Ledger Audit Report</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="{font_url}" rel="stylesheet">
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>{style}</style>
</head>
<body>
  <main class="dashboard-shell">
    <header class="dashboard-header">
      <div>
        <p class="eyebrow">US Ledger Audit Report</p>
        <h1>投資回測 Dashboard</h1>
        <p class="header-copy">
          用 raw price、股息、成本、稅與 USD/TWD 匯率重建可審計現金流，
          第一屏先確認設定，再閱讀績效與明細。
        </p>
      </div>
      <div class="header-meta">
        <span>產生時間</span>
        <strong>{generated_at}</strong>
      </div>
    </header>

    <section class="section-block section-tight" aria-labelledby="settings-title">
      <div class="section-heading">
        <div>
          <p class="eyebrow">Run Setup</p>
          <h2 id="settings-title">設定總覽</h2>
        </div>
        <p>先確認期間、標的、策略、DCA、股息、成本與稅率；這些假設會直接影響所有圖表。</p>
      </div>
      {settings_html}
    </section>

    {navigation_html}

    {family_sections}

    {normalized_section}

    {audit_html}

    {warning_html}
  </main>
</body>
</html>
"""


def _render_ledger_navigation() -> str:
    items = [
        ("#view-buy-hold", "B&H 一次投入", "同一筆初始本金的長期持有結果"),
        ("#view-dca", "DCA 定期投入", "多次外部現金流，和 B&H 分開閱讀"),
        ("#view-rebalance", "再平衡", "同一筆初始本金的目標權重配置"),
        ("#view-leverage-risk", "槓桿風險", "此報表只標示位置，細節看槓桿報表"),
        ("#view-audit", "Audit 明細", "交易、股息、現金流與 CSV 下載"),
    ]
    nav_links = "\n".join(
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
    <p>第一層先選策略族群；每個區塊內再用標的、股息模式與情境表作第二層篩選。</p>
  </div>
  <div class="nav-grid">{nav_links}</div>
</nav>"""


def _render_ledger_family_sections(
    equity: pd.DataFrame,
    positions: pd.DataFrame,
    metrics: pd.DataFrame,
    cash_flows: pd.DataFrame,
    rebalance: pd.DataFrame,
    target_weights: Any,
) -> str:
    sections = [
        _render_ledger_family_section(
            section_id="view-buy-hold",
            eyebrow="Lump Sum",
            title="B&H 一次投入",
            description=(
                "這裡只比較 ledger_buy_and_hold。它和 DCA 的本金口徑不同，"
                "所以不和 DCA 放進同一個期末資產排名。"
            ),
            strategy="ledger_buy_and_hold",
            metrics=metrics,
            equity=equity,
            positions=positions,
            target_weights=target_weights,
            focus="buy_hold",
        ),
        _render_ledger_family_section(
            section_id="view-dca",
            eyebrow="Cash Flow",
            title="DCA 定期投入",
            description=(
                "DCA 有外部現金流，主要看累計投入、期末資產與 Simple Cash Return；"
                "CAGR / Sharpe 在 v1 不作主要結論。"
            ),
            strategy="ledger_dca",
            metrics=metrics,
            equity=equity,
            positions=positions,
            target_weights=target_weights,
            focus="dca",
            extra_html=_render_family_cash_flow_preview(cash_flows, "ledger_dca"),
        ),
        _render_ledger_family_section(
            section_id="view-rebalance",
            eyebrow="Portfolio",
            title="再平衡",
            description="用 SPY/QQQ 目標權重檢查月初再平衡後的權重漂移與交易成本。",
            strategy="ledger_rebalance",
            metrics=metrics,
            equity=equity,
            positions=positions,
            target_weights=target_weights,
            focus="rebalance",
            extra_html=_render_rebalance_audit_preview(rebalance),
        ),
        _render_static_leverage_placeholder(),
    ]
    return "\n".join(sections)


def _render_ledger_family_section(
    *,
    section_id: str,
    eyebrow: str,
    title: str,
    description: str,
    strategy: str,
    metrics: pd.DataFrame,
    equity: pd.DataFrame,
    positions: pd.DataFrame,
    target_weights: Any,
    focus: str,
    extra_html: str = "",
) -> str:
    family_metrics = _filter_strategy(metrics, strategy)
    family_equity = _filter_strategy(equity, strategy)
    family_positions = _filter_strategy(positions, strategy)
    chips = _render_family_chips(family_metrics)
    kpis = _render_family_kpi_cards(family_metrics, focus=focus)
    table = _render_family_metrics_table(family_metrics, focus=focus)
    charts = _render_family_charts(
        equity=family_equity,
        positions=family_positions,
        metrics=family_metrics,
        target_weights=target_weights,
        focus=focus,
    )
    return f"""<section class="section-block family-section" id="{section_id}">
  <div class="section-heading">
    <div>
      <p class="eyebrow">{escape(eyebrow)}</p>
      <h2>{escape(title)}</h2>
    </div>
    <p>{escape(description)}</p>
  </div>
  {chips}
  {kpis}
  {table}
  {charts}
  {extra_html}
</section>"""


def _render_static_leverage_placeholder() -> str:
    return """<section class="section-block family-section" id="view-leverage-risk">
  <div class="section-heading">
    <div>
      <p class="eyebrow">Risk Lens</p>
      <h2>槓桿風險</h2>
    </div>
    <p>
      這份 ledger 報表是非槓桿主線。槓桿借款、利息、安全緩衝與 margin call
      請看 leverage dashboard。
    </p>
  </div>
  <div class="notice-card">
    <strong>本金口徑提醒</strong>
    <p>B&H 與再平衡是一筆初始投入；DCA 是多次外部投入。槓桿報表也會沿用這個分區原則。</p>
  </div>
</section>"""


def _render_family_chips(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        return '<div class="filter-row"><span>目前沒有這個策略族群的資料</span></div>'
    tickers = sorted(metrics["ticker"].dropna().unique()) if "ticker" in metrics else []
    modes = (
        sorted(metrics["dividend_mode"].dropna().unique())
        if "dividend_mode" in metrics
        else []
    )
    basis = sorted(metrics["basis"].dropna().unique()) if "basis" in metrics else []
    chips = [f"標的: {', '.join(map(str, tickers))}", f"股息: {', '.join(map(str, modes))}"]
    if basis:
        chips.append(f"幣別: {', '.join(map(str, basis))}")
    return "<div class=\"filter-row\">" + "".join(
        f"<span>{escape(chip)}</span>" for chip in chips
    ) + "</div>"


def _render_family_kpi_cards(metrics: pd.DataFrame, *, focus: str) -> str:
    if metrics.empty:
        return '<p class="empty-state">沒有可顯示的策略資料。</p>'
    focus_row = metrics.loc[metrics["ending_equity"].astype(float).idxmax()]
    if focus == "dca":
        cards = [
            ("累計投入", _format_money_or_blank(focus_row.total_contributed), str(focus_row.basis)),
            ("期末資產", _format_money_or_blank(focus_row.ending_equity), str(focus_row.basis)),
            (
                "Simple Cash Return",
                _format_percent_or_blank(focus_row.simple_cash_return),
                "DCA v1 主要報酬口徑",
            ),
            ("稅前股息", _format_money_or_blank(focus_row.gross_dividends), str(focus_row.basis)),
            ("預扣稅", _format_money_or_blank(focus_row.withholding_tax), str(focus_row.basis)),
            ("最後持股", _format_shares_or_blank(focus_row.final_shares), str(focus_row.ticker)),
        ]
    elif focus == "rebalance":
        cards = [
            ("期末資產", _format_money_or_blank(focus_row.ending_equity), str(focus_row.basis)),
            ("投入本金", _format_money_or_blank(focus_row.total_contributed), str(focus_row.basis)),
            ("CAGR", _format_percent_or_blank(focus_row.cagr), "同一筆初始本金"),
            ("Sharpe", _format_number_or_blank(focus_row.sharpe), "價格路徑風險"),
            ("最大回撤", _format_percent_or_blank(focus_row.max_drawdown), "USD/TWD 依 basis"),
            ("最後權重", str(focus_row.final_weights), "目標配置檢查"),
        ]
    else:
        cards = [
            ("期末資產", _format_money_or_blank(focus_row.ending_equity), str(focus_row.basis)),
            ("投入本金", _format_money_or_blank(focus_row.total_contributed), str(focus_row.basis)),
            ("CAGR", _format_percent_or_blank(focus_row.cagr), "同一筆初始本金"),
            ("Sharpe", _format_number_or_blank(focus_row.sharpe), "價格路徑風險"),
            ("最大回撤", _format_percent_or_blank(focus_row.max_drawdown), "歷史最大跌幅"),
            (
                "Simple Return",
                _format_percent_or_blank(focus_row.simple_cash_return),
                "期末 / 投入 - 1",
            ),
        ]
    cards_html = "\n".join(
        f"""<article class="kpi-card">
  <span>{escape(label)}</span>
  <strong>{escape(value)}</strong>
  <small>{escape(note)}</small>
</article>"""
        for label, value, note in cards
    )
    return f'<div class="kpi-grid family-kpis">{cards_html}</div>'


def _render_family_metrics_table(metrics: pd.DataFrame, *, focus: str) -> str:
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
    for column in [
        "simple_cash_return",
        "total_return",
        "cagr",
        "max_drawdown",
        "volatility",
    ]:
        if column in display.columns:
            display[column] = display[column].map(_format_percent_or_blank)
    if "sharpe" in display.columns:
        display["sharpe"] = display["sharpe"].map(_format_number_or_blank)
    display["calmar_display"] = [
        _format_calmar(row.cagr, row.max_drawdown) for row in metrics.itertuples()
    ]
    for column in [
        "ending_equity",
        "total_contributed",
        "gross_dividends",
        "withholding_tax",
        "fees_paid",
        "cash",
    ]:
        if column in display.columns:
            display[column] = display[column].map(_format_money_or_blank)
    if "final_shares" in display.columns:
        display["final_shares"] = display["final_shares"].map(_format_shares_or_blank)
    if focus == "dca":
        columns = [
            ("scenario", "情境"),
            ("basis", "幣別"),
            ("total_contributed", "累計投入"),
            ("ending_equity", "期末資產"),
            ("simple_cash_return", "Simple Cash Return"),
            ("gross_dividends", "稅前股息"),
            ("withholding_tax", "預扣稅"),
            ("fees_paid", "費用"),
            ("final_shares", "最後持股"),
            ("cash", "現金"),
        ]
    else:
        columns = [
            ("scenario", "情境"),
            ("basis", "幣別"),
            ("total_contributed", "投入本金"),
            ("ending_equity", "期末資產"),
            ("simple_cash_return", "Simple Return"),
            ("cagr", "CAGR"),
            ("sharpe", "Sharpe"),
            ("calmar_display", "Calmar"),
            ("max_drawdown", "最大回撤"),
            ("gross_dividends", "稅前股息"),
            ("withholding_tax", "預扣稅"),
            ("fees_paid", "費用"),
            ("final_weights", "最後權重"),
        ]
    return _html_table(display, columns, css_class="wide-table")


def _render_family_charts(
    *,
    equity: pd.DataFrame,
    positions: pd.DataFrame,
    metrics: pd.DataFrame,
    target_weights: Any,
    focus: str,
) -> str:
    import plotly.graph_objects as go

    color_map = _scenario_color_map(equity)
    chart_specs: list[tuple[str, str, str, Any]] = [
        (
            "USD 權益曲線",
            "同一策略族群內比較，避免一次投入與定期投入混在一起。",
            _build_equity_figure(
                go,
                equity,
                value_column="total_equity",
                yaxis_title="USD",
                color_map=color_map,
            ),
        ),
        (
            "投入本金 vs 期末資產",
            "DCA 看累計投入，B&H / 再平衡看初始投入。",
            _build_contribution_figure(go, metrics),
        ),
    ]
    if focus != "dca":
        chart_specs.append(
            (
                "回撤路徑",
                "同本金口徑下觀察歷史下跌深度。",
                _build_drawdown_figure(go, equity, color_map),
            )
        )
    else:
        chart_specs.append(
            (
                "現金與持股市值",
                "檢查每期投入、股息留存與再投入造成的現金變化。",
                _build_cash_market_figure(go, equity, color_map),
            )
        )
    if focus == "rebalance":
        chart_specs.append(
            (
                "權重漂移",
                "檢查再平衡後 SPY/QQQ 是否回到目標權重附近。",
                _build_weight_drift_figure(go, positions, target_weights),
            )
        )
    return "\n".join(
        _render_chart_article(f"{focus}-{index}", title, description, figure)
        for index, (title, description, figure) in enumerate(chart_specs)
    )


def _render_family_cash_flow_preview(cash_flows: pd.DataFrame, strategy: str) -> str:
    family_flows = _filter_strategy(cash_flows, strategy)
    table = _render_event_table(
        family_flows,
        [
            ("date", "日期"),
            ("ticker", "標的"),
            ("dividend_mode", "股息模式"),
            ("kind", "類型"),
            ("amount", "金額"),
            ("currency", "幣別"),
            ("note", "備註"),
        ],
    )
    return f"""<article class="detail-card">
  <h3>DCA 現金流預覽</h3>
  <p>這裡只列最近幾筆外部投入；完整資料請到 Audit 區下載 cash flows CSV。</p>
  {table}
</article>"""


def _render_rebalance_audit_preview(rebalance: pd.DataFrame) -> str:
    if rebalance.empty:
        return ""
    recent = rebalance.tail(8).copy()
    for column in ["traded_value", "fees"]:
        if column in recent.columns:
            recent[column] = recent[column].map(_format_money_or_blank)
    if "post_rebalance_max_abs_drift" in recent.columns:
        recent["post_rebalance_max_abs_drift"] = recent[
            "post_rebalance_max_abs_drift"
        ].map(_format_percent_or_blank)
    table = _html_table(
        recent,
        [
            ("date", "日期"),
            ("ticker", "組合"),
            ("dividend_mode", "股息模式"),
            ("trade_count", "交易數"),
            ("traded_value", "交易金額"),
            ("post_rebalance_max_abs_drift", "調整後最大偏離"),
            ("post_rebalance_weights", "調整後權重"),
            ("trade_reasons", "買賣原因"),
        ],
        css_class="compact-table",
    )
    return f"""<article class="detail-card">
  <h3>再平衡讀法</h3>
  <p>最近再平衡摘要：交易數、調整後權重與買賣原因。完整資料請下載再平衡摘要 CSV。</p>
  {table}
</article>"""


def _render_ledger_normalized_section(equity: pd.DataFrame) -> str:
    import plotly.graph_objects as go

    figure = _build_normalized_equity_figure(go, equity)
    chart = _render_chart_article(
        "normalized-equity",
        "10,000 USD 標準化權益曲線",
        "所有線都從 10,000 開始，僅比較路徑形狀、波動和回撤，不代表實際投入結果。",
        figure,
    )
    return f"""<section class="section-block" id="view-normalized">
  <div class="section-heading">
    <div>
      <p class="eyebrow">Shape Only</p>
      <h2>標準化比較</h2>
    </div>
    <p>這個區塊是輔助視角。B&H、DCA、再平衡的實際投入本金不同，不能用這張圖判定誰賺最多。</p>
  </div>
  <div class="notice-card warning-notice">
    <strong>非實際投入結果，不可當作本金報酬排名。</strong>
    <p>標準化曲線只回答「哪條路徑比較抖、回撤比較深、復原比較慢」。實際績效請回到各策略族群閱讀。</p>
  </div>
  {chart}
</section>"""


def _build_normalized_equity_figure(go: Any, equity: pd.DataFrame) -> Any:
    figure = go.Figure()
    has_trace = False
    color_map = _scenario_color_map(equity)
    for key, group in _iter_equity_groups(equity):
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
                line={"color": color_map.get(key, "#3f5f73"), "width": 2.2},
                hovertemplate=(
                    f"{_scenario_full(*key)}<br>%{{x|%Y-%m-%d}}"
                    "<br>標準化資產: %{y:,.2f}<extra></extra>"
                ),
            )
        )
    if not has_trace:
        _add_empty_annotation(figure, "沒有可標準化的權益曲線資料")
    return _style_plotly_figure(figure, yaxis_title="Normalized USD")


def _render_chart_article(
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


def _render_dashboard_charts(
    equity: pd.DataFrame,
    positions: pd.DataFrame,
    metrics: pd.DataFrame,
    target_weights: Any,
) -> str:
    import plotly.graph_objects as go

    color_map = _scenario_color_map(equity)
    charts = [
        (
            "twd-equity-chart",
            "TWD 權益曲線",
            "用 USD/TWD 匯率換算後的主要報表幣別結果。",
            _build_equity_figure(
                go,
                equity,
                value_column="total_equity_twd",
                yaxis_title="TWD",
                color_map=color_map,
            ),
        ),
        (
            "usd-equity-chart",
            "USD 原幣權益曲線",
            "不含換匯影響，用來觀察標的本身與策略現金流。",
            _build_equity_figure(
                go,
                equity,
                value_column="total_equity",
                yaxis_title="USD",
                color_map=color_map,
            ),
        ),
        (
            "drawdown-chart",
            "最大回撤路徑",
            "以 USD 權益曲線計算，DCA 因外部現金流會以輔助視角閱讀。",
            _build_drawdown_figure(go, equity, color_map),
        ),
        (
            "contribution-chart",
            "投入本金 vs 期末資產",
            "DCA 以每期投入日匯率換算投入本金；B&H 以期初投入資金換算。",
            _build_contribution_figure(go, metrics),
        ),
        (
            "frictions-chart",
            "股息 / 稅 / 費用",
            "比較各情境收到的稅前股息、股息預扣稅與交易成本。",
            _build_frictions_figure(go, metrics),
        ),
        (
            "cash-market-chart",
            "現金與市值",
            "實線是持股市值，虛線是現金；可檢查股息留存與再投入差異。",
            _build_cash_market_figure(go, equity, color_map),
        ),
        (
            "weight-drift-chart",
            "權重漂移",
            "再平衡情境可用來檢查 SPY/QQQ 權重是否按月拉回目標。",
            _build_weight_drift_figure(go, positions, target_weights),
        ),
    ]

    sections: list[str] = []
    for index, (chart_id, title, description, figure) in enumerate(charts):
        include_plotly = "cdn" if index == 0 else False
        chart_html = figure.to_html(
            full_html=False,
            include_plotlyjs=include_plotly,
            config={"displaylogo": False, "responsive": True},
        )
        sections.append(
            f"""<article class="chart-card" id="{chart_id}">
  <div class="chart-heading">
    <h3>{escape(title)}</h3>
    <p>{escape(description)}</p>
  </div>
  {chart_html}
</article>"""
        )
    return "\n".join(sections)


def _build_equity_figure(
    go: Any,
    equity: pd.DataFrame,
    *,
    value_column: str,
    yaxis_title: str,
    color_map: dict[tuple[str, str, str], str],
) -> Any:
    figure = go.Figure()
    has_trace = False
    if value_column in equity.columns:
        for key, group in _iter_equity_groups(equity):
            group = group.dropna(subset=[value_column])
            if group.empty:
                continue
            has_trace = True
            dates = pd.to_datetime(group["date"])
            figure.add_trace(
                go.Scatter(
                    x=dates,
                    y=group[value_column],
                    mode="lines",
                    name=_scenario_short(*key),
                    legendgroup=_scenario_id(*key),
                    line={"color": color_map[key], "width": 2.2},
                    hovertemplate=(
                        f"{_scenario_full(*key)}<br>%{{x|%Y-%m-%d}}"
                        f"<br>{yaxis_title}: %{{y:,.2f}}<extra></extra>"
                    ),
                )
            )
    if not has_trace:
        _add_empty_annotation(figure, "沒有可繪製的權益曲線資料")
    return _style_plotly_figure(figure, yaxis_title=yaxis_title)


def _build_drawdown_figure(
    go: Any,
    equity: pd.DataFrame,
    color_map: dict[tuple[str, str, str], str],
) -> Any:
    figure = go.Figure()
    has_trace = False
    for key, group in _iter_equity_groups(equity):
        if "total_equity" not in group.columns:
            continue
        has_trace = True
        dates = pd.to_datetime(group["date"])
        figure.add_trace(
            go.Scatter(
                x=dates,
                y=_drawdown(group["total_equity"]),
                mode="lines",
                name=_scenario_short(*key),
                legendgroup=_scenario_id(*key),
                line={"color": color_map[key], "width": 2},
                hovertemplate=(
                    f"{_scenario_full(*key)}<br>%{{x|%Y-%m-%d}}"
                    "<br>回撤: %{y:.2%}<extra></extra>"
                ),
            )
        )
    if not has_trace:
        _add_empty_annotation(figure, "沒有可繪製的回撤資料")
    figure = _style_plotly_figure(figure, yaxis_title="Drawdown")
    figure.update_yaxes(tickformat=".0%")
    return figure


def _build_contribution_figure(go: Any, metrics: pd.DataFrame) -> Any:
    figure = go.Figure()
    if metrics.empty:
        _add_empty_annotation(figure, "沒有可比較的投入與期末資產資料")
        return _style_plotly_figure(figure, yaxis_title="Amount")

    x = [
        _scenario_short(row.ticker, row.strategy, row.dividend_mode)
        for row in metrics.itertuples()
    ]
    figure.add_trace(
        go.Bar(
            x=x,
            y=metrics["total_contributed"],
            name="投入本金",
            marker_color="#6f8375",
            hovertemplate="投入本金: %{y:,.2f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=x,
            y=metrics["ending_equity"],
            name="期末資產",
            marker_color="#3f5f73",
            hovertemplate="期末資產: %{y:,.2f}<extra></extra>",
        )
    )
    figure.update_layout(barmode="group")
    return _style_plotly_figure(figure, yaxis_title=str(metrics["basis"].iloc[0]))


def _build_frictions_figure(go: Any, metrics: pd.DataFrame) -> Any:
    figure = go.Figure()
    if metrics.empty:
        _add_empty_annotation(figure, "沒有股息、稅或費用資料")
        return _style_plotly_figure(figure, yaxis_title="Amount")

    x = [
        _scenario_short(row.ticker, row.strategy, row.dividend_mode)
        for row in metrics.itertuples()
    ]
    series = [
        ("稅前股息", "gross_dividends", "#6f8375"),
        ("股息預扣稅", "withholding_tax", "#b66f52"),
        ("交易費用", "fees_paid", "#7f6d9d"),
    ]
    for name, column, color in series:
        figure.add_trace(
            go.Bar(
                x=x,
                y=metrics[column],
                name=name,
                marker_color=color,
                hovertemplate=f"{name}: %{{y:,.2f}}<extra></extra>",
            )
        )
    figure.update_layout(barmode="group")
    return _style_plotly_figure(figure, yaxis_title=str(metrics["basis"].iloc[0]))


def _build_cash_market_figure(
    go: Any,
    equity: pd.DataFrame,
    color_map: dict[tuple[str, str, str], str],
) -> Any:
    figure = go.Figure()
    has_trace = False
    for key, group in _iter_equity_groups(equity):
        dates = pd.to_datetime(group["date"])
        color = color_map[key]
        if "market_value" in group.columns:
            has_trace = True
            figure.add_trace(
                go.Scatter(
                    x=dates,
                    y=group["market_value"],
                    mode="lines",
                    name=f"{_scenario_short(*key)} 市值",
                    legendgroup=_scenario_id(*key),
                    line={"color": color, "width": 2},
                    hovertemplate=(
                        f"{_scenario_full(*key)}<br>%{{x|%Y-%m-%d}}"
                        "<br>市值: %{y:,.2f}<extra></extra>"
                    ),
                )
            )
        if "cash" in group.columns:
            has_trace = True
            figure.add_trace(
                go.Scatter(
                    x=dates,
                    y=group["cash"],
                    mode="lines",
                    name=f"{_scenario_short(*key)} 現金",
                    legendgroup=_scenario_id(*key),
                    line={"color": color, "width": 1.8, "dash": "dash"},
                    hovertemplate=(
                        f"{_scenario_full(*key)}<br>%{{x|%Y-%m-%d}}"
                        "<br>現金: %{y:,.2f}<extra></extra>"
                    ),
                )
            )
    if not has_trace:
        _add_empty_annotation(figure, "沒有可繪製的現金與市值資料")
    return _style_plotly_figure(figure, yaxis_title="USD")


def _build_weight_drift_figure(
    go: Any,
    positions: pd.DataFrame,
    target_weights: Any,
) -> Any:
    figure = go.Figure()
    if positions.empty or "weight" not in positions.columns:
        _add_empty_annotation(figure, "沒有可繪製的部位權重資料")
        return _style_plotly_figure(figure, yaxis_title="Weight")

    portfolio_positions = positions[positions["strategy"] == "ledger_rebalance"].copy()
    if portfolio_positions.empty:
        _add_empty_annotation(figure, "目前沒有 ledger_rebalance 權重資料")
        return _style_plotly_figure(figure, yaxis_title="Weight")

    for (ticker, mode, asset), group in portfolio_positions.groupby(
        ["ticker", "dividend_mode", "asset"],
        sort=True,
    ):
        dates = pd.to_datetime(group["date"])
        trace_name = f"{ticker} {_mode_short(str(mode))} {asset}"
        figure.add_trace(
            go.Scatter(
                x=dates,
                y=group["weight"],
                mode="lines",
                name=trace_name,
                hovertemplate=(
                    f"{ticker} · ledger_rebalance · {asset} · {_mode_label(str(mode))}"
                    "<br>%{x|%Y-%m-%d}<br>權重: %{y:.2%}<extra></extra>"
                ),
            )
        )
    clean_targets = _clean_target_weights(target_weights)
    if clean_targets:
        min_date = pd.to_datetime(portfolio_positions["date"]).min()
        max_date = pd.to_datetime(portfolio_positions["date"]).max()
        for asset, target_weight in sorted(clean_targets.items()):
            figure.add_trace(
                go.Scatter(
                    x=[min_date, max_date],
                    y=[target_weight, target_weight],
                    mode="lines",
                    name=f"{asset} target",
                    line={"dash": "dash", "width": 1.2, "color": "#8a948d"},
                    hovertemplate=(
                        f"{asset} target<br>%{{x|%Y-%m-%d}}"
                        f"<br>target: {target_weight:.2%}<extra></extra>"
                    ),
                    showlegend=True,
                )
            )
    figure = _style_plotly_figure(figure, yaxis_title="Weight")
    figure.update_yaxes(tickformat=".0%")
    return figure


def _style_plotly_figure(figure: Any, *, yaxis_title: str) -> Any:
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={
            "family": "Noto Sans TC, Noto Sans JP, Segoe UI, sans-serif",
            "color": "#202521",
            "size": 12,
        },
        margin={"l": 60, "r": 24, "t": 28, "b": 56},
        hovermode="x unified",
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


def _render_settings_overview(
    metrics: pd.DataFrame,
    report_context: dict[str, Any],
    config_path: Path,
) -> str:
    start = report_context.get("start_date") or _first_metric_value(metrics, "start")
    end = report_context.get("end_date") or _first_metric_value(metrics, "end")
    tickers = report_context.get("tickers") or sorted(metrics["ticker"].dropna().unique())
    strategy_ids = sorted(metrics["strategy"].dropna().unique()) if "strategy" in metrics else []
    mode_ids = (
        sorted(metrics["dividend_mode"].dropna().unique()) if "dividend_mode" in metrics else []
    )
    price_sources = (
        sorted(metrics["price_source"].dropna().unique()) if "price_source" in metrics else []
    )
    dividend_sources = (
        sorted(metrics["dividend_source"].dropna().unique()) if "dividend_source" in metrics else []
    )
    cost_summary = report_context.get("cost_summary", "成本設定未提供；請回看 config。")
    tax_summary = report_context.get("tax_summary", "稅率設定未提供；請回看 config。")

    items = [
        ("期間", f"{start} → {end}"),
        ("標的", _join_display_values(tickers)),
        (
            "策略",
            _join_display_values(_strategy_label_with_id(strategy) for strategy in strategy_ids),
        ),
        (
            "股息模式",
            _join_display_values(_mode_label_with_id(mode) for mode in mode_ids),
        ),
        ("初始資金", _money_with_currency(report_context.get("initial_cash"), "USD")),
        (
            "DCA",
            (
                f"{_money_with_currency(report_context.get('dca_contribution'), 'USD')} / "
                f"{report_context.get('dca_frequency', '未提供')}"
            ),
        ),
        (
            "再平衡",
            (
                f"{report_context.get('rebalance_frequency', '未提供')} · "
                f"{_format_target_weights(report_context.get('target_weights'))}"
            ),
        ),
        (
            "幣別",
            (
                f"account={report_context.get('account_currency', 'USD')} · "
                f"base={report_context.get('base_currency', 'TWD')}"
            ),
        ),
        (
            "資料來源",
            (
                f"price={_join_display_values(price_sources)} / "
                f"dividend={_join_display_values(dividend_sources)}"
            ),
        ),
        ("成本", cost_summary),
        ("稅率", tax_summary),
        ("設定檔", str(config_path)),
    ]
    cards = "\n".join(
        f"""<div class="setting-card">
  <dt>{escape(label)}</dt>
  <dd>{escape(str(value))}</dd>
</div>"""
        for label, value in items
    )
    return f"<dl class=\"settings-grid\">{cards}</dl>"


def _render_kpi_cards(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        return '<p class="empty-state">沒有可顯示的 KPI。</p>'

    focus = metrics.loc[metrics["ending_equity"].astype(float).idxmax()]
    comparable_drawdowns = metrics["max_drawdown"].dropna()
    drawdown = focus["max_drawdown"]
    drawdown_note = "焦點情境"
    if pd.isna(drawdown) and not comparable_drawdowns.empty:
        drawdown = comparable_drawdowns.min()
        drawdown_note = "取可比較情境"

    cards = [
        (
            "焦點情境",
            _scenario_full(focus.ticker, focus.strategy, focus.dividend_mode),
            "完整比較見情境表",
        ),
        ("期末資產", _format_money_or_blank(focus.ending_equity), str(focus.basis)),
        ("投入本金", _format_money_or_blank(focus.total_contributed), "外部現金流已納入"),
        (
            "Simple Cash Return",
            _format_percent_or_blank(focus.simple_cash_return),
            "適合 DCA 第一版閱讀",
        ),
        ("最大回撤", _format_percent_or_blank(drawdown), drawdown_note),
        ("稅前股息", _format_money_or_blank(focus.gross_dividends), str(focus.basis)),
        ("預扣稅", _format_money_or_blank(focus.withholding_tax), str(focus.basis)),
        ("交易費用", _format_money_or_blank(focus.fees_paid), str(focus.basis)),
        ("最後持股", _format_shares_or_blank(focus.final_shares), str(focus.ticker)),
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


def _render_metrics_html_table(metrics: pd.DataFrame) -> str:
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
    for column in ["simple_cash_return", "total_return", "cagr", "max_drawdown"]:
        display[column] = display[column].map(_format_percent_or_blank)
    for column in [
        "ending_equity",
        "total_contributed",
        "gross_dividends",
        "withholding_tax",
        "fees_paid",
        "cash",
    ]:
        display[column] = display[column].map(_format_money_or_blank)
    display["final_shares"] = display["final_shares"].map(_format_shares_or_blank)
    columns = [
        ("scenario", "情境"),
        ("basis", "幣別"),
        ("ending_equity", "期末資產"),
        ("total_contributed", "投入本金"),
        ("simple_cash_return", "Simple Return"),
        ("max_drawdown", "最大回撤"),
        ("gross_dividends", "稅前股息"),
        ("withholding_tax", "預扣稅"),
        ("fees_paid", "費用"),
        ("final_shares", "最後持股"),
        ("final_weights", "最後權重"),
        ("cash", "現金"),
    ]
    return _html_table(display, columns, css_class="wide-table")


def _render_rebalance_section(rebalance: pd.DataFrame) -> str:
    if rebalance.empty:
        return ""
    recent = rebalance.tail(10).copy()
    for column in ["traded_value", "fees"]:
        if column in recent.columns:
            recent[column] = recent[column].map(_format_money_or_blank)
    if "post_rebalance_max_abs_drift" in recent.columns:
        recent["post_rebalance_max_abs_drift"] = recent[
            "post_rebalance_max_abs_drift"
        ].map(_format_percent_or_blank)
    table = _html_table(
        recent,
        [
            ("date", "日期"),
            ("ticker", "組合"),
            ("dividend_mode", "股息模式"),
            ("trade_count", "交易數"),
            ("buy_count", "買入"),
            ("sell_count", "賣出"),
            ("traded_value", "交易金額"),
            ("fees", "費用"),
            ("post_rebalance_max_abs_drift", "調整後最大偏離"),
            ("post_rebalance_weights", "調整後權重"),
            ("trade_reasons", "買賣原因"),
        ],
        css_class="compact-table",
    )
    return f"""<section class="section-block" aria-labelledby="rebalance-title">
  <div class="section-heading">
    <div>
      <p class="eyebrow">Rebalance Audit</p>
      <h2 id="rebalance-title">再平衡讀法</h2>
    </div>
    <p>這裡把每次再平衡的買賣原因攤開：交易數、買賣方向、調整後權重，以及離目標權重還差多少。</p>
  </div>
  {table}
</section>"""


def _render_audit_section(
    *,
    output_path: Path,
    metrics_path: Path,
    trades_path: Path,
    dividends_path: Path,
    cash_flows_path: Path,
    equity_path: Path,
    positions_path: Path,
    rebalance_path: Path,
    trades: pd.DataFrame,
    dividends: pd.DataFrame,
    cash_flows: pd.DataFrame,
    positions: pd.DataFrame,
    rebalance: pd.DataFrame,
) -> str:
    links = [
        ("指標 CSV", metrics_path),
        ("交易明細 CSV", trades_path),
        ("股息明細 CSV", dividends_path),
        ("外部現金流 CSV", cash_flows_path),
        ("權益曲線 CSV", equity_path),
        ("部位權重 CSV", positions_path),
        ("再平衡摘要 CSV", rebalance_path),
    ]
    link_parts = []
    for label, path in links:
        href = escape(_relative_report_path(path, output_path))
        link_parts.append(f'<a class="download-link" href="{href}">{escape(label)}</a>')
    link_html = "\n".join(link_parts)
    trades_table = _render_event_table(
        trades,
        [
            ("date", "日期"),
            ("ticker", "標的"),
            ("strategy", "策略"),
            ("dividend_mode", "股息模式"),
            ("side", "方向"),
            ("quantity", "股數"),
            ("price", "價格"),
            ("fees", "費用"),
            ("net_cash_flow", "現金流"),
            ("note", "備註"),
        ],
    )
    dividend_table = _render_event_table(
        dividends,
        [
            ("date", "日期"),
            ("ticker", "標的"),
            ("strategy", "策略"),
            ("dividend_mode", "股息模式"),
            ("shares", "股數"),
            ("gross_amount", "稅前股息"),
            ("withholding_tax", "預扣稅"),
            ("net_amount", "淨額"),
            ("reinvested_quantity", "再投入股數"),
        ],
    )
    cash_flow_table = _render_event_table(
        cash_flows,
        [
            ("date", "日期"),
            ("ticker", "標的"),
            ("strategy", "策略"),
            ("dividend_mode", "股息模式"),
            ("kind", "類型"),
            ("amount", "金額"),
            ("currency", "幣別"),
            ("note", "備註"),
        ],
    )
    position_table = _render_event_table(
        positions,
        [
            ("date", "日期"),
            ("ticker", "情境"),
            ("strategy", "策略"),
            ("dividend_mode", "股息模式"),
            ("asset", "資產"),
            ("quantity", "股數"),
            ("price", "價格"),
            ("market_value", "市值"),
            ("weight", "權重"),
        ],
    )
    rebalance_table = _render_event_table(
        rebalance,
        [
            ("date", "日期"),
            ("ticker", "組合"),
            ("dividend_mode", "股息模式"),
            ("trade_count", "交易數"),
            ("buy_count", "買入"),
            ("sell_count", "賣出"),
            ("traded_value", "交易金額"),
            ("fees", "費用"),
            ("post_rebalance_max_abs_drift", "最大偏離"),
            ("post_rebalance_weights", "調整後權重"),
            ("trade_reasons", "買賣原因"),
        ],
    )
    return f"""<section class="section-block" aria-labelledby="audit-title">
  <div class="section-heading">
    <div>
      <p class="eyebrow">Audit Trail</p>
      <h2 id="audit-title">審計明細與 CSV 下載</h2>
    </div>
    <p>HTML 用來閱讀，CSV 用來追查每一筆交易、股息、投入與每日權益。</p>
  </div>
  <div class="download-row" aria-label="CSV 下載">{link_html}</div>
  <div class="audit-grid">
    <article class="audit-card"><h3>再平衡摘要</h3>{rebalance_table}</article>
    <article class="audit-card"><h3>最近交易</h3>{trades_table}</article>
    <article class="audit-card"><h3>最近股息</h3>{dividend_table}</article>
    <article class="audit-card"><h3>最近現金流</h3>{cash_flow_table}</article>
    <article class="audit-card"><h3>最近部位權重</h3>{position_table}</article>
  </div>
</section>"""


def _render_event_table(df: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    if df.empty:
        return '<p class="empty-state">沒有資料。</p>'
    available = [(column, label) for column, label in columns if column in df.columns]
    return _html_table(df.tail(8), available, css_class="compact-table")


def _render_warning_section(warnings: list[str]) -> str:
    warning_items = "".join(
        f"<li>{escape(warning)}</li>" for warning in warnings
    ) or "<li>目前沒有資料對齊警告。</li>"
    return f"""<section class="section-block limitations" aria-labelledby="warning-title">
  <div class="section-heading">
    <div>
      <p class="eyebrow">Limitations</p>
      <h2 id="warning-title">警告與限制</h2>
    </div>
    <p>這是研究報表，不是投資建議；正式決策前仍需確認資料授權、資料品質與稅務假設。</p>
  </div>
  <ul>
    {warning_items}
    <li>
      yfinance dividend date 在 v1 先視為可入帳日期；
      精確 ex-date/payment-date 差異留到後續強化。
    </li>
    <li>v1 支援 fractional shares，不模擬券商是否允許碎股。</li>
    <li>DCA 的 CAGR/Sharpe 暫不作主要結論，避免外部現金流造成誤讀。</li>
  </ul>
</section>"""


def _html_table(
    df: pd.DataFrame,
    columns: list[tuple[str, str]],
    *,
    css_class: str,
) -> str:
    headers = "".join(f"<th>{escape(label)}</th>" for _, label in columns)
    rows: list[str] = []
    for _, row in df.iterrows():
        cells = "".join(
            f"<td>{escape(_format_html_cell(row.get(column)))}</td>"
            for column, _ in columns
        )
        rows.append(f"<tr>{cells}</tr>")
    body = "\n".join(rows) if rows else '<tr><td colspan="99">沒有資料。</td></tr>'
    return f"""<div class="table-wrap">
  <table class="{css_class}">
    <thead><tr>{headers}</tr></thead>
    <tbody>{body}</tbody>
  </table>
</div>"""


def _format_html_cell(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if hasattr(value, "isoformat") and not isinstance(value, str):
        try:
            return value.isoformat()
        except TypeError:
            pass
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.4f}".rstrip("0").rstrip(".")
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return str(value)


def _preferred_basis_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or "basis" not in metrics.columns:
        return metrics.copy()
    for basis in ["TWD", "USD"]:
        preferred = metrics[metrics["basis"] == basis]
        if not preferred.empty:
            return preferred.copy()
    return metrics.copy()


def _iter_equity_groups(equity: pd.DataFrame) -> list[tuple[tuple[str, str, str], pd.DataFrame]]:
    if equity.empty:
        return []
    group_columns = ["ticker", "strategy", "dividend_mode"]
    return [
        ((str(ticker), str(strategy), str(mode)), group.sort_values("date"))
        for (ticker, strategy, mode), group in equity.groupby(group_columns, sort=True)
    ]


def _scenario_color_map(equity: pd.DataFrame) -> dict[tuple[str, str, str], str]:
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
        for index, (key, _) in enumerate(_iter_equity_groups(equity))
    }


def _scenario_id(ticker: str, strategy: str, mode: str) -> str:
    return f"{ticker}-{strategy}-{mode}"


def _scenario_short(ticker: str, strategy: str, mode: str) -> str:
    return f"{ticker} {_strategy_short(strategy)} {_mode_short(mode)}"


def _scenario_full(ticker: str, strategy: str, mode: str) -> str:
    return f"{ticker} · {_strategy_label(strategy)} ({strategy}) · {_mode_label(mode)} ({mode})"


def _strategy_short(strategy: str) -> str:
    return {
        "ledger_buy_and_hold": "B&H",
        "ledger_dca": "DCA",
        "ledger_rebalance": "Rebal",
    }.get(strategy, strategy)


def _strategy_label(strategy: str) -> str:
    return {
        "ledger_buy_and_hold": "Buy and Hold",
        "ledger_dca": "定期定額",
        "ledger_rebalance": "定期再平衡",
    }.get(strategy, strategy)


def _strategy_label_with_id(strategy: str) -> str:
    return f"{_strategy_label(strategy)} ({strategy})"


def _mode_short(mode: str) -> str:
    return {"cash": "現金", "reinvest": "再投"}.get(mode, mode)


def _mode_label(mode: str) -> str:
    return {"cash": "現金股息", "reinvest": "股息再投入"}.get(mode, mode)


def _mode_label_with_id(mode: str) -> str:
    return f"{_mode_label(mode)} ({mode})"


def _first_metric_value(metrics: pd.DataFrame, column: str) -> str:
    if metrics.empty or column not in metrics.columns:
        return "未提供"
    value = metrics[column].dropna()
    return str(value.iloc[0]) if not value.empty else "未提供"


def _join_display_values(values: Any) -> str:
    if isinstance(values, str):
        return values
    values = list(values)
    return "、".join(str(value) for value in values) if values else "未提供"


def _money_with_currency(value: Any, currency: str) -> str:
    if value is None or pd.isna(value):
        return "未提供"
    return f"{float(value):,.2f} {currency}"


def _format_target_weights(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "未提供"
    return "、".join(f"{ticker} {float(weight):.0%}" for ticker, weight in value.items())


def _clean_target_weights(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    clean: dict[str, float] = {}
    for ticker, weight in value.items():
        try:
            clean[str(ticker)] = float(weight)
        except (TypeError, ValueError):
            continue
    return clean


def _format_rebalance_weights(
    positions: pd.DataFrame,
    target_weights: dict[str, float],
) -> str:
    if positions.empty:
        return ""
    parts: list[str] = []
    for row in positions.itertuples():
        asset = str(row.asset)
        weight = float(row.weight)
        if asset in target_weights:
            drift = weight - target_weights[asset]
            parts.append(
                f"{asset}={weight:.2%} target={target_weights[asset]:.0%} drift={drift:+.2%}"
            )
        else:
            parts.append(f"{asset}={weight:.2%}")
    return "; ".join(parts)


def _post_rebalance_drift_values(
    positions: pd.DataFrame,
    target_weights: dict[str, float],
) -> list[float]:
    drift_values: list[float] = []
    if positions.empty:
        return drift_values
    for row in positions.itertuples():
        asset = str(row.asset)
        if asset in target_weights:
            drift_values.append(abs(float(row.weight) - target_weights[asset]))
    return drift_values


def _format_trade_reasons(trades: pd.DataFrame) -> str:
    reasons: list[str] = []
    for row in trades.sort_values(["side", "asset"]).itertuples():
        note = str(getattr(row, "note", ""))
        current = _format_note_percent(note, "current_weight")
        target = _format_note_percent(note, "target_weight")
        drift = _format_note_percent(note, "drift", signed=True)
        action = _note_value(note, "action")
        side = str(getattr(row, "side", "")).lower()
        asset = str(getattr(row, "asset", ""))
        pieces = [f"{side.upper()} {asset}"]
        if action:
            pieces.append(action)
        if current and target:
            pieces.append(f"{current}->{target}")
        if drift:
            pieces.append(f"drift {drift}")
        details = "; ".join(pieces[1:])
        reasons.append(f"{pieces[0]}: {details}" if details else pieces[0])
    return " | ".join(reasons)


def _filter_strategy(frame: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if frame.empty or "strategy" not in frame.columns:
        return frame.copy()
    return frame[frame["strategy"] == strategy].copy()


def _note_value(note: str, key: str) -> str:
    prefix = f"{key}="
    for part in note.split(";"):
        part = part.strip()
        if part.startswith(prefix):
            return part[len(prefix) :]
    return ""


def _format_note_percent(note: str, key: str, *, signed: bool = False) -> str:
    value = _note_value(note, key)
    if not value:
        return ""
    try:
        number = float(value)
    except ValueError:
        return value
    sign = "+" if signed and number >= 0 else ""
    return f"{sign}{number:.2%}"


def _relative_report_path(path: Path, output_path: Path) -> str:
    try:
        return str(path.relative_to(output_path.parent))
    except ValueError:
        return path.name


def _ledger_dashboard_css() -> str:
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
  --gold: #c09a4c;
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
    linear-gradient(180deg, rgba(111, 131, 117, 0.08), rgba(247, 246, 241, 0) 420px),
    var(--bg);
  color: var(--ink);
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
  background: rgba(255, 255, 255, 0.82);
  box-shadow: var(--shadow);
}

.dashboard-header h1 {
  margin: 6px 0 10px;
  font-size: clamp(2rem, 4vw, 3.2rem);
  line-height: 1.05;
  font-weight: 700;
}

.header-copy {
  max-width: 760px;
  margin: 0;
  color: var(--muted);
  line-height: 1.7;
}

.header-meta {
  min-width: 190px;
  padding: 16px;
  border-left: 3px solid var(--sage);
  background: var(--surface-soft);
  border-radius: 8px;
}

.header-meta span,
.eyebrow,
.kpi-card span,
.setting-card dt {
  display: block;
  color: var(--muted);
  font-size: 0.78rem;
  font-weight: 700;
}

.header-meta strong {
  display: block;
  margin-top: 8px;
  font-size: 0.95rem;
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

.dashboard-nav {
  position: sticky;
  top: 0;
  z-index: 10;
  background: rgba(255, 255, 255, 0.94);
  backdrop-filter: blur(10px);
}

.nav-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 10px;
}

.nav-card {
  display: block;
  min-height: 88px;
  padding: 13px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfbf7;
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

.section-heading {
  display: flex;
  justify-content: space-between;
  gap: 24px;
  align-items: end;
  margin-bottom: 18px;
}

.section-heading h2 {
  margin: 4px 0 0;
  font-size: 1.25rem;
}

.section-heading p {
  max-width: 640px;
  margin: 0;
  color: var(--muted);
  line-height: 1.65;
}

.eyebrow {
  margin: 0;
  color: var(--indigo);
}

.settings-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin: 0;
}

.setting-card,
.kpi-card,
.audit-card,
.chart-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface);
}

.setting-card {
  min-height: 92px;
  padding: 14px;
  background: #fbfbf7;
}

.setting-card dd {
  margin: 8px 0 0;
  color: var(--ink);
  line-height: 1.5;
  word-break: break-word;
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}

.kpi-card {
  padding: 16px;
  background: linear-gradient(180deg, #ffffff, #fafaf6);
}

.kpi-card strong {
  display: block;
  margin-top: 8px;
  font-size: clamp(1.2rem, 2.2vw, 1.8rem);
  line-height: 1.15;
  color: var(--ink);
  overflow-wrap: anywhere;
}

.kpi-card small {
  display: block;
  margin-top: 8px;
  color: var(--muted);
  line-height: 1.45;
}

.legend-guide,
.download-row,
.filter-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 14px;
}

.legend-guide span,
.download-link,
.filter-row span {
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

.filter-row span {
  background: #f7f8f2;
  color: var(--muted);
}

.download-link:hover {
  border-color: var(--indigo);
  color: var(--indigo);
}

.notice-card,
.detail-card {
  margin-top: 14px;
  padding: 16px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfbf7;
}

.notice-card strong,
.detail-card h3 {
  display: block;
  margin: 0 0 8px;
  font-size: 1rem;
}

.notice-card p,
.detail-card p {
  margin: 0 0 12px;
  color: var(--muted);
  line-height: 1.65;
}

.warning-notice {
  border-color: rgba(182, 111, 82, 0.45);
  background: #fff8f3;
}

.chart-card {
  margin-top: 14px;
  padding: 18px;
}

.chart-heading {
  display: flex;
  justify-content: space-between;
  gap: 20px;
  margin-bottom: 8px;
}

.chart-heading h3,
.audit-card h3 {
  margin: 0;
  font-size: 1rem;
}

.chart-heading p {
  max-width: 620px;
  margin: 0;
  color: var(--muted);
  line-height: 1.6;
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

td {
  color: var(--ink);
}

.wide-table td:first-child {
  min-width: 270px;
  white-space: normal;
}

.audit-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 12px;
}

.audit-card {
  padding: 16px;
}

.audit-card h3 {
  margin-bottom: 12px;
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
    grid-template-columns: 1fr;
    display: block;
  }

  .header-meta,
  .section-heading p,
  .chart-heading p {
    margin-top: 14px;
  }

  .settings-grid,
  .kpi-grid,
  .nav-grid {
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

  .settings-grid,
  .kpi-grid,
  .nav-grid {
    grid-template-columns: 1fr;
  }

  th,
  td {
    padding: 9px;
  }
}
"""


def _metric_record(
    result: LedgerRunResult,
    *,
    basis: str,
    equity: pd.Series,
    fx_rate: pd.Series | None,
) -> dict[str, Any]:
    equity = pd.Series(equity).astype(float)
    ending_equity = float(equity.iloc[-1])
    total_contributed = _total_contributed(result, basis=basis, fx_rate=fx_rate)
    simple_cash_return = (
        ending_equity / total_contributed - 1.0 if total_contributed else np.nan
    )

    if result.strategy == "ledger_dca":
        total_return = np.nan
        cagr = np.nan
        volatility = np.nan
        sharpe = np.nan
        drawdown = np.nan
    else:
        returns = equity.pct_change().fillna(0.0)
        summary = performance_summary(returns)
        total_return = simple_cash_return
        days = max((pd.Timestamp(equity.index[-1]) - pd.Timestamp(equity.index[0])).days, 1)
        cagr = (1.0 + total_return) ** (365.25 / days) - 1.0 if total_return > -1 else np.nan
        volatility = summary["volatility"]
        sharpe = summary["sharpe"]
        drawdown = max_drawdown(returns)

    dividends = result.ledger.dividends
    gross_dividends = float(dividends["gross_amount"].sum()) if not dividends.empty else 0.0
    withholding_tax = (
        float(dividends["withholding_tax"].sum()) if not dividends.empty else 0.0
    )
    fees = result.ledger.total_fees_paid

    if basis == "TWD" and fx_rate is not None:
        final_fx = float(fx_rate.iloc[-1])
        gross_dividends *= final_fx
        withholding_tax *= final_fx
        fees *= final_fx

    return {
        "ticker": result.ticker,
        "strategy": result.strategy,
        "dividend_mode": result.dividend_mode.value,
        "basis": basis,
        "start": pd.Timestamp(equity.index[0]).date().isoformat(),
        "end": pd.Timestamp(equity.index[-1]).date().isoformat(),
        "ending_equity": ending_equity,
        "total_contributed": total_contributed,
        "simple_cash_return": simple_cash_return,
        "total_return": total_return,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "gross_dividends": gross_dividends,
        "withholding_tax": withholding_tax,
        "fees_paid": fees,
        "final_shares": _final_shares(result),
        "final_weights": _final_weights(result),
        "cash": result.ledger.cash * (float(fx_rate.iloc[-1]) if fx_rate is not None else 1.0),
        "price_source": result.price_source,
        "dividend_source": result.dividend_source,
    }


def _combine_event_frames(results: list[LedgerRunResult], name: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for result in results:
        frame = getattr(result.ledger, name).copy()
        if frame.empty:
            continue
        frame.insert(0, "ticker", result.ticker)
        frame.insert(1, "strategy", result.strategy)
        frame.insert(2, "dividend_mode", result.dividend_mode.value)
        frame = frame.dropna(axis=1, how="all")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _format_metrics_for_markdown(metrics: pd.DataFrame) -> pd.DataFrame:
    formatted = metrics.copy()
    for column in ["total_return", "cagr", "volatility", "max_drawdown", "simple_cash_return"]:
        formatted[column] = formatted[column].map(_format_percent_or_blank)
    for column in [
        "ending_equity",
        "total_contributed",
        "gross_dividends",
        "withholding_tax",
        "fees_paid",
        "cash",
    ]:
        formatted[column] = formatted[column].map(_format_money_or_blank)
    formatted["sharpe"] = formatted["sharpe"].map(_format_number_or_blank)
    formatted["final_shares"] = formatted["final_shares"].map(_format_shares_or_blank)
    if "final_weights" not in formatted.columns:
        formatted["final_weights"] = ""
    return formatted[
        [
            "ticker",
            "strategy",
            "dividend_mode",
            "basis",
            "ending_equity",
            "total_contributed",
            "simple_cash_return",
            "total_return",
            "cagr",
            "max_drawdown",
            "sharpe",
            "gross_dividends",
            "withholding_tax",
            "fees_paid",
            "final_shares",
            "final_weights",
            "cash",
        ]
    ]


def _total_contributed(
    result: LedgerRunResult,
    *,
    basis: str,
    fx_rate: pd.Series | None,
) -> float:
    if result.strategy == "ledger_dca":
        cash_flows = result.ledger.cash_flows
        if cash_flows.empty:
            return 0.0
        if basis == "USD":
            return float(cash_flows["amount"].sum())
        if fx_rate is None:
            raise ValueError("FX rate is required for non-USD DCA contribution conversion.")
        flow_dates = pd.to_datetime(cash_flows["date"])
        aligned_fx = align_fx_rate(fx_rate, flow_dates)
        return float((cash_flows["amount"].to_numpy() * aligned_fx.to_numpy()).sum())

    if basis == "USD":
        return float(result.ledger.starting_cash)
    if fx_rate is None:
        raise ValueError("FX rate is required for non-USD initial capital conversion.")
    return float(result.ledger.starting_cash * float(fx_rate.iloc[0]))


def _final_shares(result: LedgerRunResult) -> float:
    quantity = getattr(result.ledger, "quantity", np.nan)
    return float(quantity) if not pd.isna(quantity) else np.nan


def _final_weights(result: LedgerRunResult) -> str:
    positions = _positions_frame_for_result(result)
    if positions.empty or "weight" not in positions.columns:
        return ""
    positions = positions.copy()
    positions["date"] = pd.to_datetime(positions["date"])
    final_date = positions["date"].max()
    final_positions = positions[positions["date"] == final_date].sort_values("asset")
    weights = [
        f"{row.asset}={float(row.weight):.2%}"
        for row in final_positions.itertuples()
        if float(row.weight) > 1e-8
    ]
    return ", ".join(weights)


def _format_percent_or_blank(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def _format_money_or_blank(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):,.2f}"


def _format_number_or_blank(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def _format_calmar(cagr: Any, max_drawdown_value: Any) -> str:
    if pd.isna(cagr) or pd.isna(max_drawdown_value) or float(max_drawdown_value) == 0:
        return ""
    return f"{float(cagr) / abs(float(max_drawdown_value)):.2f}"


def _format_shares_or_blank(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.6f}"


def _drawdown(equity: pd.Series) -> pd.Series:
    equity = pd.Series(equity).astype(float)
    return equity / equity.cummax() - 1.0


__all__ = [
    "LedgerReportResult",
    "LedgerRunResult",
    "align_dividends_to_trading_dates",
    "build_equity_export",
    "build_positions_export",
    "build_rebalance_export",
    "dca_contribution_dates",
    "ledger_metrics_records",
    "rebalance_schedule_dates",
    "render_ledger_markdown",
    "run_buy_and_hold_ledger",
    "run_dca_ledger",
    "run_rebalance_ledger",
    "write_ledger_html",
    "write_ledger_report",
]
