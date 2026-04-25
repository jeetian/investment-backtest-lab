from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from investment_backtest_lab.costs import CostModel
from investment_backtest_lab.data.fx import align_fx_rate
from investment_backtest_lab.ledger import AccountLedger
from investment_backtest_lab.models import DividendFrame, DividendMode, PriceFrame
from investment_backtest_lab.reports import max_drawdown, performance_summary


@dataclass(frozen=True)
class LedgerRunResult:
    ticker: str
    strategy: str
    dividend_mode: DividendMode
    ledger: AccountLedger
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
    warnings: list[str]
    markdown_path: Path
    metrics_path: Path
    trades_path: Path
    dividends_path: Path
    cash_flows_path: Path
    equity_path: Path
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


def write_ledger_report(
    *,
    results: list[LedgerRunResult],
    base_currency: str,
    usd_twd: pd.Series | None,
    output_dir: Path,
    slug: str,
    config_path: Path,
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

    markdown_path = output_dir / f"ledger_{slug}.md"
    metrics_path = output_dir / f"ledger_{slug}_metrics.csv"
    trades_path = output_dir / f"ledger_{slug}_trades.csv"
    dividends_path = output_dir / f"ledger_{slug}_dividends.csv"
    cash_flows_path = output_dir / f"ledger_{slug}_cash_flows.csv"
    equity_path = output_dir / f"ledger_{slug}_equity.csv"
    html_path = output_dir / f"ledger_{slug}.html"

    metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    trades.to_csv(trades_path, index=False, encoding="utf-8")
    dividends.to_csv(dividends_path, index=False, encoding="utf-8")
    cash_flows.to_csv(cash_flows_path, index=False, encoding="utf-8")
    equity.to_csv(equity_path, index=False, encoding="utf-8")
    markdown_path.write_text(
        render_ledger_markdown(
            config_path=config_path,
            metrics=metrics,
            warnings=warnings,
            metrics_path=metrics_path,
            trades_path=trades_path,
            dividends_path=dividends_path,
            cash_flows_path=cash_flows_path,
            equity_path=equity_path,
            html_path=html_path,
        ),
        encoding="utf-8",
    )
    write_ledger_html(
        equity=equity,
        dividends=dividends,
        cash_flows=cash_flows,
        output_path=html_path,
    )

    return LedgerReportResult(
        metrics=metrics,
        trades=trades,
        dividends=dividends,
        cash_flows=cash_flows,
        equity=equity,
        warnings=warnings,
        markdown_path=markdown_path,
        metrics_path=metrics_path,
        trades_path=trades_path,
        dividends_path=dividends_path,
        cash_flows_path=cash_flows_path,
        equity_path=equity_path,
        html_path=html_path,
    )


def render_ledger_markdown(
    *,
    config_path: Path,
    metrics: pd.DataFrame,
    warnings: list[str],
    metrics_path: Path,
    trades_path: Path,
    dividends_path: Path,
    cash_flows_path: Path,
    equity_path: Path,
    html_path: Path,
) -> str:
    metrics_md = _format_metrics_for_markdown(metrics).to_markdown(
        index=False,
        disable_numparse=True,
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
- 圖表 HTML：`{html_path}`

## 怎麼讀

- 本報表使用 raw price 加上明確股息現金流，避免 adjusted price 與股息重複計算。
- `dividend_mode=cash`：股息扣除預扣稅後留在現金。
- `dividend_mode=reinvest`：股息扣除預扣稅後，用對齊後交易日收盤價再投入。
- `strategy=ledger_dca`：每期外部投入會記錄在 cash flows，不用 CAGR/Sharpe 當主要結論。
- `basis=USD`：原幣結果；`basis=TWD`：用 USD/TWD 匯率換算後結果。

## 指標摘要

{metrics_md}

## 警告與限制

{warning_lines}
- yfinance dividend date 在 v1 先視為可入帳日期；精確 ex-date/payment-date 差異留到後續強化。
- v1 支援 fractional shares，不模擬券商是否允許碎股。
- 這是研究報表，不是投資建議。
"""


def write_ledger_html(
    *,
    equity: pd.DataFrame,
    dividends: pd.DataFrame,
    cash_flows: pd.DataFrame,
    output_path: Path,
) -> Path:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "TWD Equity",
            "USD Equity",
            "Drawdown",
            "Cash / Market Value / Dividends",
        ),
        vertical_spacing=0.08,
    )

    group_columns = ["ticker", "strategy", "dividend_mode"]
    for (ticker, strategy, mode), group in equity.groupby(group_columns):
        name = f"{ticker} {strategy} {mode}"
        dates = pd.to_datetime(group["date"])
        figure.add_trace(
            go.Scatter(x=dates, y=group["total_equity_twd"], mode="lines", name=f"{name} TWD"),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(x=dates, y=group["total_equity"], mode="lines", name=f"{name} USD"),
            row=2,
            col=1,
        )
        drawdown = _drawdown(group["total_equity"])
        figure.add_trace(
            go.Scatter(x=dates, y=drawdown, mode="lines", name=f"{name} drawdown"),
            row=3,
            col=1,
        )
        figure.add_trace(
            go.Scatter(x=dates, y=group["cash"], mode="lines", name=f"{name} cash"),
            row=4,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=dates,
                y=group["market_value"],
                mode="lines",
                name=f"{name} market value",
            ),
            row=4,
            col=1,
        )

    if not dividends.empty:
        dividends = dividends.copy()
        dividends["date"] = pd.to_datetime(dividends["date"])
        for (ticker, strategy, mode), group in dividends.groupby(group_columns):
            figure.add_trace(
                go.Bar(
                    x=group["date"],
                    y=group["net_amount"],
                    name=f"{ticker} {strategy} {mode} net dividend",
                    opacity=0.45,
                ),
                row=4,
                col=1,
            )
            figure.add_trace(
                go.Bar(
                    x=group["date"],
                    y=group["withholding_tax"],
                    name=f"{ticker} {strategy} {mode} withholding tax",
                    opacity=0.45,
                ),
                row=4,
                col=1,
            )

    if not cash_flows.empty:
        cash_flows = cash_flows.copy()
        cash_flows["date"] = pd.to_datetime(cash_flows["date"])
        for (ticker, strategy, mode), group in cash_flows.groupby(group_columns):
            figure.add_trace(
                go.Bar(
                    x=group["date"],
                    y=group["amount"],
                    name=f"{ticker} {strategy} {mode} deposit",
                    opacity=0.30,
                ),
                row=4,
                col=1,
            )

    figure.update_layout(
        title="US Ledger Audit Report",
        hovermode="x unified",
        barmode="stack",
        height=1100,
        legend=dict(orientation="h"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output_path), include_plotlyjs="cdn")
    return output_path


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
        "final_shares": result.ledger.quantity,
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
    "dca_contribution_dates",
    "ledger_metrics_records",
    "render_ledger_markdown",
    "run_buy_and_hold_ledger",
    "run_dca_ledger",
    "write_ledger_html",
    "write_ledger_report",
]
