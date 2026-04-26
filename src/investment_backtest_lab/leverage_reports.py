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
from investment_backtest_lab.ledger_reports import dca_contribution_dates, rebalance_schedule_dates
from investment_backtest_lab.leverage import MarginLoanLedger, PortfolioMarginLedger
from investment_backtest_lab.models import LeverageConfig, PriceFrame
from investment_backtest_lab.reports import max_drawdown, performance_summary


@dataclass(frozen=True)
class LeveragedRunResult:
    ticker: str
    strategy: str
    ledger: MarginLoanLedger | PortfolioMarginLedger
    equity_curve: pd.DataFrame
    price_source: str
    total_contributed: float
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class LeverageReportResult:
    metrics: pd.DataFrame
    trades: pd.DataFrame
    interest: pd.DataFrame
    leverage_events: pd.DataFrame
    cash_flows: pd.DataFrame
    curves: pd.DataFrame
    positions: pd.DataFrame
    warnings: list[str]
    markdown_path: Path
    metrics_path: Path
    trades_path: Path
    interest_path: Path
    events_path: Path
    cash_flows_path: Path
    curves_path: Path
    positions_path: Path
    html_path: Path


def run_buy_hold_leveraged(
    *,
    price_frame: PriceFrame,
    cost_model: CostModel,
    initial_cash: float,
    leverage: LeverageConfig,
) -> LeveragedRunResult:
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
    _run_fixed_target_leverage_path(ledger, prices)
    return LeveragedRunResult(
        ticker=price_frame.asset.ticker,
        strategy="buy_hold_leveraged",
        ledger=ledger,
        equity_curve=ledger.equity_curve,
        price_source=price_frame.source,
        total_contributed=float(initial_cash),
        warnings=tuple(warnings),
    )


def run_dca_leveraged(
    *,
    price_frame: PriceFrame,
    cost_model: CostModel,
    contribution: float,
    frequency: str,
    leverage: LeverageConfig,
) -> LeveragedRunResult:
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
    contribution_dates = dca_contribution_dates(prices.index, frequency=frequency)
    previous_date: pd.Timestamp | None = None
    total_contributed = 0.0
    for current_date, price in prices.items():
        current_date = pd.Timestamp(current_date)
        current_price = float(price)
        if previous_date is not None:
            days = max(1, (current_date - previous_date).days)
            ledger.accrue_interest(current_date, days=days)
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

    return LeveragedRunResult(
        ticker=price_frame.asset.ticker,
        strategy="dca_leveraged",
        ledger=ledger,
        equity_curve=ledger.equity_curve,
        price_source=price_frame.source,
        total_contributed=float(total_contributed),
        warnings=tuple(warnings),
    )


def run_rebalance_leveraged(
    *,
    price_frames: list[PriceFrame],
    cost_model: CostModel,
    initial_cash: float,
    target_weights: dict[str, float],
    frequency: str,
    leverage: LeverageConfig,
) -> LeveragedRunResult:
    if len(price_frames) < 2:
        raise ValueError("Leveraged rebalance requires at least two price frames.")
    tickers = [frame.asset.ticker for frame in price_frames]
    prices = pd.concat([frame.close() for frame in price_frames], axis=1).dropna().sort_index()
    if prices.empty:
        raise ValueError("Leveraged rebalance requires overlapping non-empty price data.")

    warnings: list[str] = []
    for frame in price_frames:
        warnings.extend(_price_warnings(frame))
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

    return LeveragedRunResult(
        ticker="_".join(tickers),
        strategy="rebalance_leveraged",
        ledger=ledger,
        equity_curve=ledger.equity_curve,
        price_source="; ".join(f"{frame.asset.ticker}:{frame.source}" for frame in price_frames),
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
    leverage_events = _collect_frames(results, "leverage_events")
    cash_flows = _collect_frames(results, "cash_flows")
    curves = _collect_curves(results, usd_twd=usd_twd, base_currency=base_currency)
    positions = _collect_positions(results)

    prefix = f"leverage_{slug}"
    metrics_path = output_dir / f"{prefix}_metrics.csv"
    trades_path = output_dir / f"{prefix}_trades.csv"
    interest_path = output_dir / f"{prefix}_interest.csv"
    events_path = output_dir / f"{prefix}_events.csv"
    cash_flows_path = output_dir / f"{prefix}_cash_flows.csv"
    curves_path = output_dir / f"{prefix}_curve.csv"
    positions_path = output_dir / f"{prefix}_positions.csv"
    markdown_path = output_dir / f"{prefix}.md"
    html_path = output_dir / f"{prefix}.html"

    metrics.to_csv(metrics_path, index=False)
    trades.to_csv(trades_path, index=False)
    interest.to_csv(interest_path, index=False)
    leverage_events.to_csv(events_path, index=False)
    cash_flows.to_csv(cash_flows_path, index=False)
    curves.to_csv(curves_path, index=False)
    positions.to_csv(positions_path, index=False)

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
                "events": events_path,
                "cash_flows": cash_flows_path,
                "curve": curves_path,
                "positions": positions_path,
                "html": html_path,
            },
        ),
        encoding="utf-8",
    )
    html_path.write_text(
        _render_html(
            metrics=metrics,
            curves=curves,
            warnings=warnings,
            report_context=report_context,
            paths={
                "metrics": metrics_path,
                "trades": trades_path,
                "interest": interest_path,
                "events": events_path,
                "cash_flows": cash_flows_path,
                "curve": curves_path,
                "positions": positions_path,
            },
        ),
        encoding="utf-8",
    )

    return LeverageReportResult(
        metrics=metrics,
        trades=trades,
        interest=interest,
        leverage_events=leverage_events,
        cash_flows=cash_flows,
        curves=curves,
        positions=positions,
        warnings=warnings,
        markdown_path=markdown_path,
        metrics_path=metrics_path,
        trades_path=trades_path,
        interest_path=interest_path,
        events_path=events_path,
        cash_flows_path=cash_flows_path,
        curves_path=curves_path,
        positions_path=positions_path,
        html_path=html_path,
    )


def _run_fixed_target_leverage_path(
    ledger: MarginLoanLedger,
    prices: pd.Series,
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
        ledger.check_margin_risk(current_date, price=current_price)
        ledger.snapshot(current_date, price=current_price)
        previous_date = current_date


def _price_warnings(price_frame: PriceFrame) -> list[str]:
    warnings: list[str] = []
    if price_frame.adjusted:
        warnings.append(
            f"{price_frame.asset.ticker}: leveraged margin report should use raw prices; "
            "adjusted prices can hide tradable price path details."
        )
    warnings.append(
        f"{price_frame.asset.ticker}: margin loan v1 does not yet model dividends; "
        "use it as a price-path risk model, not a full total-return audit."
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
        rows.append(
            {
                "ticker": result.ticker,
                "strategy": result.strategy,
                "target_leverage": result.ledger.leverage.target_leverage,
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
        frame.insert(0, "ticker", result.ticker)
        frame.insert(1, "strategy", result.strategy)
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
        f"- 目標槓桿: {report_context.get('target_leverage')}",
        f"- 借款年利率: {report_context.get('annual_borrow_rate')}",
        f"- 維持率: {report_context.get('maintenance_requirement')}",
        f"- 安全緩衝門檻: {report_context.get('min_safety_buffer')}",
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
    warnings: list[str],
    report_context: dict[str, Any],
    paths: dict[str, Path],
) -> str:
    charts = _render_charts(curves)
    metrics_html = metrics.to_html(index=False, classes="data-table", border=0)
    links = "\n".join(
        f'<a href="{escape(path.name)}">{escape(label)}</a>' for label, path in paths.items()
    )
    warnings_html = "".join(f"<li>{escape(warning)}</li>" for warning in warnings)
    font_href = (
        "https://fonts.googleapis.com/css2?"
        "family=Noto+Sans+JP:wght@400;500;700&"
        "family=Noto+Sans+TC:wght@400;500;700&display=swap"
    )
    period_value = f"{report_context.get('start_date')} 到 {report_context.get('end_date')}"
    tickers_value = ", ".join(report_context.get("tickers", []))
    strategies_value = ", ".join(report_context.get("strategies", []))
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>輕槓桿風險模型報表</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="{font_href}" rel="stylesheet">
  <style>
    :root {{
      --paper: #f7f5ef;
      --ink: #252525;
      --muted: #6f716d;
      --line: #dedad0;
      --indigo: #42526e;
      --sage: #7b8f7a;
      --copper: #b8845b;
      --panel: #fffdf8;
    }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: "Noto Sans TC", "Noto Sans JP", system-ui, sans-serif;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 20px 48px; }}
    header {{
      display: grid;
      gap: 12px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 18px;
    }}
    h1 {{ margin: 0; font-size: 30px; font-weight: 700; letter-spacing: 0; }}
    h2 {{ margin: 26px 0 12px; font-size: 18px; letter-spacing: 0; }}
    .subtle {{ color: var(--muted); line-height: 1.7; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 10px;
    }}
    .tile {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      box-shadow: 0 8px 20px rgba(37, 37, 37, 0.04);
    }}
    .label {{ color: var(--muted); font-size: 12px; }}
    .value {{ margin-top: 5px; font-weight: 700; }}
    .links {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .links a {{
      color: var(--indigo);
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 7px 10px;
      text-decoration: none;
    }}
    .data-table {{ width: 100%; border-collapse: collapse; background: var(--panel); }}
    .data-table th, .data-table td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: right;
      font-size: 12px;
      white-space: nowrap;
    }}
    .data-table th {{ color: var(--muted); font-weight: 700; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }}
    .warning {{ color: #7a4b2b; }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>輕槓桿風險模型報表</h1>
    <div class="subtle">
      先看是否安全，再談報酬最佳化。這份報表檢查 margin loan 借款、
      每日利息、維持率、安全緩衝與自動降槓桿。
    </div>
  </header>
  <section>
    <h2>設定總覽</h2>
    <div class="grid">
      {_setting_tile("期間", period_value)}
      {_setting_tile("標的", tickers_value)}
      {_setting_tile("策略", strategies_value)}
      {_setting_tile("目標槓桿", str(report_context.get("target_leverage")))}
      {_setting_tile("借款年利率", str(report_context.get("annual_borrow_rate")))}
      {_setting_tile("維持率", str(report_context.get("maintenance_requirement")))}
      {_setting_tile("安全緩衝門檻", str(report_context.get("min_safety_buffer")))}
      {_setting_tile("降槓桿目標", str(report_context.get("deleverage_to")))}
    </div>
  </section>
  <section>
    <h2>關鍵指標</h2>
    <div class="table-wrap">{metrics_html}</div>
  </section>
  <section>
    <h2>風險曲線</h2>
    {charts}
  </section>
  <section>
    <h2>Audit CSV</h2>
    <div class="links">{links}</div>
  </section>
  <section>
    <h2>限制與警告</h2>
    <ul class="warning">{warnings_html}</ul>
  </section>
</main>
</body>
</html>
"""


def _setting_tile(label: str, value: str) -> str:
    return (
        '<div class="tile">'
        f'<div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(value)}</div>'
        "</div>"
    )


def _render_charts(curves: pd.DataFrame) -> str:
    if curves.empty:
        return "<p>No curve data.</p>"
    curve = curves.copy()
    curve["date"] = pd.to_datetime(curve["date"])
    figure_equity = go.Figure()
    figure_safety = go.Figure()
    for (ticker, strategy), group in curve.groupby(["ticker", "strategy"]):
        name = f"{ticker} {strategy}"
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
    "run_buy_hold_leveraged",
    "run_dca_leveraged",
    "run_rebalance_leveraged",
    "write_leverage_report",
]
