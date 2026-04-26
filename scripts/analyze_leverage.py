from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from investment_backtest_lab.config import load_backtest_config
from investment_backtest_lab.costs import CostModel
from investment_backtest_lab.data import MarketDataLoader
from investment_backtest_lab.leverage_reports import (
    run_buy_hold_leveraged,
    run_dca_leveraged,
    run_rebalance_leveraged,
    write_leverage_report,
)
from investment_backtest_lab.models import (
    AssetSpec,
    AssetType,
    DataSource,
    LeverageKind,
    Market,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a US ETF margin-loan risk report.")
    parser.add_argument("--config", default="configs/mvp_example.yaml")
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ"])
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["buy_hold_leveraged", "dca_leveraged", "rebalance_leveraged"],
        choices=[
            "buy_hold",
            "dca",
            "rebalance",
            "buy_hold_leveraged",
            "dca_leveraged",
            "rebalance_leveraged",
        ],
    )
    args = parser.parse_args()

    config = load_backtest_config(args.config)
    if not config.leverage.enabled:
        raise ValueError("leverage.enabled must be true before running analyze_leverage.py.")
    if config.leverage.kind != LeverageKind.MARGIN_LOAN:
        raise ValueError(
            "analyze_leverage.py models margin loans only. leveraged_etf_product should be "
            "backtested as a normal price series with a separate limitation note."
        )

    cost_model = CostModel.from_dict(config.cost_model)
    loader = MarketDataLoader(use_cache=True)
    selected_assets = select_us_etf_assets(config.universe, args.tickers)
    start_date = config.start_date.isoformat()
    end_date = config.end_date.isoformat()
    normalized_strategies = normalize_strategies(args.strategies)
    usd_twd = load_usd_twd_if_needed(
        loader=loader,
        assets=selected_assets,
        base_currency=config.ledger.base_currency,
        start_date=start_date,
        end_date=end_date,
    )

    price_frames = [
        loader.load_asset(
            asset,
            start_date=start_date,
            end_date=end_date,
            adjusted=False,
        )
        for asset in selected_assets
    ]

    results = []
    for price_frame in price_frames:
        if "buy_hold_leveraged" in normalized_strategies:
            results.append(
                run_buy_hold_leveraged(
                    price_frame=price_frame,
                    cost_model=cost_model,
                    initial_cash=config.ledger.initial_cash,
                    leverage=config.leverage,
                )
            )
        if "dca_leveraged" in normalized_strategies:
            results.append(
                run_dca_leveraged(
                    price_frame=price_frame,
                    cost_model=cost_model,
                    contribution=config.dca.contribution,
                    frequency=config.dca.frequency,
                    leverage=config.leverage,
                )
            )
    if "rebalance_leveraged" in normalized_strategies:
        target_weights = selected_target_weights(config.rebalance.target_weights, selected_assets)
        results.append(
            run_rebalance_leveraged(
                price_frames=price_frames,
                cost_model=cost_model,
                initial_cash=config.ledger.initial_cash,
                target_weights=target_weights,
                frequency=config.rebalance.frequency,
                leverage=config.leverage,
            )
        )

    slug = "_".join(ticker.lower().replace("/", "_").replace("=", "_") for ticker in args.tickers)
    report = write_leverage_report(
        results=results,
        output_dir=Path(args.output_dir),
        slug=slug,
        base_currency=config.ledger.base_currency,
        usd_twd=usd_twd,
        config_path=Path(args.config),
        report_context=build_report_context(config, selected_assets, normalized_strategies),
    )
    print_terminal_summary(report.metrics, report.warnings)
    print(f"Markdown report: {report.markdown_path}")
    print(f"Metrics CSV:     {report.metrics_path}")
    print(f"Trades CSV:      {report.trades_path}")
    print(f"Interest CSV:    {report.interest_path}")
    print(f"Events CSV:      {report.events_path}")
    print(f"Cash flows CSV:  {report.cash_flows_path}")
    print(f"Curve CSV:       {report.curves_path}")
    print(f"Positions CSV:   {report.positions_path}")
    print(f"HTML report:     {report.html_path}")


def normalize_strategies(strategies: list[str]) -> list[str]:
    aliases = {
        "buy_hold": "buy_hold_leveraged",
        "dca": "dca_leveraged",
        "rebalance": "rebalance_leveraged",
        "buy_hold_leveraged": "buy_hold_leveraged",
        "dca_leveraged": "dca_leveraged",
        "rebalance_leveraged": "rebalance_leveraged",
    }
    return sorted({aliases[strategy] for strategy in strategies})


def build_report_context(
    config: Any,
    selected_assets: list[AssetSpec],
    strategies: list[str],
) -> dict[str, Any]:
    return {
        "start_date": config.start_date.isoformat(),
        "end_date": config.end_date.isoformat(),
        "tickers": [asset.ticker for asset in selected_assets],
        "strategies": strategies,
        "target_leverage": f"{config.leverage.target_leverage:.2f}x",
        "max_leverage": f"{config.leverage.max_leverage:.2f}x",
        "annual_borrow_rate": f"{config.leverage.annual_borrow_rate:.2%}",
        "maintenance_requirement": f"{config.leverage.maintenance_requirement:.2%}",
        "min_safety_buffer": f"{config.leverage.min_safety_buffer:.2%}",
        "deleverage_to": f"{config.leverage.deleverage_to:.2f}x",
        "initial_cash": config.ledger.initial_cash,
        "dca_contribution": config.dca.contribution,
        "account_currency": config.ledger.account_currency,
        "base_currency": config.ledger.base_currency,
        "generated_at": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M:%S %Z"),
    }


def select_us_etf_assets(universe: list[AssetSpec], tickers: list[str]) -> list[AssetSpec]:
    by_ticker = {asset.ticker.upper(): asset for asset in universe}
    selected: list[AssetSpec] = []
    missing: list[str] = []
    invalid: list[str] = []
    for ticker in tickers:
        asset = by_ticker.get(ticker.upper())
        if asset is None:
            missing.append(ticker)
            continue
        if (
            asset.market != Market.US
            or asset.currency.upper() != "USD"
            or asset.asset_type != AssetType.ETF
        ):
            invalid.append(ticker)
            continue
        selected.append(asset)
    if missing:
        available = ", ".join(sorted(by_ticker))
        raise ValueError(f"Tickers not found in config universe: {missing}. Available: {available}")
    if invalid:
        raise ValueError(f"Leverage v1 supports only USD US ETFs, got: {invalid}")
    return selected


def selected_target_weights(
    target_weights: dict[str, float],
    selected_assets: list[AssetSpec],
) -> dict[str, float]:
    selected_tickers = [asset.ticker for asset in selected_assets]
    missing = [ticker for ticker in selected_tickers if ticker not in target_weights]
    if missing:
        raise ValueError(f"Missing leverage rebalance target weights for: {missing}")
    weights = {ticker: float(target_weights[ticker]) for ticker in selected_tickers}
    total_weight = sum(weights.values())
    if not 0.999 <= total_weight <= 1.001:
        raise ValueError(
            "Selected leverage rebalance target weights must sum to 1.0, "
            f"got {total_weight:.4f}."
        )
    return weights


def load_usd_twd_if_needed(
    *,
    loader: MarketDataLoader,
    assets: list[AssetSpec],
    base_currency: str,
    start_date: str,
    end_date: str,
) -> pd.Series | None:
    needs_usd_twd = any(asset.currency.upper() == "USD" for asset in assets)
    if base_currency.upper() != "TWD" or not needs_usd_twd:
        return None
    fx_asset = AssetSpec("USDTWD=X", Market.FX, AssetType.FX, "TWD", DataSource.YFINANCE)
    return loader.load_asset(fx_asset, start_date=start_date, end_date=end_date).close()


def print_terminal_summary(metrics: pd.DataFrame, warnings: list[str]) -> None:
    print("Leverage risk summary")
    display = metrics[
        [
            "ticker",
            "strategy",
            "ending_equity_usd",
            "simple_cash_return",
            "max_drawdown",
            "interest_paid",
            "final_debt",
            "max_actual_leverage",
            "worst_safety_buffer",
            "margin_call_count",
            "forced_deleverage_count",
        ]
    ].copy()
    for column in ["simple_cash_return", "max_drawdown", "worst_safety_buffer"]:
        display[column] = display[column].map(format_percent_or_blank)
    for column in [
        "ending_equity_usd",
        "interest_paid",
        "final_debt",
        "max_actual_leverage",
    ]:
        display[column] = display[column].map(format_number_or_blank)
    print(display.to_string(index=False))
    if warnings:
        print("")
        print("Warnings")
        for warning in warnings:
            print(f"- {warning}")


def format_percent_or_blank(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def format_number_or_blank(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):,.2f}"


if __name__ == "__main__":
    main()
