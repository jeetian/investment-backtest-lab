from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from investment_backtest_lab.config import load_backtest_config
from investment_backtest_lab.costs import CostModel
from investment_backtest_lab.data import MarketDataLoader
from investment_backtest_lab.ledger_reports import run_buy_and_hold_ledger, write_ledger_report
from investment_backtest_lab.models import AssetSpec, AssetType, DataSource, DividendMode, Market


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an auditable US ledger report.")
    parser.add_argument("--config", default="configs/mvp_example.yaml")
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ"])
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument(
        "--dividend-modes",
        nargs="+",
        default=[DividendMode.CASH.value, DividendMode.REINVEST.value],
        choices=[mode.value for mode in DividendMode],
    )
    args = parser.parse_args()

    config = load_backtest_config(args.config)
    cost_model = CostModel.from_dict(config.cost_model)
    loader = MarketDataLoader(use_cache=True)
    selected_assets = select_assets(config.universe, args.tickers)
    start_date = config.start_date.isoformat()
    end_date = config.end_date.isoformat()

    usd_twd = load_usd_twd_if_needed(
        loader=loader,
        assets=selected_assets,
        base_currency=config.ledger.base_currency,
        start_date=start_date,
        end_date=end_date,
    )

    results = []
    for asset in selected_assets:
        if asset.market != Market.US or asset.currency.upper() != "USD":
            raise ValueError(f"Ledger report v1 supports only USD US assets, got {asset.ticker}.")
        price_frame = loader.load_asset(
            asset,
            start_date=start_date,
            end_date=end_date,
            adjusted=False,
        )
        dividend_frame = loader.load_dividends(
            asset,
            start_date=start_date,
            end_date=end_date,
        )
        for mode in args.dividend_modes:
            results.append(
                run_buy_and_hold_ledger(
                    price_frame=price_frame,
                    dividend_frame=dividend_frame,
                    cost_model=cost_model,
                    initial_cash=config.ledger.initial_cash,
                    dividend_mode=mode,
                    withholding_rate=config.tax.us.dividend_withholding_rate,
                )
            )

    slug = "_".join(ticker.lower().replace("/", "_").replace("=", "_") for ticker in args.tickers)
    report = write_ledger_report(
        results=results,
        base_currency=config.ledger.base_currency,
        usd_twd=usd_twd,
        output_dir=Path(args.output_dir),
        slug=slug,
        config_path=Path(args.config),
    )
    print_terminal_summary(report.metrics, report.warnings)
    print(f"Markdown report: {report.markdown_path}")
    print(f"Metrics CSV:     {report.metrics_path}")
    print(f"Trades CSV:      {report.trades_path}")
    print(f"Dividends CSV:   {report.dividends_path}")
    print(f"Equity CSV:      {report.equity_path}")
    print(f"HTML report:     {report.html_path}")


def select_assets(universe: list[AssetSpec], tickers: list[str]) -> list[AssetSpec]:
    by_ticker = {asset.ticker.upper(): asset for asset in universe}
    selected: list[AssetSpec] = []
    missing: list[str] = []
    for ticker in tickers:
        asset = by_ticker.get(ticker.upper())
        if asset is None:
            missing.append(ticker)
        else:
            selected.append(asset)
    if missing:
        available = ", ".join(sorted(by_ticker))
        raise ValueError(f"Tickers not found in config universe: {missing}. Available: {available}")
    return selected


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
    print("Ledger 指標摘要")
    display = metrics[
        [
            "ticker",
            "dividend_mode",
            "basis",
            "ending_equity",
            "total_return",
            "cagr",
            "max_drawdown",
            "gross_dividends",
            "withholding_tax",
            "fees_paid",
            "final_shares",
            "cash",
        ]
    ].copy()
    for column in ["total_return", "cagr", "max_drawdown"]:
        display[column] = display[column].map(lambda value: f"{float(value):.2%}")
    for column in [
        "ending_equity",
        "gross_dividends",
        "withholding_tax",
        "fees_paid",
        "final_shares",
        "cash",
    ]:
        display[column] = display[column].map(lambda value: f"{float(value):,.2f}")
    print(display.to_string(index=False))
    if warnings:
        print("")
        print("警告")
        for warning in warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
