from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from investment_backtest_lab.config import load_backtest_config
from investment_backtest_lab.data import MarketDataLoader
from investment_backtest_lab.leveraged_etf_lab import (
    DEFAULT_AUDIT_SCENARIO_ID,
    ProductSpec,
    build_leveraged_etf_lab_outputs,
    lab_config_for_scan_mode,
    resolve_scan_mode,
    synthetic_daily_reset_prices,
    write_leveraged_etf_lab_report,
)
from investment_backtest_lab.models import AssetSpec, AssetType, DataSource, Market


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a QQQ/QLD/TQQQ leveraged ETF product optimization report."
    )
    parser.add_argument("--config", default="configs/mvp_example.yaml")
    parser.add_argument("--family", default="qqq")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument(
        "--cash-flow-mode",
        choices=["lump_sum", "dca", "both"],
        default=None,
        help="Run lump-sum, DCA, or both cash-flow modes. Defaults to config.",
    )
    parser.add_argument(
        "--audit-scenario",
        default=DEFAULT_AUDIT_SCENARIO_ID,
        help="Scenario id to render in the Extreme Scenario Audit section.",
    )
    parser.add_argument(
        "--scan-mode",
        choices=["fast", "full"],
        default=None,
        help="fast is the default; full uses the complete configured grid.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--fast", action="store_true", help="Shortcut for --scan-mode fast.")
    mode_group.add_argument("--full", action="store_true", help="Shortcut for --scan-mode full.")
    args = parser.parse_args()

    config = load_backtest_config(args.config)
    scan_mode = normalize_scan_mode(args)
    lab = lab_config_for_scan_mode(config.leveraged_etf_lab, scan_mode)
    family = args.family.lower()
    if family != lab.family:
        raise ValueError(
            f"Config leveraged_etf_lab.family is {lab.family!r}; got --family {family!r}."
        )

    products = [ProductSpec.from_config(product) for product in lab.products.values()]
    loader = MarketDataLoader(use_cache=True)
    actual_prices = load_actual_product_prices(
        loader=loader,
        universe=config.universe,
        products=products,
        start_date=lab.actual_start_date or config.start_date.isoformat(),
        end_date=config.end_date.isoformat(),
    )
    synthetic_prices = load_synthetic_product_prices(
        loader=loader,
        universe=config.universe,
        products=products,
        start_date=lab.synthetic_start_date or config.start_date.isoformat(),
        end_date=config.end_date.isoformat(),
    )
    outputs = build_leveraged_etf_lab_outputs(
        actual_prices=actual_prices,
        synthetic_prices=synthetic_prices,
        products=products,
        lab_config=lab,
        scan_mode=scan_mode,
        cash_flow_mode=args.cash_flow_mode,
    )
    result = write_leveraged_etf_lab_report(
        outputs=outputs,
        output_dir=Path(args.output_dir),
        family=family,
        config_path=Path(args.config),
        audit_scenario_id=args.audit_scenario,
    )
    print_terminal_summary(result.metrics)
    print(f"Scan mode:        {scan_mode}")
    print(f"Cash flow mode:   {args.cash_flow_mode or lab.cash_flow_mode}")
    print(f"HTML report:      {result.html_path}")
    print(f"Metrics CSV:      {result.metrics_path}")
    print(f"Curves CSV:       {result.curves_path}")
    print(f"Allocations CSV:  {result.allocations_path}")
    print(f"Extreme Audit:    {result.extreme_audit_path}")
    print(f"DCA Optimizer:    {result.dca_optimizer_path}")
    print(f"Payload JSON:     {result.payload_path}")


def load_actual_product_prices(
    *,
    loader: MarketDataLoader,
    universe: list[AssetSpec],
    products: list[ProductSpec],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    assets = [select_or_create_us_etf(universe, product.ticker) for product in products]
    frames = [
        loader.load_asset(
            asset,
            start_date=start_date,
            end_date=end_date,
            adjusted=True,
        ).close()
        for asset in assets
    ]
    return pd.concat(frames, axis=1).dropna(how="any").sort_index()


def load_synthetic_product_prices(
    *,
    loader: MarketDataLoader,
    universe: list[AssetSpec],
    products: list[ProductSpec],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    base = products[0]
    base_asset = select_or_create_us_etf(universe, base.ticker)
    base_close = loader.load_asset(
        base_asset,
        start_date=start_date,
        end_date=end_date,
        adjusted=True,
    ).close()
    return synthetic_daily_reset_prices(base_close, products)


def select_or_create_us_etf(universe: list[AssetSpec], ticker: str) -> AssetSpec:
    by_ticker = {asset.ticker.upper(): asset for asset in universe}
    asset = by_ticker.get(ticker.upper())
    if asset is None:
        return AssetSpec(ticker.upper(), Market.US, AssetType.ETF, "USD", DataSource.YFINANCE)
    if asset.market != Market.US or asset.currency != "USD" or asset.asset_type != AssetType.ETF:
        raise ValueError(f"Leveraged ETF lab supports only USD US ETFs, got {asset}.")
    return asset


def print_terminal_summary(metrics: pd.DataFrame) -> None:
    print("Leveraged ETF product lab summary")
    display = (
        metrics.sort_values(["data_mode", "cash_flow_mode", "rank"])
        .groupby(["data_mode", "cash_flow_mode"])
        .head(8)[
            [
                "data_mode",
                "cash_flow_mode",
                "rank",
                "robust_rank",
                "scenario_label",
                "total_contributed",
                "ending_equity",
                "simple_cash_return",
                "xirr",
                "cagr",
                "max_drawdown",
                "robust_score",
                "calmar",
                "sortino",
                "risk_flag",
            ]
        ]
        .copy()
    )
    for column in ["simple_cash_return", "xirr", "cagr", "max_drawdown"]:
        display[column] = display[column].map(format_percent_or_blank)
    for column in ["calmar", "sortino", "robust_score"]:
        display[column] = display[column].map(format_number_or_blank)
    for column in ["total_contributed", "ending_equity"]:
        display[column] = display[column].map(format_money_or_blank)
    print(display.to_string(index=False))


def normalize_scan_mode(args: argparse.Namespace) -> str:
    return resolve_scan_mode(
        getattr(args, "scan_mode", None),
        fast=bool(getattr(args, "fast", False)),
        full=bool(getattr(args, "full", False)),
    )


def format_percent_or_blank(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def format_number_or_blank(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def format_money_or_blank(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):,.2f}"


if __name__ == "__main__":
    main()
