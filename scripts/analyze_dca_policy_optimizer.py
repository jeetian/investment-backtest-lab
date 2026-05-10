from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from investment_backtest_lab.config import load_backtest_config
from investment_backtest_lab.costs import CostModel
from investment_backtest_lab.data import MarketDataLoader
from investment_backtest_lab.dca_policy_optimizer import (
    build_dca_policy_optimizer_outputs,
    policy_config_for_scan_mode,
    write_dca_policy_optimizer_report,
)
from investment_backtest_lab.leveraged_etf_lab import (
    ProductSpec,
    resolve_scan_mode,
    synthetic_daily_reset_prices,
)
from investment_backtest_lab.models import AssetSpec, AssetType, DataSource, Market
from investment_backtest_lab.tw_total_return import (
    TW50_FAMILY,
    build_tw50_total_return_inputs,
    write_tw50_audit_files,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a QQQ/QLD/TQQQ DCA policy optimization report."
    )
    parser.add_argument("--config", default="configs/mvp_example.yaml")
    parser.add_argument("--family", default="qqq")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument(
        "--scan-mode",
        choices=["fast", "full"],
        default=None,
        help="fast is the default; full uses the complete configured policy grid.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--fast", action="store_true", help="Shortcut for --scan-mode fast.")
    mode_group.add_argument("--full", action="store_true", help="Shortcut for --scan-mode full.")
    cohort_group = parser.add_mutually_exclusive_group()
    cohort_group.add_argument(
        "--cohort-validation",
        dest="cohort_validation",
        action="store_true",
        default=None,
        help="Enable rolling cohort validation.",
    )
    cohort_group.add_argument(
        "--no-cohort-validation",
        dest="cohort_validation",
        action="store_false",
        help="Skip rolling cohort validation for a faster run.",
    )
    args = parser.parse_args()

    config = load_backtest_config(args.config)
    scan_mode = resolve_scan_mode(
        args.scan_mode,
        fast=bool(args.fast),
        full=bool(args.full),
    )
    optimizer_config = policy_config_for_scan_mode(config.dca_policy_optimizer, scan_mode)
    family = args.family.lower()
    if family != optimizer_config.family:
        raise ValueError(
            f"Config dca_policy_optimizer.family is {optimizer_config.family!r}; "
            f"got --family {family!r}."
        )
    products = [
        ProductSpec.from_config(product)
        for product in config.leveraged_etf_lab.products.values()
    ]
    cost_model = CostModel.from_dict(config.cost_model)
    product_assets = product_asset_map(config.universe, products)
    loader = MarketDataLoader(use_cache=True)
    output_dir = Path(args.output_dir)
    if family == TW50_FAMILY:
        tw50 = build_tw50_total_return_inputs(
            start_date=config.leveraged_etf_lab.synthetic_start_date
            or config.start_date.isoformat(),
            end_date=config.end_date.isoformat(),
            products=products,
        )
        actual_prices = tw50.actual_prices
        synthetic_prices = tw50.hybrid_prices
        write_tw50_audit_files(output_dir=output_dir, result=tw50)
    else:
        actual_prices = load_actual_product_prices(
            loader=loader,
            universe=config.universe,
            products=products,
            start_date=config.leveraged_etf_lab.actual_start_date
            or config.start_date.isoformat(),
            end_date=config.end_date.isoformat(),
        )
        synthetic_prices = load_synthetic_product_prices(
            loader=loader,
            universe=config.universe,
            products=products,
            start_date=config.leveraged_etf_lab.synthetic_start_date
            or config.start_date.isoformat(),
            end_date=config.end_date.isoformat(),
        )
    outputs = build_dca_policy_optimizer_outputs(
        actual_prices=actual_prices,
        synthetic_prices=synthetic_prices,
        products=products,
        config=optimizer_config,
        scan_mode=scan_mode,
        cohort_validation=args.cohort_validation,
        cost_model=cost_model,
        product_assets=product_assets,
    )
    result = write_dca_policy_optimizer_report(
        outputs=outputs,
        output_dir=output_dir,
        family=family,
        config_path=Path(args.config),
    )
    print_terminal_summary(result.metrics, result.allocation_signal)
    print(f"Scan mode:       {scan_mode}")
    print("Cost mode:       net_of_cost")
    print(f"HTML report:     {result.html_path}")
    print(f"Metrics CSV:     {result.metrics_path}")
    print(f"Policy CSV:      {result.policy_path}")
    print(f"Walk-forward:    {result.walk_forward_path}")
    print(f"Cohorts CSV:     {result.cohorts_path}")
    print(f"Cohort summary:  {result.cohort_summary_path}")
    print(f"Allocation:      {result.allocation_signal_path}")
    print(f"Explainability:  {result.signal_explainability_path}")
    print(f"Payload JSON:    {result.payload_path}")


def load_actual_product_prices(
    *,
    loader: MarketDataLoader,
    universe: list[AssetSpec],
    products: list[ProductSpec],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    assets = [select_or_create_supported_asset(universe, product.ticker) for product in products]
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
    base_asset = select_or_create_supported_asset(universe, base.ticker)
    base_close = loader.load_asset(
        base_asset,
        start_date=start_date,
        end_date=end_date,
        adjusted=True,
    ).close()
    return synthetic_daily_reset_prices(base_close, products)


def select_or_create_supported_asset(universe: list[AssetSpec], ticker: str) -> AssetSpec:
    by_ticker = {asset.ticker.upper(): asset for asset in universe}
    asset = by_ticker.get(ticker.upper())
    if asset is None:
        return AssetSpec(ticker.upper(), Market.US, AssetType.ETF, "USD", DataSource.YFINANCE)
    if asset.asset_type != AssetType.ETF:
        raise ValueError(f"DCA policy optimizer supports ETF product assets, got {asset}.")
    return asset


def product_asset_map(
    universe: list[AssetSpec],
    products: list[ProductSpec],
) -> dict[str, AssetSpec]:
    return {
        product.ticker: select_or_create_supported_asset(universe, product.ticker)
        for product in products
    }


def print_terminal_summary(metrics: pd.DataFrame, current_signal: pd.DataFrame) -> None:
    print("DCA policy optimizer summary")
    display = (
        metrics.sort_values(["data_mode", "rank"])
        .groupby("data_mode")
        .head(8)[
            [
                "data_mode",
                "rank",
                "scenario_label",
                "validation_status",
                "risk_flag",
                "total_contributed",
                "ending_equity",
                "total_trade_cost",
                "cost_drag_on_contributed",
                "xirr",
                "max_drawdown",
                "effective_leverage_avg",
                "effective_leverage_max",
            ]
        ]
        .copy()
    )
    for column in ["xirr", "max_drawdown", "cost_drag_on_contributed"]:
        display[column] = display[column].map(format_percent_or_blank)
    for column in ["total_contributed", "ending_equity", "total_trade_cost"]:
        display[column] = display[column].map(format_money_or_blank)
    for column in ["effective_leverage_avg", "effective_leverage_max"]:
        display[column] = display[column].map(format_leverage_or_blank)
    print(display.to_string(index=False))
    if not current_signal.empty:
        signal = current_signal.iloc[0]
        print()
        print("Top monthly allocation research signal")
        print(f"as_of:     {signal['as_of_date']}")
        print(f"strategy:  {signal['scenario_label']}")
        print(f"regime:    {signal['regime']}")
        print(f"target:    {float(signal['target_effective_leverage']):.2f}x")
        print(f"rebalance: {signal.get('next_rebalance_date', '')}")
        print(f"monitor:   {signal.get('next_monitor_date', '')}")
        print(f"weights:   {format_weight_summary(signal)}")


def format_percent_or_blank(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def format_money_or_blank(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):,.2f}"


def format_leverage_or_blank(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2f}x"


def format_weight_summary(row: pd.Series) -> str:
    parts = []
    for column in row.index:
        if (
            str(column).endswith("_weight")
            and str(column) != "cash_weight"
            and not str(column).startswith("previous_")
        ):
            ticker = str(column).removesuffix("_weight")
            value = row.get(column, 0.0)
            if value == "":
                continue
            parts.append(f"{ticker} {float(value):.0%}")
    return ", ".join(parts)


if __name__ == "__main__":
    main()
