from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from investment_backtest_lab.config import load_backtest_config
from investment_backtest_lab.data import MarketDataLoader
from investment_backtest_lab.dca_policy_optimizer import (
    DATA_MODE_ACTUAL,
    DATA_MODE_SYNTHETIC,
    policy_config_for_scan_mode,
)
from investment_backtest_lab.leveraged_etf_lab import (
    ProductSpec,
    resolve_scan_mode,
    synthetic_daily_reset_prices,
)
from investment_backtest_lab.models import AssetSpec, AssetType, DataSource, Market
from investment_backtest_lab.monthly_decision_replay import (
    SELECTOR_ACTUAL_PRIMARY,
    SELECTOR_SYNTHETIC_PRIMARY,
    build_monthly_decision_replay_outputs,
    write_monthly_decision_replay_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a monthly decision replay report with synthetic-primary ranking."
    )
    parser.add_argument("--config", default="configs/mvp_example.yaml")
    parser.add_argument("--family", default="qqq")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument(
        "--selector",
        choices=[SELECTOR_ACTUAL_PRIMARY, SELECTOR_SYNTHETIC_PRIMARY],
        default=None,
        help="actual_primary keeps old priority; synthetic_primary uses stress-first ranking.",
    )
    parser.add_argument(
        "--scan-mode",
        choices=["fast", "full"],
        default=None,
        help="fast is the default; full uses the complete configured policy grid.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--fast", action="store_true", help="Shortcut for --scan-mode fast.")
    mode_group.add_argument("--full", action="store_true", help="Shortcut for --scan-mode full.")
    args = parser.parse_args()

    config = load_backtest_config(args.config)
    replay_config = config.monthly_decision_replay
    selector = (args.selector or replay_config.selector).lower()
    scan_mode = resolve_scan_mode(
        args.scan_mode,
        fast=bool(args.fast),
        full=bool(args.full),
    )
    family = args.family.lower()
    if family != replay_config.family:
        raise ValueError(
            f"Config monthly_decision_replay.family is {replay_config.family!r}; "
            f"got --family {family!r}."
        )
    optimizer_config = policy_config_for_scan_mode(config.dca_policy_optimizer, scan_mode)
    if family != optimizer_config.family:
        raise ValueError(
            f"Config dca_policy_optimizer.family is {optimizer_config.family!r}; "
            f"got --family {family!r}."
        )

    products = [
        ProductSpec.from_config(product)
        for product in config.leveraged_etf_lab.products.values()
    ]
    loader = MarketDataLoader(use_cache=True)
    mode_prices = load_prices_for_selector(
        selector=selector,
        loader=loader,
        universe=config.universe,
        products=products,
        actual_start_date=config.leveraged_etf_lab.actual_start_date
        or config.start_date.isoformat(),
        synthetic_start_date=config.leveraged_etf_lab.synthetic_start_date
        or config.start_date.isoformat(),
        end_date=config.end_date.isoformat(),
    )
    outputs = build_monthly_decision_replay_outputs(
        mode_prices=mode_prices,
        products=products,
        optimizer_config=optimizer_config,
        replay_config=replay_config,
        scan_mode=scan_mode,
        selector=selector,
    )
    output_dir = Path(args.output_dir)
    actual_primary_signal = load_actual_primary_signal(output_dir, family)
    result = write_monthly_decision_replay_report(
        outputs=outputs,
        output_dir=output_dir,
        family=family,
        actual_primary_signal=actual_primary_signal,
    )
    print_terminal_summary(result.ranking, selector=selector, scan_mode=scan_mode)
    print(f"HTML report: {result.html_path}")
    print(f"Decisions:   {result.decisions_path}")
    print(f"Cohorts:     {result.cohorts_path}")
    print(f"Ranking:     {result.ranking_path}")
    print(f"Equity:      {result.equity_path}")


def load_prices_for_selector(
    *,
    selector: str,
    loader: MarketDataLoader,
    universe: list[AssetSpec],
    products: list[ProductSpec],
    actual_start_date: str,
    synthetic_start_date: str,
    end_date: str,
) -> dict[str, pd.DataFrame]:
    if selector == SELECTOR_ACTUAL_PRIMARY:
        return {
            DATA_MODE_ACTUAL: load_actual_product_prices(
                loader=loader,
                universe=universe,
                products=products,
                start_date=actual_start_date,
                end_date=end_date,
            )
        }
    if selector == SELECTOR_SYNTHETIC_PRIMARY:
        return {
            DATA_MODE_SYNTHETIC: load_synthetic_product_prices(
                loader=loader,
                universe=universe,
                products=products,
                start_date=synthetic_start_date,
                end_date=end_date,
            )
        }
    raise ValueError(f"selector must be actual_primary or synthetic_primary, got {selector!r}.")


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
        return AssetSpec(
            ticker.upper(),
            Market.US,
            AssetType.ETF,
            "USD",
            DataSource.YFINANCE,
        )
    if asset.market != Market.US or asset.currency != "USD" or asset.asset_type != AssetType.ETF:
        raise ValueError(f"Monthly decision replay supports only USD US ETFs, got {asset}.")
    return asset


def load_actual_primary_signal(output_dir: Path, family: str) -> pd.DataFrame:
    path = output_dir / f"dca_policy_optimizer_{family.lower()}_allocation_signal.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def print_terminal_summary(ranking: pd.DataFrame, *, selector: str, scan_mode: str) -> None:
    print("Monthly decision replay summary")
    print(f"Selector:  {selector}")
    print(f"Scan mode: {scan_mode}")
    if ranking.empty:
        print("No replay ranking rows were produced.")
        return
    display = ranking.head(8)[
        [
            "replay_rank",
            "scenario_label",
            "eligible_for_monthly_signal",
            "win_rate_vs_qqq_dca",
            "median_xirr",
            "worst_xirr",
            "worst_max_drawdown",
            "drawdown_breach_rate",
        ]
    ].copy()
    for column in [
        "win_rate_vs_qqq_dca",
        "median_xirr",
        "worst_xirr",
        "worst_max_drawdown",
        "drawdown_breach_rate",
    ]:
        display[column] = display[column].map(format_percent_or_blank)
    print(display.to_string(index=False))


def format_percent_or_blank(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2%}"


if __name__ == "__main__":
    main()
