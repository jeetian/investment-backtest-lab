from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

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
    DATA_MODE_HYBRID,
    SELECTOR_ACTUAL_PRIMARY,
    SELECTOR_HYBRID_PRIMARY,
    SELECTOR_SYNTHETIC_PRIMARY,
    build_hybrid_actual_preferred_prices,
    build_monthly_decision_replay_outputs,
    write_monthly_decision_replay_report,
)


@dataclass(frozen=True)
class ReplayPriceLoadResult:
    mode_prices: dict[str, pd.DataFrame]
    source_coverage: pd.DataFrame


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a monthly decision replay report with hybrid-primary Monte Carlo ranking."
        )
    )
    parser.add_argument("--config", default="configs/mvp_example.yaml")
    parser.add_argument("--family", default="qqq")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument(
        "--selector",
        choices=[SELECTOR_ACTUAL_PRIMARY, SELECTOR_SYNTHETIC_PRIMARY, SELECTOR_HYBRID_PRIMARY],
        default=None,
        help=(
            "hybrid_primary uses actual ETF data with synthetic backfill; "
            "actual_primary and synthetic_primary remain audit selectors."
        ),
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
    started_at = datetime.now(UTC).replace(microsecond=0)
    started = perf_counter()

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

    log_stage(
        "start",
        started,
        (
            f"Monthly decision replay started at {started_at.isoformat()} "
            f"selector={selector} scan_mode={scan_mode} "
            f"horizons={replay_config.horizons_years}"
        ),
    )
    products = [
        ProductSpec.from_config(product)
        for product in config.leveraged_etf_lab.products.values()
    ]
    loader = MarketDataLoader(use_cache=True)
    log_stage("load-prices", started, "Loading replay price data.")
    price_result = load_prices_for_selector(
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
    log_stage("load-prices", started, price_frame_summary(price_result.mode_prices))
    if not price_result.source_coverage.empty:
        log_stage("load-prices", started, source_coverage_summary(price_result.source_coverage))
    log_stage("replay", started, "Building full rolling cohort replay. This may take minutes.")
    outputs = build_monthly_decision_replay_outputs(
        mode_prices=price_result.mode_prices,
        products=products,
        optimizer_config=optimizer_config,
        replay_config=replay_config,
        scan_mode=scan_mode,
        selector=selector,
        source_coverage=price_result.source_coverage,
        progress=lambda message: log_stage("replay", started, message),
    )
    output_dir = Path(args.output_dir)
    log_stage("write-report", started, "Loading actual-primary signal and writing report files.")
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
    print(f"MC summary:  {result.mc_summary_path}")
    print(f"MC trials:   {result.mc_trials_path}")
    print(f"Coverage:    {result.source_coverage_path}")
    log_stage("done", started, f"Completed in {perf_counter() - started:.1f} seconds.")


def load_prices_for_selector(
    *,
    selector: str,
    loader: MarketDataLoader,
    universe: list[AssetSpec],
    products: list[ProductSpec],
    actual_start_date: str,
    synthetic_start_date: str,
    end_date: str,
) -> ReplayPriceLoadResult:
    if selector == SELECTOR_ACTUAL_PRIMARY:
        prices = load_actual_product_prices(
            loader=loader,
            universe=universe,
            products=products,
            start_date=actual_start_date,
            end_date=end_date,
        )
        return ReplayPriceLoadResult(
            mode_prices={DATA_MODE_ACTUAL: prices},
            source_coverage=source_coverage_from_aligned_prices(prices, products),
        )
    if selector == SELECTOR_SYNTHETIC_PRIMARY:
        prices = load_synthetic_product_prices(
            loader=loader,
            universe=universe,
            products=products,
            start_date=synthetic_start_date,
            end_date=end_date,
        )
        return ReplayPriceLoadResult(
            mode_prices={DATA_MODE_SYNTHETIC: prices},
            source_coverage=source_coverage_from_aligned_prices(prices, products),
        )
    if selector == SELECTOR_HYBRID_PRIMARY:
        actual = load_actual_product_price_columns(
            loader=loader,
            universe=universe,
            products=products,
            start_date=synthetic_start_date,
            end_date=end_date,
        )
        synthetic = load_synthetic_product_prices(
            loader=loader,
            universe=universe,
            products=products,
            start_date=synthetic_start_date,
            end_date=end_date,
        )
        hybrid = build_hybrid_actual_preferred_prices(
            actual_prices=actual,
            synthetic_prices=synthetic,
            products=products,
        )
        return ReplayPriceLoadResult(
            mode_prices={DATA_MODE_HYBRID: hybrid.prices},
            source_coverage=hybrid.coverage,
        )
    raise ValueError(
        "selector must be actual_primary, synthetic_primary, or hybrid_primary; "
        f"got {selector!r}."
    )


def load_actual_product_prices(
    *,
    loader: MarketDataLoader,
    universe: list[AssetSpec],
    products: list[ProductSpec],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    return load_actual_product_price_columns(
        loader=loader,
        universe=universe,
        products=products,
        start_date=start_date,
        end_date=end_date,
    ).dropna(how="any").sort_index()


def load_actual_product_price_columns(
    *,
    loader: MarketDataLoader,
    universe: list[AssetSpec],
    products: list[ProductSpec],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    frames = []
    for product in products:
        asset = select_or_create_us_etf(universe, product.ticker)
        close = loader.load_asset(
            asset,
            start_date=start_date,
            end_date=end_date,
            adjusted=True,
        ).close()
        frames.append(close.rename(product.ticker))
    return pd.concat(frames, axis=1).sort_index()


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
            "cohort_gate_passed",
            "win_rate_vs_qqq_dca",
            "expected_xirr",
            "median_xirr",
            "p05_xirr",
            "expected_max_drawdown",
            "drawdown_breach_rate",
        ]
    ].copy()
    for column in [
        "win_rate_vs_qqq_dca",
        "expected_xirr",
        "median_xirr",
        "p05_xirr",
        "expected_max_drawdown",
        "drawdown_breach_rate",
    ]:
        display[column] = display[column].map(format_percent_or_blank)
    print(display.to_string(index=False))


def price_frame_summary(mode_prices: dict[str, pd.DataFrame]) -> str:
    parts: list[str] = []
    for mode, frame in mode_prices.items():
        if frame.empty:
            parts.append(f"{mode}: empty")
            continue
        parts.append(
            f"{mode}: rows={len(frame):,}, "
            f"start={frame.index.min().date()}, end={frame.index.max().date()}"
        )
    return "Loaded replay price data. " + "; ".join(parts)


def source_coverage_summary(source_coverage: pd.DataFrame) -> str:
    pieces = []
    for row in source_coverage.itertuples(index=False):
        row_dict = row._asdict()
        pieces.append(
            f"{row_dict.get('ticker')}: replay={row_dict.get('replay_start')}.."
            f"{row_dict.get('replay_end')}, actual={row_dict.get('actual_start')}.."
            f"{row_dict.get('actual_end')}, backfill="
            f"{row_dict.get('synthetic_backfill_start')}.."
            f"{row_dict.get('synthetic_backfill_end')}"
        )
    return "Source coverage. " + "; ".join(pieces)


def source_coverage_from_aligned_prices(
    prices: pd.DataFrame,
    products: list[ProductSpec],
) -> pd.DataFrame:
    rows = []
    for product in products:
        series = prices.get(product.ticker)
        clean = pd.Series(dtype="float64") if series is None else pd.Series(series).dropna()
        rows.append(
            {
                "ticker": product.ticker,
                "replay_start": (
                    "" if clean.empty else pd.Timestamp(clean.index.min()).date().isoformat()
                ),
                "replay_end": (
                    "" if clean.empty else pd.Timestamp(clean.index.max()).date().isoformat()
                ),
                "actual_start": "",
                "actual_end": "",
                "synthetic_backfill_start": "",
                "synthetic_backfill_end": "",
                "splice_date": "",
                "splice_scale": "",
            }
        )
    return pd.DataFrame(rows)


def log_stage(stage: str, started: float, message: str) -> None:
    elapsed = perf_counter() - started
    print(f"[{elapsed:8.1f}s] {stage}: {message}", flush=True)


def format_percent_or_blank(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2%}"


if __name__ == "__main__":
    main()
