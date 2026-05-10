from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import pandas as pd

from investment_backtest_lab.config import load_backtest_config
from investment_backtest_lab.costs import CostModel
from investment_backtest_lab.data import MarketDataLoader
from investment_backtest_lab.dca_policy_optimizer import (
    DATA_MODE_ACTUAL,
    DATA_MODE_SYNTHETIC,
    policy_config_for_scan_mode,
)
from investment_backtest_lab.external_signals import (
    external_signal_columns_for_feature_set,
    load_external_signal_features,
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
from investment_backtest_lab.strategy_search import load_frozen_sentiment
from investment_backtest_lab.tw_total_return import (
    TW50_FAMILY,
    build_tw50_total_return_inputs,
    write_tw50_audit_files,
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
    parser.add_argument(
        "--optuna-scenarios",
        default=None,
        help="Optional Optuna shortlist CSV to append as replay scenarios.",
    )
    parser.add_argument(
        "--cost-stress-multiplier",
        type=float,
        default=1.0,
        help="Stress-only multiplier applied to estimated trade costs. Default: 1.0.",
    )
    parser.add_argument(
        "--report-suffix",
        default="",
        help="Optional suffix for replay output files, e.g. stress_2x.",
    )
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
    cost_model = CostModel.from_dict(config.cost_model)
    product_assets = product_asset_map(config.universe, products)
    loader = MarketDataLoader(use_cache=True)
    output_dir = Path(args.output_dir)
    if args.cost_stress_multiplier <= 0:
        raise ValueError("--cost-stress-multiplier must be positive.")
    cost_note = "Cost mode=net_of_cost."
    if args.cost_stress_multiplier != 1.0:
        cost_note += f" Stress multiplier={args.cost_stress_multiplier:g}x."
    log_stage("start", started, cost_note)
    log_stage("load-prices", started, "Loading replay price data.")
    price_result = load_prices_for_selector(
        selector=selector,
        family=family,
        output_dir=output_dir,
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
    optuna_scenarios = load_optuna_scenarios(args.optuna_scenarios)
    optuna_sentiment = None
    optuna_external_signals = None
    if not optuna_scenarios.empty:
        trading_index = next(iter(price_result.mode_prices.values())).index
        if config.strategy_search.include_sentiment:
            optuna_sentiment = load_frozen_sentiment(
                Path(config.strategy_search.sentiment_path),
                trading_index=trading_index,
            )
        if config.strategy_search.include_external_signals:
            optuna_external_signals = load_external_signal_features(
                Path(config.strategy_search.external_signals_path),
                trading_index=trading_index,
                feature_columns=external_signal_columns_for_feature_set(
                    config.strategy_search.external_signal_feature_set
                ),
                require_complete=config.strategy_search.external_signal_feature_set == "core",
            )
            if config.strategy_search.external_signal_feature_set == "core":
                price_result = trim_price_result_to_external_coverage(
                    price_result=price_result,
                    external_signals=optuna_external_signals,
                )
                optuna_external_signals = optuna_external_signals.reindex(
                    next(iter(price_result.mode_prices.values())).index
                )
                log_stage(
                    "replay",
                    started,
                    (
                        "Using core external official replay window "
                        f"{optuna_external_signals.index.min().date().isoformat()}.."
                        f"{optuna_external_signals.index.max().date().isoformat()}."
                    ),
                )
        log_stage(
            "replay",
            started,
            f"Loaded Optuna replay shortlist: {len(optuna_scenarios)} scenarios.",
        )
    log_stage("replay", started, "Building full rolling cohort replay. This may take minutes.")
    outputs = build_monthly_decision_replay_outputs(
        mode_prices=price_result.mode_prices,
        products=products,
        optimizer_config=optimizer_config,
        replay_config=replay_config,
        scan_mode=scan_mode,
        selector=selector,
        source_coverage=price_result.source_coverage,
        cost_model=cost_model,
        product_assets=product_assets,
        optuna_scenarios=optuna_scenarios,
        optuna_sentiment=optuna_sentiment,
        optuna_external_signals=optuna_external_signals,
        cost_multiplier=float(args.cost_stress_multiplier),
        progress=lambda message: log_stage("replay", started, message),
    )
    log_stage("write-report", started, "Loading actual-primary signal and writing report files.")
    actual_primary_signal = load_actual_primary_signal(output_dir, family)
    result = write_monthly_decision_replay_report(
        outputs=outputs,
        output_dir=output_dir,
        family=family,
        actual_primary_signal=actual_primary_signal,
        report_suffix=str(args.report_suffix),
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
    print(f"Compare:     {result.compare_payload_path}")
    print(f"Trade audit: {result.trade_audit_path}")
    log_stage("done", started, f"Completed in {perf_counter() - started:.1f} seconds.")


def trim_price_result_to_external_coverage(
    *,
    price_result: ReplayPriceLoadResult,
    external_signals: pd.DataFrame,
) -> ReplayPriceLoadResult:
    if external_signals.empty:
        raise ValueError("Core external replay requires non-empty external signal coverage.")
    start = pd.Timestamp(external_signals.index.min())
    end = pd.Timestamp(external_signals.index.max())
    trimmed_prices = {
        mode: prices.loc[(prices.index >= start) & (prices.index <= end)].copy()
        for mode, prices in price_result.mode_prices.items()
    }
    for mode, prices in trimmed_prices.items():
        if prices.empty:
            raise ValueError(
                "Core external replay produced empty prices after applying official "
                f"window {start.date().isoformat()}..{end.date().isoformat()} for {mode}."
            )
    coverage = price_result.source_coverage.copy()
    if not coverage.empty:
        coverage["replay_start"] = start.date().isoformat()
        coverage["replay_end"] = end.date().isoformat()
        note = (
            "official_core_external_replay_window: "
            f"{start.date().isoformat()}..{end.date().isoformat()}"
        )
        if "source_notes" in coverage.columns:
            coverage["source_notes"] = coverage["source_notes"].fillna("").astype(str)
            coverage["source_notes"] = coverage["source_notes"].map(
                lambda value: f"{value}; {note}" if value else note
            )
        else:
            coverage["source_notes"] = note
    return ReplayPriceLoadResult(mode_prices=trimmed_prices, source_coverage=coverage)


def load_prices_for_selector(
    *,
    selector: str,
    family: str,
    output_dir: Path,
    loader: MarketDataLoader,
    universe: list[AssetSpec],
    products: list[ProductSpec],
    actual_start_date: str,
    synthetic_start_date: str,
    end_date: str,
) -> ReplayPriceLoadResult:
    if family == TW50_FAMILY:
        result = build_tw50_total_return_inputs(
            start_date=synthetic_start_date,
            end_date=end_date,
            products=products,
        )
        write_tw50_audit_files(output_dir=output_dir, result=result)
        if selector == SELECTOR_ACTUAL_PRIMARY:
            return ReplayPriceLoadResult(
                mode_prices={DATA_MODE_ACTUAL: result.actual_prices},
                source_coverage=result.source_coverage,
            )
        if selector == SELECTOR_SYNTHETIC_PRIMARY:
            return ReplayPriceLoadResult(
                mode_prices={DATA_MODE_SYNTHETIC: result.hybrid_prices},
                source_coverage=result.source_coverage,
            )
        if selector == SELECTOR_HYBRID_PRIMARY:
            return ReplayPriceLoadResult(
                mode_prices={DATA_MODE_HYBRID: result.hybrid_prices},
                source_coverage=result.source_coverage,
            )
        raise ValueError(
            "selector must be actual_primary, synthetic_primary, or hybrid_primary; "
            f"got {selector!r}."
        )

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
        asset = select_or_create_supported_asset(universe, product.ticker)
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
        return AssetSpec(
            ticker.upper(),
            Market.US,
            AssetType.ETF,
            "USD",
            DataSource.YFINANCE,
        )
    if asset.asset_type != AssetType.ETF:
        raise ValueError(f"Monthly decision replay supports ETF product assets, got {asset}.")
    return asset


def product_asset_map(
    universe: list[AssetSpec],
    products: list[ProductSpec],
) -> dict[str, AssetSpec]:
    return {
        product.ticker: select_or_create_supported_asset(universe, product.ticker)
        for product in products
    }


def load_actual_primary_signal(output_dir: Path, family: str) -> pd.DataFrame:
    path = output_dir / f"dca_policy_optimizer_{family.lower()}_allocation_signal.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_optuna_scenarios(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    scenario_path = Path(path)
    if not scenario_path.exists():
        raise FileNotFoundError(f"Missing Optuna scenario CSV: {scenario_path}")
    return pd.read_csv(scenario_path)


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
            "win_rate_vs_benchmark",
            "expected_xirr",
            "median_xirr",
            "p05_xirr",
            "expected_max_drawdown",
            "drawdown_breach_rate",
            "total_trade_cost",
            "cost_drag_on_contributed",
        ]
    ].copy()
    for column in [
        "win_rate_vs_benchmark",
        "expected_xirr",
        "median_xirr",
        "p05_xirr",
        "expected_max_drawdown",
        "drawdown_breach_rate",
        "cost_drag_on_contributed",
    ]:
        display[column] = display[column].map(format_percent_or_blank)
    display["total_trade_cost"] = display["total_trade_cost"].map(format_number_or_blank)
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


def format_number_or_blank(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):,.2f}"


if __name__ == "__main__":
    main()
