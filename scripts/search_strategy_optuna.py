from __future__ import annotations

import argparse
from pathlib import Path

from investment_backtest_lab.config import load_backtest_config
from investment_backtest_lab.costs import CostModel
from investment_backtest_lab.external_signals import (
    external_signal_audit_status,
    write_external_signal_audit,
)
from investment_backtest_lab.leveraged_etf_lab import ProductSpec
from investment_backtest_lab.models import AssetSpec, AssetType, DataSource, Market
from investment_backtest_lab.pre_optimization_audit import AUDIT_FAIL, write_pre_optimization_audit
from investment_backtest_lab.strategy_search import (
    SUPPORTED_OBJECTIVE_PROFILES,
    build_strategy_search_context,
    run_optuna_strategy_search,
)
from investment_backtest_lab.tw_total_return import TW50_FAMILY, build_tw50_total_return_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Optuna-first TW50 strategy search.")
    parser.add_argument("--config", default="configs/tw50_example.yaml")
    parser.add_argument("--family", default="tw50")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--study-name", default="tw50_v1")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--storage", default=None)
    parser.add_argument(
        "--objective-profile",
        choices=SUPPORTED_OBJECTIVE_PROFILES,
        default=None,
        help=(
            "Search/report profile. return_first prioritizes XIRR and win rate "
            "vs the configured benchmark."
        ),
    )
    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=None,
        help="Stop adding new trials after this many hours, then export reports.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Print one progress line after this many completed trials; use 0 to silence.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Rebuild reports from an existing SQLite study without adding trials.",
    )
    parser.add_argument("--force", action="store_true", help="Run even if audit has fail gates.")
    sentiment_group = parser.add_mutually_exclusive_group()
    sentiment_group.add_argument("--include-sentiment", action="store_true", default=None)
    sentiment_group.add_argument("--no-sentiment", action="store_false", dest="include_sentiment")
    external_group = parser.add_mutually_exclusive_group()
    external_group.add_argument("--include-external-signals", action="store_true", default=None)
    external_group.add_argument(
        "--no-external-signals",
        action="store_false",
        dest="include_external_signals",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_backtest_config(config_path)
    family = args.family.lower()
    if family != TW50_FAMILY:
        raise SystemExit("Optuna strategy search v1 supports only --family tw50.")

    output_dir = Path(args.output_dir)
    audit = write_pre_optimization_audit(
        config=config,
        config_path=config_path,
        family=family,
        output_dir=output_dir,
    )
    if audit.overall_status == AUDIT_FAIL and not args.force:
        raise SystemExit(
            "Pre-optimization audit failed. "
            f"Review {audit.html_path} or rerun with --force for research experiments."
        )
    include_external = (
        config.strategy_search.include_external_signals
        if args.include_external_signals is None
        else args.include_external_signals
    )
    if include_external:
        external_checks, _csv, _md, external_html = write_external_signal_audit(
            family=family,
            external_dir=Path("data/external"),
            output_dir=output_dir,
            feature_set=config.strategy_search.external_signal_feature_set,
        )
        external_status = external_signal_audit_status(external_checks)
        if external_status == AUDIT_FAIL and not args.force:
            raise SystemExit(
                "External signal audit failed. "
                f"Review {external_html} or rerun with --no-external-signals."
            )

    products = [
        ProductSpec.from_config(product)
        for product in config.leveraged_etf_lab.products.values()
    ]
    tw50 = build_tw50_total_return_inputs(
        start_date=config.leveraged_etf_lab.synthetic_start_date
        or config.start_date.isoformat(),
        end_date=config.end_date.isoformat(),
        products=products,
    )
    try:
        context = build_strategy_search_context(
            prices=tw50.hybrid_prices,
            products=products,
            product_assets=product_asset_map(config.universe, products),
            cost_model=CostModel.from_dict(config.cost_model),
            config=config,
            output_dir=output_dir,
            include_sentiment=args.include_sentiment,
            include_external_signals=include_external,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Strategy search input error: {exc}") from None
    storage = Path(args.storage or config.strategy_search.storage_path)
    if args.trials == 0 and not args.export_only:
        raise SystemExit("Use --export-only to rebuild reports without adding trials.")
    trials = 0 if args.export_only else int(
        config.strategy_search.n_trials if args.trials is None else args.trials
    )
    result = run_optuna_strategy_search(
        context=context,
        study_name=args.study_name,
        trials=trials,
        storage_path=storage,
        export_only=args.export_only,
        objective_profile=args.objective_profile,
        timeout_hours=args.timeout_hours,
        progress_interval_trials=args.progress_interval,
    )
    print("Optuna Strategy Search")
    print(f"study:       {result.study_name}")
    print(f"profile:     {result.objective_profile}")
    print(f"features:    {result.study_status.get('external_signal_feature_set', 'all')}")
    print(
        "coverage:    "
        f"{result.study_status.get('external_signal_common_start', '')} "
        f"to {result.study_status.get('external_signal_common_end', '')}"
    )
    print(f"trials:      {len(result.trials):,}")
    print(f"pareto:      {len(result.pareto):,}")
    print(f"triage:      {len(result.candidate_triage):,}")
    print(f"hard limit:  {result.dynamic_drawdown_limit:.2%}")
    print(f"completed:   {result.study_status.get('completed_trials', 0):,}")
    print(f"failed:      {result.study_status.get('failed_trials', 0):,}")
    print(f"last trial:  {result.study_status.get('last_trial_number', '')}")
    print(f"storage:     {result.storage_path}")
    print(f"Trials CSV:  {result.trials_path}")
    print(f"Pareto CSV:  {result.pareto_path}")
    print(f"Best CSV:    {result.best_candidates_path}")
    print(f"Triage CSV:  {result.candidate_triage_path}")
    print(f"HTML:        {result.html_path}")


def select_or_create_supported_asset(universe: list[AssetSpec], ticker: str) -> AssetSpec:
    by_ticker = {asset.ticker.upper(): asset for asset in universe}
    asset = by_ticker.get(ticker.upper())
    if asset is None:
        return AssetSpec(ticker.upper(), Market.US, AssetType.ETF, "USD", DataSource.YFINANCE)
    if asset.asset_type != AssetType.ETF:
        raise ValueError(f"Strategy search supports ETF product assets, got {asset}.")
    return asset


def product_asset_map(
    universe: list[AssetSpec],
    products: list[ProductSpec],
) -> dict[str, AssetSpec]:
    return {
        product.ticker: select_or_create_supported_asset(universe, product.ticker)
        for product in products
    }


if __name__ == "__main__":
    main()
