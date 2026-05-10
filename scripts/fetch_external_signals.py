from __future__ import annotations

import argparse
from pathlib import Path

from investment_backtest_lab.config import load_backtest_config
from investment_backtest_lab.external_signals import ExternalSignalError, fetch_external_signals


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch frozen external regime signals.")
    parser.add_argument("--config", default="configs/tw50_example.yaml")
    parser.add_argument("--family", default="tw50")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--external-dir", default="data/external")
    parser.add_argument(
        "--allow-missing-fear-greed",
        action="store_true",
        help="Allow non-F&G smoke fetches when CNN blocks the Fear & Greed API.",
    )
    args = parser.parse_args()

    config = load_backtest_config(Path(args.config))
    end_date = args.end_date or config.end_date.isoformat()
    try:
        require_fear_greed = (
            config.strategy_search.external_signal_feature_set != "core"
            and not args.allow_missing_fear_greed
        )
        result = fetch_external_signals(
            config=config,
            family=args.family,
            external_dir=Path(args.external_dir),
            end_date=end_date,
            require_fear_greed=require_fear_greed,
        )
    except ExternalSignalError as exc:
        raise SystemExit(f"External signal fetch error: {exc}") from None

    print("External Signals Fetch")
    print(f"family:     {result.family}")
    print(f"rows:       {len(result.features):,}")
    print(f"features:   {result.features_path}")
    print(f"manifest:   {result.manifest_path}")
    for name, path in result.raw_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
