from __future__ import annotations

import argparse
from pathlib import Path

from investment_backtest_lab.config import load_backtest_config
from investment_backtest_lab.external_signals import (
    external_signal_audit_status,
    write_external_signal_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen external regime signals.")
    parser.add_argument("--config", default="configs/tw50_example.yaml")
    parser.add_argument("--family", default="tw50")
    parser.add_argument("--external-dir", default="data/external")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    config = load_backtest_config(Path(args.config))
    checks, csv_path, md_path, html_path = write_external_signal_audit(
        family=args.family,
        external_dir=Path(args.external_dir),
        output_dir=Path(args.output_dir),
        feature_set=config.strategy_search.external_signal_feature_set,
    )
    print("External Signal Audit")
    print(f"family:   {args.family.lower()}")
    print(f"status:   {external_signal_audit_status(checks)}")
    print(f"CSV:      {csv_path}")
    print(f"Markdown: {md_path}")
    print(f"HTML:     {html_path}")


if __name__ == "__main__":
    main()
