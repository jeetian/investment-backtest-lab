from __future__ import annotations

import argparse
from pathlib import Path

from investment_backtest_lab.master_dashboard import write_master_dashboard_from_reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the USD master dashboard from ledger and leverage reports."
    )
    parser.add_argument("--config", default="configs/mvp_example.yaml")
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ"])
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    slug = "_".join(ticker.lower().replace("/", "_").replace("=", "_") for ticker in args.tickers)
    result = write_master_dashboard_from_reports(
        output_dir=Path(args.output_dir),
        slug=slug,
        config_path=Path(args.config),
    )
    print("Master dashboard summary")
    print(f"Scenarios:     {len(result.scenarios)}")
    print(f"HTML report:   {result.html_path}")
    print(f"Scenarios CSV: {result.scenarios_path}")
    print(f"Payload JSON:  {result.payload_path}")


if __name__ == "__main__":
    main()
