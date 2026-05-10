from __future__ import annotations

import argparse
from pathlib import Path

from investment_backtest_lab.config import load_backtest_config
from investment_backtest_lab.pre_optimization_audit import write_pre_optimization_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pre-optimization audit gates.")
    parser.add_argument("--config", default="configs/tw50_example.yaml")
    parser.add_argument("--family", default="tw50")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    config_path = Path(args.config)
    result = write_pre_optimization_audit(
        config=load_backtest_config(config_path),
        config_path=config_path,
        family=args.family,
        output_dir=Path(args.output_dir),
    )
    print("Pre-Optimization Audit")
    print(f"family:       {result.family}")
    print(f"status:       {result.overall_status}")
    print(f"base 1x MDD:  {result.base_1x_stress_max_drawdown:.2%}")
    print(f"hard limit:   {result.dynamic_drawdown_limit:.2%}")
    print(f"Markdown:     {result.markdown_path}")
    print(f"CSV:          {result.csv_path}")
    print(f"HTML:         {result.html_path}")


if __name__ == "__main__":
    main()
