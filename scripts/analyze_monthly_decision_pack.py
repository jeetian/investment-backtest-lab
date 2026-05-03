from __future__ import annotations

import argparse
from pathlib import Path

from investment_backtest_lab.config import load_backtest_config
from investment_backtest_lab.monthly_decision_pack import (
    write_monthly_decision_pack_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a monthly allocation decision pack from optimizer CSV outputs."
    )
    parser.add_argument("--config", default="configs/mvp_example.yaml")
    parser.add_argument("--family", default="qqq")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    config = load_backtest_config(args.config)
    pack_config = config.monthly_decision_pack
    family = args.family.lower()
    if family != pack_config.family:
        raise ValueError(
            f"Config monthly_decision_pack.family is {pack_config.family!r}; "
            f"got --family {family!r}."
        )
    result = write_monthly_decision_pack_report(
        output_dir=Path(args.output_dir),
        family=family,
        config=pack_config,
    )
    signal = result.decision.iloc[0]
    print("Monthly Decision Pack")
    print(f"as_of:     {signal['as_of_date']}")
    print(f"strategy:  {signal['scenario_label']}")
    print(f"regime:    {signal['regime']}")
    print(f"target:    {float(signal['target_effective_leverage']):.2f}x")
    print(
        "weights:   "
        f"QQQ {float(signal.get('QQQ_weight', 0.0)):.0%}, "
        f"QLD {float(signal.get('QLD_weight', 0.0)):.0%}, "
        f"TQQQ {float(signal.get('TQQQ_weight', 0.0)):.0%}, "
        f"CASH {float(signal.get('CASH_weight', 0.0)):.0%}"
    )
    print(f"review:    {bool(signal['manual_review_required'])}")
    print(f"HTML:      {result.html_path}")
    print(f"CSV:       {result.csv_path}")
    print(f"History:   {result.history_path}")


if __name__ == "__main__":
    main()
