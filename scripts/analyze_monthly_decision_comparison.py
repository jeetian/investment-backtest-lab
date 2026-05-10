from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

from investment_backtest_lab.config import load_backtest_config
from investment_backtest_lab.monthly_decision_comparison import (
    validate_monthly_decision_comparison_as_of,
    write_monthly_decision_comparison_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the monthly decision authority and replay-primary comparison."
    )
    parser.add_argument("--config", default="configs/mvp_example.yaml")
    parser.add_argument("--family", default="qqq")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument(
        "--expected-as-of",
        default=None,
        help=(
            "Required recommended_as_of_date. Defaults to config end_date minus one day "
            "because yfinance end is exclusive."
        ),
    )
    args = parser.parse_args()

    config = load_backtest_config(args.config)
    family = args.family.lower()
    if family != config.monthly_decision_pack.family:
        raise ValueError(
            f"Config monthly_decision_pack.family is {config.monthly_decision_pack.family!r}; "
            f"got --family {family!r}."
        )
    if family != config.monthly_decision_replay.family:
        raise ValueError(
            f"Config monthly_decision_replay.family is {config.monthly_decision_replay.family!r}; "
            f"got --family {family!r}."
        )

    result = write_monthly_decision_comparison_report(
        output_dir=Path(args.output_dir),
        family=family,
        top_n=args.top_n,
    )
    row = result.comparison.iloc[0]
    expected_as_of = args.expected_as_of or (config.end_date - timedelta(days=1)).isoformat()
    validate_monthly_decision_comparison_as_of(
        result.comparison,
        expected_as_of=expected_as_of,
    )
    print("Monthly Decision Comparison")
    print(f"authority:   {row['decision_authority']}")
    print(f"research:    {row['replay_primary_scenario_label']}")
    print(f"actionable:  {row['actionable_default_scenario_label']}")
    print(f"recommended: {row['recommended_scenario_label']}")
    print(f"weights:     {format_weight_summary(row)}")
    print(f"cost mode:   {row['actionable_default_cost_mode']}")
    cost_drag = format_percent_or_blank(row["actionable_default_cost_drag_on_contributed"])
    print(f"cost drag:   {cost_drag}")
    print(f"replay top:  {row['replay_primary_scenario_label']}")
    print(f"actual ref:  {row['actual_primary_scenario_label']}")
    print(f"disagrees:   {bool(row['actual_disagrees_with_authority'])}")
    print(f"review:      {bool(row['manual_review_required'])}")
    print(f"reasons:     {row['review_reasons'] or 'None'}")
    print(f"as-of check: {row['recommended_as_of_date']} == {expected_as_of}")
    print(f"HTML:        {result.html_path}")
    print(f"CSV:         {result.csv_path}")
    print(f"Top N:       {result.top_candidates_path}")


def format_weight_summary(row) -> str:
    raw = row.get("recommended_weights_json", "")
    weights = {}
    if raw:
        try:
            parsed = json.loads(str(raw))
            if isinstance(parsed, dict):
                weights = {str(key): float(value) for key, value in parsed.items()}
        except (TypeError, ValueError, json.JSONDecodeError):
            weights = {}
    if not weights:
        weights = {
            ticker: float(row.get(f"recommended_{ticker}_weight", 0.0))
            for ticker in ["QQQ", "QLD", "TQQQ", "CASH"]
        }
    clean = {ticker: value for ticker, value in weights.items() if value == value}
    if not clean:
        return "no tradable weights"
    return ", ".join(f"{ticker} {value:.0%}" for ticker, value in clean.items())


def format_percent_or_blank(value) -> str:
    try:
        if value != value:
            return ""
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return ""


if __name__ == "__main__":
    main()
