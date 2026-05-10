from __future__ import annotations

import argparse
import json
from pathlib import Path

from investment_backtest_lab.monthly_decision_review import (
    REVIEW_STATUSES,
    SELECTED_LAYERS,
    write_monthly_decision_review_files,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record the human review status for the monthly decision comparison."
    )
    parser.add_argument("--family", default="qqq")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--status", choices=REVIEW_STATUSES, default="pending_review")
    parser.add_argument("--selected-layer", choices=SELECTED_LAYERS, default="actionable_default")
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    result = write_monthly_decision_review_files(
        output_dir=Path(args.output_dir),
        family=args.family.lower(),
        status=args.status,
        selected_layer=args.selected_layer,
        reviewer=args.reviewer,
        notes=args.notes,
    )
    row = result.review.iloc[0]
    print("Monthly Decision Review")
    print(f"family:      {row['family']}")
    print(f"as_of:       {row['as_of_date']}")
    print(f"status:      {row['review_status']}")
    print(f"layer:       {row['selected_layer']}")
    print(f"reviewer:    {row['reviewer'] or 'Unspecified'}")
    print(f"recommended: {row['recommended_scenario_label']}")
    print(f"weights:     {format_weight_summary(row)}")
    print(f"cost mode:   {row['recommended_cost_mode']}")
    print(f"cost drag:   {format_percent_or_blank(row['recommended_cost_drag_on_contributed'])}")
    print(f"review flag: {bool(row['manual_review_required'])}")
    print(f"reasons:     {row['review_reasons'] or 'None'}")
    print(f"CSV:         {result.csv_path}")
    print(f"Markdown:    {result.markdown_path}")


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
