from __future__ import annotations

import argparse
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
    print(
        "weights:     "
        f"QQQ {float(row['recommended_QQQ_weight']):.0%}, "
        f"QLD {float(row['recommended_QLD_weight']):.0%}, "
        f"TQQQ {float(row['recommended_TQQQ_weight']):.0%}, "
        f"CASH {float(row['recommended_CASH_weight']):.0%}"
    )
    print(f"review flag: {bool(row['manual_review_required'])}")
    print(f"reasons:     {row['review_reasons'] or 'None'}")
    print(f"CSV:         {result.csv_path}")
    print(f"Markdown:    {result.markdown_path}")


if __name__ == "__main__":
    main()
