from __future__ import annotations

import argparse
from pathlib import Path

from investment_backtest_lab.optuna_shortlist import write_optuna_shortlist


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a replay shortlist from an Optuna candidate triage CSV."
    )
    parser.add_argument("--study-name", default="tw50_v2_return_first_external")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--input", default=None, help="Candidate triage CSV path.")
    parser.add_argument("--max-candidates", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    triage_path = Path(
        args.input
        or output_dir / f"optuna_{args.study_name}_candidate_triage.csv"
    )
    result = write_optuna_shortlist(
        triage_path=triage_path,
        output_dir=output_dir,
        study_name=args.study_name,
        max_candidates=args.max_candidates,
    )
    print("Optuna candidate shortlist")
    print(f"study:      {args.study_name}")
    print(f"candidates: {len(result.shortlist):,}")
    print(f"CSV:        {result.csv_path}")
    print(f"HTML:       {result.html_path}")


if __name__ == "__main__":
    main()
