from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REVIEW_STATUSES = ("pending_review", "accepted", "deferred", "rejected", "override")
SELECTED_LAYERS = ("actionable_default", "research_authority")
REVIEW_COLUMNS = [
    "generated_at",
    "family",
    "as_of_date",
    "decision_authority",
    "recommended_scenario_label",
    "recommended_QQQ_weight",
    "recommended_QLD_weight",
    "recommended_TQQQ_weight",
    "recommended_CASH_weight",
    "recommended_weights_json",
    "recommended_cost_mode",
    "recommended_total_trade_cost",
    "recommended_cost_drag_on_contributed",
    "recommended_turnover_sum",
    "manual_review_required",
    "review_reasons",
    "actual_primary_scenario_label",
    "strategies_differ",
    "review_status",
    "selected_layer",
    "reviewer",
    "review_notes",
    "comparison_html_path",
    "mc_summary_path",
    "source_coverage_path",
]


@dataclass(frozen=True)
class MonthlyDecisionReviewResult:
    review: pd.DataFrame
    csv_path: Path
    markdown_path: Path


def write_monthly_decision_review_files(
    *,
    output_dir: Path,
    family: str,
    status: str = "pending_review",
    selected_layer: str = "actionable_default",
    reviewer: str = "",
    notes: str = "",
    generated_at: str | None = None,
) -> MonthlyDecisionReviewResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = family.lower()
    comparison_path = output_dir / f"monthly_decision_comparison_{prefix}.csv"
    comparison = load_monthly_decision_comparison_csv(comparison_path)
    review = build_monthly_decision_review_record(
        comparison=comparison,
        output_dir=output_dir,
        family=prefix,
        status=status,
        selected_layer=selected_layer,
        reviewer=reviewer,
        notes=notes,
        generated_at=generated_at,
    )
    csv_path = output_dir / f"monthly_decision_review_{prefix}.csv"
    markdown_path = output_dir / f"monthly_decision_review_{prefix}.md"
    review.to_csv(csv_path, index=False)
    markdown_path.write_text(render_monthly_decision_review_markdown(review), encoding="utf-8")
    return MonthlyDecisionReviewResult(
        review=review,
        csv_path=csv_path,
        markdown_path=markdown_path,
    )


def load_monthly_decision_comparison_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            "Monthly decision review requires existing comparison CSV.\n"
            f"Missing file: {path}\n"
            "Run first:\n"
            "python -m uv run python scripts\\analyze_monthly_decision_comparison.py "
            "--config configs\\mvp_example.yaml --family qqq"
        )
    comparison = pd.read_csv(path)
    if comparison.empty:
        raise ValueError(f"Monthly decision comparison CSV is empty: {path}")
    return comparison


def build_monthly_decision_review_record(
    *,
    comparison: pd.DataFrame,
    output_dir: Path,
    family: str,
    status: str = "pending_review",
    selected_layer: str = "actionable_default",
    reviewer: str = "",
    notes: str = "",
    generated_at: str | None = None,
) -> pd.DataFrame:
    if status not in REVIEW_STATUSES:
        allowed = ", ".join(REVIEW_STATUSES)
        raise ValueError(f"review status must be one of {allowed}; got {status!r}.")
    if selected_layer not in SELECTED_LAYERS:
        allowed = ", ".join(SELECTED_LAYERS)
        raise ValueError(f"selected layer must be one of {allowed}; got {selected_layer!r}.")
    if selected_layer == "research_authority" and status != "override":
        raise ValueError(
            "research_authority can only be selected with review_status=override. "
            "Use selected_layer=actionable_default for accepted, pending, deferred, or rejected."
        )
    if comparison.empty:
        raise ValueError("Monthly decision review requires a non-empty comparison CSV.")

    row = comparison.iloc[0]
    prefix = family.lower()
    record = {
        "generated_at": generated_at
        or datetime.now(UTC).replace(microsecond=0).isoformat(),
        "family": prefix,
        "as_of_date": _first_non_empty(
            row.get("recommended_as_of_date"),
            row.get("actual_primary_as_of_date"),
        ),
        "decision_authority": row.get("decision_authority", ""),
        "recommended_scenario_label": _safe_text(row.get("recommended_scenario_label")),
        "recommended_QQQ_weight": _safe_float(row.get("recommended_QQQ_weight")),
        "recommended_QLD_weight": _safe_float(row.get("recommended_QLD_weight")),
        "recommended_TQQQ_weight": _safe_float(row.get("recommended_TQQQ_weight")),
        "recommended_CASH_weight": _safe_float(row.get("recommended_CASH_weight")),
        "recommended_weights_json": row.get(
            "recommended_weights_json",
            _legacy_weights_json(row),
        ),
        "recommended_cost_mode": _safe_text(row.get("actionable_default_cost_mode")),
        "recommended_total_trade_cost": _safe_float(
            row.get("actionable_default_total_trade_cost")
        ),
        "recommended_cost_drag_on_contributed": _safe_float(
            row.get("actionable_default_cost_drag_on_contributed")
        ),
        "recommended_turnover_sum": _safe_float(row.get("actionable_default_turnover_sum")),
        "manual_review_required": _safe_bool(row.get("manual_review_required")),
        "review_reasons": _safe_text(row.get("review_reasons")),
        "actual_primary_scenario_label": _safe_text(row.get("actual_primary_scenario_label")),
        "strategies_differ": _safe_bool(row.get("strategies_differ")),
        "review_status": status,
        "selected_layer": selected_layer,
        "reviewer": reviewer,
        "review_notes": notes,
        "comparison_html_path": str(output_dir / f"monthly_decision_comparison_{prefix}.html"),
        "mc_summary_path": str(output_dir / f"monthly_decision_replay_{prefix}_mc_summary.csv"),
        "source_coverage_path": str(
            output_dir / f"monthly_decision_replay_{prefix}_source_coverage.csv"
        ),
    }
    return pd.DataFrame([record], columns=REVIEW_COLUMNS)


def render_monthly_decision_review_markdown(review: pd.DataFrame) -> str:
    if review.empty:
        raise ValueError("Monthly decision review markdown requires a non-empty review frame.")
    row = review.iloc[0]
    weights = _format_weights(row)
    accepted_with_flags = (
        row["review_status"] == "accepted" and _safe_bool(row["manual_review_required"])
    )
    override_research = (
        row["review_status"] == "override" and row["selected_layer"] == "research_authority"
    )
    warning = (
        "\n> WARNING: accepted despite review flags. "
        "Review reasons must be checked before acting on this record.\n"
        if accepted_with_flags
        else ""
    )
    override_warning = (
        "\n> OVERRIDE: research authority selected instead of the zero-breach actionable "
        "default. This can include Monte Carlo drawdown breach risk.\n"
        if override_research
        else ""
    )
    return f"""# Monthly Decision Review

{warning}
{override_warning}
## Decision

- Family: `{row["family"]}`
- As of date: `{row["as_of_date"]}`
- Authority: `{row["decision_authority"]}`
- Selected layer: `{row["selected_layer"]}`
- Recommended strategy: `{row["recommended_scenario_label"]}`
- Weights: {weights}
- Cost mode: `{row["recommended_cost_mode"]}`
- Total trade cost: `{_format_number(row["recommended_total_trade_cost"])}`
- Cost drag on contributed: `{_format_ratio_percent(row["recommended_cost_drag_on_contributed"])}`
- Turnover sum: `{_format_number(row["recommended_turnover_sum"])}`

## Review

- Status: `{row["review_status"]}`
- Reviewer: `{row["reviewer"]}`
- Manual review required: `{row["manual_review_required"]}`
- Review reasons: {row["review_reasons"] or "None"}
- Actual-primary reference: `{row["actual_primary_scenario_label"]}`
- Actual-primary differs from research: `{row["strategies_differ"]}`
- Notes: {row["review_notes"] or "None"}

## Evidence

- Comparison: `{row["comparison_html_path"]}`
- MC summary: `{row["mc_summary_path"]}`
- Source coverage: `{row["source_coverage_path"]}`
"""


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _safe_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _first_non_empty(*values: Any) -> str:
    for value in values:
        if value is not None and not pd.isna(value) and str(value) != "":
            return str(value)
    return ""


def _safe_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _legacy_weights_json(row: pd.Series) -> str:
    weights = {
        ticker: _safe_float(row.get(f"recommended_{ticker}_weight"))
        for ticker in ["QQQ", "QLD", "TQQQ", "CASH"]
    }
    weights = {key: value for key, value in weights.items() if not pd.isna(value)}
    return json.dumps(weights, ensure_ascii=False, sort_keys=True)


def _format_weights(row: pd.Series) -> str:
    raw = row.get("recommended_weights_json", "")
    weights: dict[str, float] = {}
    if raw and not pd.isna(raw):
        try:
            parsed = json.loads(str(raw))
            if isinstance(parsed, dict):
                weights = {str(key): _safe_float(value) for key, value in parsed.items()}
        except json.JSONDecodeError:
            weights = {}
    if not weights:
        weights = {
            ticker: _safe_float(row.get(f"recommended_{ticker}_weight"))
            for ticker in ["QQQ", "QLD", "TQQQ", "CASH"]
        }
    return ", ".join(
        f"{ticker} {_format_percent(value)}"
        for ticker, value in weights.items()
        if not pd.isna(value)
    )


def _format_percent(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.0%}"


def _format_ratio_percent(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def _format_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):,.2f}"


__all__ = [
    "REVIEW_COLUMNS",
    "REVIEW_STATUSES",
    "SELECTED_LAYERS",
    "MonthlyDecisionReviewResult",
    "build_monthly_decision_review_record",
    "load_monthly_decision_comparison_csv",
    "render_monthly_decision_review_markdown",
    "write_monthly_decision_review_files",
]
