from pathlib import Path

import pandas as pd
import pytest

from investment_backtest_lab.monthly_decision_review import (
    REVIEW_COLUMNS,
    build_monthly_decision_review_record,
    render_monthly_decision_review_markdown,
    write_monthly_decision_review_files,
)


def test_monthly_review_writes_csv_and_markdown_with_default_status(tmp_path: Path):
    sample_comparison().to_csv(tmp_path / "monthly_decision_comparison_qqq.csv", index=False)

    result = write_monthly_decision_review_files(
        output_dir=tmp_path,
        family="qqq",
        reviewer="Ian",
        generated_at="2026-05-05T00:00:00+00:00",
    )
    row = result.review.iloc[0]

    assert result.csv_path.exists()
    assert result.markdown_path.exists()
    assert result.review.columns.tolist() == REVIEW_COLUMNS
    assert row["review_status"] == "pending_review"
    assert row["selected_layer"] == "actionable_default"
    assert row["reviewer"] == "Ian"
    assert row["recommended_scenario_label"] == "Vol Target 63D 35%"
    assert row["recommended_QLD_weight"] == 0.96
    assert row["comparison_html_path"].endswith("monthly_decision_comparison_qqq.html")
    markdown = result.markdown_path.read_text(encoding="utf-8")
    assert "Monthly Decision Review" in markdown
    assert "pending_review" in markdown
    assert "actionable_default" in markdown
    assert "Vol Target 63D 35%" in markdown


def test_monthly_review_accepts_with_review_flag_but_discloses_warning():
    review = build_monthly_decision_review_record(
        comparison=sample_comparison(manual_review_required=True),
        output_dir=Path("reports"),
        family="qqq",
        status="accepted",
        reviewer="Ian",
        notes="Reviewed actual-primary divergence.",
        generated_at="2026-05-05T00:00:00+00:00",
    )
    markdown = render_monthly_decision_review_markdown(review)

    assert review["review_status"].iloc[0] == "accepted"
    assert "accepted despite review flags" in markdown
    assert "Reviewed actual-primary divergence." in markdown


def test_monthly_review_requires_override_for_research_authority_layer():
    with pytest.raises(ValueError, match="research_authority can only be selected"):
        build_monthly_decision_review_record(
            comparison=sample_comparison(manual_review_required=True),
            output_dir=Path("reports"),
            family="qqq",
            status="accepted",
            selected_layer="research_authority",
        )


def test_monthly_review_override_research_authority_discloses_warning():
    review = build_monthly_decision_review_record(
        comparison=sample_comparison(manual_review_required=True),
        output_dir=Path("reports"),
        family="qqq",
        status="override",
        selected_layer="research_authority",
        reviewer="Ian",
        notes="Explicitly accepting higher tail risk.",
        generated_at="2026-05-05T00:00:00+00:00",
    )
    markdown = render_monthly_decision_review_markdown(review)

    assert review["selected_layer"].iloc[0] == "research_authority"
    assert "OVERRIDE: research authority selected" in markdown
    assert "Explicitly accepting higher tail risk." in markdown


def test_monthly_review_raises_clear_error_when_comparison_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="requires existing comparison CSV"):
        write_monthly_decision_review_files(output_dir=tmp_path, family="qqq")


def test_monthly_review_rejects_unknown_status():
    with pytest.raises(ValueError, match="review status must be one of"):
        build_monthly_decision_review_record(
            comparison=sample_comparison(),
            output_dir=Path("reports"),
            family="qqq",
            status="maybe",
        )


def sample_comparison(*, manual_review_required: bool = True) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "generated_at": "2026-05-05T00:00:00+00:00",
                "decision_authority": "hybrid_primary_monte_carlo",
                "actual_primary_as_of_date": "2025-12-30",
                "actual_primary_scenario_label": (
                    "Momentum+Trend 126D/200MA 3.0x to 1.0x"
                ),
                "recommended_as_of_date": "2025-12-30",
                "recommended_scenario_label": "Vol Target 63D 35%",
                "recommended_QQQ_weight": 0.04,
                "recommended_QLD_weight": 0.96,
                "recommended_TQQQ_weight": 0.0,
                "recommended_CASH_weight": 0.0,
                "manual_review_required": manual_review_required,
                "review_reasons": (
                    "Actual-primary reference pack requires review"
                    if manual_review_required
                    else ""
                ),
                "strategies_differ": True,
            }
        ]
    )
