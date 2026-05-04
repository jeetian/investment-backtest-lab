from pathlib import Path

import pandas as pd
import pytest

from investment_backtest_lab.monthly_decision_comparison import (
    COMPARISON_COLUMNS,
    build_monthly_decision_comparison,
    validate_monthly_decision_comparison_as_of,
    write_monthly_decision_comparison_report,
)


def test_comparison_uses_replay_authority_and_actual_policy_weights():
    outputs = build_monthly_decision_comparison(
        monthly_decision=sample_monthly_pack(manual_review_required=False, review_reasons=""),
        replay_ranking=sample_replay_ranking(),
        actual_policy=sample_actual_policy(),
        source_coverage=sample_source_coverage(),
        generated_at="2026-05-04T00:00:00+00:00",
    )
    row = outputs.comparison.iloc[0]

    assert outputs.comparison.columns.tolist() == COMPARISON_COLUMNS
    assert row["decision_authority"] == "hybrid_primary_monte_carlo"
    assert row["recommended_scenario_label"] == "Vol Target 63D 35%"
    assert row["recommended_QLD_weight"] == 0.96
    assert bool(row["actual_disagrees_with_authority"]) is True
    assert bool(row["manual_review_required"]) is False
    assert outputs.top_candidates.iloc[0]["scenario_label"] == "Vol Target 63D 35%"
    assert outputs.source_coverage["ticker"].tolist() == ["QQQ", "QLD", "TQQQ"]


def test_comparison_requires_review_when_recommended_actual_policy_missing():
    outputs = build_monthly_decision_comparison(
        monthly_decision=sample_monthly_pack(manual_review_required=False, review_reasons=""),
        replay_ranking=sample_replay_ranking(),
        actual_policy=sample_actual_policy().iloc[0:0],
        generated_at="2026-05-04T00:00:00+00:00",
    )
    row = outputs.comparison.iloc[0]

    assert bool(row["manual_review_required"]) is True
    assert "no latest actual ETF tradable weight row" in row["review_reasons"]


def test_comparison_requires_review_when_mc_breaches_drawdown():
    ranking = sample_replay_ranking()
    ranking.loc[0, "drawdown_breach_rate"] = 0.20

    outputs = build_monthly_decision_comparison(
        monthly_decision=sample_monthly_pack(manual_review_required=False, review_reasons=""),
        replay_ranking=ranking,
        actual_policy=sample_actual_policy(),
        generated_at="2026-05-04T00:00:00+00:00",
    )
    row = outputs.comparison.iloc[0]

    assert bool(row["manual_review_required"]) is True
    assert "Monte Carlo replay has drawdown breach paths" in row["review_reasons"]


def test_comparison_report_writes_html_and_csv(tmp_path: Path):
    sample_monthly_pack().to_csv(tmp_path / "monthly_decision_pack_qqq.csv", index=False)
    sample_replay_ranking().to_csv(
        tmp_path / "monthly_decision_replay_qqq_ranking.csv",
        index=False,
    )
    sample_actual_policy().to_csv(tmp_path / "dca_policy_optimizer_qqq_policy.csv", index=False)
    sample_source_coverage().to_csv(
        tmp_path / "monthly_decision_replay_qqq_source_coverage.csv",
        index=False,
    )

    result = write_monthly_decision_comparison_report(output_dir=tmp_path, family="qqq")

    assert result.html_path.exists()
    assert result.csv_path.exists()
    assert result.top_candidates_path.exists()
    assert result.comparison["recommended_scenario_label"].iloc[0] == "Vol Target 63D 35%"
    assert result.source_coverage["ticker"].tolist() == ["QQQ", "QLD", "TQQQ"]
    html = result.html_path.read_text(encoding="utf-8")
    assert "Monthly Decision Pack" in html
    assert "Tradable Weights" in html
    assert "Source Coverage" in html
    assert "MC summary" in html
    assert "Vol Target 63D 35%" in html
    assert "Momentum+Trend 126D/200MA 3.0x to 1.0x" in html
    assert "2006-06-21" in html


def test_comparison_as_of_check_rejects_stale_output():
    outputs = build_monthly_decision_comparison(
        monthly_decision=sample_monthly_pack(manual_review_required=False, review_reasons=""),
        replay_ranking=sample_replay_ranking(),
        actual_policy=sample_actual_policy(),
        generated_at="2026-05-04T00:00:00+00:00",
    )

    validate_monthly_decision_comparison_as_of(
        outputs.comparison,
        expected_as_of="2025-12-30",
    )
    with pytest.raises(ValueError, match="not fresh enough"):
        validate_monthly_decision_comparison_as_of(
            outputs.comparison,
            expected_as_of="2026-04-30",
        )


def sample_monthly_pack(
    *,
    scenario_id: str = "actual_etf--dca-policy-momentum-trend-126d-200ma-3p0-1p0",
    scenario_label: str = "Momentum+Trend 126D/200MA 3.0x to 1.0x",
    manual_review_required: bool = True,
    review_reasons: str = "Synthetic stress is high risk",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "as_of_date": "2025-12-30",
                "scenario_id": scenario_id,
                "scenario_label": scenario_label,
                "regime": "momentum_trend_on",
                "reason": "momentum and trend are both positive",
                "target_effective_leverage": 3.0,
                "QQQ_weight": 0.0,
                "QLD_weight": 0.0,
                "TQQQ_weight": 1.0,
                "CASH_weight": 0.0,
                "manual_review_required": manual_review_required,
                "review_reasons": review_reasons,
            }
        ]
    )


def sample_replay_ranking() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "selector": "hybrid_primary",
                "ranking_method": "monte_carlo",
                "replay_rank": 1,
                "scenario_id": "hybrid_primary--dca-policy-vol-target-63d-35pct",
                "scenario_label": "Vol Target 63D 35%",
                "eligible_for_monthly_signal": True,
                "cohort_gate_passed": True,
                "win_rate_vs_qqq_dca": 0.9985,
                "expected_xirr": 0.2810,
                "median_xirr": 0.2713,
                "p05_xirr": -0.0500,
                "expected_max_drawdown": -0.5100,
                "p05_max_drawdown": -0.7224,
                "drawdown_breach_rate": 0.0,
            },
            {
                "selector": "hybrid_primary",
                "ranking_method": "monte_carlo",
                "replay_rank": 2,
                "scenario_id": "hybrid_primary--dca-policy-momentum-trend-126d-200ma-2p0-1p0",
                "scenario_label": "Momentum+Trend 126D/200MA 2.0x to 1.0x",
                "eligible_for_monthly_signal": True,
                "cohort_gate_passed": True,
                "win_rate_vs_qqq_dca": 0.9884,
                "expected_xirr": 0.2400,
                "median_xirr": 0.2373,
                "p05_xirr": -0.0700,
                "expected_max_drawdown": -0.6000,
                "p05_max_drawdown": -0.8931,
                "drawdown_breach_rate": 0.0,
            },
        ]
    )


def sample_actual_policy() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2025-12-29",
                "data_mode": "actual_etf",
                "scenario_id": "actual_etf--dca-policy-vol-target-63d-35pct",
                "scenario_label": "Vol Target 63D 35%",
                "regime": "vol_target",
                "reason": "prior 63D realized volatility",
                "target_effective_leverage": 1.96,
                "QQQ_weight": 0.04,
                "QLD_weight": 0.96,
                "TQQQ_weight": 0.0,
                "CASH_weight": 0.0,
            },
            {
                "date": "2025-12-30",
                "data_mode": "actual_etf",
                "scenario_id": "actual_etf--dca-policy-vol-target-63d-35pct",
                "scenario_label": "Vol Target 63D 35%",
                "regime": "vol_target",
                "reason": "prior 63D realized volatility",
                "target_effective_leverage": 1.96,
                "QQQ_weight": 0.04,
                "QLD_weight": 0.96,
                "TQQQ_weight": 0.0,
                "CASH_weight": 0.0,
            },
        ]
    )


def sample_source_coverage() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "QQQ",
                "replay_start": "1999-03-10",
                "replay_end": "2025-12-30",
                "actual_start": "1999-03-10",
                "actual_end": "2025-12-30",
                "synthetic_backfill_start": "",
                "synthetic_backfill_end": "",
                "splice_date": "1999-03-10",
                "splice_scale": 1.0,
            },
            {
                "ticker": "QLD",
                "replay_start": "1999-03-10",
                "replay_end": "2025-12-30",
                "actual_start": "2006-06-21",
                "actual_end": "2025-12-30",
                "synthetic_backfill_start": "1999-03-10",
                "synthetic_backfill_end": "2006-06-20",
                "splice_date": "2006-06-21",
                "splice_scale": 0.5,
            },
            {
                "ticker": "TQQQ",
                "replay_start": "1999-03-10",
                "replay_end": "2025-12-30",
                "actual_start": "2010-02-11",
                "actual_end": "2025-12-30",
                "synthetic_backfill_start": "1999-03-10",
                "synthetic_backfill_end": "2010-02-10",
                "splice_date": "2010-02-11",
                "splice_scale": 0.25,
            },
        ]
    )
