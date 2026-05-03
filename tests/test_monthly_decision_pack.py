from pathlib import Path

import pandas as pd
import pytest

from investment_backtest_lab.models import MonthlyDecisionPackConfig
from investment_backtest_lab.monthly_decision_pack import (
    build_monthly_decision_pack,
    load_optimizer_outputs,
    optimizer_output_paths,
    write_monthly_decision_pack_report,
)


def test_monthly_decision_pack_first_run_without_manual_review():
    decision = build_monthly_decision_pack(
        allocation_signal=sample_allocation_signal(),
        metrics=sample_metrics(),
        cohort_summary=sample_cohort_summary(),
        history=pd.DataFrame(),
        config=MonthlyDecisionPackConfig(),
        generated_at="2026-01-01T00:00:00+00:00",
    )

    row = decision.iloc[0]

    assert bool(row["first_run"]) is True
    assert bool(row["manual_review_required"]) is False
    assert row["weight_sum"] == pytest.approx(1.0)
    assert bool(row["weight_sum_ok"]) is True
    assert row["weight_change_summary"] == "first run: no previous signal history"


def test_monthly_decision_pack_compares_with_previous_history():
    history = build_monthly_decision_pack(
        allocation_signal=sample_allocation_signal(tqqq_weight=0.0, cash_weight=1.0),
        metrics=sample_metrics(),
        cohort_summary=sample_cohort_summary(),
        history=pd.DataFrame(),
        config=MonthlyDecisionPackConfig(),
        generated_at="2026-01-01T00:00:00+00:00",
    )

    decision = build_monthly_decision_pack(
        allocation_signal=sample_allocation_signal(tqqq_weight=1.0, cash_weight=0.0),
        metrics=sample_metrics(),
        cohort_summary=sample_cohort_summary(),
        history=history,
        config=MonthlyDecisionPackConfig(),
        generated_at="2026-02-01T00:00:00+00:00",
    )

    row = decision.iloc[0]

    assert bool(row["first_run"]) is False
    assert bool(row["allocation_changed"]) is True
    assert row["previous_CASH_weight"] == pytest.approx(1.0)
    assert "TQQQ +100.0%pt" in row["weight_change_summary"]


def test_monthly_decision_pack_requires_manual_review_on_risk_flags():
    decision = build_monthly_decision_pack(
        allocation_signal=sample_allocation_signal(review_now=True),
        metrics=sample_metrics(synthetic_risk_flag="high_drawdown", synthetic_drawdown=-0.90),
        cohort_summary=sample_cohort_summary(drawdown_breach_rate=0.10),
        history=pd.DataFrame(),
        config=MonthlyDecisionPackConfig(),
        generated_at="2026-01-01T00:00:00+00:00",
    )

    row = decision.iloc[0]

    assert bool(row["manual_review_required"]) is True
    assert "Synthetic stress" in row["review_reasons"]
    assert "Cohort robustness" in row["review_reasons"]
    assert "Weekly monitor" in row["review_reasons"]


def test_monthly_decision_pack_requires_weights_to_sum_to_one():
    decision = build_monthly_decision_pack(
        allocation_signal=sample_allocation_signal(tqqq_weight=0.90, cash_weight=0.0),
        metrics=sample_metrics(),
        cohort_summary=sample_cohort_summary(),
        history=pd.DataFrame(),
        config=MonthlyDecisionPackConfig(),
        generated_at="2026-01-01T00:00:00+00:00",
    )

    row = decision.iloc[0]

    assert bool(row["weight_sum_ok"]) is False
    assert bool(row["manual_review_required"]) is True
    assert "Allocation weights" in row["review_reasons"]


def test_missing_optimizer_outputs_raise_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="requires existing DCA optimizer outputs"):
        load_optimizer_outputs(output_dir=tmp_path, family="qqq", require_outputs=True)


def test_monthly_decision_pack_writes_html_csv_and_history(tmp_path):
    write_optimizer_fixture(tmp_path)

    result = write_monthly_decision_pack_report(
        output_dir=tmp_path,
        family="qqq",
        config=MonthlyDecisionPackConfig(),
    )

    assert result.html_path.exists()
    assert result.csv_path.exists()
    assert result.history_path.exists()
    html = result.html_path.read_text(encoding="utf-8")
    assert "Monthly Decision Pack" in html
    assert "目前研究配置" in html
    assert "上期 vs 本期" in html
    assert "Decision Checklist" in html
    assert "研究訊號，不是投資建議" in html
    assert "manual_review_required" in html

    result_again = write_monthly_decision_pack_report(
        output_dir=tmp_path,
        family="qqq",
        config=MonthlyDecisionPackConfig(),
    )

    assert len(result_again.history) == 2
    assert bool(result_again.decision.iloc[0]["first_run"]) is False


def test_monthly_decision_pack_repairs_malformed_history_on_append(tmp_path):
    write_optimizer_fixture(tmp_path)
    history_path = tmp_path / "monthly_decision_pack_qqq_signal_history.csv"
    history_path.write_text(
        "as_of_date,scenario_id,QQQ_weight\n"
        "2025-01-01,old,1.0\n"
        "bad,row,with,too,many,fields\n",
        encoding="utf-8",
    )

    result = write_monthly_decision_pack_report(
        output_dir=tmp_path,
        family="qqq",
        config=MonthlyDecisionPackConfig(),
    )

    repaired = pd.read_csv(history_path)
    assert len(repaired) == 2
    assert len(result.history) == 2


def write_optimizer_fixture(output_dir: Path) -> None:
    paths = optimizer_output_paths(output_dir, "qqq")
    sample_metrics().to_csv(paths.metrics, index=False)
    sample_cohort_summary().to_csv(paths.cohort_summary, index=False)
    sample_allocation_signal().to_csv(paths.allocation_signal, index=False)
    pd.DataFrame({"date": ["2026-01-01"]}).to_csv(paths.policy, index=False)
    pd.DataFrame({"fold_index": [1]}).to_csv(paths.walk_forward, index=False)
    pd.DataFrame({"cohort_start": ["2020-01-01"]}).to_csv(paths.cohorts, index=False)


def sample_allocation_signal(
    *,
    tqqq_weight: float = 1.0,
    cash_weight: float = 0.0,
    review_now: bool = False,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "as_of_date": "2025-12-30",
                "data_mode": "actual_etf",
                "scenario_id": "actual_etf--dca-policy-test",
                "scenario_label": "Test Policy",
                "rank": 1,
                "validation_status": "stable",
                "regime": "trend_on",
                "reason": "test reason",
                "target_effective_leverage": 3.0,
                "QQQ_weight": 0.0,
                "QLD_weight": 0.0,
                "TQQQ_weight": tqqq_weight,
                "CASH_weight": cash_weight,
                "xirr": 0.25,
                "max_drawdown": -0.60,
                "next_rebalance_date": "2026-01-02",
                "next_monitor_date": "2026-01-06",
                "rebalance_cadence": "monthly",
                "monitor_cadence": "weekly",
                "review_now": review_now,
                "weight_sum": tqqq_weight + cash_weight,
            }
        ]
    )


def sample_metrics(
    *,
    synthetic_risk_flag: str = "ok",
    synthetic_drawdown: float = -0.70,
) -> pd.DataFrame:
    rows = [
        metric_row(
            data_mode="actual_etf",
            scenario_id="actual_etf--dca-policy-test",
            risk_flag="ok",
            max_drawdown=-0.60,
        ),
        metric_row(
            data_mode="synthetic_stress",
            scenario_id="synthetic_stress--dca-policy-test",
            risk_flag=synthetic_risk_flag,
            max_drawdown=synthetic_drawdown,
        ),
    ]
    return pd.DataFrame(rows)


def metric_row(
    *,
    data_mode: str,
    scenario_id: str,
    risk_flag: str,
    max_drawdown: float,
) -> dict[str, object]:
    return {
        "data_mode": data_mode,
        "scenario_id": scenario_id,
        "scenario_label": "Test Policy",
        "xirr": 0.25,
        "max_drawdown": max_drawdown,
        "risk_flag": risk_flag,
        "risk_failed": False,
        "validation_status": "stable",
        "validation_folds": 5,
        "walk_forward_pass_rate": 0.80,
        "eligible_for_candidate": True,
        "policy_key": "dca-policy-test",
        "rank": 1,
    }


def sample_cohort_summary(*, drawdown_breach_rate: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "data_mode": "actual_etf",
                "scenario_id": "actual_etf--dca-policy-test",
                "scenario_label": "Test Policy",
                "strategy_family": "test",
                "cohort_count": 12,
                "median_cohort_xirr": 0.20,
                "worst_cohort_xirr": 0.05,
                "median_cohort_max_drawdown": -0.50,
                "worst_cohort_max_drawdown": -0.70,
                "top3_hit_rate": 0.80,
                "median_rank": 1,
                "rank_iqr": 1,
                "drawdown_breach_rate": drawdown_breach_rate,
            }
        ]
    )
