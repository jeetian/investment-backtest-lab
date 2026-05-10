from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from investment_backtest_lab.optuna_loop import (
    LOOP_STATUS_PROMISING,
    LOOP_STATUS_WATCHLIST,
    LOOP_STATUS_WEAK,
    LoopPaths,
    build_final_shortlist,
    build_loop_summary_row,
    classify_loop_row,
    copy_cycle_snapshot,
    next_snapshot_dir,
    write_loop_summary,
    write_snapshot_index,
)


def test_loop_status_classification_promising_watchlist_and_weak():
    baseline = {
        "baseline_expected_xirr": 0.12,
        "baseline_win_rate": 0.72,
        "baseline_p05_xirr": -0.05,
        "baseline_drawdown_breach_rate": 0.01,
        "baseline_cost_drag": 0.36,
    }
    promising, _reason = classify_loop_row(
        {
            **baseline,
            "replay_expected_xirr": 0.15,
            "replay_win_rate": 0.88,
            "replay_p05_xirr": -0.04,
            "replay_drawdown_breach_rate": 0.0,
            "replay_cost_drag": 0.30,
        }
    )
    watchlist, reason = classify_loop_row(
        {
            **baseline,
            "replay_expected_xirr": 0.15,
            "replay_win_rate": 0.88,
            "replay_p05_xirr": -0.04,
            "replay_drawdown_breach_rate": 0.0,
            "replay_cost_drag": 0.70,
        }
    )
    weak, _reason = classify_loop_row(
        {
            **baseline,
            "replay_expected_xirr": 0.10,
            "replay_win_rate": 0.50,
            "replay_p05_xirr": -0.06,
            "replay_drawdown_breach_rate": 0.02,
            "replay_cost_drag": 0.20,
        }
    )

    assert promising == LOOP_STATUS_PROMISING
    assert watchlist == LOOP_STATUS_WATCHLIST
    assert "cost" in reason
    assert weak == LOOP_STATUS_WEAK


def test_loop_status_treats_failed_cost_stress_as_watchlist():
    status, reason = classify_loop_row(
        {
            "baseline_expected_xirr": 0.12,
            "baseline_win_rate": 0.72,
            "baseline_p05_xirr": -0.05,
            "baseline_drawdown_breach_rate": 0.01,
            "baseline_cost_drag": 0.36,
            "replay_expected_xirr": 0.15,
            "replay_win_rate": 0.88,
            "replay_p05_xirr": -0.04,
            "replay_drawdown_breach_rate": 0.0,
            "replay_cost_drag": 0.30,
            "stress_2x_pass": False,
        }
    )

    assert status == LOOP_STATUS_WATCHLIST
    assert "2x cost stress" in reason


def test_build_loop_summary_row_and_outputs(tmp_path: Path):
    started = datetime(2026, 5, 9, 1, 0, tzinfo=UTC)
    ended = started + timedelta(minutes=90)
    row = build_loop_summary_row(
        cycle=1,
        study_name="study",
        started_at=started,
        ended_at=ended,
        triage=sample_triage(),
        shortlist=sample_shortlist(),
        replay_ranking=sample_ranking(),
        stress_ranking=sample_stress_ranking(),
    )
    paths = LoopPaths(tmp_path, "study", "tw50")
    write_loop_summary(paths, pd.DataFrame([row]))

    assert row["completed_trials"] == 3
    assert row["candidate_count"] == 2
    assert row["replay_top_trial_number"] == 101
    assert row["loop_status"] == LOOP_STATUS_PROMISING
    assert row["stress_2x_pass"] is True
    assert paths.summary_csv.exists()
    assert paths.summary_md.exists()
    assert "Optuna Replay Loop Summary" in paths.summary_html.read_text(encoding="utf-8")


def test_final_candidates_are_unique_and_limited():
    summary = pd.DataFrame(
        [
            {
                "cycle": 1,
                "loop_status": LOOP_STATUS_PROMISING,
                "replay_top_trial_number": 101,
                "beats_baseline_xirr": True,
                "beats_baseline_win_rate": True,
                "replay_expected_xirr": 0.15,
                "replay_win_rate": 0.88,
                "replay_p05_xirr": -0.02,
                "replay_drawdown_breach_rate": 0.0,
                "replay_cost_drag": 0.20,
            },
            {
                "cycle": 2,
                "loop_status": LOOP_STATUS_PROMISING,
                "replay_top_trial_number": 102,
                "beats_baseline_xirr": True,
                "beats_baseline_win_rate": True,
                "replay_expected_xirr": 0.14,
                "replay_win_rate": 0.86,
                "replay_p05_xirr": -0.03,
                "replay_drawdown_breach_rate": 0.0,
                "replay_cost_drag": 0.21,
            },
        ]
    )
    triage = sample_triage()

    final = build_final_shortlist(
        summary=summary,
        triage=triage,
        study_name="study",
        max_candidates=1,
    )

    assert len(final) == 1
    assert final["trial_number"].iloc[0] == 101
    assert final["scenario_name"].iloc[0] == "optuna_trial_101"


def test_final_candidates_fall_back_to_triage_without_loop_summary():
    final = build_final_shortlist(
        summary=pd.DataFrame(),
        triage=sample_triage(),
        study_name="study",
        max_candidates=2,
    )

    assert final["trial_number"].tolist() == [101, 102]
    assert final["shortlist_reason"].tolist() == ["triage_fallback", "triage_fallback"]


def test_snapshot_path_does_not_overwrite_existing(tmp_path: Path):
    paths = LoopPaths(tmp_path, "study", "tw50")
    (tmp_path / "optuna_study.html").write_text("html", encoding="utf-8")
    first = copy_cycle_snapshot(paths, cycle=1)
    second = next_snapshot_dir(paths.snapshot_root, 1)

    assert first.exists()
    assert second.name == "cycle_0001_2"


def test_snapshot_index_links_reports_and_key_candidates(tmp_path: Path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    sample_ranking().to_csv(
        snapshot / "monthly_decision_replay_tw50_ranking.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "review_reasons": "Research candidate only",
                "actual_primary_scenario_label": "Vol Target 63D 25%",
                "actual_primary_target_effective_leverage": 0.81,
            }
        ]
    ).to_csv(snapshot / "monthly_decision_comparison_tw50.csv", index=False)
    (snapshot / "monthly_decision_comparison_tw50.html").write_text(
        "comparison",
        encoding="utf-8",
    )
    (snapshot / "monthly_decision_replay_tw50.html").write_text(
        "replay",
        encoding="utf-8",
    )
    (snapshot / "monthly_decision_replay_tw50_trade_audit.html").write_text(
        "audit",
        encoding="utf-8",
    )
    sample_stress_ranking().to_csv(
        snapshot / "monthly_decision_replay_tw50_stress_2x_ranking.csv",
        index=False,
    )
    (snapshot / "monthly_decision_replay_tw50_stress_2x.html").write_text(
        "stress",
        encoding="utf-8",
    )

    index = write_snapshot_index(
        snapshot=snapshot,
        study_name="study",
        family="tw50",
        title="Snapshot Preview",
        cycle=1,
    )

    html = index.read_text(encoding="utf-8")
    assert "Snapshot Preview" in html
    assert "Monthly decision comparison + compare lab" in html
    assert "Optuna V3 #101" in html
    assert "Vol Target 63D 25%" in html
    assert "0.81x" in html
    assert "2x Transaction Cost Check" in html
    assert "Trade cost audit" in html


def sample_triage() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trial_number": 101,
                "state": "COMPLETE",
                "triage_status": "candidate",
                "expected_xirr": 0.16,
                "win_rate_vs_0050_dca": 0.90,
                "p05_xirr": 0.01,
                "p05_max_drawdown": -0.50,
                "cost_drag_on_contributed": 0.20,
                "turnover_sum": 30.0,
                "params_json": '{"a": 1}',
            },
            {
                "trial_number": 102,
                "state": "COMPLETE",
                "triage_status": "candidate",
                "expected_xirr": 0.14,
                "win_rate_vs_0050_dca": 0.85,
                "p05_xirr": 0.00,
                "p05_max_drawdown": -0.55,
                "cost_drag_on_contributed": 0.21,
                "turnover_sum": 32.0,
                "params_json": '{"a": 2}',
            },
            {
                "trial_number": 103,
                "state": "COMPLETE",
                "triage_status": "rejected",
                "expected_xirr": 0.30,
                "params_json": '{"a": 3}',
            },
        ]
    )


def sample_shortlist() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trial_number": 101,
                "scenario_label": "Optuna V3 #101",
                "expected_xirr": 0.16,
                "win_rate_vs_0050_dca": 0.90,
                "p05_xirr": 0.01,
            }
        ]
    )


def sample_ranking() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "replay_rank": 1,
                "scenario_id": "hybrid_primary--dca-policy-optuna_trial_101",
                "scenario_label": "Optuna V3 #101",
                "expected_xirr": 0.15,
                "win_rate_vs_benchmark": 0.88,
                "p05_xirr": -0.02,
                "p05_max_drawdown": -0.60,
                "drawdown_breach_rate": 0.0,
                "cost_drag_on_contributed": 0.20,
                "turnover_sum": 40.0,
            },
            {
                "replay_rank": 2,
                "scenario_id": "hybrid_primary--dca-policy-vol_target_63_25",
                "scenario_label": "Vol Target 63D 25%",
                "expected_xirr": 0.12,
                "win_rate_vs_benchmark": 0.72,
                "p05_xirr": -0.05,
                "p05_max_drawdown": -0.75,
                "drawdown_breach_rate": 0.01,
                "cost_drag_on_contributed": 0.36,
                "turnover_sum": 160.0,
            },
        ]
    )


def sample_stress_ranking() -> pd.DataFrame:
    ranking = sample_ranking().copy()
    ranking.loc[
        ranking["scenario_id"].astype(str).str.contains("optuna_trial_101"),
        ["expected_xirr", "win_rate_vs_benchmark", "p05_xirr", "drawdown_breach_rate"],
    ] = [0.13, 0.80, -0.04, 0.0]
    ranking.loc[
        ranking["scenario_label"].astype(str).eq("Vol Target 63D 25%"),
        ["expected_xirr", "win_rate_vs_benchmark", "p05_xirr", "drawdown_breach_rate"],
    ] = [0.12, 0.72, -0.05, 0.01]
    return ranking
