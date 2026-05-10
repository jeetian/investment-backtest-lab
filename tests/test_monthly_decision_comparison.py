import json
from pathlib import Path

import pandas as pd
import pytest

from investment_backtest_lab.monthly_decision_comparison import (
    COMPARISON_COLUMNS,
    build_monthly_decision_comparison,
    validate_monthly_decision_comparison_as_of,
    write_monthly_decision_comparison_report,
)


def test_comparison_uses_research_authority_and_actionable_default_weights():
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
    assert row["replay_primary_scenario_label"] == "Vol Target 63D 35%"
    assert row["actionable_default_scenario_label"] == "Vol Target 63D 25%"
    assert row["recommended_scenario_label"] == "Vol Target 63D 25%"
    assert row["recommended_QLD_weight"] == 0.27
    assert row["replay_primary_cost_mode"] == "net_of_cost"
    assert row["actionable_default_cost_mode"] == "net_of_cost"
    assert row["actionable_default_total_trade_cost"] == pytest.approx(1123.45)
    assert row["actionable_default_cost_drag_on_contributed"] == pytest.approx(0.0123)
    assert bool(row["recommendation_layers_differ"]) is True
    assert bool(row["actual_disagrees_with_authority"]) is True
    assert bool(row["manual_review_required"]) is True
    assert "Research authority has Monte Carlo drawdown breach paths" in row["review_reasons"]
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
    assert "Research authority has Monte Carlo drawdown breach paths" in row["review_reasons"]


def test_comparison_requires_review_when_no_zero_breach_actionable_default():
    ranking = sample_replay_ranking()
    ranking["drawdown_breach_rate"] = 0.01
    ranking["manual_review_required"] = True

    outputs = build_monthly_decision_comparison(
        monthly_decision=sample_monthly_pack(manual_review_required=False, review_reasons=""),
        replay_ranking=ranking,
        actual_policy=sample_actual_policy(),
        generated_at="2026-05-04T00:00:00+00:00",
    )
    row = outputs.comparison.iloc[0]

    assert row["recommended_policy_source"] == "missing_actual_etf_policy_state"
    assert row["recommended_scenario_label"] == ""
    assert bool(row["manual_review_required"]) is True
    assert "No zero-breach actionable default is available" in row["review_reasons"]


def test_comparison_supports_dynamic_tw50_weight_schema():
    outputs = build_monthly_decision_comparison(
        monthly_decision=sample_monthly_pack(
            scenario_id="actual_etf--dca-policy-vol-target-63d-20pct",
            scenario_label="Vol Target 63D 20%",
            manual_review_required=False,
            review_reasons="",
        ).assign(
            QQQ_weight=pd.NA,
            QLD_weight=pd.NA,
            TQQQ_weight=pd.NA,
            **{"0050_weight": 0.60, "00631L_weight": 0.40},
        ),
        replay_ranking=sample_replay_ranking().assign(
            win_rate_vs_benchmark=lambda frame: frame["win_rate_vs_qqq_dca"]
        ),
        actual_policy=pd.DataFrame(
            [
                {
                    "date": "2026-04-30",
                    "data_mode": "actual_etf",
                    "scenario_id": "actual_etf--dca-policy-vol-target-63d-25pct",
                    "scenario_label": "Vol Target 63D 25%",
                    "regime": "vol_target",
                    "reason": "prior 63D realized volatility",
                    "target_effective_leverage": 1.4,
                    "0050_weight": 0.60,
                    "00631L_weight": 0.40,
                    "CASH_weight": 0.0,
                }
            ]
        ),
        source_coverage=sample_tw50_source_coverage(),
        generated_at="2026-05-04T00:00:00+00:00",
    )

    row = outputs.comparison.iloc[0]
    weights = json.loads(row["recommended_weights_json"])
    actual_weights = json.loads(row["actual_primary_weights_json"])

    assert weights == {"0050": 0.6, "00631L": 0.4, "CASH": 0.0}
    assert actual_weights["0050"] == 0.6
    assert row["recommended_weight_sum"] == pytest.approx(1.0)
    assert "price_proxy_not_total_return" in outputs.source_coverage["source_notes"].iloc[0]


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
    assert result.comparison["replay_primary_scenario_label"].iloc[0] == "Vol Target 63D 35%"
    assert result.comparison["recommended_scenario_label"].iloc[0] == "Vol Target 63D 25%"
    assert result.source_coverage["ticker"].tolist() == ["QQQ", "QLD", "TQQQ"]
    html = result.html_path.read_text(encoding="utf-8")
    assert "Monthly Decision Pack" in html
    assert "Research Authority" in html
    assert "Actionable Default" in html
    assert "Actionable Default Weights" in html
    assert "Cost mode" in html
    assert "Cost Drag" in html
    assert "Why Costs Are High" in html
    assert "Cost / Final Equity" in html
    assert "Source Coverage" in html
    assert "No replay compare payload is available" in html
    assert "MC summary" in html
    assert "Vol Target 63D 35%" in html
    assert "Vol Target 63D 25%" in html
    assert "Momentum+Trend 126D/200MA 3.0x to 1.0x" in html
    assert "2006-06-21" in html


def test_tw50_comparison_report_explains_settings_reference_weights_and_chart(
    tmp_path: Path,
):
    sample_monthly_pack(
        scenario_id="actual_etf--dca-policy-vol-target-63d-25pct",
        scenario_label="Vol Target 63D 25%",
        manual_review_required=False,
        review_reasons="",
    ).assign(
        QQQ_weight=pd.NA,
        QLD_weight=pd.NA,
        TQQQ_weight=pd.NA,
        **{"0050_weight": 0.813, "00631L_weight": 0.0},
    ).to_csv(tmp_path / "monthly_decision_pack_tw50.csv", index=False)
    ranking = sample_replay_ranking().copy()
    ranking["drawdown_breach_rate"] = [0.01, 0.02]
    ranking["manual_review_required"] = True
    ranking.to_csv(tmp_path / "monthly_decision_replay_tw50_ranking.csv", index=False)
    sample_tw50_actual_policy().to_csv(
        tmp_path / "dca_policy_optimizer_tw50_policy.csv",
        index=False,
    )
    sample_tw50_source_coverage().to_csv(
        tmp_path / "monthly_decision_replay_tw50_source_coverage.csv",
        index=False,
    )
    (tmp_path / "dca_policy_optimizer_tw50_compare_payload.json").write_text(
        json.dumps(sample_compare_payload(), ensure_ascii=False),
        encoding="utf-8",
    )
    sample_optuna_candidates().to_csv(
        tmp_path / "optuna_tw50_v1_best_candidates.csv",
        index=False,
    )
    sample_external_signal_audit().to_csv(
        tmp_path / "external_signal_audit_tw50.csv",
        index=False,
    )
    (tmp_path / "external_signal_audit_tw50.html").write_text(
        "<html>external</html>",
        encoding="utf-8",
    )
    (tmp_path / "optuna_tw50_v1.html").write_text("<html>optuna</html>", encoding="utf-8")

    result = write_monthly_decision_comparison_report(output_dir=tmp_path, family="tw50")

    row = result.comparison.iloc[0]
    assert row["recommended_scenario_label"] == ""
    assert bool(row["manual_review_required"]) is True
    html = result.html_path.read_text(encoding="utf-8")
    assert "Weekly Execution Decision Pack" in html
    assert "0050 / 00631L / CASH" in html
    assert "Initial 100,000; monthly contribution 10,000" in html
    assert "trade cadence monthly core + weekly delta trigger; contribution cadence monthly" in html
    assert "Fixed 1.5x DCA" in html
    assert "Trade cadence" in html
    assert "Contribution cadence" in html
    assert "2014-10-31 ~ 2026-04-30; total contributed 1,490,000.00" in html
    assert "1999-03-10 ~ 2026-04-30; total contributed 3,360,000.00" in html
    assert "TW ETF commission 0.1425% * 28%" in html
    assert "minimum 1 TWD" in html
    assert "Manual review reference, not an automatic order" in html
    assert "0050" in html
    assert "81.3%" in html
    assert "p05 XIRR" in html
    assert "The bad 5% Monte Carlo annualized return" in html
    assert "p05 drawdown" in html
    assert "cost drag" in html
    assert "turnover" in html
    assert "MC breach" in html
    assert "Replay Compare Lab" in html
    assert "0050 base DCA" in html
    assert "Effective leverage" in html
    assert "Cumulative trade cost" in html
    assert "Turnover" in html
    assert "comparison-compare-payload" in html
    assert "Optuna Research Candidates" in html
    assert "Pareto rank" in html
    assert "watchlist" in html
    assert "Open Optuna HTML report" in html
    assert "External Regime Signals" in html
    assert "Open external signal audit" in html
    return
    assert "初始 100,000；每月 10,000" in html
    assert "2014-10-31 ~ 2026-04-30；總投入 1,490,000.00" in html
    assert "1999-03-10 ~ 2026-04-30；總投入 3,360,000.00" in html
    assert "台股 ETF 手續費 0.1425% * 28%" in html
    assert "人工 review 參考，不是自動下單建議" in html
    assert "0050" in html
    assert "81.3%" in html
    assert "p05 XIRR" in html
    assert "偏壞 5% 情境的年化報酬" in html
    assert "p05 drawdown" in html
    assert "cost drag" in html
    assert "turnover" in html
    assert "MC breach" in html
    assert "候選策略趨勢圖" in html
    assert "comparison-compare-payload" in html
    assert "累積交易成本" in html
    assert "換倉強度" in html
    assert "Optuna Research Candidates" in html
    assert "Pareto rank" in html
    assert "watchlist" in html
    assert "Open Optuna HTML report" in html
    assert "External Regime Signals" in html
    assert "Open external signal audit" in html


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
                "drawdown_breach_rate": 0.0001666667,
                "cost_mode": "net_of_cost",
                "total_trade_cost": 2345.67,
                "cost_drag_on_contributed": 0.0250,
                "cost_to_final_equity": 0.004,
                "trade_days": 24,
                "max_single_day_trade_cost": 500.0,
                "turnover_sum": 51.23,
                "manual_review_required": True,
            },
            {
                "selector": "hybrid_primary",
                "ranking_method": "monte_carlo",
                "replay_rank": 2,
                "scenario_id": "hybrid_primary--dca-policy-vol-target-63d-25pct",
                "scenario_label": "Vol Target 63D 25%",
                "eligible_for_monthly_signal": True,
                "cohort_gate_passed": True,
                "win_rate_vs_qqq_dca": 0.9884,
                "expected_xirr": 0.2400,
                "median_xirr": 0.2373,
                "p05_xirr": -0.0700,
                "expected_max_drawdown": -0.6000,
                "p05_max_drawdown": -0.8931,
                "drawdown_breach_rate": 0.0,
                "cost_mode": "net_of_cost",
                "total_trade_cost": 1123.45,
                "cost_drag_on_contributed": 0.0123,
                "cost_to_final_equity": 0.003,
                "trade_days": 18,
                "max_single_day_trade_cost": 320.0,
                "turnover_sum": 32.10,
                "manual_review_required": False,
            },
        ]
    )


def sample_actual_policy() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2025-12-29",
                "data_mode": "actual_etf",
                "scenario_id": "actual_etf--dca-policy-vol-target-63d-25pct",
                "scenario_label": "Vol Target 63D 25%",
                "regime": "vol_target",
                "reason": "prior 63D realized volatility",
                "target_effective_leverage": 1.27,
                "QQQ_weight": 0.73,
                "QLD_weight": 0.27,
                "TQQQ_weight": 0.0,
                "CASH_weight": 0.0,
            },
            {
                "date": "2025-12-30",
                "data_mode": "actual_etf",
                "scenario_id": "actual_etf--dca-policy-vol-target-63d-25pct",
                "scenario_label": "Vol Target 63D 25%",
                "regime": "vol_target",
                "reason": "prior 63D realized volatility",
                "target_effective_leverage": 1.27,
                "QQQ_weight": 0.73,
                "QLD_weight": 0.27,
                "TQQQ_weight": 0.0,
                "CASH_weight": 0.0,
            },
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


def sample_tw50_source_coverage() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "0050",
                "replay_start": "1999-03-10",
                "replay_end": "2026-04-30",
                "actual_start": "2003-06-30",
                "actual_end": "2026-04-30",
                "synthetic_backfill_start": "",
                "synthetic_backfill_end": "",
                "proxy_start": "1999-03-10",
                "proxy_end": "2002-12-31",
                "total_return_proxy_start": "2003-01-02",
                "total_return_proxy_end": "2003-06-27",
                "splice_date": "2003-06-30",
                "source_notes": "price_proxy_not_total_return",
            },
        ]
    )


def sample_tw50_actual_policy() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-04-30",
                "data_mode": "actual_etf",
                "scenario_id": "actual_etf--dca-policy-vol-target-63d-25pct",
                "scenario_label": "Vol Target 63D 25%",
                "regime": "vol_target",
                "reason": "prior 63D realized volatility",
                "target_effective_leverage": 0.813,
                "0050_weight": 0.813,
                "00631L_weight": 0.0,
                "CASH_weight": 0.187,
            }
        ]
    )


def sample_compare_payload() -> dict[str, object]:
    return {
        "scan_mode": "fast",
        "metrics": {
            "total_equity": {"label": "Net equity", "axis": "Equity", "format": "money"},
            "normalized_equity": {"label": "Normalized", "axis": "Index", "format": "money"},
            "drawdown": {"label": "Drawdown", "axis": "Drawdown", "format": "percent"},
            "effective_leverage": {"label": "Leverage", "axis": "Leverage", "format": "number"},
            "total_contributed": {
                "label": "Contribution",
                "axis": "Contribution",
                "format": "money",
            },
            "cumulative_trade_cost": {"label": "Cost", "axis": "Cost", "format": "money"},
            "trade_cost": {"label": "Trade cost", "axis": "Cost", "format": "money"},
            "commission": {"label": "Commission", "axis": "Cost", "format": "money"},
            "transaction_tax": {"label": "Tax", "axis": "Cost", "format": "money"},
            "slippage": {"label": "Slippage", "axis": "Cost", "format": "money"},
            "turnover": {"label": "Turnover", "axis": "Turnover", "format": "number"},
        },
        "scenarios": [
            payload_scenario(
                key="actual_etf--dca-policy-vol-target-63d-25pct",
                data_mode="actual_etf",
                final_contributed=1_490_000.0,
            ),
            payload_scenario(
                key="synthetic_stress--dca-policy-vol-target-63d-35pct",
                data_mode="synthetic_stress",
                final_contributed=3_360_000.0,
            ),
            payload_scenario(
                key="synthetic_stress--dca-policy-constant_1p0",
                data_mode="synthetic_stress",
                final_contributed=3_360_000.0,
            ),
        ],
    }


def sample_optuna_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pareto_rank": 1,
                "trial_number": 42,
                "candidate_status": "watchlist",
                "candidate_reason": "Higher cost drag without clear risk improvement.",
                "expected_xirr": 0.08,
                "p05_xirr": -0.02,
                "p05_max_drawdown": -0.55,
                "cost_drag_on_contributed": 0.12,
                "turnover_sum": 99.0,
            }
        ]
    )


def sample_external_signal_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "category": "artifact",
                "check_id": "features",
                "status": "pass",
                "summary": "Found market_regime_features_tw50.csv",
                "details": "",
            },
            {
                "category": "coverage",
                "check_id": "fear_greed_history_depth",
                "status": "warn",
                "summary": "Fear & Greed coverage is limited",
                "details": "",
            },
        ]
    )


def payload_scenario(
    *,
    key: str,
    data_mode: str,
    final_contributed: float,
) -> dict[str, object]:
    return {
        "key": key,
        "short": key.rsplit("-", maxsplit=1)[-1],
        "full": key,
        "data_mode": data_mode,
        "strategy_family": "test",
        "risk_flag": "ok",
        "validation_status": "stable",
        "default": False,
        "dates": ["2014-10-31", "2026-04-30"]
        if data_mode == "actual_etf"
        else ["1999-03-10", "2026-04-30"],
        "series": {
            "total_equity": [100_000.0, 1_000_000.0],
            "normalized_equity": [10_000.0, 20_000.0],
            "drawdown": [0.0, -0.1],
            "effective_leverage": [1.0, 1.2],
            "total_contributed": [100_000.0, final_contributed],
            "cumulative_trade_cost": [0.0, 12_345.0],
            "trade_cost": [0.0, 123.0],
            "commission": [0.0, 30.0],
            "transaction_tax": [0.0, 80.0],
            "slippage": [0.0, 13.0],
            "turnover": [0.0, 0.5],
        },
    }
