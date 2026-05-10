from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from investment_backtest_lab.models import (
    AssetSpec,
    AssetType,
    BacktestConfig,
    DataSource,
    Market,
    StrategyConfig,
    StrategySearchConfig,
)
from investment_backtest_lab.pre_optimization_audit import (
    AUDIT_FAIL,
    build_pre_optimization_audit_checks,
    dynamic_drawdown_limit_from_checks,
    overall_audit_status,
    write_pre_optimization_audit,
)


def test_pre_optimization_audit_writes_reports_and_dynamic_limit(tmp_path: Path):
    config = sample_config()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("strategy_search:\n  drawdown_limit_multiplier: 1.2\n", encoding="utf-8")
    write_required_artifacts(tmp_path)

    result = write_pre_optimization_audit(
        config=config,
        config_path=config_path,
        family="tw50",
        output_dir=tmp_path,
    )

    assert result.csv_path.exists()
    assert result.markdown_path.exists()
    assert result.html_path.exists()
    assert result.base_1x_stress_max_drawdown == -0.60
    assert result.dynamic_drawdown_limit == -0.72
    assert "dynamic_drawdown_limit" in set(result.checks["check_id"])
    assert overall_audit_status(result.checks) in {"pass", "warn"}


def test_pre_optimization_audit_missing_artifact_fails(tmp_path: Path):
    config = sample_config()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("config", encoding="utf-8")

    checks = build_pre_optimization_audit_checks(
        config=config,
        config_path=config_path,
        family="tw50",
        output_dir=tmp_path,
    )

    assert overall_audit_status(checks) == AUDIT_FAIL
    assert "optimizer_metrics" in set(checks["check_id"])


def test_dynamic_drawdown_limit_from_checks():
    checks = pd.DataFrame(
        [
            {
                "category": "risk_gate",
                "check_id": "base_1x_stress_max_drawdown",
                "status": "pass",
                "summary": "",
                "details": "-0.66",
            },
            {
                "category": "risk_gate",
                "check_id": "dynamic_drawdown_limit",
                "status": "pass",
                "summary": "",
                "details": "-0.792",
            },
        ]
    )

    base, limit = dynamic_drawdown_limit_from_checks(checks)

    assert base == -0.66
    assert limit == -0.792


def write_required_artifacts(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "data_mode": "synthetic_stress",
                "scenario_id": "synthetic_stress--dca-policy-constant_1p0",
                "scenario_label": "Constant 1.0x",
                "xirr": 0.1,
                "max_drawdown": -0.60,
                "cost_drag_on_contributed": 0.01,
                "turnover_sum": 1.0,
            }
        ]
    ).to_csv(path / "dca_policy_optimizer_tw50_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "date": "2026-04-30",
                "scenario_id": "synthetic_stress--dca-policy-constant_1p0",
                "target_effective_leverage": 1.0,
            }
        ]
    ).to_csv(path / "dca_policy_optimizer_tw50_policy.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_id": "hybrid_primary--dca-policy-constant_1p0",
                "expected_xirr": 0.1,
                "p05_xirr": -0.01,
                "p05_max_drawdown": -0.55,
                "drawdown_breach_rate": 0.0,
                "cohort_gate_passed": True,
            }
        ]
    ).to_csv(path / "monthly_decision_replay_tw50_ranking.csv", index=False)
    pd.DataFrame(
        [
            {
                "decision_authority": "hybrid_primary_monte_carlo",
                "manual_review_required": True,
                "recommended_as_of_date": "2026-04-30",
            }
        ]
    ).to_csv(path / "monthly_decision_comparison_tw50.csv", index=False)
    pd.DataFrame(
        [
            {
                "ticker": "0050",
                "source_notes": "price_proxy_not_total_return",
                "splice_date": "2003-01-02",
            },
            {
                "ticker": "00631L",
                "source_notes": "synthetic_backfill",
                "splice_date": "2014-10-31",
            },
        ]
    ).to_csv(path / "tw50_total_return_sources.csv", index=False)
    pd.DataFrame(
        [
            {
                "ticker": "0050",
                "ex_dividend_date": "2025-07-01",
                "split_adjusted_cash_dividend": 1.0,
            }
        ]
    ).to_csv(path / "tw50_dividend_audit.csv", index=False)


def sample_config() -> BacktestConfig:
    return BacktestConfig(
        universe=[
            AssetSpec("0050", Market.TW, AssetType.ETF, "TWD", DataSource.FINMIND),
            AssetSpec("00631L", Market.TW, AssetType.ETF, "TWD", DataSource.FINMIND),
        ],
        start_date=date(1999, 3, 10),
        end_date=date(2026, 5, 1),
        base_currency="TWD",
        strategy=StrategyConfig("test"),
        cost_model={
            "tw": {
                "commission_rate": 0.001425,
                "commission_discount": 0.28,
                "min_commission": 20.0,
                "etf_transaction_tax_rate": 0.001,
                "slippage_bps": 1.0,
            }
        },
        strategy_search=StrategySearchConfig(family="tw50"),
    )
