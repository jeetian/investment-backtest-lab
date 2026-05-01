from pathlib import Path

import pandas as pd
import pytest

from investment_backtest_lab.dca_policy_optimizer import (
    VALIDATION_STABLE,
    VALIDATION_WATCHLIST,
    PolicyScenarioSpec,
    apply_cross_mode_candidate_filter,
    apply_policy_validation,
    build_dca_policy_optimizer_outputs,
    build_policy_weights,
    policy_config_for_scan_mode,
    rank_policy_metrics,
    target_leverage_to_product_weights,
    write_dca_policy_optimizer_report,
)
from investment_backtest_lab.leveraged_etf_lab import CASH, ProductSpec
from investment_backtest_lab.models import DCAPolicyOptimizerConfig


def test_target_leverage_to_product_weights_interpolates_between_products():
    products = sample_products()

    assert target_leverage_to_product_weights(0.5, products) == {
        "QQQ": 0.5,
        "QLD": 0.0,
        "TQQQ": 0.0,
        CASH: 0.5,
    }
    assert target_leverage_to_product_weights(1.5, products) == {
        "QQQ": 0.5,
        "QLD": 0.5,
        "TQQQ": 0.0,
        CASH: 0.0,
    }
    assert target_leverage_to_product_weights(2.5, products) == {
        "QQQ": 0.0,
        "QLD": 0.5,
        "TQQQ": 0.5,
        CASH: 0.0,
    }


def test_trend_policy_uses_shifted_signal_to_avoid_lookahead():
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    prices = pd.DataFrame(
        {
            "QQQ": [100.0, 100.0, 120.0, 130.0],
            "QLD": [100.0, 100.0, 140.0, 160.0],
            "TQQQ": [100.0, 100.0, 160.0, 190.0],
        },
        index=dates,
    )
    spec = PolicyScenarioSpec(
        name="trend_test",
        label="Trend Test",
        short_label="Trend",
        family="trend_ladder",
        kind="trend_ladder",
        params={"window": 2, "risk_on": 3.0, "risk_off": 0.0},
    )

    weights, policy = build_policy_weights(
        prices=prices,
        spec=spec,
        products=sample_products(),
        config=DCAPolicyOptimizerConfig(),
    )

    assert policy.iloc[2]["target_effective_leverage"] == pytest.approx(0.0)
    assert policy.iloc[3]["target_effective_leverage"] == pytest.approx(3.0)
    assert weights.sum(axis=1).to_numpy() == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_policy_optimizer_outputs_dca_metrics_and_current_signal():
    config = small_config()
    outputs = build_dca_policy_optimizer_outputs(
        actual_prices=sample_long_prices(),
        synthetic_prices=pd.DataFrame(),
        products=sample_products(),
        config=config,
        scan_mode="fast",
    )

    assert not outputs.metrics.empty
    assert not outputs.policy.empty
    assert not outputs.current_signal.empty
    assert outputs.metrics["effective_leverage_max"].max() <= 3.0
    assert outputs.metrics["total_contributed"].min() >= config.dca_initial_cash
    assert {
        "xirr",
        "ending_equity",
        "total_contributed",
        "simple_cash_return",
        "max_drawdown",
        "recovery_days",
        "effective_leverage_avg",
        "effective_leverage_max",
        "validation_status",
        "eligible_for_candidate",
    }.issubset(outputs.metrics.columns)
    assert {
        "regime",
        "reason",
        "target_effective_leverage",
        "QQQ_weight",
        "QLD_weight",
        "TQQQ_weight",
        "CASH_weight",
        "trend_value",
        "momentum_value",
        "volatility_value",
        "drawdown",
    }.issubset(outputs.policy.columns)


def test_policy_validation_prefers_stable_strategy_over_high_xirr_bad_test():
    metrics = pd.DataFrame(
        [
            metric_row("actual_etf--dca-policy-stable", 0.08, -0.30),
            metric_row("actual_etf--dca-policy-fragile", 0.80, -0.40),
        ]
    )
    walk_forward = pd.DataFrame(
        [
            walk_row("actual_etf--dca-policy-stable", True, 0.08, -0.30),
            walk_row("actual_etf--dca-policy-stable", True, 0.07, -0.35),
            walk_row("actual_etf--dca-policy-fragile", False, -0.10, -0.96),
            walk_row("actual_etf--dca-policy-fragile", False, 0.70, -0.80),
        ]
    )

    ranked = rank_policy_metrics(
        apply_policy_validation(metrics, walk_forward, config=DCAPolicyOptimizerConfig())
    )

    assert ranked.iloc[0]["scenario_id"] == "actual_etf--dca-policy-stable"
    assert ranked.iloc[0]["validation_status"] == VALIDATION_STABLE
    assert ranked.iloc[-1]["validation_status"] != VALIDATION_STABLE


def test_drawdown_below_limit_is_not_candidate_but_high_risk_band_can_remain_watchlist():
    metrics = pd.DataFrame(
        [
            metric_row("actual_etf--dca-policy-breach", 0.50, -0.951),
            metric_row("actual_etf--dca-policy-high-risk", 0.20, -0.90),
        ]
    )
    walk_forward = pd.DataFrame(
        [
            walk_row("actual_etf--dca-policy-breach", True, 0.50, -0.80),
            walk_row("actual_etf--dca-policy-high-risk", True, 0.20, -0.90),
        ]
    )

    validated = apply_policy_validation(metrics, walk_forward, config=DCAPolicyOptimizerConfig())
    by_id = validated.set_index("scenario_id")

    assert bool(by_id.loc["actual_etf--dca-policy-breach", "risk_failed"]) is True
    assert bool(by_id.loc["actual_etf--dca-policy-breach", "eligible_for_candidate"]) is False
    assert by_id.loc["actual_etf--dca-policy-high-risk", "risk_flag"] == "high_drawdown"
    assert by_id.loc["actual_etf--dca-policy-high-risk", "validation_status"] in {
        VALIDATION_STABLE,
        VALIDATION_WATCHLIST,
    }


def test_cross_mode_filter_blocks_actual_candidate_when_synthetic_breaches_limit():
    metrics = pd.DataFrame(
        [
            metric_row("actual_etf--dca-policy-constant_3p0", 0.40, -0.80),
            metric_row("synthetic_stress--dca-policy-constant_3p0", 0.30, -0.99),
        ]
    )
    metrics["validation_status"] = VALIDATION_STABLE
    metrics["eligible_for_candidate"] = True

    filtered = apply_cross_mode_candidate_filter(
        metrics,
        config=DCAPolicyOptimizerConfig(),
    )

    assert filtered["cross_mode_drawdown_breach"].all()
    assert not filtered["eligible_for_candidate"].any()


def test_walk_forward_splits_do_not_overlap_and_test_dca_restarts():
    outputs = build_dca_policy_optimizer_outputs(
        actual_prices=sample_long_prices(),
        synthetic_prices=pd.DataFrame(),
        products=sample_products(),
        config=small_config(),
        scan_mode="fast",
    )

    walk = outputs.walk_forward
    assert not walk.empty
    train_end = pd.to_datetime(walk["train_end"])
    test_start = pd.to_datetime(walk["test_start"])
    assert (train_end <= test_start).all()
    assert walk["test_total_contributed"].min() >= 22_000.0
    expected = walk["test_ending_equity"] / walk["test_total_contributed"] - 1.0
    assert walk["test_simple_cash_return"].to_numpy() == pytest.approx(expected.to_numpy())


def test_dca_policy_optimizer_report_writes_html_and_csv(tmp_path):
    outputs = build_dca_policy_optimizer_outputs(
        actual_prices=sample_long_prices(),
        synthetic_prices=sample_long_prices(),
        products=sample_products(),
        config=small_config(),
        scan_mode="fast",
    )

    result = write_dca_policy_optimizer_report(
        outputs=outputs,
        output_dir=tmp_path,
        family="qqq",
        config_path=Path("configs/mvp_example.yaml"),
    )

    assert result.html_path.exists()
    assert result.metrics_path.exists()
    assert result.policy_path.exists()
    assert result.walk_forward_path.exists()
    assert result.current_signal_path.exists()
    html = result.html_path.read_text(encoding="utf-8")
    assert "DCA Policy Optimizer" in html
    assert "Current Signal Board" in html
    assert "Best Candidates" in html
    assert "Compare Lab" in html
    assert "不是投資建議" in html
    assert "policy CSV" in html


def test_fast_policy_config_reduces_scan_space():
    config = DCAPolicyOptimizerConfig()
    fast = policy_config_for_scan_mode(config, "fast")
    full = policy_config_for_scan_mode(config, "full")

    assert len(fast.trend_windows) < len(full.trend_windows)
    assert fast.top_n == config.fast_top_n
    assert full.top_n == config.top_n


def metric_row(scenario_id: str, xirr: float, max_drawdown: float) -> dict[str, object]:
    return {
        "data_mode": "actual_etf",
        "scenario_id": scenario_id,
        "scenario_label": scenario_id,
        "short_label": scenario_id,
        "strategy_family": "test",
        "start_date": "2010-01-01",
        "end_date": "2020-01-01",
        "total_contributed": 100_000.0,
        "ending_equity": 200_000.0,
        "simple_cash_return": 1.0,
        "xirr": xirr,
        "cagr": xirr,
        "volatility": 0.2,
        "sharpe": 1.0,
        "sortino": 1.0,
        "calmar": 1.0,
        "max_drawdown": max_drawdown,
        "recovery_days": 100,
        "effective_leverage_avg": 2.0,
        "effective_leverage_max": 3.0,
        "risk_flag": "drawdown_limit_breach" if max_drawdown < -0.95 else "high_drawdown",
        "risk_failed": max_drawdown < -0.95,
    }


def walk_row(
    scenario_id: str,
    passed: bool,
    test_xirr: float,
    test_drawdown: float,
) -> dict[str, object]:
    return {
        "data_mode": "actual_etf",
        "fold_index": 1,
        "scenario_id": scenario_id,
        "scenario_label": scenario_id,
        "strategy_family": "test",
        "train_start": "2010-01-01",
        "train_end": "2015-01-01",
        "test_start": "2015-01-01",
        "test_end": "2017-01-01",
        "train_rank": 1,
        "test_total_contributed": 34_000.0,
        "test_ending_equity": 40_000.0,
        "test_simple_cash_return": 40_000.0 / 34_000.0 - 1.0,
        "test_xirr": test_xirr,
        "test_max_drawdown": test_drawdown,
        "test_recovery_days": 100,
        "test_risk_flag": "ok" if passed else "drawdown_limit_breach",
        "test_risk_failed": not passed,
        "passed_fold": passed,
    }


def small_config() -> DCAPolicyOptimizerConfig:
    return DCAPolicyOptimizerConfig(
        target_leverage_grid=(0.0, 1.0, 2.0, 3.0),
        trend_windows=(20,),
        momentum_windows=(20,),
        volatility_windows=(20,),
        volatility_targets=(0.25,),
        drawdown_guards=(-0.10, -0.20, -0.30, -0.50),
        top_n=6,
        fast_top_n=4,
        walk_forward_top_n=2,
    )


def sample_products() -> list[ProductSpec]:
    return [
        ProductSpec("QQQ", 1.0, "QQQ 1x"),
        ProductSpec("QLD", 2.0, "QLD 2x"),
        ProductSpec("TQQQ", 3.0, "TQQQ 3x"),
    ]


def sample_long_prices() -> pd.DataFrame:
    dates = pd.date_range("2010-01-04", "2018-12-31", freq="B")
    base_return = pd.Series(0.0004, index=dates)
    base_return.iloc[300:420] = -0.002
    base_return.iloc[900:980] = -0.003
    qqq = 100.0 * (1.0 + base_return).cumprod()
    qld = 100.0 * (1.0 + base_return * 2.0).cumprod()
    tqqq = 100.0 * (1.0 + base_return * 3.0).cumprod()
    return pd.DataFrame({"QQQ": qqq, "QLD": qld, "TQQQ": tqqq}, index=dates)
