from pathlib import Path

import pandas as pd
import pytest

from investment_backtest_lab.dca_policy_optimizer import DATA_MODE_SYNTHETIC
from investment_backtest_lab.leveraged_etf_lab import ProductSpec
from investment_backtest_lab.models import DCAPolicyOptimizerConfig, MonthlyDecisionReplayConfig
from investment_backtest_lab.monthly_decision_replay import (
    SELECTOR_SYNTHETIC_PRIMARY,
    build_monthly_decision_replay_outputs,
    build_replay_ranking,
    write_monthly_decision_replay_report,
)


def test_replay_ranking_uses_qqq_dca_tie_rule_and_hard_drawdown_limit():
    cohorts = pd.DataFrame(
        [
            cohort_row("synthetic_stress--dca-policy-constant_1p0", 0.05, -0.40, tie=True),
            cohort_row("synthetic_stress--dca-policy-constant_1p0", 0.05, -0.35, tie=True),
            cohort_row("synthetic_stress--dca-policy-better", 0.08, -0.60, win=True),
            cohort_row("synthetic_stress--dca-policy-better", 0.04, -0.70),
            cohort_row("synthetic_stress--dca-policy-breach", 0.40, -0.96, win=True),
            cohort_row("synthetic_stress--dca-policy-breach", 0.30, -0.90, win=True),
        ]
    )

    ranking = build_replay_ranking(
        cohorts,
        selector=SELECTOR_SYNTHETIC_PRIMARY,
        config=MonthlyDecisionReplayConfig(),
    ).set_index("scenario_id")

    baseline = ranking.loc["synthetic_stress--dca-policy-constant_1p0"]
    breach = ranking.loc["synthetic_stress--dca-policy-breach"]

    assert baseline["win_rate_vs_qqq_dca"] == pytest.approx(0.50)
    assert bool(breach["eligible_for_monthly_signal"]) is False
    assert breach["drawdown_breach_rate"] == pytest.approx(0.50)


def test_synthetic_primary_replay_uses_synthetic_source_and_restarts_dca():
    outputs = build_monthly_decision_replay_outputs(
        mode_prices={DATA_MODE_SYNTHETIC: sample_long_prices()},
        products=sample_products(),
        optimizer_config=small_optimizer_config(),
        replay_config=small_replay_config(),
        scan_mode="fast",
        selector=SELECTOR_SYNTHETIC_PRIMARY,
    )

    assert not outputs.cohorts.empty
    assert set(outputs.cohorts["data_mode"]) == {DATA_MODE_SYNTHETIC}
    assert set(outputs.cohorts["selector"]) == {SELECTOR_SYNTHETIC_PRIMARY}
    assert pd.to_datetime(outputs.cohorts["cohort_end"]).max() <= sample_long_prices().index.max()
    expected = outputs.cohorts["ending_equity"] / outputs.cohorts["total_contributed"] - 1.0
    assert outputs.cohorts["simple_cash_return"].to_numpy() == pytest.approx(
        expected.to_numpy()
    )


def test_monthly_decision_replay_writes_html_and_csv(tmp_path: Path):
    outputs = build_monthly_decision_replay_outputs(
        mode_prices={DATA_MODE_SYNTHETIC: sample_long_prices()},
        products=sample_products(),
        optimizer_config=small_optimizer_config(),
        replay_config=small_replay_config(),
        scan_mode="fast",
        selector=SELECTOR_SYNTHETIC_PRIMARY,
    )

    result = write_monthly_decision_replay_report(
        outputs=outputs,
        output_dir=tmp_path,
        family="qqq",
        actual_primary_signal=pd.DataFrame(
            [{"scenario_label": "Actual Primary Test", "data_mode": "actual_etf"}]
        ),
    )

    assert result.html_path.exists()
    assert result.decisions_path.exists()
    assert result.cohorts_path.exists()
    assert result.ranking_path.exists()
    assert result.equity_path.exists()
    html = result.html_path.read_text(encoding="utf-8")
    assert "Synthetic Primary Replay" in html
    assert "QQQ DCA benchmark" in html
    assert "win rate" in html
    assert "drawdown breach rate" in html
    assert "Actual Primary Test" in html


def cohort_row(
    scenario_id: str,
    xirr: float,
    max_drawdown: float,
    *,
    win: bool = False,
    tie: bool = False,
) -> dict[str, object]:
    return {
        "selector": SELECTOR_SYNTHETIC_PRIMARY,
        "data_mode": DATA_MODE_SYNTHETIC,
        "horizon_years": 5,
        "cohort_start": "2010-01-04",
        "cohort_end": "2015-01-05",
        "scenario_id": scenario_id,
        "scenario_label": scenario_id,
        "strategy_family": "test",
        "cohort_rank": 1,
        "total_contributed": 70_000.0,
        "ending_equity": 100_000.0,
        "simple_cash_return": 100_000.0 / 70_000.0 - 1.0,
        "xirr": xirr,
        "max_drawdown": max_drawdown,
        "drawdown_breach": max_drawdown < -0.95,
        "benchmark_scenario_id": "synthetic_stress--dca-policy-constant_1p0",
        "benchmark_xirr": 0.05,
        "benchmark_ending_equity": 90_000.0,
        "win_vs_benchmark": win,
        "tie_vs_benchmark": tie,
    }


def small_optimizer_config() -> DCAPolicyOptimizerConfig:
    return DCAPolicyOptimizerConfig(
        target_leverage_grid=(1.0, 2.0),
        trend_windows=(20,),
        momentum_windows=(20,),
        volatility_windows=(20,),
        volatility_targets=(0.25,),
        drawdown_guards=(-0.10, -0.20, -0.30, -0.50),
        dca_initial_cash=10_000.0,
        dca_contribution=1_000.0,
        top_n=6,
        fast_top_n=4,
        walk_forward_top_n=2,
        cohort_horizons_years=(2,),
    )


def small_replay_config() -> MonthlyDecisionReplayConfig:
    return MonthlyDecisionReplayConfig(
        selector=SELECTOR_SYNTHETIC_PRIMARY,
        horizons_years=(2,),
        initial_cash=10_000.0,
        monthly_contribution=1_000.0,
        max_drawdown_limit=-0.95,
        min_win_rate=0.50,
    )


def sample_products() -> list[ProductSpec]:
    return [
        ProductSpec("QQQ", 1.0, "QQQ 1x"),
        ProductSpec("QLD", 2.0, "QLD 2x"),
        ProductSpec("TQQQ", 3.0, "TQQQ 3x"),
    ]


def sample_long_prices() -> pd.DataFrame:
    dates = pd.date_range("2010-01-04", "2016-12-30", freq="B")
    base_return = pd.Series(0.0005, index=dates)
    base_return.iloc[260:360] = -0.0025
    base_return.iloc[900:980] = -0.0020
    qqq = 100.0 * (1.0 + base_return).cumprod()
    qld = 100.0 * (1.0 + base_return * 2.0).cumprod()
    tqqq = 100.0 * (1.0 + base_return * 3.0).cumprod()
    return pd.DataFrame({"QQQ": qqq, "QLD": qld, "TQQQ": tqqq}, index=dates)
