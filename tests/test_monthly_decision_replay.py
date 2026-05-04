from pathlib import Path

import pandas as pd
import pytest

from investment_backtest_lab.dca_policy_optimizer import DATA_MODE_SYNTHETIC
from investment_backtest_lab.leveraged_etf_lab import ProductSpec
from investment_backtest_lab.models import DCAPolicyOptimizerConfig, MonthlyDecisionReplayConfig
from investment_backtest_lab.monthly_decision_replay import (
    DATA_MODE_HYBRID,
    SELECTOR_HYBRID_PRIMARY,
    SELECTOR_SYNTHETIC_PRIMARY,
    build_hybrid_actual_preferred_prices,
    build_monte_carlo_replay_ranking,
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
    assert not outputs.mc_trials.empty
    assert set(outputs.mc_trials["block_length_days"]) == {21, 63}


def test_hybrid_prices_use_scaled_synthetic_backfill_before_actual_listing():
    dates = pd.date_range("2020-01-01", "2020-01-10", freq="B")
    synthetic = pd.DataFrame(
        {
            "QQQ": [100, 101, 102, 103, 104, 105, 106, 107],
            "QLD": [100, 102, 104, 106, 108, 110, 112, 114],
            "TQQQ": [100, 103, 106, 109, 112, 115, 118, 121],
        },
        index=dates,
    )
    actual = pd.DataFrame(
        {
            "QQQ": [200, 202, 204, 206, 208, 210, 212, 214],
            "QLD": [pd.NA, pd.NA, pd.NA, 53, 54, 55, 56, 57],
            "TQQQ": [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 23, 24, 25],
        },
        index=dates,
    )

    result = build_hybrid_actual_preferred_prices(
        actual_prices=actual,
        synthetic_prices=synthetic,
        products=sample_products(),
    )

    assert result.prices.loc[dates[2], "QLD"] == pytest.approx(104 * (53 / 106))
    assert result.prices.loc[dates[3], "QLD"] == pytest.approx(53)
    assert result.prices.loc[dates[4], "TQQQ"] == pytest.approx(112 * (23 / 115))
    assert result.prices.loc[dates[5], "TQQQ"] == pytest.approx(23)
    coverage = result.coverage.set_index("ticker")
    assert coverage.loc["QLD", "synthetic_backfill_start"] == "2020-01-01"
    assert coverage.loc["QLD", "synthetic_backfill_end"] == "2020-01-03"
    assert coverage.loc["TQQQ", "splice_date"] == "2020-01-08"


def test_hybrid_primary_replay_uses_mc_ranking_and_source_coverage():
    hybrid = build_hybrid_actual_preferred_prices(
        actual_prices=sample_actual_with_late_listings(),
        synthetic_prices=sample_long_prices(),
        products=sample_products(),
    )
    outputs = build_monthly_decision_replay_outputs(
        mode_prices={DATA_MODE_HYBRID: hybrid.prices},
        products=sample_products(),
        optimizer_config=small_optimizer_config(),
        replay_config=small_replay_config(selector=SELECTOR_HYBRID_PRIMARY),
        scan_mode="fast",
        selector=SELECTOR_HYBRID_PRIMARY,
        source_coverage=hybrid.coverage,
    )

    assert outputs.ranking_method == "monte_carlo"
    assert not outputs.ranking.empty
    assert "expected_xirr" in outputs.ranking.columns
    assert set(outputs.mc_trials["selector"]) == {SELECTOR_HYBRID_PRIMARY}
    assert set(outputs.source_coverage["ticker"]) == {"QQQ", "QLD", "TQQQ"}


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
    assert result.mc_summary_path.exists()
    assert result.mc_trials_path.exists()
    assert result.source_coverage_path.exists()
    assert result.mc_trials_path.suffixes[-2:] == [".csv", ".gz"]
    assert "expected_xirr" in result.mc_summary.columns
    html = result.html_path.read_text(encoding="utf-8")
    assert "Hybrid-Primary Monte Carlo Replay" in html
    assert "MC summary CSV" in html
    assert "compressed MC trials CSV" in html
    assert "win rate" in html.lower()
    assert "drawdown breach rate" in html
    assert "Actual Primary Test" in html


def test_monte_carlo_ranking_is_reproducible_and_cohort_gate_blocks_candidate():
    trials = pd.DataFrame(
        [
            mc_row("hybrid_primary--dca-policy-good", 0.10, -0.40, win=True),
            mc_row("hybrid_primary--dca-policy-good", 0.08, -0.35, win=True),
            mc_row("hybrid_primary--dca-policy-breached", 0.40, -0.20, win=True),
            mc_row("hybrid_primary--dca-policy-breached", 0.35, -0.30, win=True),
        ]
    )
    cohorts = pd.DataFrame(
        [
            cohort_row("hybrid_primary--dca-policy-good", 0.09, -0.50, win=True),
            cohort_row("hybrid_primary--dca-policy-breached", 0.30, -0.96, win=True),
        ]
    )

    ranking = build_monte_carlo_replay_ranking(
        trials,
        cohorts=cohorts,
        selector=SELECTOR_HYBRID_PRIMARY,
        config=MonthlyDecisionReplayConfig(),
    ).set_index("scenario_id")

    assert bool(ranking.loc["hybrid_primary--dca-policy-good", "eligible_for_monthly_signal"])
    assert not bool(
        ranking.loc["hybrid_primary--dca-policy-breached", "eligible_for_monthly_signal"]
    )
    assert ranking.loc["hybrid_primary--dca-policy-good", "expected_xirr"] == pytest.approx(0.09)


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


def mc_row(
    scenario_id: str,
    xirr: float,
    max_drawdown: float,
    *,
    win: bool = False,
    tie: bool = False,
) -> dict[str, object]:
    return {
        "selector": SELECTOR_HYBRID_PRIMARY,
        "data_mode": DATA_MODE_HYBRID,
        "mc_path_id": "h5_b63_s0001",
        "horizon_years": 5,
        "block_length_days": 63,
        "sample_index": 1,
        "scenario_id": scenario_id,
        "scenario_label": scenario_id,
        "strategy_family": "test",
        "total_contributed": 70_000.0,
        "ending_equity": 100_000.0,
        "simple_cash_return": 100_000.0 / 70_000.0 - 1.0,
        "xirr": xirr,
        "max_drawdown": max_drawdown,
        "drawdown_breach": max_drawdown < -0.95,
        "benchmark_scenario_id": "hybrid_primary--dca-policy-constant_1p0",
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


def small_replay_config(selector: str = SELECTOR_SYNTHETIC_PRIMARY) -> MonthlyDecisionReplayConfig:
    return MonthlyDecisionReplayConfig(
        selector=selector,
        horizons_years=(2,),
        initial_cash=10_000.0,
        monthly_contribution=1_000.0,
        max_drawdown_limit=-0.95,
        min_win_rate=0.50,
        monte_carlo_enabled=True,
        monte_carlo_seed=7,
        monte_carlo_block_lengths_days=(21, 63),
        monte_carlo_fast_samples_per_scale=2,
        monte_carlo_full_samples_per_scale=3,
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


def sample_actual_with_late_listings() -> pd.DataFrame:
    prices = sample_long_prices()
    actual = prices.copy()
    actual.loc[actual.index < "2011-01-03", "QLD"] = pd.NA
    actual.loc[actual.index < "2012-01-03", "TQQQ"] = pd.NA
    return actual
