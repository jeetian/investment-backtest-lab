import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from investment_backtest_lab.costs import CostModel, USCostConfig
from investment_backtest_lab.dca_policy_optimizer import DATA_MODE_SYNTHETIC
from investment_backtest_lab.leveraged_etf_lab import ProductSpec
from investment_backtest_lab.models import (
    AssetSpec,
    AssetType,
    DataSource,
    DCAPolicyOptimizerConfig,
    Market,
    MonthlyDecisionReplayConfig,
)
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

_REPLAY_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "analyze_monthly_decision_replay.py"
)
_REPLAY_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "_test_analyze_monthly_decision_replay",
    _REPLAY_SCRIPT_PATH,
)
assert _REPLAY_SCRIPT_SPEC is not None
assert _REPLAY_SCRIPT_SPEC.loader is not None
_REPLAY_SCRIPT = importlib.util.module_from_spec(_REPLAY_SCRIPT_SPEC)
sys.modules[_REPLAY_SCRIPT_SPEC.name] = _REPLAY_SCRIPT
_REPLAY_SCRIPT_SPEC.loader.exec_module(_REPLAY_SCRIPT)
ReplayPriceLoadResult = _REPLAY_SCRIPT.ReplayPriceLoadResult
trim_price_result_to_external_coverage = (
    _REPLAY_SCRIPT.trim_price_result_to_external_coverage
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
    assert baseline["win_rate_vs_benchmark"] == pytest.approx(0.50)
    assert bool(breach["eligible_for_monthly_signal"]) is False
    assert breach["drawdown_breach_rate"] == pytest.approx(0.50)


def test_replay_accepts_base_dca_benchmark_alias():
    outputs = build_monthly_decision_replay_outputs(
        mode_prices={DATA_MODE_SYNTHETIC: sample_long_prices()},
        products=sample_products(),
        optimizer_config=small_optimizer_config(),
        replay_config=small_replay_config(benchmark="base_dca"),
        scan_mode="fast",
        selector=SELECTOR_SYNTHETIC_PRIMARY,
    )

    assert not outputs.ranking.empty
    assert "win_rate_vs_benchmark" in outputs.ranking.columns


def test_replay_accepts_fixed_1p5x_benchmark():
    optimizer_config = DCAPolicyOptimizerConfig(
        **{
            **small_optimizer_config().__dict__,
            "target_leverage_grid": (1.0, 1.5, 2.0),
        }
    )
    outputs = build_monthly_decision_replay_outputs(
        mode_prices={DATA_MODE_SYNTHETIC: sample_long_prices()},
        products=sample_products(),
        optimizer_config=optimizer_config,
        replay_config=small_replay_config(benchmark="fixed_1p5x_dca"),
        scan_mode="fast",
        selector=SELECTOR_SYNTHETIC_PRIMARY,
    )

    assert not outputs.ranking.empty
    baseline = outputs.ranking[
        outputs.ranking["scenario_id"].astype(str).str.endswith("constant_1p5")
    ].iloc[0]
    assert baseline["win_rate_vs_benchmark"] == pytest.approx(0.50)
    assert "win_rate_vs_fixed_1p5x_dca" in outputs.ranking.columns
    assert "win_rate_vs_0050_dca" in outputs.ranking.columns


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


def test_monthly_replay_ranking_includes_net_cost_summary():
    cost_model = CostModel(us=USCostConfig(slippage_bps=10.0))
    products = sample_products()
    outputs = build_monthly_decision_replay_outputs(
        mode_prices={DATA_MODE_SYNTHETIC: sample_long_prices()},
        products=products,
        optimizer_config=small_optimizer_config(),
        replay_config=small_replay_config(),
        scan_mode="fast",
        selector=SELECTOR_SYNTHETIC_PRIMARY,
        cost_model=cost_model,
        product_assets=us_product_assets(products),
    )

    assert set(outputs.ranking["cost_mode"]) == {"net_of_cost"}
    assert outputs.ranking["total_trade_cost"].max() > 0.0
    assert "cost_drag_on_contributed" in outputs.ranking.columns
    assert "cost_to_final_equity" in outputs.ranking.columns
    assert "trade_days" in outputs.ranking.columns
    assert not outputs.trade_audit.empty
    assert "max_single_day_trade_cost" in outputs.trade_audit.columns


def test_monthly_replay_can_append_optuna_shortlist_scenarios():
    products = sample_products()
    outputs = build_monthly_decision_replay_outputs(
        mode_prices={DATA_MODE_SYNTHETIC: sample_long_prices()},
        products=products,
        optimizer_config=small_optimizer_config(),
        replay_config=small_replay_config(),
        scan_mode="fast",
        selector=SELECTOR_SYNTHETIC_PRIMARY,
        cost_model=CostModel(us=USCostConfig(slippage_bps=1.0)),
        product_assets=us_product_assets(products),
        optuna_scenarios=pd.DataFrame(
            [
                {
                    "trial_number": 12076,
                    "scenario_name": "optuna_trial_12076",
                    "scenario_label": "Optuna V2 #12076",
                    "params_json": json.dumps(sample_optuna_params(), sort_keys=True),
                }
            ]
        ),
    )

    assert "synthetic_stress--dca-policy-optuna_trial_12076" in set(
        outputs.ranking["scenario_id"]
    )
    optuna_row = outputs.ranking[
        outputs.ranking["scenario_id"].astype(str).str.contains("optuna_trial_12076")
    ].iloc[0]
    assert optuna_row["strategy_family"] == "optuna_return_first"
    assert optuna_row["scenario_label"] == "Optuna V5 #12076"
    assert "cost_drag_on_contributed" in outputs.ranking.columns
    payload_keys = {scenario["key"] for scenario in outputs.compare_payload["scenarios"]}
    assert "synthetic_stress--dca-policy-optuna_trial_12076" in payload_keys
    assert "effective_leverage" in outputs.compare_payload["metrics"]
    assert "cumulative_trade_cost" in outputs.compare_payload["metrics"]
    assert "commission" in outputs.compare_payload["metrics"]
    assert "transaction_tax" in outputs.compare_payload["metrics"]
    assert "slippage" in outputs.compare_payload["metrics"]


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
    assert result.compare_payload_path.exists()
    assert result.trade_audit_path.exists()
    assert result.trade_audit_html_path.exists()
    assert result.mc_trials_path.suffixes[-2:] == [".csv", ".gz"]
    assert "expected_xirr" in result.mc_summary.columns
    payload = json.loads(result.compare_payload_path.read_text(encoding="utf-8"))
    assert payload["metrics"]["turnover"]["label"] == "Turnover"
    assert payload["metrics"]["commission"]["label"] == "Commission"
    html = result.html_path.read_text(encoding="utf-8")
    assert "Hybrid-Primary Monte Carlo Replay" in html
    assert "Trade Cost Audit" in html
    assert "MC summary CSV" in html
    assert "compressed MC trials CSV" in html
    assert "win rate" in html.lower()
    assert "drawdown breach rate" in html
    assert "Actual Primary Test" in html


def test_core_external_replay_trims_prices_to_common_feature_coverage():
    prices = sample_long_prices()
    feature_dates = prices.index[
        (prices.index >= "2011-07-22") & (prices.index <= "2012-01-31")
    ]
    external_signals = pd.DataFrame(
        {
            "vix_percentile_252": 0.5,
            "usdtwd_return_63d_percentile_252": 0.5,
            "tw_margin_balance_percentile_252": 0.5,
            "tw_institutional_net_buy_21d_percentile_252": 0.5,
        },
        index=feature_dates,
    )
    coverage = pd.DataFrame(
        [
            {
                "ticker": "QQQ",
                "replay_start": "1999-03-10",
                "replay_end": "2005-12-30",
                "source_notes": "",
            }
        ]
    )

    result = trim_price_result_to_external_coverage(
        price_result=ReplayPriceLoadResult(
            mode_prices={DATA_MODE_SYNTHETIC: prices},
            source_coverage=coverage,
        ),
        external_signals=external_signals,
    )

    trimmed = result.mode_prices[DATA_MODE_SYNTHETIC]
    assert trimmed.index.min() == feature_dates.min()
    assert trimmed.index.max() == feature_dates.max()
    row = result.source_coverage.iloc[0]
    assert row["replay_start"] == "2011-07-22"
    assert row["replay_end"] == "2012-01-31"
    assert "official_core_external_replay_window" in row["source_notes"]


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


def test_monte_carlo_ranking_keeps_high_xirr_breach_candidate_as_research_top():
    trials = pd.DataFrame(
        [
            mc_row("hybrid_primary--dca-policy-safe", 0.10, -0.40, win=True),
            mc_row("hybrid_primary--dca-policy-safe", 0.08, -0.35, win=True),
            mc_row("hybrid_primary--dca-policy-high-return", 0.40, -0.20, win=True),
            mc_row("hybrid_primary--dca-policy-high-return", 0.35, -0.96, win=True),
        ]
    )
    cohorts = pd.DataFrame(
        [
            cohort_row("hybrid_primary--dca-policy-safe", 0.09, -0.50, win=True),
            cohort_row("hybrid_primary--dca-policy-high-return", 0.30, -0.70, win=True),
        ]
    )

    ranking = build_monte_carlo_replay_ranking(
        trials,
        cohorts=cohorts,
        selector=SELECTOR_HYBRID_PRIMARY,
        config=MonthlyDecisionReplayConfig(),
    )

    top = ranking.iloc[0]
    assert top["scenario_id"] == "hybrid_primary--dca-policy-high-return"
    assert bool(top["eligible_for_monthly_signal"]) is True
    assert bool(top["manual_review_required"]) is True
    assert top["drawdown_breach_rate"] == pytest.approx(0.50)


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


def small_replay_config(
    selector: str = SELECTOR_SYNTHETIC_PRIMARY,
    benchmark: str = "qqq_dca",
) -> MonthlyDecisionReplayConfig:
    return MonthlyDecisionReplayConfig(
        selector=selector,
        benchmark=benchmark,
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


def us_product_assets(products: list[ProductSpec]) -> dict[str, AssetSpec]:
    return {
        product.ticker: AssetSpec(
            product.ticker,
            Market.US,
            AssetType.ETF,
            "USD",
            DataSource.YFINANCE,
        )
        for product in products
    }


def sample_optuna_params() -> dict[str, object]:
    return {
        "trend_window": 100,
        "slope_window": 21,
        "momentum_window": 63,
        "vol_window": 21,
        "vol_percentile_window": 252,
        "vol_target": 0.25,
        "risk_on_leverage": 2.0,
        "risk_off_leverage": 0.5,
        "cash_leverage": 0.0,
        "drawdown_guard": -0.2,
        "severe_drawdown_guard": -0.5,
        "vol_spike_quantile": 0.8,
        "rebalance_cadence": "monthly",
        "rebalance_threshold": 0.05,
        "min_holding_days": 21,
        "signal_hysteresis": 0.01,
        "max_annual_turnover": 20.0,
        "sentiment_mode": "off",
        "fear_threshold": 20.0,
        "greed_threshold": 80.0,
        "sentiment_deleverage": 0.0,
    }


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
