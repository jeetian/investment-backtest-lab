from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import investment_backtest_lab.strategy_search as strategy_search
from investment_backtest_lab.costs import CostModel
from investment_backtest_lab.leveraged_etf_lab import ProductSpec
from investment_backtest_lab.models import (
    AssetSpec,
    AssetType,
    BacktestConfig,
    DataSource,
    Market,
    MonthlyDecisionReplayConfig,
    StrategyConfig,
    StrategySearchConfig,
)
from investment_backtest_lab.strategy_search import (
    apply_external_signal_overlay,
    apply_trade_filters,
    build_candidate_triage,
    build_strategy_search_context,
    classify_best_candidates,
    external_signal_set,
    load_frozen_sentiment,
    run_optuna_strategy_search,
)


def test_frozen_sentiment_is_shifted_and_validated(tmp_path: Path):
    path = tmp_path / "fear_greed.csv"
    pd.DataFrame(
        {
            "date": ["2018-01-01", "2018-01-02", "2018-01-03"],
            "score": [10, 20, 30],
            "rating": ["fear", "neutral", "greed"],
        }
    ).to_csv(path, index=False)
    trading_index = pd.to_datetime(["2018-01-01", "2018-01-02", "2018-01-03"])

    sentiment = load_frozen_sentiment(path, trading_index=trading_index)

    assert pd.isna(sentiment.iloc[0])
    assert sentiment.iloc[1] == 10
    assert sentiment.iloc[2] == 20


def test_frozen_sentiment_missing_file_fails(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Missing frozen sentiment CSV"):
        load_frozen_sentiment(
            tmp_path / "missing.csv",
            trading_index=pd.date_range("2020-01-01", periods=3),
        )


def test_classify_best_candidates_blocks_higher_cost_without_risk_improvement():
    pareto = pd.DataFrame(
        [
            {
                "trial_number": 1,
                "cohort_gate_passed": True,
                "expected_xirr": 0.2,
                "p05_xirr": -0.04,
                "p05_max_drawdown": -0.75,
                "cost_drag_on_contributed": 0.50,
                "turnover_sum": 100.0,
            }
        ]
    )

    classified = classify_best_candidates(
        pareto,
        baseline={
            "p05_xirr": -0.03,
            "p05_max_drawdown": -0.70,
            "cost_drag_on_contributed": 0.30,
        },
    )

    assert classified["candidate_status"].iloc[0] == "watchlist"
    assert "Higher cost drag" in classified["candidate_reason"].iloc[0]


def test_classify_best_candidates_uses_triage_status_when_available():
    pareto = pd.DataFrame(
        [
            {
                "trial_number": 12641,
                "cohort_gate_passed": True,
                "expected_xirr": 0.20,
                "win_rate_vs_base_dca": 1.0,
                "p05_xirr": 0.10,
                "p05_max_drawdown": -0.50,
                "cost_drag_on_contributed": 0.01,
                "turnover_sum": 8.0,
            }
        ]
    )
    triage = pd.DataFrame(
        [
            {
                "trial_number": 12641,
                "triage_status": "rejected",
                "triage_reason": "Final holdout expected XIRR is negative.",
            }
        ]
    )

    classified = classify_best_candidates(
        pareto,
        baseline={},
        objective_profile="return_first",
        triage=triage,
    )

    assert classified["candidate_status"].iloc[0] == "rejected"
    assert "Final holdout" in classified["candidate_reason"].iloc[0]


def test_external_signal_overlay_can_deleverage_or_turn_off():
    target = pd.Series([1.5, 1.5, 1.5], index=pd.date_range("2020-01-01", periods=3))
    signals = pd.DataFrame(
        {
            "vix_percentile_252": [10.0, 90.0, 95.0],
            "fear_greed_score": [20.0, 50.0, 80.0],
        },
        index=target.index,
    )

    adjusted = apply_external_signal_overlay(
        target,
        external_signals=signals,
        params={
            "vix_percentile_252_mode": "risk_off",
            "vix_percentile_252_low_threshold": 20.0,
            "vix_percentile_252_high_threshold": 80.0,
            "vix_percentile_252_deleverage": 0.5,
            "fear_greed_score_mode": "off",
        },
    )

    assert adjusted.iloc[0] == 1.5
    assert adjusted.iloc[1] == 1.0
    assert adjusted.iloc[2] == 1.0


def test_optuna_search_writes_outputs_and_can_resume(tmp_path: Path):
    config = sample_config(tmp_path)
    prices = sample_prices()
    context = build_strategy_search_context(
        prices=prices,
        products=sample_products(),
        product_assets=sample_product_assets(),
        cost_model=CostModel.from_dict(config.cost_model),
        config=config,
        output_dir=tmp_path,
        include_sentiment=False,
    )
    storage = tmp_path / "study.db"

    first = run_optuna_strategy_search(
        context=context,
        study_name="tw50_test",
        trials=3,
        storage_path=storage,
    )
    second = run_optuna_strategy_search(
        context=context,
        study_name="tw50_test",
        trials=2,
        storage_path=storage,
    )

    assert storage.exists()
    assert first.trials_path.exists()
    assert second.trials_path.exists()
    assert len(second.trials) >= 5
    assert set(
        [
            "expected_xirr",
            "p05_xirr",
            "sharpe",
            "sortino",
            "drawdown_breach_rate",
            "cost_drag_on_contributed",
            "turnover_sum",
        ]
    ).issubset(second.trials.columns)
    assert "external_signal_set" in second.trials.columns
    assert "external_signal_modes_json" in second.trials.columns
    assert "win_rate_vs_0050_dca" in second.trials.columns
    assert "return_first_rank" in second.trials.columns
    assert second.candidate_triage_path.exists()
    assert "triage_status" in second.candidate_triage.columns
    assert second.study_status["completed_trials"] >= 5


def test_optuna_export_only_rebuilds_reports_without_adding_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = sample_config(tmp_path)
    context = build_strategy_search_context(
        prices=sample_prices(),
        products=sample_products(),
        product_assets=sample_product_assets(),
        cost_model=CostModel.from_dict(config.cost_model),
        config=config,
        output_dir=tmp_path,
        include_sentiment=False,
    )
    storage = tmp_path / "study.db"
    run_optuna_strategy_search(
        context=context,
        study_name="tw50_export_test",
        trials=3,
        storage_path=storage,
    )
    monkeypatch.setattr(
        strategy_search,
        "evaluate_strategy_params",
        lambda **_kwargs: pytest.fail("export-only should reuse cached holdout metrics"),
    )
    exported = run_optuna_strategy_search(
        context=context,
        study_name="tw50_export_test",
        trials=0,
        storage_path=storage,
        export_only=True,
    )

    assert exported.export_only is True
    assert exported.study_status["completed_trials"] == 3
    assert len(exported.trials) == 3
    assert exported.html_path.exists()
    assert exported.candidate_triage_path.exists()


def test_optuna_return_first_profile_writes_ranked_outputs(tmp_path: Path):
    config = sample_config(tmp_path)
    context = build_strategy_search_context(
        prices=sample_prices(),
        products=sample_products(),
        product_assets=sample_product_assets(),
        cost_model=CostModel.from_dict(config.cost_model),
        config=config,
        output_dir=tmp_path,
        include_sentiment=False,
    )

    result = run_optuna_strategy_search(
        context=context,
        study_name="tw50_return_first_test",
        trials=3,
        storage_path=tmp_path / "return_first.db",
        objective_profile="return_first",
        timeout_hours=1.0,
    )

    assert result.objective_profile == "return_first"
    assert "return_first_rank" in result.trials.columns
    assert "return_first_score" in result.candidate_triage.columns
    assert "win_rate_vs_0050_dca" in result.best_candidates.columns
    assert "Return Leaders" in result.html_path.read_text(encoding="utf-8")


def test_core_external_search_uses_complete_core_window_without_fear_greed(tmp_path: Path):
    feature_path = tmp_path / "market_regime_features_tw50.csv"
    dates = pd.bdate_range("2017-01-02", periods=760)
    pd.DataFrame(
        {
            "date": dates,
            "fear_greed_score": np.linspace(10.0, 90.0, len(dates)),
            "vix_percentile_252": np.linspace(20.0, 80.0, len(dates)),
            "usdtwd_return_63d_percentile_252": [
                np.nan if index < 40 else 50.0 for index in range(len(dates))
            ],
            "tw_margin_balance_percentile_252": np.linspace(30.0, 70.0, len(dates)),
            "tw_institutional_net_buy_21d_percentile_252": np.linspace(
                40.0,
                60.0,
                len(dates),
            ),
        }
    ).to_csv(feature_path, index=False)
    config = sample_config(tmp_path)
    config = BacktestConfig(
        **{
            **config.__dict__,
            "strategy_search": StrategySearchConfig(
                family="tw50",
                n_trials=1,
                include_sentiment=False,
                include_external_signals=True,
                external_signal_feature_set="core",
                external_signals_path=str(feature_path),
                storage_path=str(tmp_path / "study.db"),
                final_holdout_start="2019-01-01",
                final_holdout_end="2019-12-31",
            ),
        }
    )

    context = build_strategy_search_context(
        prices=sample_prices(),
        products=sample_products(),
        product_assets=sample_product_assets(),
        cost_model=CostModel.from_dict(config.cost_model),
        config=config,
        output_dir=tmp_path,
    )
    result = run_optuna_strategy_search(
        context=context,
        study_name="tw50_core_test",
        trials=1,
        storage_path=tmp_path / "core.db",
        objective_profile="return_first",
    )

    assert "fear_greed_score" not in context.external_signals.columns
    assert "fear_greed_score" not in external_signal_set(context)
    assert context.external_signal_common_start == context.prices.index.min().date().isoformat()
    assert "fear_greed_score" not in result.trials["params_json"].dropna().iloc[0]
    assert "fear_threshold" not in result.trials["params_json"].dropna().iloc[0]
    assert result.study_status["external_signal_feature_set"] == "core"
    assert "vix_percentile_252" in result.study_status["external_signal_set"]


def test_weekly_execution_search_forces_weekly_cadence(tmp_path: Path):
    config = sample_config(tmp_path)
    config = BacktestConfig(
        **{
            **config.__dict__,
            "strategy_search": StrategySearchConfig(
                family="tw50",
                n_trials=1,
                include_sentiment=False,
                storage_path=str(tmp_path / "weekly.db"),
                final_holdout_start="2019-01-01",
                final_holdout_end="2019-12-31",
                execution_profile="weekly_execution",
                execution_cadence="weekly",
                contribution_cadence="monthly",
                objective_profile="return_first",
            ),
        }
    )
    context = build_strategy_search_context(
        prices=sample_prices(),
        products=sample_products(),
        product_assets=sample_product_assets(),
        cost_model=CostModel.from_dict(config.cost_model),
        config=config,
        output_dir=tmp_path,
        include_sentiment=False,
    )

    result = run_optuna_strategy_search(
        context=context,
        study_name="tw50_weekly_test",
        trials=1,
        storage_path=tmp_path / "weekly.db",
        objective_profile="return_first",
    )

    params = result.trials["params_json"].dropna().map(json.loads).iloc[0]
    assert params["rebalance_cadence"] == "weekly"
    assert params["min_holding_days"] in {0, 5}
    assert result.study_status["execution_cadence"] == "weekly"
    assert result.study_status["contribution_cadence"] == "monthly"
    assert result.study_status["study_fingerprint_sha256"]


def test_monthly_core_weekly_delta_profile_samples_threshold(tmp_path: Path):
    config = sample_config(tmp_path)
    config = BacktestConfig(
        **{
            **config.__dict__,
            "monthly_decision_replay": MonthlyDecisionReplayConfig(
                family="tw50",
                benchmark="fixed_1p5x_dca",
                initial_cash=100_000,
                monthly_contribution=10_000,
                max_drawdown_limit=-0.85,
            ),
            "strategy_search": StrategySearchConfig(
                family="tw50",
                benchmark="fixed_1p5x_dca",
                n_trials=1,
                include_sentiment=False,
                storage_path=str(tmp_path / "v5.db"),
                final_holdout_start="2019-01-01",
                final_holdout_end="2019-12-31",
                execution_profile="monthly_core_weekly_delta",
                execution_cadence="monthly_core_weekly_delta",
                contribution_cadence="monthly",
                objective_profile="return_first",
            ),
        }
    )
    context = build_strategy_search_context(
        prices=sample_prices(),
        products=sample_products(),
        product_assets=sample_product_assets(),
        cost_model=CostModel.from_dict(config.cost_model),
        config=config,
        output_dir=tmp_path,
        include_sentiment=False,
    )

    result = run_optuna_strategy_search(
        context=context,
        study_name="tw50_v5_test",
        trials=1,
        storage_path=tmp_path / "v5.db",
        objective_profile="return_first",
    )

    params = result.trials["params_json"].dropna().map(json.loads).iloc[0]
    assert params["rebalance_cadence"] == "monthly_core_weekly_delta"
    assert 0.15 <= params["weekly_delta_threshold"] <= 0.60
    assert "win_rate_vs_fixed_1p5x_dca" in result.trials.columns
    assert result.study_status["execution_cadence"] == "monthly_core_weekly_delta"


def test_monthly_core_weekly_delta_filter_trades_monthly_and_large_weekly_moves():
    index = pd.bdate_range("2026-01-02", periods=11)
    target = pd.Series(
        [1.0, 1.05, 1.10, 1.52, 1.55, 1.56, 1.70, 1.18, 1.70, 1.72, 1.30],
        index=index,
    )
    filtered = apply_trade_filters(
        target,
        params={
            "rebalance_cadence": "monthly_core_weekly_delta",
            "rebalance_threshold": 0.0,
            "weekly_delta_threshold": 0.40,
            "min_holding_days": 0,
        },
    )

    assert filtered.iloc[0] == pytest.approx(1.0)
    assert filtered.loc[pd.Timestamp("2026-01-07")] == pytest.approx(1.0)
    assert filtered.loc[pd.Timestamp("2026-01-12")] == pytest.approx(1.70)


def test_study_fingerprint_blocks_appending_different_execution_profile(tmp_path: Path):
    config = sample_config(tmp_path)
    first_context = build_strategy_search_context(
        prices=sample_prices(),
        products=sample_products(),
        product_assets=sample_product_assets(),
        cost_model=CostModel.from_dict(config.cost_model),
        config=config,
        output_dir=tmp_path,
        include_sentiment=False,
    )
    storage = tmp_path / "fingerprint.db"
    run_optuna_strategy_search(
        context=first_context,
        study_name="tw50_fingerprint_test",
        trials=1,
        storage_path=storage,
    )
    weekly_config = BacktestConfig(
        **{
            **config.__dict__,
            "strategy_search": StrategySearchConfig(
                family="tw50",
                n_trials=1,
                include_sentiment=False,
                storage_path=str(storage),
                final_holdout_start="2019-01-01",
                final_holdout_end="2019-12-31",
                execution_profile="weekly_execution",
                execution_cadence="weekly",
            ),
        }
    )
    weekly_context = build_strategy_search_context(
        prices=sample_prices(),
        products=sample_products(),
        product_assets=sample_product_assets(),
        cost_model=CostModel.from_dict(weekly_config.cost_model),
        config=weekly_config,
        output_dir=tmp_path,
        include_sentiment=False,
    )

    with pytest.raises(ValueError, match="fingerprint"):
        run_optuna_strategy_search(
            context=weekly_context,
            study_name="tw50_fingerprint_test",
            trials=1,
            storage_path=storage,
        )


def test_candidate_triage_classifies_candidates_watchlist_and_rejections():
    trials = pd.DataFrame(
        [
            {
                "trial_number": 1,
                "state": "COMPLETE",
                "cohort_gate_passed": True,
                "drawdown_breach_rate": 0.0,
                "expected_xirr": 0.10,
                "p05_xirr": 0.03,
                "p05_max_drawdown": -0.30,
                "cost_drag_on_contributed": 0.01,
                "turnover_sum": 5.0,
                "holdout_expected_xirr": 0.04,
            },
            {
                "trial_number": 2,
                "state": "COMPLETE",
                "cohort_gate_passed": True,
                "drawdown_breach_rate": 0.10,
                "expected_xirr": 0.20,
                "p05_xirr": 0.04,
                "p05_max_drawdown": -0.20,
                "cost_drag_on_contributed": 0.01,
                "turnover_sum": 5.0,
                "holdout_expected_xirr": 0.04,
            },
            {
                "trial_number": 3,
                "state": "COMPLETE",
                "cohort_gate_passed": True,
                "drawdown_breach_rate": 0.0,
                "expected_xirr": 0.09,
                "p05_xirr": 0.025,
                "p05_max_drawdown": -0.34,
                "cost_drag_on_contributed": 0.10,
                "turnover_sum": 5.0,
                "holdout_expected_xirr": 0.04,
            },
        ]
    )

    triage = build_candidate_triage(
        trials,
        baseline={
            "p05_xirr": 0.02,
            "p05_max_drawdown": -0.35,
            "cost_drag_on_contributed": 0.02,
            "turnover_sum": 5.0,
        },
    )

    statuses = dict(zip(triage["trial_number"], triage["triage_status"], strict=False))
    assert statuses[1] == "candidate"
    assert statuses[2] == "rejected"
    assert statuses[3] == "watchlist"


def test_return_first_triage_prioritizes_xirr_then_win_rate():
    trials = pd.DataFrame(
        [
            {
                "trial_number": 1,
                "state": "COMPLETE",
                "cohort_gate_passed": True,
                "drawdown_breach_rate": 0.0,
                "expected_xirr": 0.20,
                "win_rate_vs_base_dca": 0.50,
                "p05_xirr": -0.08,
                "p05_max_drawdown": -0.70,
                "cost_drag_on_contributed": 0.01,
                "turnover_sum": 5.0,
                "holdout_expected_xirr": 0.04,
            },
            {
                "trial_number": 2,
                "state": "COMPLETE",
                "cohort_gate_passed": True,
                "drawdown_breach_rate": 0.0,
                "expected_xirr": 0.18,
                "win_rate_vs_base_dca": 1.00,
                "p05_xirr": 0.08,
                "p05_max_drawdown": -0.50,
                "cost_drag_on_contributed": 0.01,
                "turnover_sum": 5.0,
                "holdout_expected_xirr": 0.04,
            },
        ]
    )

    triage = build_candidate_triage(
        trials,
        baseline={
            "expected_xirr": 0.12,
            "win_rate_vs_base_dca": 0.70,
            "p05_xirr": -0.05,
            "p05_max_drawdown": -0.75,
            "cost_drag_on_contributed": 0.02,
            "turnover_sum": 5.0,
        },
        objective_profile="return_first",
    )

    assert triage["trial_number"].tolist() == [1, 2]
    assert triage["return_first_rank"].iloc[0] == 1
    assert "win_rate_vs_0050_dca" in triage.columns
    assert "return_first_score" in triage.columns
    assert set(triage["triage_status"]) == {"candidate"}


def test_return_first_triage_marks_high_cost_as_watchlist():
    trials = pd.DataFrame(
        [
            {
                "trial_number": 1,
                "state": "COMPLETE",
                "cohort_gate_passed": True,
                "drawdown_breach_rate": 0.0,
                "expected_xirr": 0.30,
                "win_rate_vs_base_dca": 1.00,
                "p05_xirr": 0.08,
                "p05_max_drawdown": -0.50,
                "cost_drag_on_contributed": 0.05,
                "turnover_sum": 5.0,
                "holdout_expected_xirr": 0.04,
            }
        ]
    )

    triage = build_candidate_triage(
        trials,
        baseline={
            "expected_xirr": 0.12,
            "win_rate_vs_base_dca": 0.70,
            "p05_xirr": -0.05,
            "p05_max_drawdown": -0.75,
            "cost_drag_on_contributed": 0.02,
            "turnover_sum": 5.0,
        },
        objective_profile="return_first",
    )

    assert triage["triage_status"].iloc[0] == "watchlist"
    assert "Higher cost drag" in triage["triage_reason"].iloc[0]


def sample_prices() -> pd.DataFrame:
    index = pd.bdate_range("2017-01-02", periods=760)
    returns = np.full(len(index), 0.0005)
    returns[120:150] = -0.01
    base = 100.0 * pd.Series(1.0 + returns, index=index).cumprod()
    leveraged = 100.0 * pd.Series(1.0 + 2.0 * returns, index=index).clip(lower=0.01).cumprod()
    return pd.DataFrame({"0050": base, "00631L": leveraged}, index=index)


def sample_config(tmp_path: Path) -> BacktestConfig:
    return BacktestConfig(
        universe=list(sample_product_assets().values()),
        start_date=date(2017, 1, 1),
        end_date=date(2019, 12, 31),
        base_currency="TWD",
        strategy=StrategyConfig("test"),
        cost_model={
            "tw": {
                "commission_rate": 0.001425,
                "commission_discount": 0.28,
                "min_commission": 1.0,
                "etf_transaction_tax_rate": 0.001,
                "slippage_bps": 1.0,
            }
        },
        monthly_decision_replay=MonthlyDecisionReplayConfig(
            family="tw50",
            benchmark="base_dca",
            initial_cash=100_000,
            monthly_contribution=10_000,
            max_drawdown_limit=-0.85,
        ),
        strategy_search=StrategySearchConfig(
            family="tw50",
            n_trials=5,
            include_sentiment=False,
            storage_path=str(tmp_path / "study.db"),
            final_holdout_start="2019-01-01",
            final_holdout_end="2019-12-31",
        ),
    )


def sample_products() -> list[ProductSpec]:
    return [
        ProductSpec("0050", 1.0, "0050 1x"),
        ProductSpec("00631L", 2.0, "00631L 2x"),
    ]


def sample_product_assets() -> dict[str, AssetSpec]:
    return {
        "0050": AssetSpec("0050", Market.TW, AssetType.ETF, "TWD", DataSource.FINMIND),
        "00631L": AssetSpec("00631L", Market.TW, AssetType.ETF, "TWD", DataSource.FINMIND),
    }
