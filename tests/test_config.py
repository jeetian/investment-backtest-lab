from investment_backtest_lab.config import load_backtest_config
from investment_backtest_lab.models import DataSource, DividendMode, LeverageKind, Market


def test_load_mvp_config():
    config = load_backtest_config("configs/mvp_example.yaml")

    assert config.base_currency == "TWD"
    assert config.end_date.isoformat() == "2026-05-01"
    assert config.universe[0].market == Market.US
    assert config.universe[4].data_source == DataSource.FINMIND
    assert config.rebalance.target_weights["SPY"] == 0.60
    assert config.rebalance.target_weights["QQQ"] == 0.40
    assert config.dca.contribution == 1_000
    assert config.tax.us.dividend_withholding_rate == 0.30
    assert config.dividend.mode == DividendMode.CASH
    assert config.ledger.base_currency == "TWD"
    assert config.ledger.account_currency == "USD"
    assert config.ledger.initial_cash == 10_000
    assert config.leverage.enabled is True
    assert config.leverage.kind == LeverageKind.MARGIN_LOAN
    assert config.leverage.target_leverage == 1.30
    assert config.leverage.maintenance_requirement == 0.35
    assert config.dynamic_leverage.enabled is True
    assert config.dynamic_leverage.trend_window == 200
    assert config.dynamic_leverage.risk_off_leverage == 1.00
    assert config.leveraged_etf_lab.family == "qqq"
    assert config.leveraged_etf_lab.initial_cash == 10_000
    assert config.leveraged_etf_lab.dca_initial_cash == 10_000
    assert config.leveraged_etf_lab.dca_contribution == 1_000
    assert config.leveraged_etf_lab.cash_flow_mode == "both"
    assert config.leveraged_etf_lab.robust_ranking_enabled is True
    assert config.leveraged_etf_lab.fast_grid_step == 0.25
    assert config.leveraged_etf_lab.full_grid_step == 0.10
    assert config.leveraged_etf_lab.fast_top_n == 12
    assert config.leveraged_etf_lab.products["TQQQ"].leverage == 3.0
    assert config.leveraged_etf_lab.synthetic_failure_drawdown == -0.85
    assert config.monthly_decision_replay.enabled is True
    assert config.monthly_decision_replay.family == "qqq"
    assert config.monthly_decision_replay.selector == "hybrid_primary"
    assert config.monthly_decision_replay.benchmark == "qqq_dca"
    assert config.monthly_decision_replay.horizons_years == (5, 10, 15, 20)
    assert config.monthly_decision_replay.initial_cash == 10_000
    assert config.monthly_decision_replay.monthly_contribution == 1_000
    assert config.monthly_decision_replay.max_drawdown_limit == -0.95
    assert config.monthly_decision_replay.min_win_rate == 0.50
    assert config.monthly_decision_replay.monte_carlo_enabled is True
    assert config.monthly_decision_replay.monte_carlo_seed == 20260504
    assert config.monthly_decision_replay.monte_carlo_block_lengths_days == (63, 252, 504)
    assert config.monthly_decision_replay.monte_carlo_fast_samples_per_scale == 100
    assert config.monthly_decision_replay.monte_carlo_full_samples_per_scale == 500
    assert config.strategy_search.family == "tw50"
    assert config.strategy_search.engine == "optuna"
    assert config.strategy_search.drawdown_limit_multiplier == 1.20
    assert config.strategy_search.sentiment_path == "data/external/fear_greed.csv"
    assert config.strategy_search.external_signal_feature_set == "all"
    assert config.strategy_search.external_signals_path.endswith("market_regime_features_tw50.csv")


def test_load_tw50_config_uses_core_external_signals():
    config = load_backtest_config("configs/tw50_example.yaml")

    assert config.strategy_search.family == "tw50"
    assert config.strategy_search.include_external_signals is True
    assert config.strategy_search.include_sentiment is False
    assert config.strategy_search.external_signal_feature_set == "core"
    assert config.monthly_decision_replay.benchmark == "fixed_1p5x_dca"
    assert config.strategy_search.benchmark == "fixed_1p5x_dca"
    assert config.strategy_search.execution_cadence == "monthly_core_weekly_delta"
