from investment_backtest_lab.config import load_backtest_config
from investment_backtest_lab.models import DataSource, DividendMode, LeverageKind, Market


def test_load_mvp_config():
    config = load_backtest_config("configs/mvp_example.yaml")

    assert config.base_currency == "TWD"
    assert config.universe[0].market == Market.US
    assert config.universe[2].data_source == DataSource.FINMIND
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
