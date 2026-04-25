import pandas as pd
import pytest

from investment_backtest_lab.costs import CostModel
from investment_backtest_lab.ledger import AccountLedger
from investment_backtest_lab.models import AssetSpec, AssetType, DataSource, Market


def spy_asset() -> AssetSpec:
    return AssetSpec("SPY", Market.US, AssetType.ETF, "USD", DataSource.YFINANCE)


def zero_cost_model() -> CostModel:
    return CostModel.from_dict(
        {
            "us": {
                "commission_per_share": 0.0,
                "min_commission": 0.0,
                "sec_fee_rate": 0.0,
                "finra_taf_per_share": 0.0,
                "finra_taf_cap": 0.0,
                "slippage_bps": 0.0,
            }
        }
    )


def test_buy_and_snapshot_equity_are_auditable():
    ledger = AccountLedger(spy_asset(), starting_cash=1_000, cost_model=zero_cost_model())

    ledger.buy("2024-01-02", quantity=10, price=10)
    snapshot = ledger.snapshot("2024-01-03", price=12)

    assert ledger.cash == pytest.approx(900)
    assert ledger.positions["SPY"] == pytest.approx(10)
    assert snapshot.market_value == pytest.approx(120)
    assert snapshot.total_equity == pytest.approx(1_020)
    assert snapshot.total_equity == pytest.approx(snapshot.cash + snapshot.market_value)


def test_sell_updates_cash_position_and_fees():
    cost_model = CostModel.from_dict(
        {
            "us": {
                "commission_per_share": 1.0,
                "min_commission": 0.0,
                "sec_fee_rate": 0.0,
                "finra_taf_per_share": 0.0,
                "finra_taf_cap": 0.0,
                "slippage_bps": 0.0,
            }
        }
    )
    ledger = AccountLedger(spy_asset(), starting_cash=1_000, cost_model=cost_model)

    ledger.buy("2024-01-02", quantity=10, price=10)
    ledger.sell("2024-01-03", quantity=5, price=12)

    assert ledger.cash == pytest.approx(945)
    assert ledger.positions["SPY"] == pytest.approx(5)
    assert ledger.total_fees_paid == pytest.approx(15)
    assert ledger.trades.iloc[-1]["net_cash_flow"] == pytest.approx(55)


def test_cash_dividend_records_gross_withholding_and_net_cash():
    ledger = AccountLedger(spy_asset(), starting_cash=1_000, cost_model=zero_cost_model())
    ledger.buy("2024-01-02", quantity=10, price=10)

    dividend = ledger.cash_dividend(
        "2024-03-31",
        dividend_per_share=1.0,
        withholding_rate=0.30,
    )

    assert dividend.gross_amount == pytest.approx(10)
    assert dividend.withholding_tax == pytest.approx(3)
    assert dividend.net_amount == pytest.approx(7)
    assert dividend.cash_amount == pytest.approx(7)
    assert ledger.cash == pytest.approx(907)
    assert ledger.total_taxes_paid == pytest.approx(3)


def test_dividend_reinvestment_increases_shares_without_leverage():
    ledger = AccountLedger(spy_asset(), starting_cash=1_000, cost_model=zero_cost_model())
    ledger.buy("2024-01-02", quantity=10, price=10)

    dividend = ledger.cash_dividend(
        "2024-03-31",
        dividend_per_share=1.0,
        withholding_rate=0.30,
        reinvest=True,
        price=7,
    )

    assert dividend.net_amount == pytest.approx(7)
    assert dividend.reinvested_quantity == pytest.approx(1)
    assert dividend.cash_amount == pytest.approx(0)
    assert ledger.positions["SPY"] == pytest.approx(11)
    assert ledger.cash == pytest.approx(900)
    assert ledger.trades.iloc[-1]["note"] == "dividend reinvestment"


def test_buy_with_cash_uses_only_affordable_cash_budget():
    ledger = AccountLedger(spy_asset(), starting_cash=1_000, cost_model=zero_cost_model())

    trade = ledger.buy_with_cash("2024-01-02", cash_amount=250, price=100)

    assert trade is not None
    assert trade.quantity == pytest.approx(2.5)
    assert ledger.cash == pytest.approx(750)
    assert ledger.positions["SPY"] == pytest.approx(2.5)


def test_deposit_records_external_cash_flow():
    ledger = AccountLedger(spy_asset(), starting_cash=0, cost_model=zero_cost_model())

    event = ledger.deposit("2024-01-02", amount=1_000, note="DCA contribution")

    assert event.amount == pytest.approx(1_000)
    assert ledger.cash == pytest.approx(1_000)
    assert ledger.total_cash_deposited == pytest.approx(1_000)
    assert ledger.cash_flows.iloc[0]["kind"] == "deposit"


def test_build_equity_curve_snapshots_each_price_date():
    ledger = AccountLedger(spy_asset(), starting_cash=1_000, cost_model=zero_cost_model())
    ledger.buy("2024-01-02", quantity=10, price=10)
    prices = pd.Series([10.0, 11.0, 12.0], index=pd.bdate_range("2024-01-02", periods=3))

    equity = ledger.build_equity_curve(prices)

    assert len(equity) == 3
    assert equity.iloc[-1]["total_equity"] == pytest.approx(1_020)
    assert equity.iloc[-1]["total_equity"] == pytest.approx(
        equity.iloc[-1]["cash"] + equity.iloc[-1]["market_value"]
    )
