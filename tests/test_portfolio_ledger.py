import pytest

from investment_backtest_lab.costs import CostModel
from investment_backtest_lab.models import AssetSpec, AssetType, DataSource, Market
from investment_backtest_lab.portfolio_ledger import PortfolioLedger


def spy_asset() -> AssetSpec:
    return AssetSpec("SPY", Market.US, AssetType.ETF, "USD", DataSource.YFINANCE)


def qqq_asset() -> AssetSpec:
    return AssetSpec("QQQ", Market.US, AssetType.ETF, "USD", DataSource.YFINANCE)


def tw_asset() -> AssetSpec:
    return AssetSpec("0050", Market.TW, AssetType.ETF, "TWD", DataSource.FINMIND)


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


def test_initial_rebalance_allocates_shared_cash_to_target_weights():
    ledger = PortfolioLedger(
        [spy_asset(), qqq_asset()],
        starting_cash=10_000,
        cost_model=zero_cost_model(),
    )

    ledger.rebalance_to_weights(
        "2024-01-02",
        prices={"SPY": 100, "QQQ": 100},
        target_weights={"SPY": 0.6, "QQQ": 0.4},
    )
    snapshot = ledger.snapshot("2024-01-02", prices={"SPY": 100, "QQQ": 100})

    assert ledger.positions["SPY"] == pytest.approx(60)
    assert ledger.positions["QQQ"] == pytest.approx(40)
    assert ledger.cash == pytest.approx(0)
    assert snapshot.total_equity == pytest.approx(10_000)
    weights = ledger.positions_history.set_index("asset")["weight"]
    assert weights["SPY"] == pytest.approx(0.6)
    assert weights["QQQ"] == pytest.approx(0.4)


def test_second_rebalance_sells_overweight_asset_and_buys_underweight_asset():
    ledger = PortfolioLedger(
        [spy_asset(), qqq_asset()],
        starting_cash=10_000,
        cost_model=zero_cost_model(),
    )
    ledger.rebalance_to_weights(
        "2024-01-02",
        prices={"SPY": 100, "QQQ": 100},
        target_weights={"SPY": 0.6, "QQQ": 0.4},
    )

    trades = ledger.rebalance_to_weights(
        "2024-02-01",
        prices={"SPY": 200, "QQQ": 100},
        target_weights={"SPY": 0.6, "QQQ": 0.4},
    )
    snapshot = ledger.snapshot("2024-02-01", prices={"SPY": 200, "QQQ": 100})

    assert [trade.side.value for trade in trades] == ["sell", "buy"]
    assert "action=reduce overweight" in trades[0].note
    assert "drift=+0.1500" in trades[0].note
    assert "action=increase underweight" in trades[1].note
    assert "drift=-0.1500" in trades[1].note
    assert ledger.positions["SPY"] == pytest.approx(48)
    assert ledger.positions["QQQ"] == pytest.approx(64)
    assert ledger.cash == pytest.approx(0)
    assert snapshot.total_equity == pytest.approx(16_000)
    weights = ledger.positions_history[ledger.positions_history["date"] == "2024-02-01"]
    weights = weights.set_index("asset")["weight"]
    assert weights["SPY"] == pytest.approx(0.6)
    assert weights["QQQ"] == pytest.approx(0.4)


def test_rebalance_with_fees_never_overbuys_available_cash():
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
    ledger = PortfolioLedger(
        [spy_asset(), qqq_asset()],
        starting_cash=10_000,
        cost_model=cost_model,
    )

    ledger.rebalance_to_weights(
        "2024-01-02",
        prices={"SPY": 100, "QQQ": 100},
        target_weights={"SPY": 0.6, "QQQ": 0.4},
    )
    snapshot = ledger.snapshot("2024-01-02", prices={"SPY": 100, "QQQ": 100})

    assert ledger.cash >= -1e-9
    assert ledger.total_fees_paid > 0
    assert snapshot.total_equity == pytest.approx(
        ledger.cash + ledger.positions["SPY"] * 100 + ledger.positions["QQQ"] * 100
    )


def test_cash_dividend_stays_cash_until_next_rebalance_allocates_it():
    ledger = PortfolioLedger(
        [spy_asset(), qqq_asset()],
        starting_cash=10_000,
        cost_model=zero_cost_model(),
    )
    ledger.rebalance_to_weights(
        "2024-01-02",
        prices={"SPY": 100, "QQQ": 100},
        target_weights={"SPY": 0.6, "QQQ": 0.4},
    )

    dividend = ledger.cash_dividend(
        "SPY",
        "2024-01-15",
        dividend_per_share=1.0,
        withholding_rate=0.30,
    )
    assert dividend.gross_amount == pytest.approx(60)
    assert dividend.net_amount == pytest.approx(42)
    assert ledger.cash == pytest.approx(42)

    ledger.rebalance_to_weights(
        "2024-02-01",
        prices={"SPY": 100, "QQQ": 100},
        target_weights={"SPY": 0.6, "QQQ": 0.4},
    )
    assert ledger.cash == pytest.approx(0)
    assert ledger.positions["SPY"] > 60
    assert ledger.positions["QQQ"] > 40


def test_reinvested_dividend_buys_same_asset_before_rebalance():
    ledger = PortfolioLedger(
        [spy_asset(), qqq_asset()],
        starting_cash=10_000,
        cost_model=zero_cost_model(),
    )
    ledger.rebalance_to_weights(
        "2024-01-02",
        prices={"SPY": 100, "QQQ": 100},
        target_weights={"SPY": 0.6, "QQQ": 0.4},
    )

    dividend = ledger.cash_dividend(
        "SPY",
        "2024-01-15",
        dividend_per_share=1.0,
        withholding_rate=0.30,
        reinvest=True,
        price=100,
    )

    assert dividend.reinvested_quantity == pytest.approx(0.42)
    assert ledger.positions["SPY"] == pytest.approx(60.42)
    assert ledger.cash == pytest.approx(0)


def test_target_weights_must_sum_to_one():
    ledger = PortfolioLedger(
        [spy_asset(), qqq_asset()],
        starting_cash=10_000,
        cost_model=zero_cost_model(),
    )

    with pytest.raises(ValueError, match="sum to 1.0"):
        ledger.rebalance_to_weights(
            "2024-01-02",
            prices={"SPY": 100, "QQQ": 100},
            target_weights={"SPY": 0.6, "QQQ": 0.3},
        )


def test_portfolio_ledger_rejects_non_usd_us_assets():
    with pytest.raises(ValueError, match="USD-denominated US assets"):
        PortfolioLedger([spy_asset(), tw_asset()], starting_cash=10_000)
