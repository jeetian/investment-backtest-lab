import pandas as pd
import pytest

from investment_backtest_lab.costs import CostModel
from investment_backtest_lab.leverage import MarginLoanLedger, PortfolioMarginLedger
from investment_backtest_lab.leverage_reports import (
    run_buy_hold_leveraged,
    run_dca_leveraged,
    run_rebalance_leveraged,
)
from investment_backtest_lab.models import (
    AssetSpec,
    AssetType,
    DataSource,
    DividendFrame,
    DividendMode,
    LeverageConfig,
    Market,
    PriceFrame,
)


def spy_asset() -> AssetSpec:
    return AssetSpec("SPY", Market.US, AssetType.ETF, "USD", DataSource.YFINANCE)


def qqq_asset() -> AssetSpec:
    return AssetSpec("QQQ", Market.US, AssetType.ETF, "USD", DataSource.YFINANCE)


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


def leverage_config(**overrides) -> LeverageConfig:
    values = {
        "enabled": True,
        "target_leverage": 1.3,
        "max_leverage": 1.3,
        "annual_borrow_rate": 0.0,
        "maintenance_requirement": 0.35,
        "min_safety_buffer": 0.25,
        "deleverage_to": 1.1,
    }
    values.update(overrides)
    return LeverageConfig(**values)


def flat_price_frame() -> PriceFrame:
    index = pd.bdate_range("2024-01-02", periods=5)
    data = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1_000},
        index=index,
    )
    return PriceFrame(asset=spy_asset(), data=data, adjusted=False, source="raw-test")


def three_month_price_frame() -> PriceFrame:
    index = pd.bdate_range("2024-01-02", "2024-03-29")
    data = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1_000},
        index=index,
    )
    return PriceFrame(asset=spy_asset(), data=data, adjusted=False, source="raw-test")


def three_month_qqq_price_frame() -> PriceFrame:
    index = pd.bdate_range("2024-01-02", "2024-03-29")
    data = pd.DataFrame(
        {"open": 50.0, "high": 50.0, "low": 50.0, "close": 50.0, "volume": 1_000},
        index=index,
    )
    return PriceFrame(asset=qqq_asset(), data=data, adjusted=False, source="raw-test")


def empty_dividend_frame(asset: AssetSpec | None = None) -> DividendFrame:
    asset = spy_asset() if asset is None else asset
    data = pd.DataFrame(
        {"dividend_per_share": pd.Series(dtype="float64")},
        index=pd.DatetimeIndex([], name="date"),
    )
    return DividendFrame(asset=asset, data=data, currency="USD", source="empty-test")


def single_dividend_frame(asset: AssetSpec | None = None) -> DividendFrame:
    asset = spy_asset() if asset is None else asset
    data = pd.DataFrame(
        {"dividend_per_share": [1.0]},
        index=pd.DatetimeIndex(["2024-01-03"], name="date"),
    )
    return DividendFrame(asset=asset, data=data, currency="USD", source="dividend-test")


def initialized_ledger(**config_overrides) -> MarginLoanLedger:
    ledger = MarginLoanLedger(
        spy_asset(),
        starting_cash=1_000,
        leverage=leverage_config(**config_overrides),
        cost_model=zero_cost_model(),
    )
    ledger.initialize_to_target_leverage("2024-01-02", price=100)
    return ledger


def test_fixed_price_initial_leverage_buy_is_auditable():
    ledger = initialized_ledger()
    snapshot = ledger.snapshot("2024-01-02", price=100)

    assert ledger.quantity == pytest.approx(13)
    assert ledger.cash == pytest.approx(0)
    assert ledger.debt == pytest.approx(300)
    assert snapshot.market_value == pytest.approx(1_300)
    assert snapshot.total_equity == pytest.approx(1_000)
    assert snapshot.actual_leverage == pytest.approx(1.3)
    assert snapshot.equity_ratio == pytest.approx(1_000 / 1_300)


def test_daily_margin_interest_lowers_equity():
    ledger = initialized_ledger(annual_borrow_rate=0.365)

    interest = ledger.accrue_interest("2024-01-03", days=1)
    snapshot = ledger.snapshot("2024-01-03", price=100)

    assert interest == pytest.approx(0.3)
    assert ledger.debt == pytest.approx(300.3)
    assert snapshot.total_equity == pytest.approx(999.7)


def test_margin_cash_dividend_adds_after_tax_cash_and_equity():
    ledger = initialized_ledger()

    dividend = ledger.cash_dividend(
        "2024-01-03",
        dividend_per_share=1.0,
        withholding_rate=0.30,
        reinvest=False,
    )
    snapshot = ledger.snapshot("2024-01-03", price=100)

    assert dividend is not None
    assert dividend.gross_amount == pytest.approx(13)
    assert dividend.withholding_tax == pytest.approx(3.9)
    assert dividend.net_amount == pytest.approx(9.1)
    assert ledger.cash == pytest.approx(9.1)
    assert snapshot.total_equity == pytest.approx(1_009.1)


def test_margin_reinvest_dividend_increases_shares_without_new_debt():
    ledger = initialized_ledger()

    dividend = ledger.cash_dividend(
        "2024-01-03",
        dividend_per_share=1.0,
        withholding_rate=0.30,
        reinvest=True,
        price=100,
    )
    snapshot = ledger.snapshot("2024-01-03", price=100)

    assert dividend is not None
    assert dividend.reinvested_quantity == pytest.approx(0.091)
    assert ledger.quantity == pytest.approx(13.091)
    assert ledger.debt == pytest.approx(300)
    assert ledger.cash == pytest.approx(0)
    assert snapshot.total_equity == pytest.approx(1_009.1)


def test_price_down_with_enough_safety_buffer_does_not_deleverage():
    ledger = initialized_ledger()

    ledger.check_margin_risk("2024-01-03", price=80)

    assert len(ledger.trades) == 1
    assert ledger.leverage_events.iloc[-1]["event_type"] == "target_leverage_rebalance"


def test_low_safety_buffer_triggers_auto_deleverage_to_safe_target():
    ledger = initialized_ledger()

    ledger.check_margin_risk("2024-01-03", price=50)
    snapshot = ledger.snapshot("2024-01-03", price=50)

    assert snapshot.actual_leverage == pytest.approx(1.1)
    assert ledger.debt == pytest.approx(35)
    assert ledger.quantity == pytest.approx(7.7)
    assert "safety_deleverage" in set(ledger.leverage_events["event_type"])
    assert len(ledger.forced_deleveraging_trades) == 0


def test_margin_call_records_forced_deleverage():
    ledger = initialized_ledger()

    ledger.check_margin_risk("2024-01-03", price=20)
    snapshot = ledger.snapshot("2024-01-03", price=20)

    assert "margin_call" in set(ledger.leverage_events["event_type"])
    assert "forced_deleverage" in set(ledger.leverage_events["event_type"])
    assert len(ledger.forced_deleveraging_trades) == 1
    assert ledger.quantity == pytest.approx(0)
    assert snapshot.total_equity == pytest.approx(-40)


def test_snapshot_equity_identity_holds_every_day():
    result = run_buy_hold_leveraged(
        price_frame=flat_price_frame(),
        dividend_frame=empty_dividend_frame(),
        cost_model=zero_cost_model(),
        initial_cash=1_000,
        leverage=leverage_config(),
        dividend_mode=DividendMode.CASH,
        withholding_rate=0.30,
    )

    curve = result.equity_curve
    identity = curve["cash"] + curve["market_value"] - curve["debt"]
    assert identity.to_numpy() == pytest.approx(curve["total_equity"].to_numpy())


def test_one_times_leverage_matches_unleveraged_price_path():
    result = run_buy_hold_leveraged(
        price_frame=flat_price_frame(),
        dividend_frame=empty_dividend_frame(),
        cost_model=zero_cost_model(),
        initial_cash=1_000,
        leverage=leverage_config(
            target_leverage=1.0,
            max_leverage=1.0,
            deleverage_to=1.0,
        ),
        dividend_mode=DividendMode.CASH,
        withholding_rate=0.30,
    )

    assert result.ledger.debt == pytest.approx(0)
    assert result.ledger.quantity == pytest.approx(10)
    assert result.equity_curve.iloc[-1]["total_equity"] == pytest.approx(1_000)


def test_dca_leveraged_records_contributions_and_target_debt():
    result = run_dca_leveraged(
        price_frame=three_month_price_frame(),
        dividend_frame=empty_dividend_frame(),
        cost_model=zero_cost_model(),
        contribution=1_000,
        frequency="MS",
        leverage=leverage_config(),
        dividend_mode=DividendMode.CASH,
        withholding_rate=0.30,
    )

    assert len(result.ledger.cash_flows) == 3
    assert result.total_contributed == pytest.approx(3_000)
    assert result.ledger.quantity == pytest.approx(39)
    assert result.ledger.debt == pytest.approx(900)
    assert result.equity_curve.iloc[-1]["total_equity"] == pytest.approx(3_000)


def test_buy_hold_leveraged_runner_applies_dividends():
    result = run_buy_hold_leveraged(
        price_frame=flat_price_frame(),
        dividend_frame=single_dividend_frame(),
        cost_model=zero_cost_model(),
        initial_cash=1_000,
        leverage=leverage_config(),
        dividend_mode=DividendMode.CASH,
        withholding_rate=0.30,
    )

    assert result.ledger.total_gross_dividends == pytest.approx(13)
    assert result.ledger.total_withholding_tax == pytest.approx(3.9)
    assert result.equity_curve.iloc[-1]["total_equity"] == pytest.approx(1_009.1)


def test_portfolio_margin_initial_rebalance_allocates_target_weights():
    ledger = PortfolioMarginLedger(
        [spy_asset(), qqq_asset()],
        starting_cash=10_000,
        leverage=leverage_config(),
        cost_model=zero_cost_model(),
        portfolio_label="SPY_QQQ",
    )

    ledger.rebalance_to_weights(
        "2024-01-02",
        prices={"SPY": 100, "QQQ": 50},
        target_weights={"SPY": 0.60, "QQQ": 0.40},
        target_leverage=1.3,
    )
    snapshot = ledger.snapshot("2024-01-02", prices={"SPY": 100, "QQQ": 50})

    assert ledger.positions["SPY"] == pytest.approx(78)
    assert ledger.positions["QQQ"] == pytest.approx(104)
    assert ledger.debt == pytest.approx(3_000)
    assert snapshot.market_value == pytest.approx(13_000)
    positions = ledger.position_curve
    latest = positions[positions["date"] == "2024-01-02"]
    assert latest.set_index("asset").loc["SPY", "weight"] == pytest.approx(0.60)
    assert latest.set_index("asset").loc["QQQ", "weight"] == pytest.approx(0.40)


def test_portfolio_margin_cash_dividend_records_tax_and_cash():
    ledger = PortfolioMarginLedger(
        [spy_asset(), qqq_asset()],
        starting_cash=10_000,
        leverage=leverage_config(),
        cost_model=zero_cost_model(),
        portfolio_label="SPY_QQQ",
    )
    ledger.rebalance_to_weights(
        "2024-01-02",
        prices={"SPY": 100, "QQQ": 50},
        target_weights={"SPY": 0.60, "QQQ": 0.40},
        target_leverage=1.3,
    )

    dividend = ledger.cash_dividend(
        "2024-01-03",
        ticker="SPY",
        dividend_per_share=1.0,
        withholding_rate=0.30,
        reinvest=False,
    )

    assert dividend is not None
    assert dividend.gross_amount == pytest.approx(78)
    assert dividend.withholding_tax == pytest.approx(23.4)
    assert ledger.cash == pytest.approx(54.6)


def test_portfolio_margin_rebalance_sells_overweight_asset():
    ledger = PortfolioMarginLedger(
        [spy_asset(), qqq_asset()],
        starting_cash=10_000,
        leverage=leverage_config(),
        cost_model=zero_cost_model(),
        portfolio_label="SPY_QQQ",
    )
    ledger.rebalance_to_weights(
        "2024-01-02",
        prices={"SPY": 100, "QQQ": 50},
        target_weights={"SPY": 0.60, "QQQ": 0.40},
        target_leverage=1.3,
    )

    ledger.rebalance_to_weights(
        "2024-02-01",
        prices={"SPY": 100, "QQQ": 100},
        target_weights={"SPY": 0.60, "QQQ": 0.40},
        target_leverage=1.3,
    )
    snapshot = ledger.snapshot("2024-02-01", prices={"SPY": 100, "QQQ": 100})
    latest = ledger.position_curve[ledger.position_curve["date"] == "2024-02-01"]

    assert snapshot.actual_leverage == pytest.approx(1.3)
    assert latest.set_index("asset").loc["SPY", "weight"] == pytest.approx(0.60)
    assert latest.set_index("asset").loc["QQQ", "weight"] == pytest.approx(0.40)
    assert "sell" in set(ledger.trades["side"])


def test_rebalance_leveraged_runner_outputs_portfolio_curve():
    result = run_rebalance_leveraged(
        price_frames=[three_month_price_frame(), three_month_qqq_price_frame()],
        dividend_frames=[empty_dividend_frame(), empty_dividend_frame(qqq_asset())],
        cost_model=zero_cost_model(),
        initial_cash=10_000,
        target_weights={"SPY": 0.60, "QQQ": 0.40},
        frequency="monthly",
        leverage=leverage_config(),
        dividend_mode=DividendMode.CASH,
        withholding_rate=0.30,
    )

    assert result.strategy == "rebalance_leveraged"
    assert result.ticker == "SPY_QQQ"
    assert result.equity_curve.iloc[-1]["actual_leverage"] == pytest.approx(1.3)
    assert not result.ledger.position_curve.empty


def test_portfolio_margin_weight_sum_must_be_one():
    ledger = PortfolioMarginLedger(
        [spy_asset(), qqq_asset()],
        starting_cash=10_000,
        leverage=leverage_config(),
        cost_model=zero_cost_model(),
    )

    with pytest.raises(ValueError, match="sum to 1.0"):
        ledger.rebalance_to_weights(
            "2024-01-02",
            prices={"SPY": 100, "QQQ": 50},
            target_weights={"SPY": 0.60, "QQQ": 0.30},
            target_leverage=1.3,
        )
