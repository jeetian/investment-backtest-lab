from pathlib import Path

import pandas as pd
import pytest

from investment_backtest_lab.costs import CostModel
from investment_backtest_lab.ledger_reports import (
    align_dividends_to_trading_dates,
    dca_contribution_dates,
    ledger_metrics_records,
    run_buy_and_hold_ledger,
    run_dca_ledger,
    run_rebalance_ledger,
    write_ledger_report,
)
from investment_backtest_lab.models import (
    AssetSpec,
    AssetType,
    DataSource,
    DividendFrame,
    DividendMode,
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


def flat_price_frame() -> PriceFrame:
    index = pd.bdate_range("2024-01-02", periods=5)
    data = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1_000,
        },
        index=index,
    )
    return PriceFrame(asset=spy_asset(), data=data, adjusted=False, source="raw-test")


def dividend_frame() -> DividendFrame:
    data = pd.DataFrame(
        {"dividend_per_share": [1.0]},
        index=pd.DatetimeIndex(["2024-01-06"], name="date"),
    )
    return DividendFrame(asset=spy_asset(), data=data, currency="USD", source="dividend-test")


def empty_dividend_frame() -> DividendFrame:
    data = pd.DataFrame(
        {"dividend_per_share": pd.Series(dtype="float64")},
        index=pd.DatetimeIndex([], name="date"),
    )
    return DividendFrame(asset=spy_asset(), data=data, currency="USD", source="empty-test")


def empty_qqq_dividend_frame() -> DividendFrame:
    data = pd.DataFrame(
        {"dividend_per_share": pd.Series(dtype="float64")},
        index=pd.DatetimeIndex([], name="date"),
    )
    return DividendFrame(asset=qqq_asset(), data=data, currency="USD", source="empty-test")


def three_month_price_frame() -> PriceFrame:
    index = pd.bdate_range("2024-01-02", "2024-03-29")
    data = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1_000,
        },
        index=index,
    )
    return PriceFrame(asset=spy_asset(), data=data, adjusted=False, source="raw-test")


def three_month_qqq_price_frame() -> PriceFrame:
    index = pd.bdate_range("2024-01-02", "2024-03-29")
    data = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1_000,
        },
        index=index,
    )
    return PriceFrame(asset=qqq_asset(), data=data, adjusted=False, source="raw-test")


def test_dividend_date_aligns_to_next_trading_day():
    aligned, warnings = align_dividends_to_trading_dates(
        dividend_frame().data,
        flat_price_frame().data.index,
    )

    assert not warnings
    assert aligned.iloc[0]["dividend_date"] == pd.Timestamp("2024-01-06")
    assert aligned.iloc[0]["effective_date"] == pd.Timestamp("2024-01-08")


def test_dca_contribution_dates_use_first_available_trading_day():
    dates = dca_contribution_dates(three_month_price_frame().data.index, frequency="MS")

    assert dates == {
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-02-01"),
        pd.Timestamp("2024-03-01"),
    }


def test_cash_dividend_uses_raw_price_without_double_counting():
    result = run_buy_and_hold_ledger(
        price_frame=flat_price_frame(),
        dividend_frame=dividend_frame(),
        cost_model=zero_cost_model(),
        initial_cash=1_000,
        dividend_mode=DividendMode.CASH,
        withholding_rate=0.30,
    )

    assert result.ledger.dividends.iloc[0]["gross_amount"] == pytest.approx(10)
    assert result.ledger.dividends.iloc[0]["withholding_tax"] == pytest.approx(3)
    assert result.ledger.dividends.iloc[0]["net_amount"] == pytest.approx(7)
    assert result.equity_curve.iloc[-1]["total_equity"] == pytest.approx(1_007)
    assert result.ledger.cash == pytest.approx(7)


def test_reinvest_dividend_increases_shares_and_preserves_after_tax_equity():
    result = run_buy_and_hold_ledger(
        price_frame=flat_price_frame(),
        dividend_frame=dividend_frame(),
        cost_model=zero_cost_model(),
        initial_cash=1_000,
        dividend_mode=DividendMode.REINVEST,
        withholding_rate=0.30,
    )

    assert result.ledger.dividends.iloc[0]["reinvested_quantity"] == pytest.approx(0.07)
    assert result.ledger.quantity == pytest.approx(10.07)
    assert result.ledger.cash == pytest.approx(0)
    assert result.equity_curve.iloc[-1]["total_equity"] == pytest.approx(1_007)


def test_dca_ledger_records_three_monthly_contributions_and_buys():
    result = run_dca_ledger(
        price_frame=three_month_price_frame(),
        dividend_frame=empty_dividend_frame(),
        cost_model=zero_cost_model(),
        contribution=1_000,
        frequency="MS",
        dividend_mode=DividendMode.CASH,
        withholding_rate=0.30,
    )

    assert result.strategy == "ledger_dca"
    assert len(result.ledger.cash_flows) == 3
    assert result.ledger.total_cash_deposited == pytest.approx(3_000)
    assert result.ledger.quantity == pytest.approx(30)
    assert result.ledger.cash == pytest.approx(0)
    assert result.equity_curve.iloc[-1]["total_equity"] == pytest.approx(3_000)


def test_dca_ledger_with_fees_buys_affordable_fractional_shares():
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
    result = run_dca_ledger(
        price_frame=flat_price_frame(),
        dividend_frame=empty_dividend_frame(),
        cost_model=cost_model,
        contribution=1_000,
        frequency="MS",
        dividend_mode=DividendMode.CASH,
        withholding_rate=0.30,
    )

    expected_shares = 1_000 / 101
    assert result.ledger.quantity == pytest.approx(expected_shares)
    assert result.ledger.total_fees_paid == pytest.approx(expected_shares)
    assert result.equity_curve.iloc[-1]["total_equity"] == pytest.approx(expected_shares * 100)


def test_dca_same_day_dividend_uses_pre_contribution_shares_only():
    data = pd.DataFrame(
        {"dividend_per_share": [1.0]},
        index=pd.DatetimeIndex(["2024-02-01"], name="date"),
    )
    dividends = DividendFrame(asset=spy_asset(), data=data, currency="USD", source="same-day")

    result = run_dca_ledger(
        price_frame=three_month_price_frame(),
        dividend_frame=dividends,
        cost_model=zero_cost_model(),
        contribution=1_000,
        frequency="MS",
        dividend_mode=DividendMode.CASH,
        withholding_rate=0.30,
    )

    assert result.ledger.dividends.iloc[0]["gross_amount"] == pytest.approx(10)
    assert result.ledger.dividends.iloc[0]["net_amount"] == pytest.approx(7)
    assert result.ledger.quantity == pytest.approx(30)
    assert result.ledger.cash == pytest.approx(7)
    assert result.equity_curve.iloc[-1]["total_equity"] == pytest.approx(3_007)


def test_dca_reinvested_dividend_increases_shares_before_contribution_buy():
    data = pd.DataFrame(
        {"dividend_per_share": [1.0]},
        index=pd.DatetimeIndex(["2024-02-01"], name="date"),
    )
    dividends = DividendFrame(asset=spy_asset(), data=data, currency="USD", source="same-day")

    result = run_dca_ledger(
        price_frame=three_month_price_frame(),
        dividend_frame=dividends,
        cost_model=zero_cost_model(),
        contribution=1_000,
        frequency="MS",
        dividend_mode=DividendMode.REINVEST,
        withholding_rate=0.30,
    )

    assert result.ledger.dividends.iloc[0]["reinvested_quantity"] == pytest.approx(0.07)
    assert result.ledger.quantity == pytest.approx(30.07)
    assert result.ledger.cash == pytest.approx(0)
    assert result.equity_curve.iloc[-1]["total_equity"] == pytest.approx(3_007)


def test_rebalance_ledger_runs_portfolio_with_twd_metrics():
    result = run_rebalance_ledger(
        price_frames=[three_month_price_frame(), three_month_qqq_price_frame()],
        dividend_frames=[empty_dividend_frame(), empty_qqq_dividend_frame()],
        cost_model=zero_cost_model(),
        initial_cash=10_000,
        target_weights={"SPY": 0.6, "QQQ": 0.4},
        frequency="monthly",
        dividend_mode=DividendMode.CASH,
        withholding_rate=0.30,
    )
    fx = pd.Series(30.0, index=three_month_price_frame().data.index)

    assert result.strategy == "ledger_rebalance"
    assert result.ticker == "SPY_QQQ"
    assert not result.ledger.positions_history.empty
    assert result.equity_curve.iloc[-1]["total_equity"] == pytest.approx(10_000)
    records = ledger_metrics_records(result, base_currency="TWD", usd_twd=fx)
    twd = [record for record in records if record["basis"] == "TWD"][0]
    assert twd["ending_equity"] == pytest.approx(300_000)
    assert "SPY=60.00%" in twd["final_weights"]
    assert "QQQ=40.00%" in twd["final_weights"]


def test_write_ledger_report_outputs_markdown_csv_and_html(tmp_path):
    cash = run_buy_and_hold_ledger(
        price_frame=flat_price_frame(),
        dividend_frame=dividend_frame(),
        cost_model=zero_cost_model(),
        initial_cash=1_000,
        dividend_mode=DividendMode.CASH,
        withholding_rate=0.30,
    )
    reinvest = run_buy_and_hold_ledger(
        price_frame=flat_price_frame(),
        dividend_frame=dividend_frame(),
        cost_model=zero_cost_model(),
        initial_cash=1_000,
        dividend_mode=DividendMode.REINVEST,
        withholding_rate=0.30,
    )
    dca = run_dca_ledger(
        price_frame=three_month_price_frame(),
        dividend_frame=empty_dividend_frame(),
        cost_model=zero_cost_model(),
        contribution=1_000,
        frequency="MS",
        dividend_mode=DividendMode.CASH,
        withholding_rate=0.30,
    )
    rebalance = run_rebalance_ledger(
        price_frames=[three_month_price_frame(), three_month_qqq_price_frame()],
        dividend_frames=[empty_dividend_frame(), empty_qqq_dividend_frame()],
        cost_model=zero_cost_model(),
        initial_cash=10_000,
        target_weights={"SPY": 0.6, "QQQ": 0.4},
        frequency="monthly",
        dividend_mode=DividendMode.CASH,
        withholding_rate=0.30,
    )
    fx = pd.Series(30.0, index=flat_price_frame().data.index)
    fx = pd.concat([fx, pd.Series(30.0, index=three_month_price_frame().data.index)])
    fx = fx[~fx.index.duplicated(keep="last")].sort_index()

    report = write_ledger_report(
        results=[cash, reinvest, dca, rebalance],
        base_currency="TWD",
        usd_twd=fx,
        output_dir=tmp_path,
        slug="spy",
        config_path=Path("configs/mvp_example.yaml"),
        report_context={"target_weights": {"SPY": 0.6, "QQQ": 0.4}},
    )

    assert report.markdown_path.exists()
    assert report.metrics_path.exists()
    assert report.trades_path.exists()
    assert report.dividends_path.exists()
    assert report.cash_flows_path.exists()
    assert report.equity_path.exists()
    assert report.positions_path.exists()
    assert report.rebalance_path.exists()
    assert report.html_path.exists()
    assert set(report.metrics["basis"]) == {"USD", "TWD"}
    assert "ledger_dca" in set(report.metrics["strategy"])
    assert "ledger_rebalance" in set(report.metrics["strategy"])
    dca_twd = report.metrics[
        (report.metrics["strategy"] == "ledger_dca") & (report.metrics["basis"] == "TWD")
    ].iloc[0]
    assert dca_twd["total_contributed"] == pytest.approx(90_000)
    assert dca_twd["simple_cash_return"] == pytest.approx(0)
    assert not report.cash_flows.empty
    assert not report.positions.empty
    assert "ledger_rebalance" in set(report.positions["strategy"])
    assert not report.rebalance.empty
    assert report.rebalance["trade_count"].max() >= 1
    assert "trade_reasons" in report.rebalance.columns
    assert "increase underweight" in " ".join(report.rebalance["trade_reasons"])
    assert "美股 Ledger 報表" in report.markdown_path.read_text(encoding="utf-8")
    assert "外部現金流 CSV" in report.markdown_path.read_text(encoding="utf-8")
    assert "部位權重 CSV" in report.markdown_path.read_text(encoding="utf-8")
    assert "再平衡摘要 CSV" in report.markdown_path.read_text(encoding="utf-8")
    html = report.html_path.read_text(encoding="utf-8")
    assert "US Ledger Audit Report" in html
    assert "https://fonts.googleapis.com" in html
    assert "Noto Sans TC" in html
    assert "dashboard-shell" in html
    assert "投資回測 Dashboard" in html
    assert "設定總覽" in html
    assert "雙層導覽" in html
    assert "B&amp;H 一次投入" in html
    assert "DCA 定期投入" in html
    assert "標準化比較" in html
    assert "非實際投入結果，不可當作本金報酬排名" in html
    assert "CSV 下載" in html
    assert "ledger_buy_and_hold" in html
    assert "ledger_dca" in html
    assert "ledger_rebalance" in html
    assert "cash" in html
    assert "reinvest" in html
    assert "deposit" in html
    assert "權重漂移" in html
    assert "再平衡讀法" in html
    assert "再平衡摘要 CSV" in html
    assert "買賣原因" in html
    assert "部位權重 CSV" in html
    assert html.index("B&amp;H 一次投入") < html.index("DCA 定期投入")
