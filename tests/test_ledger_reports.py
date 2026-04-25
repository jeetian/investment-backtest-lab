from pathlib import Path

import pandas as pd
import pytest

from investment_backtest_lab.costs import CostModel
from investment_backtest_lab.ledger_reports import (
    align_dividends_to_trading_dates,
    run_buy_and_hold_ledger,
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


def test_dividend_date_aligns_to_next_trading_day():
    aligned, warnings = align_dividends_to_trading_dates(
        dividend_frame().data,
        flat_price_frame().data.index,
    )

    assert not warnings
    assert aligned.iloc[0]["dividend_date"] == pd.Timestamp("2024-01-06")
    assert aligned.iloc[0]["effective_date"] == pd.Timestamp("2024-01-08")


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
    fx = pd.Series(30.0, index=flat_price_frame().data.index)

    report = write_ledger_report(
        results=[cash, reinvest],
        base_currency="TWD",
        usd_twd=fx,
        output_dir=tmp_path,
        slug="spy",
        config_path=Path("configs/mvp_example.yaml"),
    )

    assert report.markdown_path.exists()
    assert report.metrics_path.exists()
    assert report.trades_path.exists()
    assert report.dividends_path.exists()
    assert report.equity_path.exists()
    assert report.html_path.exists()
    assert set(report.metrics["basis"]) == {"USD", "TWD"}
    assert "美股 Ledger 報表" in report.markdown_path.read_text(encoding="utf-8")
    assert "US Ledger Audit Report" in report.html_path.read_text(encoding="utf-8")
