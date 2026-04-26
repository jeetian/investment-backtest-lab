import pandas as pd

from investment_backtest_lab.costs import CostModel
from investment_backtest_lab.leverage_reports import run_buy_hold_leveraged, write_leverage_report
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


def test_leverage_report_exports_margin_risk_outputs(tmp_path):
    result = run_buy_hold_leveraged(
        price_frame=price_frame(),
        dividend_frame=dividend_frame(),
        cost_model=zero_cost_model(),
        initial_cash=1_000,
        leverage=LeverageConfig(
            enabled=True,
            target_leverage=1.3,
            max_leverage=1.3,
            annual_borrow_rate=0.0,
            maintenance_requirement=0.35,
            min_safety_buffer=0.25,
            deleverage_to=1.1,
        ),
        dividend_mode=DividendMode.CASH,
        withholding_rate=0.30,
    )
    report = write_leverage_report(
        results=[result],
        output_dir=tmp_path,
        slug="spy",
        base_currency="TWD",
        usd_twd=pd.Series(
            [31.0, 31.1, 31.2],
            index=pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"]),
        ),
        config_path=tmp_path / "config.yaml",
        report_context={
            "start_date": "2024-01-02",
            "end_date": "2024-01-04",
            "tickers": ["SPY"],
            "strategies": ["buy_hold_leveraged"],
            "target_leverage": "1.30x",
            "annual_borrow_rate": "0.00%",
            "maintenance_requirement": "35.00%",
            "min_safety_buffer": "25.00%",
            "deleverage_to": "1.10x",
        },
    )

    assert report.metrics_path.exists()
    assert report.trades_path.exists()
    assert report.interest_path.exists()
    assert report.dividends_path.exists()
    assert report.events_path.exists()
    assert report.curves_path.exists()
    assert report.positions_path.exists()
    assert report.policy_path.exists()
    assert {"target_leverage", "interest_paid", "worst_safety_buffer"}.issubset(
        report.metrics.columns
    )
    assert {"gross_dividends", "withholding_tax", "dividend_mode"}.issubset(
        report.metrics.columns
    )
    html = report.html_path.read_text(encoding="utf-8")
    assert "buy_hold_leveraged" in html
    assert "安全緩衝" in html
    assert "dashboard-shell" in html
    assert "雙層導覽" in html
    assert "B&amp;H 一次投入" in html
    assert "DCA 定期投入" in html
    assert "再平衡" in html
    assert "槓桿風險" in html
    assert "scenario-selector" in html
    assert "data-scenario-panel" in html
    assert '"showlegend":false' in html
    assert "標準化比較" in html
    assert "B&amp;H 標準化路徑" in html
    assert "DCA 標準化路徑" in html
    assert "再平衡標準化路徑" in html
    assert "Debt 負債" in html
    assert "Safety Buffer 安全緩衝" in html
    assert "目標槓桿 vs 實際槓桿" in html
    assert "非實際投入結果，不可當作本金報酬排名" in html
    assert "Audit 明細與 CSV 下載" in html
    assert "policy CSV" in html
    assert "total_equity_twd" in report.curves.columns
    assert "target_leverage" in report.policy.columns


def price_frame() -> PriceFrame:
    index = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])
    data = pd.DataFrame(
        {
            "open": [100.0, 95.0, 90.0],
            "high": [100.0, 95.0, 90.0],
            "low": [100.0, 95.0, 90.0],
            "close": [100.0, 95.0, 90.0],
            "volume": [1_000, 1_000, 1_000],
        },
        index=index,
    )
    asset = AssetSpec("SPY", Market.US, AssetType.ETF, "USD", DataSource.YFINANCE)
    return PriceFrame(asset=asset, data=data, adjusted=False, source="raw-test")


def dividend_frame() -> DividendFrame:
    asset = AssetSpec("SPY", Market.US, AssetType.ETF, "USD", DataSource.YFINANCE)
    data = pd.DataFrame(
        {"dividend_per_share": [1.0]},
        index=pd.DatetimeIndex(["2024-01-03"], name="date"),
    )
    return DividendFrame(asset=asset, data=data, currency="USD", source="dividend-test")


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
