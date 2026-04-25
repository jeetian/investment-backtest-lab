import pandas as pd

from investment_backtest_lab.reports import annual_returns, max_drawdown, performance_summary


def test_performance_summary_has_required_metrics():
    returns = pd.Series(
        [0.01, -0.005, 0.002, 0.003],
        index=pd.bdate_range("2024-01-01", periods=4),
    )

    summary = performance_summary(returns)

    assert "cagr" in summary
    assert "max_drawdown" in summary
    assert summary["total_return"] != 0


def test_max_drawdown_negative_when_equity_falls():
    returns = pd.Series([0.10, -0.20, 0.05])

    assert max_drawdown(returns) < 0


def test_annual_returns_groups_by_year_end():
    returns = pd.Series(
        [0.01, 0.01, 0.02],
        index=pd.to_datetime(["2024-01-01", "2024-12-31", "2025-01-02"]),
    )

    result = annual_returns(returns)

    assert len(result) == 2
