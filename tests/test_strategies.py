import pandas as pd

from investment_backtest_lab.costs import CostModel
from investment_backtest_lab.models import AssetSpec, AssetType, DataSource, Market
from investment_backtest_lab.strategies.dca import run_dca_cash_flow
from investment_backtest_lab.strategies.vectorbt_wrappers import (
    moving_average_signals,
    position_from_signals,
)


def test_moving_average_signals_emit_entries_and_exits():
    close = pd.Series(
        [10, 10, 10, 11, 12, 13, 12, 11, 10, 9],
        index=pd.bdate_range("2024-01-01", periods=10),
    )

    entries, exits = moving_average_signals(close, fast_window=2, slow_window=3)

    assert entries.any()
    assert exits.any()


def test_position_from_signals_holds_between_entry_and_exit():
    index = pd.bdate_range("2024-01-01", periods=5)
    entries = pd.Series([False, True, False, False, False], index=index)
    exits = pd.Series([False, False, False, True, False], index=index)

    position = position_from_signals(entries, exits)

    assert not position.iloc[0]
    assert position.iloc[1]
    assert position.iloc[2]
    assert not position.iloc[3]


def test_dca_cash_flow_tracks_contributions_and_fees():
    asset = AssetSpec("0050", Market.TW, AssetType.ETF, "TWD", DataSource.FINMIND)
    close = pd.Series(100.0, index=pd.bdate_range("2024-01-01", "2024-06-30"))

    result = run_dca_cash_flow(
        close,
        asset=asset,
        contribution=10_000,
        frequency="MS",
        cost_model=CostModel(),
    )

    assert len(result.orders) == 6
    assert result.total_contributed == 60_000
    assert result.total_fees_paid > 0
    assert result.equity.iloc[-1] < result.total_contributed


def test_dca_monthly_schedule_includes_first_month_when_data_starts_after_month_start():
    asset = AssetSpec("SPY", Market.US, AssetType.ETF, "USD", DataSource.YFINANCE)
    close = pd.Series(100.0, index=pd.bdate_range("2020-01-02", "2020-03-31"))

    result = run_dca_cash_flow(
        close,
        asset=asset,
        contribution=100,
        frequency="MS",
        cost_model=CostModel(),
    )

    assert len(result.orders) == 3
    assert result.orders.index[0] == close.index[0]
