import pandas as pd
import pytest

from investment_backtest_lab.tw_total_return import (
    SplitEvent,
    build_dividend_audit,
    build_total_return_price_from_raw,
    build_tw50_source_coverage,
    splice_price_series,
    synthetic_leveraged_from_base,
)


def test_total_return_builder_reinvests_dividend_on_ex_date_and_adjusts_split():
    prices = pd.DataFrame(
        {
            "date": ["2025-01-16", "2025-01-17", "2025-06-18", "2025-06-19"],
            "close": [200.0, 198.0, 50.0, 51.0],
        }
    )
    dividends = pd.DataFrame(
        {
            "AnnouncementDate": ["2024-12-01"],
            "CashExDividendTradingDate": ["2025-01-17"],
            "CashDividendPaymentDate": ["2025-02-20"],
            "CashEarningsDistribution": [2.70],
        }
    )

    series, audit = build_total_return_price_from_raw(
        raw_prices=prices,
        dividends=dividends,
        ticker="0050",
        splits=[SplitEvent("2025-06-18", 4.0)],
        source="test",
    )

    assert audit["split_factor"].iloc[0] == pytest.approx(4.0)
    assert audit["split_adjusted_cash_dividend"].iloc[0] == pytest.approx(0.675)
    expected_ex_date_return = (198.0 / 4.0 + 0.675) / (200.0 / 4.0) - 1.0
    assert series.pct_change().loc[pd.Timestamp("2025-01-17")] == pytest.approx(
        expected_ex_date_return
    )


def test_total_return_builder_handles_empty_dividend_audit():
    prices = pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-03"],
            "close": [20.0, 20.5],
        }
    )

    series, audit = build_total_return_price_from_raw(
        raw_prices=prices,
        dividends=pd.DataFrame(),
        ticker="00631L",
        splits=[],
        source="test",
    )

    assert audit.empty
    assert series.iloc[-1] == pytest.approx(20.5)


def test_dividend_audit_skips_dividends_without_price_ex_date():
    audit = build_dividend_audit(
        dividends=pd.DataFrame(
            {
                "CashExDividendTradingDate": ["2025-01-17"],
                "CashEarningsDistribution": [1.0],
            }
        ),
        ticker="0050",
        splits=[],
        price_index=pd.DatetimeIndex([pd.Timestamp("2025-01-16")]),
    )

    assert audit.empty


def test_splice_price_series_scales_earlier_segment_without_splice_jump():
    earlier = pd.Series(
        [100.0, 110.0],
        index=pd.to_datetime(["2025-01-02", "2025-01-03"]),
        name="proxy",
    )
    later = pd.Series(
        [55.0, 56.0],
        index=pd.to_datetime(["2025-01-06", "2025-01-07"]),
        name="actual",
    )

    combined = splice_price_series(
        earlier=earlier,
        later=later,
        later_source_start=pd.Timestamp("2025-01-06"),
    )

    assert combined.loc[pd.Timestamp("2025-01-03")] == pytest.approx(55.0)
    assert combined.loc[pd.Timestamp("2025-01-06")] == pytest.approx(55.0)


def test_synthetic_leveraged_from_base_uses_daily_reset_returns():
    base = pd.Series(
        [100.0, 101.0, 99.99],
        index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
    )

    leveraged = synthetic_leveraged_from_base(base, leverage=2.0, name="00631L")

    assert leveraged.iloc[1] == pytest.approx(102.0)
    assert leveraged.iloc[2] == pytest.approx(99.96)


def test_source_coverage_marks_1999_2002_proxy_as_not_total_return():
    dates = pd.to_datetime(["1999-03-10", "2002-12-31", "2003-01-02", "2003-06-30"])
    coverage = build_tw50_source_coverage(
        twii_price=pd.Series([1.0, 2.0], index=dates[:2]),
        taiex_tr=pd.Series([2.1], index=dates[2:3]),
        base_actual=pd.Series([30.0], index=dates[3:4]),
        leveraged_actual=pd.Series([20.0], index=pd.to_datetime(["2014-10-31"])),
        hybrid_prices=pd.DataFrame(
            {"0050": [1.0, 2.0], "00631L": [1.0, 2.0]},
            index=pd.to_datetime(["1999-03-10", "2026-04-30"]),
        ),
    )

    base_row = coverage[coverage["ticker"] == "0050"].iloc[0]
    assert "price_proxy_not_total_return" in base_row["source_notes"]
    assert base_row["proxy_start"] == "1999-03-10"
