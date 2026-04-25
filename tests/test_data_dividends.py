import pandas as pd

from investment_backtest_lab.data.cache import ParquetCache
from investment_backtest_lab.models import (
    AssetSpec,
    AssetType,
    DataSource,
    DividendFrame,
    Market,
    normalize_dividends,
)


def spy_asset() -> AssetSpec:
    return AssetSpec("SPY", Market.US, AssetType.ETF, "USD", DataSource.YFINANCE)


def test_normalize_dividends_from_series():
    raw = pd.Series(
        [1.0, 1.2],
        index=pd.to_datetime(["2024-01-05", "2024-04-05"]),
        name="Dividends",
    )

    normalized = normalize_dividends(raw)

    assert list(normalized.columns) == ["dividend_per_share"]
    assert normalized.index.name == "date"
    assert normalized.iloc[0]["dividend_per_share"] == 1.0


def test_parquet_cache_round_trips_dividends(tmp_path):
    cache = ParquetCache(root=tmp_path)
    asset = spy_asset()
    data = pd.DataFrame(
        {"dividend_per_share": [1.0]},
        index=pd.DatetimeIndex(["2024-01-05"], name="date"),
    )
    frame = DividendFrame(asset=asset, data=data, currency="USD", source="unit-test")

    cache.put_dividends(frame, start_date="2024-01-01", end_date="2024-12-31")
    loaded = cache.get_dividends(asset, start_date="2024-01-01", end_date="2024-12-31")

    assert loaded is not None
    assert loaded.data.iloc[0]["dividend_per_share"] == 1.0
    assert str(tmp_path) in loaded.source
