from __future__ import annotations

from dataclasses import dataclass, field

from investment_backtest_lab.data.adapters import (
    CsvPriceAdapter,
    FinMindPriceAdapter,
    PriceDataAdapter,
    YFinancePriceAdapter,
)
from investment_backtest_lab.data.cache import ParquetCache
from investment_backtest_lab.models import AssetSpec, DataSource, PriceFrame


@dataclass
class MarketDataLoader:
    cache: ParquetCache | None = field(default_factory=ParquetCache)
    adapters: dict[DataSource, PriceDataAdapter] = field(default_factory=dict)
    use_cache: bool = True

    def __post_init__(self) -> None:
        defaults: dict[DataSource, PriceDataAdapter] = {
            DataSource.CSV: CsvPriceAdapter(),
            DataSource.FINMIND: FinMindPriceAdapter(),
            DataSource.YFINANCE: YFinancePriceAdapter(),
        }
        defaults.update(self.adapters)
        self.adapters = defaults

    def load_asset(self, asset: AssetSpec, *, start_date: str, end_date: str) -> PriceFrame:
        if self.cache and self.use_cache:
            for adjusted in (True, False):
                cached = self.cache.get(
                    asset,
                    start_date=start_date,
                    end_date=end_date,
                    adjusted=adjusted,
                )
                if cached is not None:
                    return cached

        adapter = self.adapters.get(asset.data_source)
        if adapter is None:
            raise ValueError(f"No adapter registered for {asset.data_source}.")

        price_frame = adapter.load(asset, start_date=start_date, end_date=end_date)
        if self.cache and self.use_cache:
            self.cache.put(price_frame, start_date=start_date, end_date=end_date)
        return price_frame

    def load_close_prices(
        self,
        assets: list[AssetSpec],
        *,
        start_date: str,
        end_date: str,
    ):
        frames = [
            self.load_asset(asset, start_date=start_date, end_date=end_date).close()
            for asset in assets
        ]
        import pandas as pd

        return pd.concat(frames, axis=1).sort_index()
