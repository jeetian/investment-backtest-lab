from __future__ import annotations

from dataclasses import dataclass, field

from investment_backtest_lab.data.adapters import (
    CsvPriceAdapter,
    DividendDataAdapter,
    FinMindPriceAdapter,
    PriceDataAdapter,
    YFinanceDividendAdapter,
    YFinancePriceAdapter,
)
from investment_backtest_lab.data.cache import ParquetCache
from investment_backtest_lab.models import AssetSpec, DataSource, DividendFrame, PriceFrame


@dataclass
class MarketDataLoader:
    cache: ParquetCache | None = field(default_factory=ParquetCache)
    adapters: dict[DataSource, PriceDataAdapter] = field(default_factory=dict)
    dividend_adapters: dict[DataSource, DividendDataAdapter] = field(default_factory=dict)
    use_cache: bool = True

    def __post_init__(self) -> None:
        defaults: dict[DataSource, PriceDataAdapter] = {
            DataSource.CSV: CsvPriceAdapter(),
            DataSource.FINMIND: FinMindPriceAdapter(),
            DataSource.YFINANCE: YFinancePriceAdapter(),
        }
        defaults.update(self.adapters)
        self.adapters = defaults
        dividend_defaults: dict[DataSource, DividendDataAdapter] = {
            DataSource.YFINANCE: YFinanceDividendAdapter(),
        }
        dividend_defaults.update(self.dividend_adapters)
        self.dividend_adapters = dividend_defaults

    def load_asset(
        self,
        asset: AssetSpec,
        *,
        start_date: str,
        end_date: str,
        adjusted: bool | None = None,
    ) -> PriceFrame:
        if self.cache and self.use_cache:
            adjusted_options = (True, False) if adjusted is None else (adjusted,)
            for adjusted_option in adjusted_options:
                cached = self.cache.get(
                    asset,
                    start_date=start_date,
                    end_date=end_date,
                    adjusted=adjusted_option,
                )
                if cached is not None:
                    return cached

        adapter = self.adapters.get(asset.data_source)
        if adapter is None:
            raise ValueError(f"No adapter registered for {asset.data_source}.")

        price_frame = adapter.load(
            asset,
            start_date=start_date,
            end_date=end_date,
            adjusted=adjusted,
        )
        if self.cache and self.use_cache:
            self.cache.put(price_frame, start_date=start_date, end_date=end_date)
        return price_frame

    def load_dividends(
        self,
        asset: AssetSpec,
        *,
        start_date: str,
        end_date: str,
    ) -> DividendFrame:
        if self.cache and self.use_cache:
            cached = self.cache.get_dividends(asset, start_date=start_date, end_date=end_date)
            if cached is not None:
                return cached

        adapter = self.dividend_adapters.get(asset.data_source)
        if adapter is None:
            raise ValueError(f"No dividend adapter registered for {asset.data_source}.")

        dividend_frame = adapter.load_dividends(
            asset,
            start_date=start_date,
            end_date=end_date,
        )
        if self.cache and self.use_cache:
            self.cache.put_dividends(dividend_frame, start_date=start_date, end_date=end_date)
        return dividend_frame

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
