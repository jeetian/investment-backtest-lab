from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

import pandas as pd

from investment_backtest_lab.models import (
    AssetSpec,
    DividendFrame,
    PriceFrame,
    normalize_dividends,
    normalize_ohlcv,
)


class PriceDataAdapter(Protocol):
    def load(
        self,
        asset: AssetSpec,
        *,
        start_date: str,
        end_date: str,
        adjusted: bool | None = None,
    ) -> PriceFrame: ...


class DividendDataAdapter(Protocol):
    def load_dividends(
        self,
        asset: AssetSpec,
        *,
        start_date: str,
        end_date: str,
    ) -> DividendFrame: ...


class CsvPriceAdapter:
    def __init__(self, path_template: str | Path = "data/raw/{ticker}.csv", adjusted: bool = True):
        self.path_template = str(path_template)
        self.adjusted = adjusted

    def load(
        self,
        asset: AssetSpec,
        *,
        start_date: str,
        end_date: str,
        adjusted: bool | None = None,
    ) -> PriceFrame:
        path = Path(self.path_template.format(ticker=asset.ticker, cache_key=asset.cache_key))
        if not path.exists():
            raise FileNotFoundError(f"CSV price file not found: {path}")

        raw = pd.read_csv(path)
        data = normalize_ohlcv(raw)
        data = data.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)]
        return PriceFrame(
            asset=asset,
            data=data,
            adjusted=self.adjusted if adjusted is None else adjusted,
            source=str(path),
        )


class YFinancePriceAdapter:
    def __init__(self, auto_adjust: bool = True):
        self.auto_adjust = auto_adjust

    def load(
        self,
        asset: AssetSpec,
        *,
        start_date: str,
        end_date: str,
        adjusted: bool | None = None,
    ) -> PriceFrame:
        import yfinance as yf

        auto_adjust = self.auto_adjust if adjusted is None else adjusted
        raw = yf.download(
            asset.ticker,
            start=start_date,
            end=end_date,
            auto_adjust=auto_adjust,
            progress=False,
            actions=False,
            group_by="column",
        )
        if raw.empty:
            raise ValueError(f"yfinance returned no rows for {asset.ticker}.")

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        data = normalize_ohlcv(raw.reset_index(), date_column="Date")
        return PriceFrame(
            asset=asset,
            data=data,
            adjusted=auto_adjust,
            source=f"yfinance:{asset.ticker}",
        )


class YFinanceDividendAdapter:
    def load_dividends(
        self,
        asset: AssetSpec,
        *,
        start_date: str,
        end_date: str,
    ) -> DividendFrame:
        import yfinance as yf

        raw = yf.Ticker(asset.ticker).dividends
        data = normalize_dividends(raw)
        if not data.empty:
            start = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date)
            data = data.loc[(data.index >= start) & (data.index <= end)]
        return DividendFrame(
            asset=asset,
            data=data,
            currency=asset.currency,
            source=f"yfinance:dividends:{asset.ticker}",
        )


class FinMindPriceAdapter:
    def __init__(
        self,
        *,
        token: str | None = None,
        dataset: str = "TaiwanStockPrice",
        adjusted_dataset: str = "TaiwanStockPriceAdj",
        prefer_adjusted: bool = True,
    ):
        self.token = token or os.getenv("FINMIND_TOKEN")
        self.dataset = dataset
        self.adjusted_dataset = adjusted_dataset
        self.prefer_adjusted = prefer_adjusted

    def load(
        self,
        asset: AssetSpec,
        *,
        start_date: str,
        end_date: str,
        adjusted: bool | None = None,
    ) -> PriceFrame:
        from FinMind.data import DataLoader

        loader = DataLoader()
        if self.token:
            loader.login_by_token(api_token=self.token)

        prefer_adjusted = self.prefer_adjusted if adjusted is None else adjusted
        dataset = self.adjusted_dataset if prefer_adjusted else self.dataset
        raw = self._download(loader, dataset, asset.ticker, start_date, end_date)
        adjusted = dataset == self.adjusted_dataset

        if raw.empty and prefer_adjusted:
            raw = self._download(loader, self.dataset, asset.ticker, start_date, end_date)
            adjusted = False

        if raw.empty:
            raise ValueError(f"FinMind returned no rows for {asset.ticker}.")

        data = normalize_ohlcv(
            raw,
            column_map={
                "max": "high",
                "min": "low",
                "Trading_Volume": "volume",
                "trading_volume": "volume",
            },
        )
        return PriceFrame(
            asset=asset,
            data=data,
            adjusted=adjusted,
            source=f"FinMind:{dataset}:{asset.ticker}",
        )

    @staticmethod
    def _download(
        loader: object,
        dataset: str,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        if hasattr(loader, "get_data"):
            return loader.get_data(
                dataset=dataset,
                data_id=ticker,
                start_date=start_date,
                end_date=end_date,
            )

        if dataset == "TaiwanStockPrice" and hasattr(loader, "taiwan_stock_daily"):
            return loader.taiwan_stock_daily(
                stock_id=ticker,
                start_date=start_date,
                end_date=end_date,
            )

        if dataset == "TaiwanStockPriceAdj" and hasattr(loader, "taiwan_stock_daily_adj"):
            return loader.taiwan_stock_daily_adj(
                stock_id=ticker,
                start_date=start_date,
                end_date=end_date,
            )

        raise AttributeError(
            "Installed FinMind DataLoader does not expose get_data or a matching daily method."
        )
