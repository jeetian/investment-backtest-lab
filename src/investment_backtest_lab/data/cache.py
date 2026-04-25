from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from investment_backtest_lab.models import AssetSpec, DividendFrame, PriceFrame


@dataclass(frozen=True)
class ParquetCache:
    root: Path = Path("data/cache")

    def path_for(
        self,
        asset: AssetSpec,
        *,
        start_date: str,
        end_date: str,
        adjusted: bool,
    ) -> Path:
        flag = "adjusted" if adjusted else "raw"
        return self.root / f"{asset.cache_key}_{start_date}_{end_date}_{flag}.parquet"

    def dividend_path_for(self, asset: AssetSpec, *, start_date: str, end_date: str) -> Path:
        return self.root / f"{asset.cache_key}_{start_date}_{end_date}_dividends.parquet"

    def get(
        self,
        asset: AssetSpec,
        *,
        start_date: str,
        end_date: str,
        adjusted: bool = True,
    ) -> PriceFrame | None:
        path = self.path_for(asset, start_date=start_date, end_date=end_date, adjusted=adjusted)
        if not path.exists():
            return None
        data = pd.read_parquet(path)
        return PriceFrame(asset=asset, data=data, adjusted=adjusted, source=str(path))

    def put(self, price_frame: PriceFrame, *, start_date: str, end_date: str) -> Path:
        path = self.path_for(
            price_frame.asset,
            start_date=start_date,
            end_date=end_date,
            adjusted=price_frame.adjusted,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        price_frame.data.to_parquet(path)
        return path

    def get_dividends(
        self,
        asset: AssetSpec,
        *,
        start_date: str,
        end_date: str,
    ) -> DividendFrame | None:
        path = self.dividend_path_for(asset, start_date=start_date, end_date=end_date)
        if not path.exists():
            return None
        data = pd.read_parquet(path)
        return DividendFrame(
            asset=asset,
            data=data,
            currency=asset.currency,
            source=str(path),
        )

    def put_dividends(
        self,
        dividend_frame: DividendFrame,
        *,
        start_date: str,
        end_date: str,
    ) -> Path:
        path = self.dividend_path_for(
            dividend_frame.asset,
            start_date=start_date,
            end_date=end_date,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        dividend_frame.data.to_parquet(path)
        return path
