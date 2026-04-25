from __future__ import annotations

import argparse
import os

from investment_backtest_lab.data import MarketDataLoader
from investment_backtest_lab.models import AssetSpec, AssetType, DataSource, Market


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", action="store_true", help="Run live data download checks.")
    args = parser.parse_args()

    if not args.network:
        print("Skipping live data checks. Pass --network to download data.")
        return

    loader = MarketDataLoader(use_cache=True)
    assets = [
        AssetSpec("SPY", Market.US, AssetType.ETF, "USD", DataSource.YFINANCE),
        AssetSpec("QQQ", Market.US, AssetType.ETF, "USD", DataSource.YFINANCE),
        AssetSpec("USDTWD=X", Market.FX, AssetType.FX, "TWD", DataSource.YFINANCE),
    ]

    if os.getenv("FINMIND_TOKEN"):
        assets.extend(
            [
                AssetSpec("0050", Market.TW, AssetType.ETF, "TWD", DataSource.FINMIND),
                AssetSpec("2330", Market.TW, AssetType.STOCK, "TWD", DataSource.FINMIND),
            ]
        )
    else:
        print("FINMIND_TOKEN is not set. Skipping Taiwan live data checks.")

    for asset in assets:
        price_frame = loader.load_asset(asset, start_date="2024-01-01", end_date="2024-03-31")
        print(
            f"[OK] {asset.ticker}: {len(price_frame.data)} rows, "
            f"adjusted={price_frame.adjusted}, source={price_frame.source}"
        )


if __name__ == "__main__":
    main()
