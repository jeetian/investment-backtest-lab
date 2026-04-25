from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from investment_backtest_lab.config import load_backtest_config
from investment_backtest_lab.costs import CostModel
from investment_backtest_lab.data import MarketDataLoader
from investment_backtest_lab.data.fx import align_fx_rate, fx_contribution
from investment_backtest_lab.models import AssetSpec, AssetType, DataSource, Market
from investment_backtest_lab.reports import performance_summary, rolling_cagr
from investment_backtest_lab.strategies.dca import run_dca_cash_flow
from investment_backtest_lab.strategies.vectorbt_wrappers import (
    moving_average_signals,
    position_from_signals,
)


def synthetic_prices() -> pd.Series:
    index = pd.bdate_range("2020-01-01", "2025-12-31")
    drift = 0.00025
    cycle = np.sin(np.arange(len(index)) / 60.0) * 0.002
    noise = np.random.default_rng(42).normal(0.0, 0.01, size=len(index))
    returns = pd.Series(drift + cycle + noise, index=index)
    prices = 100.0 * (1.0 + returns).cumprod()
    return prices.rename("SYNTH")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp_example.yaml")
    parser.add_argument("--offline-demo", action="store_true")
    args = parser.parse_args()

    config = load_backtest_config(args.config)
    cost_model = CostModel.from_dict(config.cost_model)

    asset = config.universe[0]
    close = synthetic_prices() if args.offline_demo else load_live_close(config, asset)
    ma_params = {
        "fast_window": config.strategy.params.get("fast_window", 50),
        "slow_window": config.strategy.params.get("slow_window", 200),
    }
    entries, exits = moving_average_signals(close, **ma_params)
    position = position_from_signals(entries, exits)
    signal_returns = close.pct_change().where(position.shift(fill_value=False), 0.0).fillna(0.0)
    dca = run_dca_cash_flow(
        close,
        asset=asset,
        contribution=config.dca.contribution,
        frequency=config.dca.frequency,
        cost_model=cost_model,
    )

    print("Moving-average signal sample:")
    print(pd.DataFrame({"close": close, "entry": entries, "exit": exits}).tail())
    print("")
    print("Signal performance summary:")
    print(performance_summary(signal_returns).round(4))

    if not args.offline_demo and asset.currency.upper() != config.base_currency:
        base_returns = convert_signal_returns_to_base(
            close=close,
            position=position,
            base_currency=config.base_currency,
        )
        print("")
        print(f"Signal performance summary in {config.base_currency}:")
        print(performance_summary(base_returns).round(4))
        print("")
        print("FX contribution tail:")
        print(fx_contribution(native_returns=signal_returns, base_returns=base_returns).tail())

    print("")
    print("DCA summary:")
    print(
        {
            "orders": len(dca.orders),
            "total_contributed": round(dca.total_contributed, 2),
            "total_fees_paid": round(dca.total_fees_paid, 2),
            "ending_equity": round(float(dca.equity.iloc[-1]), 2),
        }
    )
    print("")
    print("Rolling 3Y CAGR tail:")
    print(rolling_cagr(signal_returns, years=3).dropna().tail().round(4))


def load_live_close(config, asset: AssetSpec) -> pd.Series:
    loader = MarketDataLoader(use_cache=True)
    price_frame = loader.load_asset(
        asset,
        start_date=config.start_date.isoformat(),
        end_date=config.end_date.isoformat(),
    )
    return price_frame.close()


def convert_signal_returns_to_base(
    *,
    close: pd.Series,
    position: pd.Series,
    base_currency: str,
) -> pd.Series:
    if base_currency.upper() != "TWD":
        raise NotImplementedError("Live prototype currently wires USD assets to TWD only.")

    fx_asset = AssetSpec("USDTWD=X", Market.FX, AssetType.FX, "TWD", DataSource.YFINANCE)
    loader = MarketDataLoader(use_cache=True)
    fx_close = loader.load_asset(
        fx_asset,
        start_date=close.index.min().date().isoformat(),
        end_date=close.index.max().date().isoformat(),
    ).close()
    aligned_fx = align_fx_rate(fx_close, close.index)
    base_close = close * aligned_fx
    base_asset_returns = base_close.pct_change().fillna(0.0)
    invested = position.reindex(base_asset_returns.index).shift(fill_value=False)
    return base_asset_returns.where(invested, 0.0).rename(f"{close.name}_base_signal")


if __name__ == "__main__":
    main()
