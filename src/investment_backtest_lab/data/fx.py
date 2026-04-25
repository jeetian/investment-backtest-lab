from __future__ import annotations

import pandas as pd

from investment_backtest_lab.models import PriceFrame


def align_fx_rate(fx_close: pd.Series, index: pd.Index) -> pd.Series:
    rate = fx_close.sort_index().reindex(index).ffill().bfill()
    if rate.isna().any():
        raise ValueError("FX series could not be aligned to target index.")
    return rate


def convert_close_to_base(
    price_frame: PriceFrame,
    *,
    base_currency: str,
    usd_twd: pd.Series | None = None,
) -> pd.Series:
    close = price_frame.close()
    asset_currency = price_frame.asset.currency.upper()
    base_currency = base_currency.upper()

    if asset_currency == base_currency:
        return close

    if asset_currency == "USD" and base_currency == "TWD":
        if usd_twd is None:
            raise ValueError("USD/TWD FX series is required to convert USD assets to TWD.")
        return close * align_fx_rate(usd_twd, close.index)

    raise NotImplementedError(
        f"Currency conversion {asset_currency}->{base_currency} is not wired."
    )


def fx_contribution(
    *,
    native_returns: pd.Series,
    base_returns: pd.Series,
) -> pd.Series:
    native = native_returns.fillna(0.0)
    base = base_returns.fillna(0.0)
    return ((1.0 + base) / (1.0 + native) - 1.0).rename("fx_contribution")
