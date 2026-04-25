from __future__ import annotations

import pandas as pd


def moving_average_signals(
    close: pd.Series,
    *,
    fast_window: int = 50,
    slow_window: int = 200,
) -> tuple[pd.Series, pd.Series]:
    if fast_window >= slow_window:
        raise ValueError("fast_window must be less than slow_window.")

    price = close.dropna().sort_index()
    fast = price.rolling(fast_window).mean()
    slow = price.rolling(slow_window).mean()
    above = fast > slow
    previous_above = above.shift(1, fill_value=False)
    entries = (above & ~previous_above).rename("entries")
    exits = (~above & previous_above).rename("exits")
    return entries, exits


def position_from_signals(entries: pd.Series, exits: pd.Series) -> pd.Series:
    signals = pd.Series(pd.NA, index=entries.index.union(exits.index), dtype="boolean")
    signals.loc[entries.reindex(signals.index, fill_value=False)] = True
    signals.loc[exits.reindex(signals.index, fill_value=False)] = False
    return signals.ffill().fillna(False).astype(bool).rename("position")


def run_buy_and_hold(
    close: pd.Series | pd.DataFrame,
    *,
    init_cash: float = 100_000.0,
    fees: float = 0.0,
    freq: str = "1D",
):
    import vectorbt as vbt

    price = close.dropna()
    if hasattr(vbt.Portfolio, "from_holding"):
        return vbt.Portfolio.from_holding(price, init_cash=init_cash, fees=fees, freq=freq)

    entries = price.notna()
    if isinstance(entries, pd.Series):
        entries.iloc[1:] = False
    else:
        entries.iloc[1:, :] = False
    exits = entries & False
    return vbt.Portfolio.from_signals(
        price,
        entries,
        exits,
        size=float("inf"),
        init_cash=init_cash,
        fees=fees,
        freq=freq,
    )


def run_moving_average_timing(
    close: pd.Series,
    *,
    fast_window: int = 50,
    slow_window: int = 200,
    init_cash: float = 100_000.0,
    fees: float = 0.0,
    freq: str = "1D",
):
    import vectorbt as vbt

    entries, exits = moving_average_signals(
        close,
        fast_window=fast_window,
        slow_window=slow_window,
    )
    return vbt.Portfolio.from_signals(
        close.dropna().sort_index(),
        entries,
        exits,
        size=float("inf"),
        init_cash=init_cash,
        fees=fees,
        freq=freq,
    )
