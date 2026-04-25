from __future__ import annotations

import pandas as pd


def run_periodic_rebalance_bt(
    prices: pd.DataFrame,
    *,
    target_weights: dict[str, float],
    frequency: str = "monthly",
    name: str = "periodic_rebalance",
):
    try:
        import bt
    except ImportError as exc:
        raise ImportError(
            "The bt framework is required for periodic portfolio rebalancing. On "
            "Windows/Python 3.12 it may need Microsoft C++ Build Tools and, after "
            "installation, a terminal restart or system reboot before compilation works."
        ) from exc

    normalized_prices = prices.dropna(how="all").ffill().dropna()
    weights = {ticker: float(weight) for ticker, weight in target_weights.items()}
    total_weight = sum(weights.values())
    if not 0.999 <= total_weight <= 1.001:
        raise ValueError(f"Target weights must sum to 1.0, got {total_weight:.4f}.")

    if frequency.lower() in {"monthly", "month", "m"}:
        run_frequency = bt.algos.RunMonthly()
    elif frequency.lower() in {"quarterly", "quarter", "q"}:
        run_frequency = bt.algos.RunQuarterly()
    else:
        raise ValueError("frequency must be 'monthly' or 'quarterly'.")

    strategy = bt.Strategy(
        name,
        [
            run_frequency,
            bt.algos.SelectAll(),
            bt.algos.WeighSpecified(**weights),
            bt.algos.Rebalance(),
        ],
    )
    backtest = bt.Backtest(strategy, normalized_prices)
    return bt.run(backtest)
