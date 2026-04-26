from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from investment_backtest_lab.models import DynamicLeverageConfig, LeverageConfig


@dataclass(frozen=True)
class DynamicLeverageDecision:
    date: pd.Timestamp
    target_leverage: float
    regime: str
    reason: str
    price: float
    trend_average: float
    drawdown: float
    annualized_volatility: float
    safety_buffer: float | None = None

    def to_record(self) -> dict[str, object]:
        record = asdict(self)
        record["date"] = pd.Timestamp(self.date).date().isoformat()
        return record


def decide_dynamic_leverage(
    *,
    decision_date: pd.Timestamp,
    signal_prices: pd.Series,
    dynamic: DynamicLeverageConfig,
    leverage: LeverageConfig,
    safety_buffer: float | None = None,
) -> DynamicLeverageDecision:
    if not dynamic.enabled:
        return _decision(
            decision_date=decision_date,
            target=leverage.target_leverage,
            regime="static",
            reason="dynamic leverage disabled",
            signal_prices=signal_prices,
            dynamic=dynamic,
            safety_buffer=safety_buffer,
        )

    history = signal_prices.loc[:decision_date].dropna().astype(float)
    if history.empty:
        raise ValueError("Dynamic leverage requires non-empty signal prices.")

    trend_average = _trend_average(history, dynamic.trend_window)
    drawdown = _drawdown(history)
    annualized_volatility = _annualized_volatility(history, dynamic.volatility_window)
    latest_price = float(history.iloc[-1])

    if safety_buffer is not None and safety_buffer < dynamic.safety_buffer_guard:
        target = min(dynamic.neutral_leverage, leverage.deleverage_to)
        regime = "safety_guard"
        reason = "safety buffer below guard"
    elif drawdown <= dynamic.crash_guard:
        target = dynamic.risk_off_leverage
        regime = "crash_guard"
        reason = "drawdown beyond crash guard"
    elif latest_price < trend_average:
        target = dynamic.risk_off_leverage
        regime = "trend_off"
        reason = "price below trend average"
    elif drawdown <= dynamic.drawdown_guard or annualized_volatility >= dynamic.high_volatility:
        target = dynamic.neutral_leverage
        regime = "neutral"
        reason = "drawdown or volatility guard"
    else:
        target = dynamic.risk_on_leverage
        regime = "risk_on"
        reason = "trend positive and risk guards clear"

    target = max(dynamic.risk_off_leverage, min(target, leverage.max_leverage))
    return DynamicLeverageDecision(
        date=pd.Timestamp(decision_date),
        target_leverage=float(target),
        regime=regime,
        reason=reason,
        price=latest_price,
        trend_average=trend_average,
        drawdown=drawdown,
        annualized_volatility=annualized_volatility,
        safety_buffer=safety_buffer,
    )


def weighted_signal_series(prices: pd.DataFrame, target_weights: dict[str, float]) -> pd.Series:
    if prices.empty:
        raise ValueError("Cannot build a dynamic leverage signal from empty prices.")
    missing = [ticker for ticker in target_weights if ticker not in prices.columns]
    if missing:
        raise ValueError(f"Missing signal prices for target weights: {missing}")
    weights = pd.Series(target_weights, dtype=float)
    normalized = prices[weights.index].dropna()
    normalized = normalized / normalized.iloc[0]
    signal = normalized.mul(weights, axis=1).sum(axis=1) * 100.0
    return signal.rename("portfolio_signal")


def decisions_to_frame(decisions: list[DynamicLeverageDecision]) -> pd.DataFrame:
    return pd.DataFrame([decision.to_record() for decision in decisions])


def _decision(
    *,
    decision_date: pd.Timestamp,
    target: float,
    regime: str,
    reason: str,
    signal_prices: pd.Series,
    dynamic: DynamicLeverageConfig,
    safety_buffer: float | None,
) -> DynamicLeverageDecision:
    history = signal_prices.loc[:decision_date].dropna().astype(float)
    return DynamicLeverageDecision(
        date=pd.Timestamp(decision_date),
        target_leverage=float(target),
        regime=regime,
        reason=reason,
        price=float(history.iloc[-1]),
        trend_average=_trend_average(history, dynamic.trend_window),
        drawdown=_drawdown(history),
        annualized_volatility=_annualized_volatility(history, dynamic.volatility_window),
        safety_buffer=safety_buffer,
    )


def _trend_average(history: pd.Series, window: int) -> float:
    window = max(1, int(window))
    lookback = history.tail(window)
    return float(lookback.mean())


def _drawdown(history: pd.Series) -> float:
    latest = float(history.iloc[-1])
    peak = float(history.cummax().iloc[-1])
    if peak <= 0:
        return 0.0
    return latest / peak - 1.0


def _annualized_volatility(history: pd.Series, window: int) -> float:
    window = max(2, int(window))
    returns = history.pct_change().dropna().tail(window)
    if returns.empty:
        return 0.0
    return float(returns.std(ddof=0) * np.sqrt(252))


__all__ = [
    "DynamicLeverageDecision",
    "decide_dynamic_leverage",
    "decisions_to_frame",
    "weighted_signal_series",
]
