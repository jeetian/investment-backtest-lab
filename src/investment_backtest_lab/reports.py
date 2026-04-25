from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    total_return = float((1.0 + clean).prod())
    years = len(clean) / periods_per_year
    if years <= 0 or total_return <= 0:
        return 0.0
    return total_return ** (1.0 / years) - 1.0


def performance_summary(
    returns: pd.Series,
    *,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    clean = returns.dropna().astype(float)
    if clean.empty:
        raise ValueError("Cannot summarize an empty return series.")

    excess = clean - risk_free_rate / periods_per_year
    annual_return = cagr(clean, periods_per_year=periods_per_year)
    annual_volatility = float(clean.std(ddof=0) * np.sqrt(periods_per_year))
    downside = excess[excess < 0.0]
    downside_std = downside.std(ddof=0)
    sharpe = float(excess.mean() / clean.std(ddof=0) * np.sqrt(periods_per_year))
    sortino = (
        float(excess.mean() / downside_std * np.sqrt(periods_per_year))
        if len(downside) > 1 and downside_std > 0
        else np.nan
    )
    mdd = max_drawdown(clean)
    calmar = annual_return / abs(mdd) if mdd < 0 else np.nan

    return pd.Series(
        {
            "cagr": annual_return,
            "volatility": annual_volatility,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": mdd,
            "total_return": float((1.0 + clean).prod() - 1.0),
        }
    )


def annual_returns(returns: pd.Series) -> pd.Series:
    return returns.dropna().resample("YE").apply(lambda r: (1.0 + r).prod() - 1.0)


def monthly_returns(returns: pd.Series) -> pd.Series:
    return returns.dropna().resample("ME").apply(lambda r: (1.0 + r).prod() - 1.0)


def rolling_cagr(
    returns: pd.Series,
    *,
    years: int,
    periods_per_year: int = 252,
) -> pd.Series:
    window = years * periods_per_year
    clean = returns.dropna()
    return clean.rolling(window).apply(
        lambda r: (1.0 + r).prod() ** (periods_per_year / len(r)) - 1.0,
        raw=False,
    )


def write_quantstats_html(
    returns: pd.Series,
    *,
    output_path: str | Path,
    benchmark: pd.Series | None = None,
    title: str = "Investment Backtest Report",
) -> Path:
    import quantstats as qs

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    qs.reports.html(returns, benchmark=benchmark, output=str(path), title=title)
    return path
