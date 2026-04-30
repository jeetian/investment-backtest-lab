from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from investment_backtest_lab.config import load_backtest_config
from investment_backtest_lab.costs import CostModel
from investment_backtest_lab.data import MarketDataLoader
from investment_backtest_lab.data.fx import align_fx_rate, fx_contribution
from investment_backtest_lab.models import AssetSpec, AssetType, DataSource, Market, PriceFrame
from investment_backtest_lab.reports import performance_summary
from investment_backtest_lab.strategies.dca import run_dca_cash_flow
from investment_backtest_lab.strategies.vectorbt_wrappers import (
    moving_average_signals,
    position_from_signals,
)


@dataclass(frozen=True)
class AnalysisResult:
    quality: pd.DataFrame
    metrics: pd.DataFrame
    warnings: list[str]
    report_path: Path
    metrics_path: Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze cached/live backtest results.")
    parser.add_argument("--config", default="configs/mvp_example.yaml")
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ"])
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    result = analyze(
        config_path=args.config,
        tickers=args.tickers,
        output_dir=Path(args.output_dir),
    )
    print_terminal_summary(result)


def analyze(*, config_path: str | Path, tickers: list[str], output_dir: Path) -> AnalysisResult:
    config = load_backtest_config(config_path)
    cost_model = CostModel.from_dict(config.cost_model)
    loader = MarketDataLoader(use_cache=True)

    selected_assets = select_assets(config.universe, tickers)
    start_date = config.start_date.isoformat()
    end_date = config.end_date.isoformat()

    price_frames = [
        loader.load_asset(asset, start_date=start_date, end_date=end_date)
        for asset in selected_assets
    ]

    usd_twd = load_usd_twd_if_needed(
        loader=loader,
        assets=selected_assets,
        base_currency=config.base_currency,
        start_date=start_date,
        end_date=end_date,
    )

    quality_records = [
        quality_record(
            price_frame,
            start_date=start_date,
            end_date=end_date,
            loader=loader,
        )
        for price_frame in price_frames
    ]

    metric_records: list[dict[str, Any]] = []
    warnings: list[str] = []
    fast_window = int(config.strategy.params.get("fast_window", 50))
    slow_window = int(config.strategy.params.get("slow_window", 200))

    for price_frame in price_frames:
        metric_records.extend(
            strategy_metric_records(
                price_frame=price_frame,
                base_currency=config.base_currency,
                usd_twd=usd_twd,
                fast_window=fast_window,
                slow_window=slow_window,
                dca_contribution=config.dca.contribution,
                dca_frequency=config.dca.frequency,
                cost_model=cost_model,
                warnings=warnings,
            )
        )

    quality = pd.DataFrame(quality_records)
    metrics = pd.DataFrame(metric_records)

    output_dir.mkdir(parents=True, exist_ok=True)
    slug = "_".join(ticker.lower().replace("/", "_").replace("=", "_") for ticker in tickers)
    report_path = output_dir / f"quickstart_{slug}.md"
    metrics_path = output_dir / f"quickstart_{slug}_metrics.csv"

    metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    report_path.write_text(
        render_markdown_report(
            config_path=Path(config_path),
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            base_currency=config.base_currency,
            quality=quality,
            metrics=metrics,
            warnings=warnings,
            metrics_path=metrics_path,
        ),
        encoding="utf-8",
    )

    return AnalysisResult(
        quality=quality,
        metrics=metrics,
        warnings=warnings,
        report_path=report_path,
        metrics_path=metrics_path,
    )


def select_assets(universe: list[AssetSpec], tickers: list[str]) -> list[AssetSpec]:
    by_ticker = {asset.ticker.upper(): asset for asset in universe}
    selected: list[AssetSpec] = []
    missing: list[str] = []
    for ticker in tickers:
        asset = by_ticker.get(ticker.upper())
        if asset is None:
            missing.append(ticker)
        else:
            selected.append(asset)

    if missing:
        available = ", ".join(sorted(by_ticker))
        raise ValueError(f"Tickers not found in config universe: {missing}. Available: {available}")
    return selected


def load_usd_twd_if_needed(
    *,
    loader: MarketDataLoader,
    assets: list[AssetSpec],
    base_currency: str,
    start_date: str,
    end_date: str,
) -> pd.Series | None:
    needs_usd_twd = any(asset.currency.upper() == "USD" for asset in assets)
    if base_currency.upper() != "TWD" or not needs_usd_twd:
        return None

    fx_asset = AssetSpec("USDTWD=X", Market.FX, AssetType.FX, "TWD", DataSource.YFINANCE)
    return loader.load_asset(fx_asset, start_date=start_date, end_date=end_date).close()


def quality_record(
    price_frame: PriceFrame,
    *,
    start_date: str,
    end_date: str,
    loader: MarketDataLoader,
) -> dict[str, Any]:
    data = price_frame.data
    close = data["close"]
    cache_path = ""
    if loader.cache is not None:
        cache_path = str(
            loader.cache.path_for(
                price_frame.asset,
                start_date=start_date,
                end_date=end_date,
                adjusted=price_frame.adjusted,
            )
        )

    daily_returns = close.pct_change()
    return {
        "ticker": price_frame.asset.ticker,
        "source": price_frame.source,
        "cache_path": cache_path,
        "adjusted": price_frame.adjusted,
        "rows": len(data),
        "start": data.index.min().date().isoformat(),
        "end": data.index.max().date().isoformat(),
        "missing_close": int(close.isna().sum()),
        "duplicate_dates": int(data.index.duplicated().sum()),
        "non_positive_close": int((close <= 0).sum()),
        "large_move_gt_20pct": int((daily_returns.abs() > 0.20).sum()),
        "close_first": float(close.iloc[0]),
        "close_last": float(close.iloc[-1]),
    }


def strategy_metric_records(
    *,
    price_frame: PriceFrame,
    base_currency: str,
    usd_twd: pd.Series | None,
    fast_window: int,
    slow_window: int,
    dca_contribution: float,
    dca_frequency: str,
    cost_model: CostModel,
    warnings: list[str],
) -> list[dict[str, Any]]:
    close_native = price_frame.close().dropna().sort_index()
    close_base = convert_close_to_base_series(
        close_native,
        asset=price_frame.asset,
        base_currency=base_currency,
        usd_twd=usd_twd,
    )

    records: list[dict[str, Any]] = []
    records.append(
        metric_record_from_returns(
            ticker=price_frame.asset.ticker,
            strategy="buy_and_hold",
            basis=price_frame.asset.currency,
            returns=close_native.pct_change().fillna(0.0),
            trades=1,
            invested_days=len(close_native),
            note="native currency",
        )
    )
    if close_base is not close_native:
        records.append(
            metric_record_from_returns(
                ticker=price_frame.asset.ticker,
                strategy="buy_and_hold",
                basis=base_currency,
                returns=close_base.pct_change().fillna(0.0),
                trades=1,
                invested_days=len(close_base),
                note="converted with USD/TWD",
            )
        )

    if len(close_native) <= slow_window:
        warnings.append(
            f"{price_frame.asset.ticker}: only {len(close_native)} rows, "
            f"not enough for slow_window={slow_window} moving average."
        )
    else:
        entries, exits = moving_average_signals(
            close_native,
            fast_window=fast_window,
            slow_window=slow_window,
        )
        position = position_from_signals(entries, exits)
        native_returns = close_native.pct_change().where(position.shift(fill_value=False), 0.0)
        native_returns = native_returns.fillna(0.0)
        records.append(
            metric_record_from_returns(
                ticker=price_frame.asset.ticker,
                strategy=f"moving_average_{fast_window}_{slow_window}",
                basis=price_frame.asset.currency,
                returns=native_returns,
                trades=int(entries.sum() + exits.sum()),
                invested_days=int(position.sum()),
                note="native currency",
            )
        )

        if close_base is not close_native:
            base_returns = close_base.pct_change().where(position.shift(fill_value=False), 0.0)
            base_returns = base_returns.fillna(0.0)
            fx_tail = fx_contribution(native_returns=native_returns, base_returns=base_returns)
            records.append(
                metric_record_from_returns(
                    ticker=price_frame.asset.ticker,
                    strategy=f"moving_average_{fast_window}_{slow_window}",
                    basis=base_currency,
                    returns=base_returns,
                    trades=int(entries.sum() + exits.sum()),
                    invested_days=int(position.sum()),
                    note=f"converted with USD/TWD; last_fx_contribution={fx_tail.iloc[-1]:.4f}",
                )
            )

    dca = run_dca_cash_flow(
        close_native,
        asset=price_frame.asset,
        contribution=dca_contribution,
        frequency=dca_frequency,
        cost_model=cost_model,
    )
    records.append(
        dca_metric_record(
            ticker=price_frame.asset.ticker,
            basis=price_frame.asset.currency,
            total_contributed=dca.total_contributed,
            ending_equity=float(dca.equity.iloc[-1]),
            total_fees_paid=dca.total_fees_paid,
            orders=len(dca.orders),
            note="DCA contribution is interpreted in native currency.",
        )
    )

    if close_base is not close_native:
        fx_for_equity = align_fx_rate(usd_twd, dca.equity.index) if usd_twd is not None else None
        if fx_for_equity is None:
            raise ValueError("USD/TWD FX is required for base-currency DCA conversion.")
        base_equity = dca.equity * fx_for_equity
        if dca.orders.empty:
            total_contributed_base = 0.0
        else:
            fx_for_orders = align_fx_rate(usd_twd, dca.orders.index)
            total_contributed_base = float((dca.orders["contribution"] * fx_for_orders).sum())
        records.append(
            dca_metric_record(
                ticker=price_frame.asset.ticker,
                basis=base_currency,
                total_contributed=total_contributed_base,
                ending_equity=float(base_equity.iloc[-1]),
                total_fees_paid=dca.total_fees_paid
                * float(fx_for_equity.reindex(dca.equity.index).ffill().iloc[-1]),
                orders=len(dca.orders),
                note="Approximate TWD conversion of native-currency DCA cash flows.",
            )
        )

    return records


def convert_close_to_base_series(
    close: pd.Series,
    *,
    asset: AssetSpec,
    base_currency: str,
    usd_twd: pd.Series | None,
) -> pd.Series:
    if asset.currency.upper() == base_currency.upper():
        return close
    if asset.currency.upper() == "USD" and base_currency.upper() == "TWD":
        if usd_twd is None:
            raise ValueError("USD/TWD FX is required for USD to TWD conversion.")
        return (close * align_fx_rate(usd_twd, close.index)).rename(close.name)
    raise NotImplementedError(
        f"Currency conversion {asset.currency}->{base_currency} is not wired."
    )


def metric_record_from_returns(
    *,
    ticker: str,
    strategy: str,
    basis: str,
    returns: pd.Series,
    trades: int,
    invested_days: int,
    note: str,
) -> dict[str, Any]:
    summary = performance_summary(returns)
    record = {
        "ticker": ticker,
        "strategy": strategy,
        "basis": basis,
        "cagr": summary["cagr"],
        "volatility": summary["volatility"],
        "sharpe": summary["sharpe"],
        "sortino": summary["sortino"],
        "calmar": summary["calmar"],
        "max_drawdown": summary["max_drawdown"],
        "total_return": summary["total_return"],
        "trades_or_orders": trades,
        "invested_days": invested_days,
        "total_contributed": pd.NA,
        "ending_equity": pd.NA,
        "total_fees_paid": pd.NA,
        "simple_cash_return": pd.NA,
        "note": note,
    }
    return record


def dca_metric_record(
    *,
    ticker: str,
    basis: str,
    total_contributed: float,
    ending_equity: float,
    total_fees_paid: float,
    orders: int,
    note: str,
) -> dict[str, Any]:
    simple_return = ending_equity / total_contributed - 1.0 if total_contributed else pd.NA
    return {
        "ticker": ticker,
        "strategy": "dca",
        "basis": basis,
        "cagr": pd.NA,
        "volatility": pd.NA,
        "sharpe": pd.NA,
        "sortino": pd.NA,
        "calmar": pd.NA,
        "max_drawdown": pd.NA,
        "total_return": pd.NA,
        "trades_or_orders": orders,
        "invested_days": pd.NA,
        "total_contributed": total_contributed,
        "ending_equity": ending_equity,
        "total_fees_paid": total_fees_paid,
        "simple_cash_return": simple_return,
        "note": note,
    }


def render_markdown_report(
    *,
    config_path: Path,
    tickers: list[str],
    start_date: str,
    end_date: str,
    base_currency: str,
    quality: pd.DataFrame,
    metrics: pd.DataFrame,
    warnings: list[str],
    metrics_path: Path,
) -> str:
    quality_md = format_quality_for_markdown(quality).to_markdown(
        index=False,
        disable_numparse=True,
    )
    metrics_md = format_metrics_for_markdown(metrics).to_markdown(
        index=False,
        disable_numparse=True,
    )
    warning_lines = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- 無"

    return f"""# Quickstart SPY/QQQ 分析報表

## 摘要

- 設定檔：`{config_path}`
- 標的：{", ".join(tickers)}
- 期間：{start_date} 到 {end_date}
- 報表基準幣別：{base_currency}
- 指標 CSV：`{metrics_path}`

## 怎麼讀這份報表

- `buy_and_hold`：買進並持有，作為最基本 benchmark。
- `moving_average_*`：均線進出策略，只在快線高於慢線時持有。
- `dca`：定期投入現金流結果；這不是時間加權績效，所以看 `simple_cash_return`，不要和 CAGR 直接比較。
- `basis=USD`：原幣績效。
- `basis=TWD`：用 USD/TWD 匯率換算後的台幣績效。

## 資料品質檢查

{quality_md}

## 策略結果

{metrics_md}

## 警告與限制

{warning_lines}
- yfinance adjusted price 適合 prototype 研究，但正式研究仍要確認資料授權與調整邏輯。
- DCA 的台幣換算是根據投入日匯率近似換算，尚未模擬實際換匯手續、匯款費或稅務。
- 這份報表是第一版驗證報表，重點是確認資料與策略流程可讀、可重複，不是投資建議。
"""


def format_quality_for_markdown(quality: pd.DataFrame) -> pd.DataFrame:
    formatted = quality.copy()
    for column in ["close_first", "close_last"]:
        formatted[column] = formatted[column].map(lambda value: f"{value:,.2f}")
    return formatted[
        [
            "ticker",
            "rows",
            "start",
            "end",
            "missing_close",
            "duplicate_dates",
            "non_positive_close",
            "large_move_gt_20pct",
            "close_first",
            "close_last",
            "source",
        ]
    ]


def format_metrics_for_markdown(metrics: pd.DataFrame) -> pd.DataFrame:
    formatted = metrics.copy()
    percent_columns = [
        "cagr",
        "volatility",
        "max_drawdown",
        "total_return",
        "simple_cash_return",
    ]
    number_columns = ["sharpe", "sortino", "calmar"]
    money_columns = ["total_contributed", "ending_equity", "total_fees_paid"]

    for column in percent_columns:
        formatted[column] = formatted[column].map(format_percent_or_blank)
    for column in number_columns:
        formatted[column] = formatted[column].map(format_number_or_blank)
    for column in money_columns:
        formatted[column] = formatted[column].map(format_money_or_blank)

    return formatted[
        [
            "ticker",
            "strategy",
            "basis",
            "cagr",
            "total_return",
            "max_drawdown",
            "sharpe",
            "trades_or_orders",
            "total_contributed",
            "ending_equity",
            "total_fees_paid",
            "simple_cash_return",
            "note",
        ]
    ]


def format_percent_or_blank(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def format_number_or_blank(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def format_money_or_blank(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):,.2f}"


def print_terminal_summary(result: AnalysisResult) -> None:
    print("資料品質摘要")
    print(format_quality_for_markdown(result.quality).to_string(index=False))
    print("")
    print("策略結果摘要")
    print(format_metrics_for_markdown(result.metrics).to_string(index=False))
    print("")
    if result.warnings:
        print("警告")
        for warning in result.warnings:
            print(f"- {warning}")
        print("")
    print(f"Markdown report: {result.report_path}")
    print(f"Metrics CSV:     {result.metrics_path}")


if __name__ == "__main__":
    main()
