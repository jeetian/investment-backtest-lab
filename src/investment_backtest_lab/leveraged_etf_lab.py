from __future__ import annotations

import json
from dataclasses import dataclass, replace
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from investment_backtest_lab.models import LeveragedETFLabConfig, LeveragedETFProductConfig
from investment_backtest_lab.reports import performance_summary

CASH = "CASH"


@dataclass(frozen=True)
class ProductSpec:
    ticker: str
    leverage: float
    label: str

    @classmethod
    def from_config(cls, config: LeveragedETFProductConfig) -> ProductSpec:
        return cls(ticker=config.ticker, leverage=config.leverage, label=config.label)


@dataclass(frozen=True)
class LeveragedETFLabOutputs:
    metrics: pd.DataFrame
    curves: pd.DataFrame
    allocations: pd.DataFrame
    payload: dict[str, Any]
    scan_mode: str


@dataclass(frozen=True)
class LeveragedETFLabReportResult:
    metrics: pd.DataFrame
    curves: pd.DataFrame
    allocations: pd.DataFrame
    payload: dict[str, Any]
    html_path: Path
    metrics_path: Path
    payload_path: Path
    curves_path: Path
    allocations_path: Path
    scan_mode: str


def lab_config_for_scan_mode(
    lab_config: LeveragedETFLabConfig,
    scan_mode: str,
) -> LeveragedETFLabConfig:
    normalized = scan_mode.lower()
    if normalized == "fast":
        return replace(
            lab_config,
            grid_step=lab_config.fast_grid_step,
            top_n=lab_config.fast_top_n,
        )
    if normalized == "full":
        return replace(
            lab_config,
            grid_step=lab_config.full_grid_step,
            top_n=lab_config.top_n,
        )
    raise ValueError(f"scan_mode must be 'fast' or 'full', got {scan_mode!r}.")


def resolve_scan_mode(
    scan_mode: str | None = None,
    *,
    fast: bool = False,
    full: bool = False,
) -> str:
    if fast and full:
        raise ValueError("--fast and --full cannot be used together.")
    if full:
        return "full"
    if fast:
        return "fast"
    return scan_mode or "fast"


def build_leveraged_etf_lab_outputs(
    *,
    actual_prices: pd.DataFrame,
    synthetic_prices: pd.DataFrame,
    products: list[ProductSpec],
    lab_config: LeveragedETFLabConfig,
    scan_mode: str = "full",
) -> LeveragedETFLabOutputs:
    product_map = {product.ticker: product for product in products}
    outputs: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    if not actual_prices.empty:
        outputs.append(
            _run_mode(
                prices=actual_prices,
                data_mode="actual_etf",
                products=products,
                lab_config=lab_config,
            )
        )
    if not synthetic_prices.empty:
        outputs.append(
            _run_mode(
                prices=synthetic_prices,
                data_mode="synthetic_stress",
                products=products,
                lab_config=lab_config,
            )
        )
    if not outputs:
        raise ValueError("Leveraged ETF lab requires actual or synthetic price data.")

    metrics = pd.concat([item[0] for item in outputs], ignore_index=True)
    curves = pd.concat([item[1] for item in outputs], ignore_index=True)
    allocations = pd.concat([item[2] for item in outputs], ignore_index=True)
    metrics = rank_metrics(metrics)
    payload = build_compare_payload(
        metrics=metrics,
        curves=curves,
        products=product_map,
        top_n=lab_config.top_n,
        scan_mode=scan_mode,
    )
    return LeveragedETFLabOutputs(
        metrics=metrics,
        curves=curves,
        allocations=allocations,
        payload=payload,
        scan_mode=scan_mode,
    )


def synthetic_daily_reset_prices(
    base_close: pd.Series,
    products: list[ProductSpec],
    *,
    initial_price: float = 100.0,
) -> pd.DataFrame:
    clean = base_close.dropna().astype(float).sort_index()
    if clean.empty:
        raise ValueError("Cannot synthesize leveraged ETF prices from an empty base series.")
    returns = clean.pct_change().fillna(0.0)
    data: dict[str, pd.Series] = {}
    for product in products:
        daily_return = (returns * product.leverage).clip(lower=-0.99)
        data[product.ticker] = initial_price * (1.0 + daily_return).cumprod()
    return pd.DataFrame(data, index=clean.index).rename_axis("date")


def static_weight_grid(columns: list[str], *, step: float) -> list[dict[str, float]]:
    if step <= 0 or step > 1:
        raise ValueError("grid step must be between 0 and 1.")
    units = round(1.0 / step)
    if not np.isclose(units * step, 1.0):
        raise ValueError("grid step must divide 1.0 exactly, e.g. 0.10 or 0.25.")

    weights: list[dict[str, float]] = []

    def walk(remaining: int, index: int, current: list[int]) -> None:
        if index == len(columns) - 1:
            weights.append(
                {
                    column: round(unit * step, 10)
                    for column, unit in zip(columns, [*current, remaining], strict=True)
                }
            )
            return
        for unit in range(remaining + 1):
            walk(remaining - unit, index + 1, [*current, unit])

    walk(units, 0, [])
    return weights


def trend_guard_weights(
    prices: pd.DataFrame,
    *,
    base_ticker: str,
    risk_on_ticker: str,
    defensive_ticker: str,
    window: int,
) -> pd.DataFrame:
    base = prices[base_ticker].astype(float)
    signal = (base > base.rolling(window).mean()).shift(1)
    signal = signal.where(signal.notna(), False).astype(bool)
    return _binary_target_weights(
        index=prices.index,
        risk_on_ticker=risk_on_ticker,
        defensive_ticker=defensive_ticker,
        signal=signal.astype(bool),
    )


def drawdown_guard_weights(
    prices: pd.DataFrame,
    *,
    base_ticker: str,
    risk_on_ticker: str,
    middle_ticker: str,
    severe_ticker: str,
    mild_guard: float,
    severe_guard: float,
) -> pd.DataFrame:
    base = prices[base_ticker].astype(float)
    drawdown = (base / base.cummax() - 1.0).shift(1).fillna(0.0)
    rows: list[dict[str, float]] = []
    columns = [*prices.columns, CASH]
    for value in drawdown:
        target = {column: 0.0 for column in columns}
        if value <= severe_guard + 1e-12:
            target[severe_ticker] = 1.0
        elif value <= mild_guard + 1e-12:
            target[middle_ticker] = 1.0
        else:
            target[risk_on_ticker] = 1.0
        rows.append(target)
    return pd.DataFrame(rows, index=prices.index, columns=columns)


def simulate_weighted_strategy(
    *,
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    product_leverages: dict[str, float],
    initial_cash: float,
    rebalance_dates: set[pd.Timestamp] | None = None,
    rebalance_on_change: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_prices = prices.dropna(how="any").astype(float).sort_index()
    if clean_prices.empty:
        raise ValueError("Cannot simulate strategy with empty prices.")
    weights = _normalize_weight_frame(
        target_weights,
        clean_prices.index,
        list(clean_prices.columns),
    )
    schedule = set(rebalance_dates or set())
    dates = pd.DatetimeIndex(clean_prices.index)
    product_tickers = list(clean_prices.columns)
    price_array = clean_prices.to_numpy(dtype=float)
    weight_array = weights.to_numpy(dtype=float)
    leverage_array = np.array(
        [float(product_leverages.get(ticker, 0.0)) for ticker in product_tickers],
        dtype=float,
    )
    asset_values = np.zeros(len(product_tickers), dtype=float)
    cash_value = 0.0
    current_target = np.zeros(len(weights.columns), dtype=float)
    curve_rows: list[dict[str, Any]] = []
    allocation_rows: list[dict[str, Any]] = []

    for position, date in enumerate(dates):
        target = weight_array[position]
        if position == 0:
            equity = float(initial_cash)
            asset_values, cash_value = _rebalance_values_array(
                equity,
                target,
                len(product_tickers),
            )
            current_target = target.copy()
            allocation_rows.append(
                _allocation_row_from_array(
                    date,
                    weights.columns,
                    current_target,
                    "initial allocation",
                )
            )
        else:
            price_returns = price_array[position] / price_array[position - 1] - 1.0
            asset_values = asset_values * (1.0 + price_returns)
            equity = float(asset_values.sum() + cash_value)
            target_changed = not np.allclose(target, current_target, atol=1e-10)
            if date in schedule or (rebalance_on_change and target_changed):
                asset_values, cash_value = _rebalance_values_array(
                    equity,
                    target,
                    len(product_tickers),
                )
                current_target = target.copy()
                reason = "scheduled rebalance" if date in schedule else "signal change"
                allocation_rows.append(
                    _allocation_row_from_array(date, weights.columns, target, reason)
                )

        equity = float(asset_values.sum() + cash_value)
        asset_weights = asset_values / equity if equity else asset_values * np.nan
        effective_leverage = float(np.nansum(asset_weights * leverage_array))
        curve_rows.append(
            {
                "date": date,
                "total_equity": equity,
                "cash": cash_value,
                "cash_weight": cash_value / equity if equity else np.nan,
                "effective_product_leverage": effective_leverage,
                **{
                    f"{ticker}_weight": float(asset_weights[index])
                    for index, ticker in enumerate(product_tickers)
                },
            }
        )

    curve = pd.DataFrame(curve_rows)
    curve["drawdown"] = curve["total_equity"] / curve["total_equity"].cummax() - 1.0
    return curve, pd.DataFrame(allocation_rows)


def rank_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    ranked = metrics.copy()
    ranked["rank_score"] = ranked.apply(_risk_adjusted_score, axis=1)
    ranked["rank"] = (
        ranked.sort_values(
            ["data_mode", "risk_failed", "rank_score", "calmar", "sortino"],
            ascending=[True, True, False, False, False],
        )
        .groupby("data_mode")
        .cumcount()
        + 1
    )
    return ranked.sort_values(["data_mode", "rank", "strategy_family"]).reset_index(drop=True)


def build_compare_payload(
    *,
    metrics: pd.DataFrame,
    curves: pd.DataFrame,
    products: dict[str, ProductSpec],
    top_n: int,
    scan_mode: str,
) -> dict[str, Any]:
    selected_ids = _payload_scenario_ids(metrics, top_n=top_n)
    scenarios: list[dict[str, Any]] = []
    selected_curves = curves[curves["scenario_id"].isin(selected_ids)].copy()
    for scenario_id, group in selected_curves.groupby("scenario_id", sort=False):
        metric = metrics[metrics["scenario_id"] == scenario_id].iloc[0]
        group = group.sort_values("date")
        scenarios.append(
            {
                "key": str(scenario_id),
                "short": str(metric["short_label"]),
                "full": str(metric["scenario_label"]),
                "data_mode": str(metric["data_mode"]),
                "strategy_family": str(metric["strategy_family"]),
                "risk_flag": str(metric["risk_flag"]),
                "default": bool(metric["default_selected"]),
                "dates": [pd.Timestamp(date).date().isoformat() for date in group["date"]],
                "series": {
                    "total_equity": _json_series(group["total_equity"]),
                    "normalized_equity": _json_series(
                        _normalized_series(group["total_equity"], start_value=10_000.0)
                    ),
                    "drawdown": _json_series(group["drawdown"]),
                    "effective_product_leverage": _json_series(
                        group["effective_product_leverage"]
                    ),
                    "cash_weight": _json_series(group["cash_weight"]),
                },
            }
        )
    return {
        "scan_mode": scan_mode,
        "products": {
            ticker: {"label": product.label, "leverage": product.leverage}
            for ticker, product in products.items()
        },
        "metrics": {
            "total_equity": {"label": "淨資產", "axis": "USD", "format": "money"},
            "normalized_equity": {
                "label": "標準化 10,000",
                "axis": "Normalized USD",
                "format": "money",
            },
            "drawdown": {"label": "回撤", "axis": "Drawdown", "format": "percent"},
            "effective_product_leverage": {
                "label": "產品曝險倍數",
                "axis": "Product leverage",
                "format": "number",
            },
            "cash_weight": {"label": "現金比例", "axis": "Cash weight", "format": "percent"},
        },
        "scenarios": scenarios,
    }


def write_leveraged_etf_lab_report(
    *,
    outputs: LeveragedETFLabOutputs,
    output_dir: Path,
    family: str,
    config_path: Path,
) -> LeveragedETFLabReportResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = family.lower()
    html_path = output_dir / f"leveraged_etf_{slug}.html"
    metrics_path = output_dir / f"leveraged_etf_{slug}_metrics.csv"
    payload_path = output_dir / f"leveraged_etf_{slug}_compare_payload.json"
    curves_path = output_dir / f"leveraged_etf_{slug}_curves.csv"
    allocations_path = output_dir / f"leveraged_etf_{slug}_allocations.csv"

    outputs.metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    outputs.curves.to_csv(curves_path, index=False, encoding="utf-8")
    outputs.allocations.to_csv(allocations_path, index=False, encoding="utf-8")
    payload_path.write_text(
        json.dumps(outputs.payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    html_path.write_text(
        render_leveraged_etf_lab_html(
            metrics=outputs.metrics,
            payload=outputs.payload,
            family=family,
            scan_mode=outputs.scan_mode,
            config_path=config_path,
            metrics_path=metrics_path,
            curves_path=curves_path,
            allocations_path=allocations_path,
            payload_path=payload_path,
        ),
        encoding="utf-8",
    )
    return LeveragedETFLabReportResult(
        metrics=outputs.metrics,
        curves=outputs.curves,
        allocations=outputs.allocations,
        payload=outputs.payload,
        html_path=html_path,
        metrics_path=metrics_path,
        payload_path=payload_path,
        curves_path=curves_path,
        allocations_path=allocations_path,
        scan_mode=outputs.scan_mode,
    )


def render_leveraged_etf_lab_html(
    *,
    metrics: pd.DataFrame,
    payload: dict[str, Any],
    family: str,
    scan_mode: str,
    config_path: Path,
    metrics_path: Path,
    curves_path: Path,
    allocations_path: Path,
    payload_path: Path,
) -> str:
    payload_json = _json_for_script(payload)
    generated_at = pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M:%S %Z")
    font_url = (
        "https://fonts.googleapis.com/css2?"
        "family=Noto+Sans+JP:wght@400;500;600;700&"
        "family=Noto+Sans+TC:wght@400;500;600;700&display=swap"
    )
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Leveraged ETF Lab | {escape(family.upper())}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="{font_url}" rel="stylesheet">
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>{_dashboard_css()}</style>
</head>
<body>
  <main class="dashboard-shell">
    <header class="dashboard-header">
      <div>
        <p class="eyebrow">Leveraged ETF Product Lab</p>
        <h1>{escape(family.upper())} / 2x / 3x 優化實驗室</h1>
        <p class="header-copy">
          這頁研究 QQQ / QLD / TQQQ 這類產品型槓桿 ETF。它不是 margin loan，
          沒有 debt、margin call 或維持率；最大風險來自每日重設、長期衰減與巨大回撤。
        </p>
      </div>
      <div class="header-meta">
        <span>主口徑</span>
        <strong>USD</strong>
        <span>掃描模式</span>
        <strong>{escape(scan_mode.upper())}</strong>
        <small>generated {escape(generated_at)}</small>
      </div>
    </header>

    <section class="section-block">
      <div class="section-heading">
        <div>
          <p class="eyebrow">How To Read</p>
          <h2>這頁在回答什麼</h2>
        </div>
        <p>
          固定一個標的家族時，這份報表用 QQQ / QLD / TQQQ 比較
          1x、2x、3x 產品與防守規則，幫你找出「報酬更高但沒有在壓測中爆掉」
          的候選策略。
        </p>
      </div>
      <div class="notice-card warning-notice">
        <strong>槓桿 ETF Product 不是融資槓桿</strong>
        <p>
          這裡沒有 debt、margin call 或借款利息。QLD/TQQQ 是產品本身每日重設的
          2x/3x 暴露，風險重點是巨大回撤、波動耗損與長時間無法回到前高。
        </p>
      </div>
      {_reading_steps()}
      {_glossary_cards()}
    </section>

    <section class="section-block">
      <div class="section-heading">
        <div>
          <p class="eyebrow">Decision Board</p>
          <h2>風險調整排序</h2>
        </div>
        <p>排序以 Calmar / Sortino / Sharpe 與最大回撤為主，不用 CAGR 單獨決定最佳策略。</p>
      </div>
      <div class="notice-card warning-notice">
        <strong>重要限制</strong>
        <p>Actual ETF 使用真實 QQQ/QLD/TQQQ 價格；Synthetic stress 使用 QQQ 日報酬合成 2x/3x，
        只做 2000/2008 類壓力測試，不代表實際可交易 ETF 歷史。</p>
      </div>
      {_summary_cards(metrics)}
      {_metrics_tables_by_mode(metrics)}
    </section>

    <section class="section-block compare-lab">
      <div class="section-heading">
        <div>
          <p class="eyebrow">Compare Lab</p>
          <h2>自選情境疊圖</h2>
        </div>
        <p>預設只放 baseline 與各資料模式排名前段候選；完整結果請看 metrics CSV。</p>
      </div>
      <div class="notice-card">
        <strong>目前掃描模式：{escape(scan_mode)}</strong>
        <p>
          Fast 模式用較粗權重格點快速探索；Full 模式使用完整 10% grid。
          Compare Lab 可以自由勾選策略疊圖，建議一次不要超過 6 條線。
        </p>
      </div>
      <div class="compare-layout">
        <aside class="compare-control">
          <h3>可比較情境</h3>
          <div class="compare-checkbox-list">{_compare_checkboxes(payload["scenarios"])}</div>
        </aside>
        <div class="compare-main">
          <div class="compare-toolbar">{_metric_buttons(payload["metrics"])}</div>
          <div class="compare-selection" data-compare-selection>尚未選取情境。</div>
          <div class="compare-warning" data-compare-warning></div>
          <div class="compare-chart" data-compare-chart></div>
        </div>
      </div>
      <script type="application/json" id="leveraged-etf-payload">{payload_json}</script>
    </section>

    <section class="section-block">
      <div class="section-heading">
        <div>
          <p class="eyebrow">Audit Files</p>
          <h2>輸出檔案</h2>
        </div>
        <p>HTML 只做閱讀與比較；可重跑與稽核以 CSV/JSON 為準。</p>
      </div>
      <div class="audit-links">
        <a class="audit-link" href="{escape(metrics_path.name)}">metrics CSV</a>
        <a class="audit-link" href="{escape(curves_path.name)}">curves CSV</a>
        <a class="audit-link" href="{escape(allocations_path.name)}">allocations CSV</a>
        <a class="audit-link" href="{escape(payload_path.name)}">compare payload JSON</a>
        <span class="audit-note">config: {escape(str(config_path))}</span>
      </div>
    </section>
  </main>
  {_compare_script()}
</body>
</html>
"""


def _run_mode(
    *,
    prices: pd.DataFrame,
    data_mode: str,
    products: list[ProductSpec],
    lab_config: LeveragedETFLabConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clean_prices = prices[[product.ticker for product in products]].dropna(how="any")
    if clean_prices.empty:
        raise ValueError(f"{data_mode} has no overlapping product prices.")
    product_leverages = {product.ticker: product.leverage for product in products}
    scenarios = _scenario_targets(clean_prices, products, lab_config)
    metric_rows: list[dict[str, Any]] = []
    curve_frames: list[pd.DataFrame] = []
    allocation_frames: list[pd.DataFrame] = []

    for scenario in scenarios:
        curve, allocations = simulate_weighted_strategy(
            prices=clean_prices,
            target_weights=scenario["weights"],
            product_leverages=product_leverages,
            initial_cash=lab_config.initial_cash,
            rebalance_dates=scenario["rebalance_dates"],
            rebalance_on_change=bool(scenario["rebalance_on_change"]),
        )
        scenario_id = _scenario_id(data_mode, str(scenario["name"]))
        label = str(scenario["label"])
        curve.insert(0, "scenario_id", scenario_id)
        curve.insert(1, "data_mode", data_mode)
        curve.insert(2, "scenario_label", label)
        curve.insert(3, "strategy_family", str(scenario["family"]))
        allocations.insert(0, "scenario_id", scenario_id)
        allocations.insert(1, "data_mode", data_mode)
        allocations.insert(2, "scenario_label", label)
        allocations.insert(3, "strategy_family", str(scenario["family"]))
        metric_rows.append(
            _metrics_row(
                curve=curve,
                data_mode=data_mode,
                scenario_id=scenario_id,
                scenario_label=label,
                short_label=str(scenario["short_label"]),
                strategy_family=str(scenario["family"]),
                weights_summary=str(scenario["weights_summary"]),
                lab_config=lab_config,
            )
        )
        curve_frames.append(curve)
        allocation_frames.append(allocations)

    return (
        pd.DataFrame(metric_rows),
        pd.concat(curve_frames, ignore_index=True),
        pd.concat(allocation_frames, ignore_index=True),
    )


def _scenario_targets(
    prices: pd.DataFrame,
    products: list[ProductSpec],
    lab_config: LeveragedETFLabConfig,
) -> list[dict[str, Any]]:
    product_tickers = [product.ticker for product in products]
    base_ticker = product_tickers[0]
    scenarios: list[dict[str, Any]] = []

    for product in products:
        weights = _constant_weights(prices.index, product_tickers, {product.ticker: 1.0})
        scenarios.append(
            {
                "name": f"buy_hold_{product.ticker.lower()}",
                "label": f"Buy & Hold {product.label}",
                "short_label": f"B&H {product.ticker}",
                "family": "baseline",
                "weights": weights,
                "weights_summary": product.ticker,
                "rebalance_dates": set(),
                "rebalance_on_change": False,
            }
        )

    grid_columns = [*product_tickers, CASH]
    monthly_dates = monthly_rebalance_dates(prices.index)
    for weights_dict in static_weight_grid(grid_columns, step=lab_config.grid_step):
        if weights_dict[CASH] == 1.0:
            continue
        name = "static_" + "_".join(
            f"{ticker.lower()}{int(round(weight * 100))}"
            for ticker, weight in weights_dict.items()
            if weight > 0
        )
        label = "Static Mix " + " / ".join(
            f"{ticker} {weight:.0%}" for ticker, weight in weights_dict.items() if weight > 0
        )
        scenarios.append(
            {
                "name": name,
                "label": label,
                "short_label": _short_static_label(weights_dict),
                "family": "static_mix",
                "weights": _constant_weights(prices.index, product_tickers, weights_dict),
                "weights_summary": json.dumps(weights_dict, ensure_ascii=False),
                "rebalance_dates": monthly_dates,
                "rebalance_on_change": False,
            }
        )

    for window in lab_config.trend_windows:
        for risk_on in product_tickers[1:]:
            for defensive in [base_ticker, CASH]:
                weights = trend_guard_weights(
                    prices,
                    base_ticker=base_ticker,
                    risk_on_ticker=risk_on,
                    defensive_ticker=defensive,
                    window=window,
                )
                scenarios.append(
                    {
                        "name": f"trend_{window}_{risk_on.lower()}_to_{defensive.lower()}",
                        "label": f"Trend {window}MA {risk_on} to {defensive}",
                        "short_label": f"{window}MA {risk_on}->{defensive}",
                        "family": "trend_guard",
                        "weights": weights,
                        "weights_summary": f"{window}MA {risk_on}->{defensive}",
                        "rebalance_dates": set(),
                        "rebalance_on_change": True,
                    }
                )

    mild_guard, severe_guard = lab_config.drawdown_guards
    for risk_on in product_tickers[1:]:
        middle = base_ticker if risk_on == product_tickers[1] else product_tickers[1]
        weights = drawdown_guard_weights(
            prices,
            base_ticker=base_ticker,
            risk_on_ticker=risk_on,
            middle_ticker=middle,
            severe_ticker=CASH,
            mild_guard=mild_guard,
            severe_guard=severe_guard,
        )
        scenarios.append(
            {
                "name": f"drawdown_guard_{risk_on.lower()}",
                "label": f"Drawdown Guard {risk_on} / {middle} / CASH",
                "short_label": f"DD {risk_on}->{middle}->Cash",
                "family": "drawdown_guard",
                "weights": weights,
                "weights_summary": (
                    f"{risk_on}>{middle}>CASH guards {mild_guard:.0%}/{severe_guard:.0%}"
                ),
                "rebalance_dates": set(),
                "rebalance_on_change": True,
            }
        )
    return scenarios


def monthly_rebalance_dates(index: pd.Index) -> set[pd.Timestamp]:
    trading_index = pd.DatetimeIndex(index).sort_values()
    if trading_index.empty:
        return set()
    dates = pd.Series(trading_index, index=trading_index)
    return set(pd.Timestamp(value) for value in dates.groupby(trading_index.to_period("M")).first())


def _constant_weights(
    index: pd.Index,
    product_tickers: list[str],
    weights: dict[str, float],
) -> pd.DataFrame:
    columns = [*product_tickers, CASH]
    row = {column: float(weights.get(column, 0.0)) for column in columns}
    return pd.DataFrame([row for _ in index], index=index, columns=columns)


def _binary_target_weights(
    *,
    index: pd.Index,
    risk_on_ticker: str,
    defensive_ticker: str,
    signal: pd.Series,
) -> pd.DataFrame:
    columns = sorted({risk_on_ticker, defensive_ticker, CASH} - {CASH}) + [CASH]
    rows: list[dict[str, float]] = []
    for is_risk_on in signal.reindex(index).fillna(False):
        target = {column: 0.0 for column in columns}
        target[risk_on_ticker if is_risk_on else defensive_ticker] = 1.0
        rows.append(target)
    return pd.DataFrame(rows, index=index, columns=columns)


def _normalize_weight_frame(
    weights: pd.DataFrame,
    index: pd.Index,
    product_tickers: list[str],
) -> pd.DataFrame:
    columns = [*product_tickers, CASH]
    normalized = weights.reindex(index).ffill().fillna(0.0).copy()
    for column in columns:
        if column not in normalized.columns:
            normalized[column] = 0.0
    normalized = normalized[columns].astype(float)
    totals = normalized.sum(axis=1)
    if not np.allclose(totals, 1.0, atol=1e-8):
        raise ValueError("Target weights must sum to 1.0 for every date.")
    return normalized


def _rebalance_values(
    equity: float,
    target: pd.Series,
    product_tickers: pd.Index,
) -> tuple[pd.Series, float]:
    asset_values = pd.Series(
        {
            ticker: equity * float(target.get(ticker, 0.0))
            for ticker in product_tickers
        },
        dtype="float64",
    )
    cash_value = equity * float(target.get(CASH, 0.0))
    return asset_values, cash_value


def _rebalance_values_array(
    equity: float,
    target: np.ndarray,
    product_count: int,
) -> tuple[np.ndarray, float]:
    asset_values = equity * target[:product_count]
    cash_value = equity * float(target[product_count])
    return asset_values, cash_value


def _allocation_row(date: Any, target: pd.Series, reason: str) -> dict[str, Any]:
    return {
        "date": pd.Timestamp(date),
        "reason": reason,
        "weights": json.dumps(
            {str(key): round(float(value), 6) for key, value in target.items() if value > 0},
            ensure_ascii=False,
        ),
    }


def _allocation_row_from_array(
    date: Any,
    columns: pd.Index,
    target: np.ndarray,
    reason: str,
) -> dict[str, Any]:
    return {
        "date": pd.Timestamp(date),
        "reason": reason,
        "weights": json.dumps(
            {
                str(key): round(float(value), 6)
                for key, value in zip(columns, target, strict=True)
                if value > 0
            },
            ensure_ascii=False,
        ),
    }


def _metrics_row(
    *,
    curve: pd.DataFrame,
    data_mode: str,
    scenario_id: str,
    scenario_label: str,
    short_label: str,
    strategy_family: str,
    weights_summary: str,
    lab_config: LeveragedETFLabConfig,
) -> dict[str, Any]:
    equity = curve["total_equity"].astype(float)
    returns = equity.pct_change().dropna()
    summary = performance_summary(returns) if not returns.empty else pd.Series(dtype=float)
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else np.nan
    recovery_days, recovered = _max_recovery_days(equity, curve["date"])
    risk_flag = _risk_flag(
        data_mode=data_mode,
        max_drawdown=max_drawdown,
        high_risk_drawdown=lab_config.high_risk_drawdown,
        synthetic_failure_drawdown=lab_config.synthetic_failure_drawdown,
    )
    return {
        "data_mode": data_mode,
        "scenario_id": scenario_id,
        "scenario_label": scenario_label,
        "short_label": short_label,
        "strategy_family": strategy_family,
        "weights_summary": weights_summary,
        "start_date": pd.Timestamp(curve["date"].iloc[0]).date().isoformat(),
        "end_date": pd.Timestamp(curve["date"].iloc[-1]).date().isoformat(),
        "ending_equity": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "cagr": _summary_value(summary, "cagr"),
        "volatility": _summary_value(summary, "volatility"),
        "sharpe": _summary_value(summary, "sharpe"),
        "sortino": _summary_value(summary, "sortino"),
        "calmar": _summary_value(summary, "calmar"),
        "max_drawdown": max_drawdown,
        "max_recovery_days": recovery_days,
        "recovered": recovered,
        "risk_flag": risk_flag,
        "risk_failed": risk_flag in {"high_drawdown", "synthetic_stress_failed"},
        "default_selected": False,
    }


def _risk_flag(
    *,
    data_mode: str,
    max_drawdown: float,
    high_risk_drawdown: float,
    synthetic_failure_drawdown: float,
) -> str:
    if data_mode == "synthetic_stress" and max_drawdown <= synthetic_failure_drawdown:
        return "synthetic_stress_failed"
    if max_drawdown <= high_risk_drawdown:
        return "high_drawdown"
    return "ok"


def _risk_adjusted_score(row: pd.Series) -> float:
    calmar = _safe_metric(row.get("calmar"))
    sortino = _safe_metric(row.get("sortino"))
    sharpe = _safe_metric(row.get("sharpe"))
    score = calmar + 0.25 * sortino + 0.10 * sharpe
    if row.get("risk_flag") == "high_drawdown":
        score -= 25.0
    elif row.get("risk_flag") == "synthetic_stress_failed":
        score -= 100.0
    return float(score)


def _max_recovery_days(equity: pd.Series, dates: pd.Series) -> tuple[int, bool]:
    values = equity.astype(float).reset_index(drop=True)
    date_values = pd.to_datetime(dates).reset_index(drop=True)
    peak_value = values.iloc[0]
    drawdown_start: pd.Timestamp | None = None
    max_days = 0
    recovered = True
    for value, current_date in zip(values, date_values, strict=True):
        if value >= peak_value:
            if drawdown_start is not None:
                max_days = max(max_days, int((current_date - drawdown_start).days))
                drawdown_start = None
            peak_value = value
            recovered = True
        elif drawdown_start is None:
            drawdown_start = current_date
            recovered = False
    if drawdown_start is not None:
        max_days = max(max_days, int((date_values.iloc[-1] - drawdown_start).days))
        recovered = False
    return max_days, recovered


def _payload_scenario_ids(metrics: pd.DataFrame, *, top_n: int) -> set[str]:
    baseline = set(metrics[metrics["strategy_family"] == "baseline"]["scenario_id"])
    top = set(
        metrics.sort_values(
            ["data_mode", "risk_failed", "rank_score"],
            ascending=[True, True, False],
        )
        .groupby("data_mode")
        .head(top_n)["scenario_id"]
    )
    selected = baseline | top
    default_candidates = (
        metrics[
            (metrics["data_mode"] == "actual_etf")
            & (metrics["strategy_family"].isin(["baseline", "trend_guard", "drawdown_guard"]))
        ]
        .sort_values(["strategy_family", "rank_score"], ascending=[True, False])
        .head(4)["scenario_id"]
    )
    metrics.loc[metrics["scenario_id"].isin(default_candidates), "default_selected"] = True
    return selected


def _summary_cards(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        return ""
    best_actual = _best_metric(metrics, "actual_etf")
    best_synthetic = _best_metric(metrics, "synthetic_stress")
    high_risk_count = int((metrics["risk_flag"] != "ok").sum())
    cards = [
        ("Actual ETF 最佳候選", _card_title(best_actual), _card_note(best_actual)),
        ("Synthetic 壓測最佳候選", _card_title(best_synthetic), _card_note(best_synthetic)),
        ("高風險標記", f"{high_risk_count}", "max drawdown 超過設定門檻"),
        ("排序原則", "Risk-adjusted", "Calmar / Sortino / Sharpe / drawdown"),
    ]
    return "<div class=\"kpi-grid\">" + "".join(
        f"""<article class="kpi-card">
  <span>{escape(label)}</span>
  <strong>{escape(value)}</strong>
  <small>{escape(note)}</small>
</article>"""
        for label, value, note in cards
    ) + "</div>"


def _reading_steps() -> str:
    steps = [
        (
            "1",
            "先看 Actual ETF",
            "確認真實可買產品歷史中，候選策略是否真的比 QQQ 更有吸引力。",
        ),
        (
            "2",
            "再看 Synthetic stress",
            "用合成 2x/3x 長歷史檢查 2000/2008 類崩盤時會不會承受不了。",
        ),
        (
            "3",
            "最後用 Compare Lab 疊圖",
            "自己勾選 B&H、Trend Guard、Drawdown Guard，比較淨資產、回撤與產品曝險倍數。",
        ),
    ]
    return """<div class="reading-block">
  <h3>三步閱讀法</h3>
  <div class="guide-grid">""" + "".join(
        f"""<article class="guide-card">
  <span>{escape(number)}</span>
  <strong>{escape(title)}</strong>
  <p>{escape(copy)}</p>
</article>"""
        for number, title, copy in steps
    ) + "</div></div>"


def _glossary_cards() -> str:
    terms = [
        ("Actual ETF", "真實 QQQ / QLD / TQQQ 價格，最接近可交易產品歷史。"),
        ("Synthetic stress", "用 QQQ 日報酬合成 2x/3x，只做長歷史壓力測試。"),
        ("B&H", "Buy and hold，全程持有單一產品。"),
        ("Static Mix", "固定比例配置 QQQ / QLD / TQQQ / CASH，定期再平衡。"),
        ("Trend Guard", "用 QQQ 均線判斷風險開關，跌破時降槓桿或轉防守資產。"),
        ("Drawdown Guard", "用 QQQ 回撤分層降風險，例如 TQQQ -> QLD -> CASH。"),
        ("Calmar", "CAGR 除以最大回撤絕對值，越高代表每承受一份回撤換到更多報酬。"),
        ("Sortino", "只懲罰下行波動的風險調整指標，越高越好。"),
        ("Max Drawdown", "歷史最大跌幅，槓桿 ETF 報表裡最重要的風險欄位之一。"),
        ("Recovery Days", "從跌破前高到重新回到前高的最長等待天數。"),
        ("risk_flag", "ok、high_drawdown 或 synthetic_stress_failed，用來提醒高風險候選。"),
    ]
    return """<div class="glossary-block">
  <h3>名詞卡</h3>
  <div class="term-grid">""" + "".join(
        f"""<article class="term-card">
  <strong>{escape(term)}</strong>
  <p>{escape(copy)}</p>
</article>"""
        for term, copy in terms
    ) + "</div></div>"


def _metrics_tables_by_mode(metrics: pd.DataFrame) -> str:
    return "\n".join(
        [
            _metrics_table(
                metrics,
                data_mode="actual_etf",
                title="Actual ETF 真實產品歷史",
                copy="這張表只看真實 QQQ / QLD / TQQQ 價格，不含 2000/2008 壓力測試。",
            ),
            _metrics_table(
                metrics,
                data_mode="synthetic_stress",
                title="Synthetic stress 合成壓力測試",
                copy="這張表用 QQQ 日報酬合成 2x/3x，專門檢查長歷史崩盤風險。",
            ),
        ]
    )


def _metrics_table(
    metrics: pd.DataFrame,
    *,
    data_mode: str,
    title: str,
    copy: str,
) -> str:
    if metrics.empty:
        return '<p class="empty-state">沒有可顯示的策略結果。</p>'
    display = metrics[metrics["data_mode"] == data_mode].sort_values("rank").head(12).copy()
    if display.empty:
        return ""
    columns = [
        ("rank", "排名"),
        ("scenario_label", "情境"),
        ("total_return", "總報酬"),
        ("cagr", "CAGR"),
        ("max_drawdown", "最大回撤"),
        ("calmar", "Calmar"),
        ("sortino", "Sortino"),
        ("max_recovery_days", "最長修復天數"),
        ("risk_flag", "風險標記"),
    ]
    rows = []
    for _, row in display.iterrows():
        cells = []
        for column, _label in columns:
            value = row[column]
            if column in {"total_return", "cagr", "max_drawdown"}:
                text = _format_percent(value)
            elif column in {"calmar", "sortino"}:
                text = _format_number(value)
            else:
                text = str(value)
            cells.append(f"<td>{escape(text)}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    headers = "".join(f"<th>{escape(label)}</th>" for _, label in columns)
    return f"""<section class="mode-board">
  <div class="mode-board-heading">
    <h3>{escape(title)}</h3>
    <p>{escape(copy)}</p>
  </div>
  <div class="table-wrap">
  <table>
    <thead><tr>{headers}</tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</div>
</section>"""


def _compare_checkboxes(scenarios: list[dict[str, Any]]) -> str:
    labels = []
    for scenario in scenarios:
        checked = " checked" if scenario.get("default") else ""
        risk = f" · {scenario['risk_flag']}" if scenario.get("risk_flag") != "ok" else ""
        labels.append(
            f"""<label class="compare-option">
  <input type="checkbox" data-compare-checkbox value="{escape(str(scenario["key"]))}"{checked}>
  <span>
    <strong>{escape(str(scenario["short"]))}</strong>
    <small>
      {escape(str(scenario["data_mode"]))}{escape(risk)}
      · {escape(str(scenario["full"]))}
    </small>
  </span>
</label>"""
        )
    return "\n".join(labels)


def _metric_buttons(metrics: dict[str, dict[str, str]]) -> str:
    buttons = []
    for index, (metric, config) in enumerate(metrics.items()):
        active = " is-active" if index == 0 else ""
        buttons.append(
            f"""<button class="metric-button{active}" type="button"
  data-compare-metric="{escape(metric)}">{escape(config["label"])}</button>"""
        )
    return "\n".join(buttons)


def _compare_script() -> str:
    return """<script>
const payloadElement = document.getElementById("leveraged-etf-payload");
const chart = document.querySelector("[data-compare-chart]");
const checkboxes = Array.from(document.querySelectorAll("[data-compare-checkbox]"));
const metricButtons = Array.from(document.querySelectorAll("[data-compare-metric]"));
const selectedText = document.querySelector("[data-compare-selection]");
const warning = document.querySelector("[data-compare-warning]");
const payload = payloadElement
  ? JSON.parse(payloadElement.textContent)
  : { metrics: {}, scenarios: [] };
const scenarioMap = new Map(payload.scenarios.map((scenario) => [scenario.key, scenario]));
let activeMetric = "total_equity";

function redrawCompareChart() {
  if (!window.Plotly || !chart) return;
  const selectedKeys = checkboxes
    .filter((checkbox) => checkbox.checked)
    .map((checkbox) => checkbox.value);
  const metricConfig = payload.metrics[activeMetric] || {};
  const traces = selectedKeys
    .map((key) => scenarioMap.get(key))
    .filter(Boolean)
    .map((scenario) => ({
      x: scenario.dates,
      y: scenario.series[activeMetric],
      mode: "lines",
      type: "scatter",
      name: scenario.short,
      hovertemplate: [
        scenario.full,
        scenario.data_mode,
        "%{x}",
        `${metricConfig.label}: %{y}<extra></extra>`,
      ].join("<br>"),
    }));
  const layout = {
    template: "plotly_white",
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    height: 460,
    margin: { l: 64, r: 28, t: 24, b: 54 },
    hovermode: "x unified",
    showlegend: true,
    legend: { orientation: "h", y: 1.14, x: 0, font: { size: 11 } },
    font: { family: "Noto Sans TC, Noto Sans JP, Segoe UI, sans-serif", color: "#202521" },
    xaxis: { showgrid: false, zeroline: false },
    yaxis: { title: metricConfig.axis || activeMetric, gridcolor: "#e6e8e1", zeroline: false },
  };
  if (["percent", "drawdown"].includes(metricConfig.format)) {
    layout.yaxis.tickformat = ".0%";
  }
  Plotly.react(chart, traces, layout, { displaylogo: false, responsive: true });
  if (selectedText) {
    selectedText.textContent = selectedKeys.length
      ? `已選 ${selectedKeys.length} 個情境：${
          selectedKeys.map((key) => scenarioMap.get(key)?.short).join("、")
        }`
      : "尚未選取情境。";
  }
  if (warning) {
    warning.textContent = selectedKeys.length > 6
      ? "已超過建議最多 6 條線，可繼續比較，但建議縮小範圍。"
      : "";
  }
}

checkboxes.forEach((checkbox) => checkbox.addEventListener("change", redrawCompareChart));
metricButtons.forEach((button) => {
  button.addEventListener("click", () => {
    activeMetric = button.dataset.compareMetric;
    metricButtons.forEach((item) => item.classList.toggle("is-active", item === button));
    redrawCompareChart();
  });
});
redrawCompareChart();
</script>"""


def _dashboard_css() -> str:
    return """
:root {
  --bg: #f7f6f1;
  --surface: #ffffff;
  --surface-soft: #eeefe8;
  --ink: #202521;
  --muted: #66706a;
  --line: #d8ddd3;
  --indigo: #3f5f73;
  --sage: #6f8375;
  --copper: #b66f52;
  --shadow: 0 14px 36px rgba(32, 37, 33, 0.07);
}

* { box-sizing: border-box; }

html {
  background: var(--bg);
  color: var(--ink);
  font-family: "Noto Sans TC", "Noto Sans JP", "Segoe UI", "Microsoft JhengHei", sans-serif;
  letter-spacing: 0;
}

body {
  margin: 0;
  background:
    linear-gradient(180deg, rgba(63, 95, 115, 0.08), rgba(247, 246, 241, 0) 420px),
    var(--bg);
}

.dashboard-shell {
  width: min(1440px, calc(100% - 32px));
  margin: 0 auto;
  padding: 28px 0 56px;
}

.dashboard-header,
.section-block {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.9);
  box-shadow: var(--shadow);
}

.dashboard-header {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 24px;
  align-items: end;
  padding: 28px;
}

.dashboard-header h1 {
  margin: 6px 0 10px;
  font-size: clamp(2rem, 4vw, 3.2rem);
  line-height: 1.05;
}

.header-copy,
.section-heading p,
.notice-card p,
.compare-option small,
.kpi-card small,
.compare-selection {
  color: var(--muted);
  line-height: 1.6;
}

.header-meta {
  min-width: 190px;
  padding: 16px;
  border-left: 3px solid var(--sage);
  border-radius: 8px;
  background: var(--surface-soft);
}

.header-meta span,
.header-meta strong,
.header-meta small,
.eyebrow,
.kpi-card span {
  display: block;
}

.header-meta small {
  margin-top: 8px;
  color: var(--muted);
}

.eyebrow {
  margin: 0;
  color: var(--indigo);
  font-size: 0.78rem;
  font-weight: 700;
}

.section-block {
  margin-top: 18px;
  padding: 24px;
}

.section-heading {
  display: flex;
  justify-content: space-between;
  gap: 24px;
  align-items: end;
  margin-bottom: 18px;
}

.section-heading h2 {
  margin: 4px 0 0;
  font-size: 1.25rem;
}

.section-heading p {
  max-width: 680px;
  margin: 0;
}

.notice-card,
.kpi-card,
.compare-control,
.compare-main {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfbf7;
}

.notice-card {
  padding: 15px;
  margin-bottom: 14px;
}

.warning-notice {
  border-color: rgba(182, 111, 82, 0.45);
  background: #fff8f3;
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin-bottom: 14px;
}

.guide-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin-bottom: 16px;
}

.guide-card,
.term-card,
.mode-board {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #ffffff;
}

.guide-card {
  padding: 16px;
}

.guide-card span {
  display: inline-grid;
  place-items: center;
  width: 28px;
  height: 28px;
  margin-bottom: 12px;
  border-radius: 50%;
  background: var(--indigo);
  color: #ffffff;
  font-weight: 700;
}

.guide-card strong,
.term-card strong {
  display: block;
}

.guide-card p,
.term-card p,
.mode-board-heading p {
  margin: 8px 0 0;
  color: var(--muted);
  line-height: 1.55;
}

.reading-block,
.glossary-block {
  margin-top: 8px;
}

.reading-block h3,
.glossary-block h3,
.compare-control h3,
.mode-board h3 {
  margin: 0 0 10px;
}

.term-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
}

.term-card {
  padding: 12px;
}

.term-card p {
  font-size: 0.88rem;
}

.mode-board {
  margin-top: 14px;
  overflow: hidden;
}

.mode-board-heading {
  padding: 14px 16px;
  border-bottom: 1px solid var(--line);
  background: #f8f8f3;
}

.kpi-card {
  padding: 16px;
  background: linear-gradient(180deg, #ffffff, #fafaf6);
}

.kpi-card strong {
  display: block;
  margin-top: 8px;
  font-size: clamp(1.08rem, 2vw, 1.55rem);
}

.table-wrap {
  width: 100%;
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88rem;
}

th,
td {
  padding: 10px 11px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  white-space: nowrap;
}

th {
  color: var(--muted);
  background: #f4f5ee;
}

td:nth-child(3) {
  min-width: 260px;
  white-space: normal;
}

.compare-layout {
  display: grid;
  grid-template-columns: minmax(300px, 0.34fr) minmax(0, 1fr);
  gap: 14px;
  align-items: start;
}

.compare-control {
  max-height: 660px;
  overflow: auto;
  padding: 14px;
}

.compare-checkbox-list {
  display: grid;
  gap: 8px;
}

.compare-option {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: 9px;
  align-items: start;
  padding: 10px;
  border: 1px solid transparent;
  border-radius: 8px;
  background: var(--surface);
  cursor: pointer;
}

.compare-option:hover {
  border-color: var(--indigo);
}

.compare-option input {
  margin-top: 3px;
  accent-color: var(--indigo);
}

.compare-option strong,
.compare-option small {
  display: block;
}

.compare-main {
  min-width: 0;
  padding: 14px;
}

.compare-toolbar,
.audit-links {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 10px;
}

.metric-button,
.audit-link,
.audit-note {
  min-height: 34px;
  padding: 7px 11px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface);
  color: var(--ink);
  font: inherit;
  font-size: 0.88rem;
  text-decoration: none;
}

.metric-button {
  cursor: pointer;
}

.metric-button:hover,
.metric-button.is-active,
.audit-link:hover {
  border-color: var(--indigo);
  background: #f1f5f2;
  color: var(--indigo);
}

.compare-warning {
  min-height: 1.4em;
  margin-top: 4px;
  color: var(--copper);
  font-weight: 700;
}

.compare-chart {
  min-height: 460px;
  margin-top: 8px;
}

.empty-state {
  margin: 0;
  color: var(--muted);
}

@media (max-width: 980px) {
  .dashboard-header,
  .section-heading {
    display: block;
  }

  .header-meta,
  .section-heading p {
    margin-top: 14px;
  }

  .kpi-grid,
  .guide-grid,
  .term-grid,
  .compare-layout {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 640px) {
  .dashboard-shell {
    width: min(100% - 20px, 1440px);
    padding-top: 10px;
  }

  .dashboard-header,
  .section-block {
    padding: 18px;
  }

  .kpi-grid,
  .guide-grid,
  .term-grid,
  .compare-layout {
    grid-template-columns: 1fr;
  }
}
"""


def _best_metric(metrics: pd.DataFrame, data_mode: str) -> pd.Series | None:
    selected = metrics[metrics["data_mode"] == data_mode].sort_values("rank")
    if selected.empty:
        return None
    return selected.iloc[0]


def _card_title(row: pd.Series | None) -> str:
    return "無資料" if row is None else str(row["short_label"])


def _card_note(row: pd.Series | None) -> str:
    if row is None:
        return ""
    return f"Calmar {_format_number(row['calmar'])} / MDD {_format_percent(row['max_drawdown'])}"


def _summary_value(summary: pd.Series, key: str) -> float:
    value = summary.get(key, np.nan)
    return np.nan if pd.isna(value) else float(value)


def _safe_metric(value: Any) -> float:
    if pd.isna(value) or value in (np.inf, -np.inf):
        return 0.0
    return float(value)


def _normalized_series(values: pd.Series, *, start_value: float) -> pd.Series:
    series = values.astype(float).replace([np.inf, -np.inf], np.nan)
    valid = series.dropna()
    if valid.empty or valid.iloc[0] == 0:
        return pd.Series(np.nan, index=series.index)
    return series / valid.iloc[0] * start_value


def _json_series(values: Any) -> list[float | None]:
    output: list[float | None] = []
    for value in pd.Series(values):
        if pd.isna(value) or value in (np.inf, -np.inf):
            output.append(None)
        else:
            output.append(round(float(value), 6))
    return output


def _json_for_script(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def _scenario_id(data_mode: str, name: str) -> str:
    raw = f"{data_mode}__{name}"
    return "".join(ch if ch.isalnum() else "-" for ch in raw).strip("-").lower()


def _short_static_label(weights: dict[str, float]) -> str:
    active = [f"{ticker}{weight:.0%}" for ticker, weight in weights.items() if weight > 0]
    return "Static " + "/".join(active)


def _format_percent(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def _format_number(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2f}"


__all__ = [
    "CASH",
    "LeveragedETFLabOutputs",
    "LeveragedETFLabReportResult",
    "ProductSpec",
    "build_compare_payload",
    "build_leveraged_etf_lab_outputs",
    "drawdown_guard_weights",
    "lab_config_for_scan_mode",
    "monthly_rebalance_dates",
    "rank_metrics",
    "render_leveraged_etf_lab_html",
    "resolve_scan_mode",
    "simulate_weighted_strategy",
    "static_weight_grid",
    "synthetic_daily_reset_prices",
    "trend_guard_weights",
    "write_leveraged_etf_lab_report",
]
