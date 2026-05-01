from __future__ import annotations

import json
from dataclasses import dataclass, replace
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from investment_backtest_lab.leveraged_etf_lab import (
    CASH,
    ProductSpec,
    monthly_rebalance_dates,
    simulate_weighted_strategy,
    xirr,
)
from investment_backtest_lab.models import DCAPolicyOptimizerConfig
from investment_backtest_lab.reports import performance_summary

DATA_MODE_ACTUAL = "actual_etf"
DATA_MODE_SYNTHETIC = "synthetic_stress"
VALIDATION_STABLE = "stable"
VALIDATION_WATCHLIST = "watchlist"
VALIDATION_FRAGILE = "fragile"
VALIDATION_UNVALIDATED = "unvalidated"
DEFAULT_TRAIN_YEARS = 5
DEFAULT_TEST_YEARS = 2
DEFAULT_STEP_YEARS = 1


@dataclass(frozen=True)
class PolicyScenarioSpec:
    name: str
    label: str
    short_label: str
    family: str
    kind: str
    params: dict[str, Any]


@dataclass(frozen=True)
class DCAPolicyOptimizerOutputs:
    metrics: pd.DataFrame
    curves: pd.DataFrame
    policy: pd.DataFrame
    walk_forward: pd.DataFrame
    current_signal: pd.DataFrame
    payload: dict[str, Any]
    scan_mode: str


@dataclass(frozen=True)
class DCAPolicyOptimizerReportResult:
    metrics: pd.DataFrame
    curves: pd.DataFrame
    policy: pd.DataFrame
    walk_forward: pd.DataFrame
    current_signal: pd.DataFrame
    payload: dict[str, Any]
    html_path: Path
    metrics_path: Path
    policy_path: Path
    walk_forward_path: Path
    current_signal_path: Path
    payload_path: Path
    scan_mode: str


def policy_config_for_scan_mode(
    config: DCAPolicyOptimizerConfig,
    scan_mode: str,
) -> DCAPolicyOptimizerConfig:
    normalized = scan_mode.lower()
    if normalized == "full":
        return config
    if normalized == "fast":
        trend_windows = tuple(value for value in config.trend_windows if value in {100, 200})
        momentum_windows = tuple(
            value for value in config.momentum_windows if value in {126, 252}
        )
        volatility_windows = tuple(value for value in config.volatility_windows if value == 63)
        volatility_targets = tuple(
            value for value in config.volatility_targets if value in {0.25, 0.35}
        )
        target_grid = tuple(value for value in config.target_leverage_grid if value in {0, 1, 2, 3})
        return replace(
            config,
            trend_windows=trend_windows or config.trend_windows[:2],
            momentum_windows=momentum_windows or config.momentum_windows[:2],
            volatility_windows=volatility_windows or config.volatility_windows[:1],
            volatility_targets=volatility_targets or config.volatility_targets[:2],
            target_leverage_grid=target_grid or config.target_leverage_grid,
            top_n=config.fast_top_n,
        )
    raise ValueError(f"scan_mode must be 'fast' or 'full', got {scan_mode!r}.")


def target_leverage_to_product_weights(
    target_leverage: float,
    products: list[ProductSpec],
) -> dict[str, float]:
    product_points = sorted(
        ((product.ticker, product.leverage) for product in products),
        key=lambda item: item[1],
    )
    points = [(CASH, 0.0), *product_points]
    target = min(max(float(target_leverage), 0.0), float(points[-1][1]))
    for (low_ticker, low_leverage), (high_ticker, high_leverage) in zip(
        points[:-1], points[1:], strict=True
    ):
        if low_leverage <= target <= high_leverage:
            if np.isclose(high_leverage, low_leverage):
                high_weight = 1.0
            else:
                high_weight = (target - low_leverage) / (high_leverage - low_leverage)
            low_weight = 1.0 - high_weight
            weights = {product.ticker: 0.0 for product in products}
            weights[CASH] = 0.0
            weights[low_ticker] = weights.get(low_ticker, 0.0) + low_weight
            weights[high_ticker] = weights.get(high_ticker, 0.0) + high_weight
            return {key: round(float(value), 12) for key, value in weights.items()}
    raise ValueError(f"Cannot map target leverage {target_leverage} to product weights.")


def build_dca_policy_optimizer_outputs(
    *,
    actual_prices: pd.DataFrame,
    synthetic_prices: pd.DataFrame,
    products: list[ProductSpec],
    config: DCAPolicyOptimizerConfig,
    scan_mode: str,
) -> DCAPolicyOptimizerOutputs:
    specs = build_policy_scenario_specs(config=config, products=products, scan_mode=scan_mode)
    product_leverages = {product.ticker: product.leverage for product in products}
    mode_prices: dict[str, pd.DataFrame] = {}
    frames: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    if not actual_prices.empty:
        clean = _clean_prices(actual_prices, products)
        mode_prices[DATA_MODE_ACTUAL] = clean
        frames.append(
            _run_data_mode(
                prices=clean,
                data_mode=DATA_MODE_ACTUAL,
                specs=specs,
                products=products,
                product_leverages=product_leverages,
                config=config,
            )
        )
    if not synthetic_prices.empty:
        clean = _clean_prices(synthetic_prices, products)
        mode_prices[DATA_MODE_SYNTHETIC] = clean
        frames.append(
            _run_data_mode(
                prices=clean,
                data_mode=DATA_MODE_SYNTHETIC,
                specs=specs,
                products=products,
                product_leverages=product_leverages,
                config=config,
            )
        )
    if not frames:
        raise ValueError("DCA policy optimizer requires actual or synthetic prices.")

    metrics = pd.concat([item[0] for item in frames], ignore_index=True)
    curves = pd.concat([item[1] for item in frames], ignore_index=True)
    policy = pd.concat([item[2] for item in frames], ignore_index=True)
    walk_forward = build_policy_walk_forward_validation(
        mode_prices=mode_prices,
        specs=specs,
        products=products,
        product_leverages=product_leverages,
        config=config,
    )
    metrics = apply_policy_validation(metrics, walk_forward, config=config)
    metrics = apply_cross_mode_candidate_filter(metrics, config=config)
    metrics = rank_policy_metrics(metrics)
    current_signal = build_current_signal(metrics=metrics, policy=policy, top_n=config.top_n)
    payload = build_policy_compare_payload(
        metrics=metrics,
        curves=curves,
        top_n=config.top_n,
        scan_mode=scan_mode,
    )
    return DCAPolicyOptimizerOutputs(
        metrics=metrics,
        curves=curves,
        policy=policy,
        walk_forward=walk_forward,
        current_signal=current_signal,
        payload=payload,
        scan_mode=scan_mode,
    )


def build_policy_scenario_specs(
    *,
    config: DCAPolicyOptimizerConfig,
    products: list[ProductSpec],
    scan_mode: str,
) -> list[PolicyScenarioSpec]:
    del scan_mode
    grid = tuple(sorted(set(float(value) for value in config.target_leverage_grid)))
    risk_on_values = tuple(value for value in grid if value > 1.0)
    defensive_values = tuple(value for value in grid if value in {0.0, 1.0})
    specs: list[PolicyScenarioSpec] = []
    for target in grid:
        specs.append(
            PolicyScenarioSpec(
                name=f"constant_{_leverage_slug(target)}",
                label=f"Constant {target:.1f}x",
                short_label=f"{target:.1f}x",
                family="constant_leverage",
                kind="constant",
                params={"target": target},
            )
        )
    for window in config.trend_windows:
        for risk_on in risk_on_values:
            for risk_off in defensive_values:
                specs.append(
                    PolicyScenarioSpec(
                        name=(
                            f"trend_ladder_{window}_"
                            f"{_leverage_slug(risk_on)}_to_{_leverage_slug(risk_off)}"
                        ),
                        label=f"Trend Ladder {window}MA {risk_on:.1f}x to {risk_off:.1f}x",
                        short_label=f"{window}MA {risk_on:.1f}->{risk_off:.1f}",
                        family="trend_ladder",
                        kind="trend_ladder",
                        params={
                            "window": window,
                            "risk_on": risk_on,
                            "risk_off": risk_off,
                        },
                    )
                )
    for risk_on in risk_on_values:
        specs.append(
            PolicyScenarioSpec(
                name=f"drawdown_ladder_{_leverage_slug(risk_on)}",
                label=f"Drawdown Ladder max {risk_on:.1f}x",
                short_label=f"DD ladder {risk_on:.1f}x",
                family="drawdown_ladder",
                kind="drawdown_ladder",
                params={"risk_on": risk_on},
            )
        )
    for window in config.volatility_windows:
        for target_volatility in config.volatility_targets:
            specs.append(
                PolicyScenarioSpec(
                    name=f"vol_target_{window}_{int(round(target_volatility * 100))}",
                    label=f"Vol Target {window}D {target_volatility:.0%}",
                    short_label=f"Vol {window}D {target_volatility:.0%}",
                    family="vol_target_ladder",
                    kind="vol_target_ladder",
                    params={"window": window, "target_volatility": target_volatility},
                )
            )
    for momentum_window in config.momentum_windows:
        for trend_window in config.trend_windows:
            for risk_on in risk_on_values:
                for risk_off in defensive_values:
                    specs.append(
                        PolicyScenarioSpec(
                            name=(
                                f"momentum_trend_{momentum_window}_{trend_window}_"
                                f"{_leverage_slug(risk_on)}_to_{_leverage_slug(risk_off)}"
                            ),
                            label=(
                                f"Momentum+Trend {momentum_window}D/{trend_window}MA "
                                f"{risk_on:.1f}x to {risk_off:.1f}x"
                            ),
                            short_label=(
                                f"Mom {momentum_window}/{trend_window} "
                                f"{risk_on:.1f}->{risk_off:.1f}"
                            ),
                            family="momentum_trend_ladder",
                            kind="momentum_trend_ladder",
                            params={
                                "momentum_window": momentum_window,
                                "trend_window": trend_window,
                                "risk_on": risk_on,
                                "risk_off": risk_off,
                            },
                        )
                    )
    return specs


def build_policy_walk_forward_validation(
    *,
    mode_prices: dict[str, pd.DataFrame],
    specs: list[PolicyScenarioSpec],
    products: list[ProductSpec],
    product_leverages: dict[str, float],
    config: DCAPolicyOptimizerConfig,
    train_years: int = DEFAULT_TRAIN_YEARS,
    test_years: int = DEFAULT_TEST_YEARS,
    step_years: int = DEFAULT_STEP_YEARS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for data_mode, prices in mode_prices.items():
        folds = _walk_forward_folds(
            prices.index,
            train_years=train_years,
            test_years=test_years,
            step_years=step_years,
        )
        for fold_index, (train_start, train_end, test_start, test_end) in enumerate(
            folds,
            start=1,
        ):
            train_prices = prices.loc[(prices.index >= train_start) & (prices.index <= train_end)]
            test_prices = prices.loc[(prices.index >= test_start) & (prices.index <= test_end)]
            if len(train_prices) < 252 or len(test_prices) < 120:
                continue
            train_metrics, _, _ = _run_data_mode(
                prices=train_prices,
                data_mode=data_mode,
                specs=specs,
                products=products,
                product_leverages=product_leverages,
                config=config,
            )
            train_ranked = rank_policy_metrics(
                apply_policy_validation(train_metrics, pd.DataFrame(), config=config)
            ).head(config.walk_forward_top_n)
            selected = {
                str(row.scenario_id).split("--", maxsplit=1)[1]: int(row.rank)
                for row in train_ranked.itertuples()
            }
            test_specs = [spec for spec in specs if f"dca-policy-{spec.name}" in selected]
            if not test_specs:
                continue
            test_metrics, _, _ = _run_data_mode(
                prices=test_prices,
                data_mode=data_mode,
                specs=test_specs,
                products=products,
                product_leverages=product_leverages,
                config=config,
            )
            for row in test_metrics.itertuples():
                spec_key = str(row.scenario_id).split("--", maxsplit=1)[1]
                passed = bool(
                    not row.risk_failed
                    and pd.notna(row.xirr)
                    and float(row.xirr) > 0.0
                    and float(row.max_drawdown) >= config.max_drawdown_limit
                )
                rows.append(
                    {
                        "data_mode": data_mode,
                        "fold_index": fold_index,
                        "scenario_id": row.scenario_id,
                        "scenario_label": row.scenario_label,
                        "strategy_family": row.strategy_family,
                        "train_start": train_start.date().isoformat(),
                        "train_end": train_end.date().isoformat(),
                        "test_start": test_start.date().isoformat(),
                        "test_end": test_end.date().isoformat(),
                        "train_rank": selected.get(spec_key),
                        "test_total_contributed": row.total_contributed,
                        "test_ending_equity": row.ending_equity,
                        "test_simple_cash_return": row.simple_cash_return,
                        "test_xirr": row.xirr,
                        "test_max_drawdown": row.max_drawdown,
                        "test_recovery_days": row.recovery_days,
                        "test_risk_flag": row.risk_flag,
                        "test_risk_failed": row.risk_failed,
                        "passed_fold": passed,
                    }
                )
    return pd.DataFrame(rows)


def apply_policy_validation(
    metrics: pd.DataFrame,
    walk_forward: pd.DataFrame,
    *,
    config: DCAPolicyOptimizerConfig,
) -> pd.DataFrame:
    result = metrics.copy()
    result["validation_status"] = VALIDATION_UNVALIDATED
    result["validation_folds"] = 0
    result["walk_forward_pass_rate"] = np.nan
    result["mean_test_xirr"] = np.nan
    result["worst_test_drawdown"] = np.nan
    result["worst_test_recovery_days"] = np.nan
    if not walk_forward.empty:
        summary = _validation_summary(walk_forward, config=config)
        result = result.merge(summary, on="scenario_id", how="left", suffixes=("", "_wf"))
        for column in [
            "validation_status",
            "validation_folds",
            "walk_forward_pass_rate",
            "mean_test_xirr",
            "worst_test_drawdown",
            "worst_test_recovery_days",
        ]:
            wf_column = f"{column}_wf"
            if wf_column in result.columns:
                result[column] = result[wf_column].combine_first(result[column])
                result = result.drop(columns=[wf_column])
    result.loc[result["risk_failed"].astype(bool), "validation_status"] = VALIDATION_FRAGILE
    result["eligible_for_candidate"] = (
        (~result["risk_failed"].astype(bool))
        & (result["validation_status"] == VALIDATION_STABLE)
    )
    return result


def apply_cross_mode_candidate_filter(
    metrics: pd.DataFrame,
    *,
    config: DCAPolicyOptimizerConfig,
) -> pd.DataFrame:
    result = metrics.copy()
    result["policy_key"] = result["scenario_id"].map(_policy_key)
    result["cross_mode_worst_drawdown"] = result.groupby("policy_key")["max_drawdown"].transform(
        "min"
    )
    result["cross_mode_drawdown_breach"] = (
        result["cross_mode_worst_drawdown"] < config.max_drawdown_limit
    )
    result["eligible_for_candidate"] = (
        result["eligible_for_candidate"].astype(bool)
        & (~result["cross_mode_drawdown_breach"].astype(bool))
    )
    return result


def rank_policy_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    ranked = metrics.copy()
    if "validation_status" not in ranked.columns:
        ranked["validation_status"] = VALIDATION_UNVALIDATED
    ranked["policy_score"] = ranked.apply(_policy_score, axis=1)
    ranked["_validation_rank"] = ranked["validation_status"].map(_validation_rank).fillna(9)
    ranked = ranked.sort_values(
        [
            "data_mode",
            "risk_failed",
            "_validation_rank",
            "policy_score",
            "xirr",
            "max_drawdown",
        ],
        ascending=[True, True, True, False, False, False],
    )
    ranked["rank"] = ranked.groupby("data_mode").cumcount() + 1
    return ranked.drop(columns=["_validation_rank"]).reset_index(drop=True)


def build_current_signal(
    *,
    metrics: pd.DataFrame,
    policy: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    if metrics.empty or policy.empty:
        return pd.DataFrame()
    actual = metrics[metrics["data_mode"] == DATA_MODE_ACTUAL].copy()
    source = actual if not actual.empty else metrics.copy()
    eligible = source[source["eligible_for_candidate"].astype(bool)]
    selected = eligible if not eligible.empty else source[~source["risk_failed"].astype(bool)]
    if selected.empty:
        selected = source
    selected = selected.sort_values("rank").head(top_n)
    rows: list[dict[str, Any]] = []
    for metric in selected.itertuples():
        scenario_policy = policy[policy["scenario_id"] == metric.scenario_id].sort_values("date")
        if scenario_policy.empty:
            continue
        latest = scenario_policy.iloc[-1]
        rows.append(
            {
                "as_of_date": pd.Timestamp(latest["date"]).date().isoformat(),
                "data_mode": metric.data_mode,
                "scenario_id": metric.scenario_id,
                "scenario_label": metric.scenario_label,
                "rank": metric.rank,
                "validation_status": metric.validation_status,
                "regime": latest["regime"],
                "reason": latest["reason"],
                "target_effective_leverage": latest["target_effective_leverage"],
                "QQQ_weight": latest.get("QQQ_weight", np.nan),
                "QLD_weight": latest.get("QLD_weight", np.nan),
                "TQQQ_weight": latest.get("TQQQ_weight", np.nan),
                "CASH_weight": latest.get("CASH_weight", np.nan),
                "xirr": metric.xirr,
                "max_drawdown": metric.max_drawdown,
            }
        )
    return pd.DataFrame(rows)


def build_policy_compare_payload(
    *,
    metrics: pd.DataFrame,
    curves: pd.DataFrame,
    top_n: int,
    scan_mode: str,
) -> dict[str, Any]:
    selected_ids = (
        metrics.sort_values(["data_mode", "rank"])
        .groupby("data_mode")
        .head(top_n)["scenario_id"]
        .tolist()
    )
    scenarios: list[dict[str, Any]] = []
    for scenario_id, group in curves[curves["scenario_id"].isin(selected_ids)].groupby(
        "scenario_id",
        sort=False,
    ):
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
                "validation_status": str(metric["validation_status"]),
                "default": bool(metric["rank"] <= 3 and metric["data_mode"] == DATA_MODE_ACTUAL),
                "dates": [pd.Timestamp(value).date().isoformat() for value in group["date"]],
                "series": {
                    "total_equity": _json_series(group["total_equity"]),
                    "normalized_equity": _json_series(group["return_index"] * 10_000.0),
                    "drawdown": _json_series(group["drawdown"]),
                    "effective_leverage": _json_series(group["effective_product_leverage"]),
                    "total_contributed": _json_series(group["total_contributed"]),
                },
            }
        )
    return {
        "scan_mode": scan_mode,
        "metrics": {
            "total_equity": {"label": "Net equity USD", "axis": "USD", "format": "money"},
            "normalized_equity": {
                "label": "Normalized equity",
                "axis": "USD",
                "format": "money",
            },
            "drawdown": {"label": "Drawdown", "axis": "Drawdown", "format": "percent"},
            "effective_leverage": {
                "label": "Effective leverage",
                "axis": "Leverage",
                "format": "number",
            },
            "total_contributed": {
                "label": "Cumulative contributed",
                "axis": "USD",
                "format": "money",
            },
        },
        "scenarios": scenarios,
    }


def write_dca_policy_optimizer_report(
    *,
    outputs: DCAPolicyOptimizerOutputs,
    output_dir: Path,
    family: str,
    config_path: Path,
) -> DCAPolicyOptimizerReportResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"dca_policy_optimizer_{family}"
    html_path = output_dir / f"{prefix}.html"
    metrics_path = output_dir / f"{prefix}_metrics.csv"
    policy_path = output_dir / f"{prefix}_policy.csv"
    walk_forward_path = output_dir / f"{prefix}_walk_forward.csv"
    current_signal_path = output_dir / f"{prefix}_current_signal.csv"
    payload_path = output_dir / f"{prefix}_compare_payload.json"
    outputs.metrics.to_csv(metrics_path, index=False)
    outputs.policy.to_csv(policy_path, index=False)
    outputs.walk_forward.to_csv(walk_forward_path, index=False)
    outputs.current_signal.to_csv(current_signal_path, index=False)
    payload_path.write_text(
        json.dumps(outputs.payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    html_path.write_text(
        render_dca_policy_optimizer_html(
            metrics=outputs.metrics,
            walk_forward=outputs.walk_forward,
            current_signal=outputs.current_signal,
            payload=outputs.payload,
            metrics_path=metrics_path,
            policy_path=policy_path,
            walk_forward_path=walk_forward_path,
            current_signal_path=current_signal_path,
            payload_path=payload_path,
            config_path=config_path,
            scan_mode=outputs.scan_mode,
        ),
        encoding="utf-8",
    )
    return DCAPolicyOptimizerReportResult(
        metrics=outputs.metrics,
        curves=outputs.curves,
        policy=outputs.policy,
        walk_forward=outputs.walk_forward,
        current_signal=outputs.current_signal,
        payload=outputs.payload,
        html_path=html_path,
        metrics_path=metrics_path,
        policy_path=policy_path,
        walk_forward_path=walk_forward_path,
        current_signal_path=current_signal_path,
        payload_path=payload_path,
        scan_mode=outputs.scan_mode,
    )


def render_dca_policy_optimizer_html(
    *,
    metrics: pd.DataFrame,
    walk_forward: pd.DataFrame,
    current_signal: pd.DataFrame,
    payload: dict[str, Any],
    metrics_path: Path,
    policy_path: Path,
    walk_forward_path: Path,
    current_signal_path: Path,
    payload_path: Path,
    config_path: Path,
    scan_mode: str,
) -> str:
    actual = metrics[metrics["data_mode"] == DATA_MODE_ACTUAL].head(12)
    synthetic = metrics[metrics["data_mode"] == DATA_MODE_SYNTHETIC].head(12)
    best = metrics[metrics["eligible_for_candidate"].astype(bool)].head(8)
    if best.empty:
        best = metrics[~metrics["risk_failed"].astype(bool)].head(8)
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DCA Policy Optimizer</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link
    href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&family=Noto+Sans+JP:wght@400;500;700&display=swap"
    rel="stylesheet">
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --paper: #f7f5ef;
      --surface: #fffffc;
      --ink: #222420;
      --muted: #66706a;
      --line: #d9d6cb;
      --indigo: #53677f;
      --sage: #6f8574;
      --copper: #b8794f;
      --danger: #9b4b45;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: "Noto Sans TC", "Noto Sans JP", system-ui, sans-serif;
      line-height: 1.55;
    }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 28px; }}
    h1, h2, h3 {{ margin: 0; letter-spacing: 0; }}
    h1 {{ font-size: 30px; }}
    h2 {{ font-size: 21px; margin-bottom: 12px; }}
    h3 {{ font-size: 16px; }}
    .hero {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 18px;
      align-items: start;
      margin-bottom: 18px;
    }}
    .panel {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      box-shadow: 0 10px 22px rgba(47, 47, 43, 0.04);
      margin-bottom: 16px;
    }}
    .eyebrow {{
      text-transform: uppercase;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 6px;
    }}
    .note {{
      border-left: 4px solid var(--copper);
      background: #fff8f3;
      padding: 12px 14px;
      border-radius: 6px;
      color: #47372e;
    }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }}
    .kpi {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfaf6;
    }}
    .kpi span {{ display: block; color: var(--muted); font-size: 12px; }}
    .kpi strong {{ display: block; margin-top: 4px; font-size: 18px; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; }}
    th {{ color: var(--muted); font-weight: 700; background: #fbfaf6; }}
    .links a {{
      color: var(--indigo);
      margin-right: 14px;
      text-decoration: none;
      font-weight: 700;
    }}
    .compare-grid {{
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 16px;
    }}
    .controls {{
      max-height: 520px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfaf6;
    }}
    .check {{ display: block; margin: 8px 0; color: var(--ink); }}
    .metric-buttons button {{
      border: 1px solid var(--line);
      background: var(--surface);
      border-radius: 6px;
      padding: 7px 10px;
      margin: 0 6px 8px 0;
      cursor: pointer;
    }}
    .metric-buttons button.active {{
      background: var(--indigo);
      color: white;
      border-color: var(--indigo);
    }}
    #compare-chart {{ width: 100%; height: 520px; }}
    @media (max-width: 900px) {{
      main {{ padding: 16px; }}
      .hero, .compare-grid, .kpis {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<main>
  <section class="hero">
    <div class="panel">
      <p class="eyebrow">DCA Policy Optimizer</p>
      <h1>QQQ 槓桿 DCA 策略搜尋</h1>
      <p>
        這頁用 QQQ/QLD/TQQQ/CASH 產生可解釋的槓桿 policy，
        尋找在最大回撤不低於 -95% 條件下，DCA XIRR 較高且 walk-forward
        較穩健的候選策略。
      </p>
      <div class="note">
        這是研究訊號，不是投資建議。Synthetic stress 是壓力測試，
        不代表實際 TQQQ 歷史。
      </div>
    </div>
    <div class="panel">
      <p class="eyebrow">Run Context</p>
      <div class="kpis">
        <div class="kpi"><span>scan mode</span><strong>{escape(scan_mode)}</strong></div>
        <div class="kpi"><span>config</span><strong>{escape(str(config_path))}</strong></div>
        <div class="kpi"><span>hard drawdown line</span><strong>-95%</strong></div>
        <div class="kpi"><span>objective</span><strong>XIRR</strong></div>
      </div>
    </div>
  </section>

  <section class="panel">
    <h2>Current Signal Board</h2>
    <p>用目前排名較前的 Actual ETF 候選策略，顯示最新一天的 regime、權重與 effective leverage。</p>
    {_render_signal_kpis(current_signal)}
    <div class="table-wrap">{_render_table(current_signal.head(12), _signal_columns())}</div>
  </section>

  <section class="panel">
    <h2>Best Candidates</h2>
    <p>只把通過 hard drawdown 與 walk-forward 的策略列為候選；高風險策略仍可在完整排名中查看。</p>
    <div class="table-wrap">{_render_table(best, _metric_columns())}</div>
  </section>

  <section class="panel">
    <h2>Actual ETF Ranking</h2>
    <div class="table-wrap">{_render_table(actual, _metric_columns())}</div>
  </section>

  <section class="panel">
    <h2>Synthetic Stress Ranking</h2>
    <p>這裡專門看 2000/2008 類型壓力路徑，不與 Actual ETF 混成單一結論。</p>
    <div class="table-wrap">{_render_table(synthetic, _metric_columns())}</div>
  </section>

  <section class="panel">
    <h2>Walk-Forward Validation</h2>
    <p>每個 fold 只用 train period 選策略，再到下一段 test period 重新跑 DCA。</p>
    <div class="table-wrap">{_render_table(walk_forward.head(80), _walk_columns())}</div>
  </section>

  <section class="panel">
    <h2>Compare Lab</h2>
    <p>自由勾選候選策略疊圖。Normalized equity 只看路徑形狀，不作本金報酬排名。</p>
    <div class="compare-grid">
      <div>
        <div class="metric-buttons" id="metric-buttons"></div>
        <div class="controls" id="scenario-controls"></div>
      </div>
      <div id="compare-chart"></div>
    </div>
  </section>

  <section class="panel links">
    <h2>Audit Files</h2>
    <a href="{metrics_path.name}">metrics CSV</a>
    <a href="{policy_path.name}">policy CSV</a>
    <a href="{walk_forward_path.name}">walk-forward CSV</a>
    <a href="{current_signal_path.name}">current signal CSV</a>
    <a href="{payload_path.name}">payload JSON</a>
  </section>
</main>

<script id="compare-payload" type="application/json">
{json.dumps(payload, ensure_ascii=False)}
</script>
<script>
const payload = JSON.parse(document.getElementById('compare-payload').textContent);
let activeMetric = 'total_equity';
const controls = document.getElementById('scenario-controls');
const metricButtons = document.getElementById('metric-buttons');
function formatScenario(item) {{
  return `${{item.short}} · ${{item.data_mode}} · ${{item.validation_status}}`;
}}
Object.entries(payload.metrics).forEach(([key, config]) => {{
  const button = document.createElement('button');
  button.textContent = config.label;
  button.dataset.metric = key;
  if (key === activeMetric) button.classList.add('active');
  button.onclick = () => {{
    activeMetric = key;
    document
      .querySelectorAll('#metric-buttons button')
      .forEach((node) => node.classList.remove('active'));
    button.classList.add('active');
    drawChart();
  }};
  metricButtons.appendChild(button);
}});
payload.scenarios.forEach((item) => {{
  const label = document.createElement('label');
  label.className = 'check';
  const input = document.createElement('input');
  input.type = 'checkbox';
  input.checked = Boolean(item.default);
  input.dataset.key = item.key;
  input.onchange = drawChart;
  label.appendChild(input);
  label.append(` ${{formatScenario(item)}}`);
  controls.appendChild(label);
}});
function selectedScenarios() {{
  const selected = new Set(
    Array.from(document.querySelectorAll('#scenario-controls input:checked'))
      .map((node) => node.dataset.key)
  );
  return payload.scenarios.filter((item) => selected.has(item.key));
}}
function drawChart() {{
  const metric = payload.metrics[activeMetric];
  const traces = selectedScenarios().map((item) => ({{
    x: item.dates,
    y: item.series[activeMetric],
    mode: 'lines',
    name: item.short,
    hovertemplate: `${{item.full}}<br>%{{x}}<br>%{{y:.3f}}<extra></extra>`
  }}));
  Plotly.react('compare-chart', traces, {{
    margin: {{l: 56, r: 20, t: 20, b: 44}},
    paper_bgcolor: '#fffffc',
    plot_bgcolor: '#fffffc',
    yaxis: {{title: metric.axis, zeroline: false}},
    xaxis: {{title: ''}},
    legend: {{orientation: 'h', y: -0.18}},
  }}, {{responsive: true, displaylogo: false}});
}}
drawChart();
</script>
</body>
</html>
"""


def _run_data_mode(
    *,
    prices: pd.DataFrame,
    data_mode: str,
    specs: list[PolicyScenarioSpec],
    products: list[ProductSpec],
    product_leverages: dict[str, float],
    config: DCAPolicyOptimizerConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    curve_frames: list[pd.DataFrame] = []
    policy_frames: list[pd.DataFrame] = []
    contribution_dates = monthly_rebalance_dates(prices.index)
    for spec in specs:
        weights, policy = build_policy_weights(
            prices=prices,
            spec=spec,
            products=products,
            config=config,
        )
        curve, _allocations = simulate_weighted_strategy(
            prices=prices,
            target_weights=weights,
            product_leverages=product_leverages,
            initial_cash=config.dca_initial_cash,
            rebalance_on_change=True,
            contribution_dates=contribution_dates,
            contribution_amount=config.dca_contribution,
        )
        scenario_id = f"{data_mode}--dca-policy-{spec.name}"
        curve.insert(0, "scenario_id", scenario_id)
        curve.insert(1, "data_mode", data_mode)
        curve.insert(2, "scenario_label", spec.label)
        curve.insert(3, "short_label", spec.short_label)
        curve.insert(4, "strategy_family", spec.family)
        policy.insert(0, "scenario_id", scenario_id)
        policy.insert(1, "data_mode", data_mode)
        policy.insert(2, "scenario_label", spec.label)
        policy.insert(3, "short_label", spec.short_label)
        policy.insert(4, "strategy_family", spec.family)
        metric_rows.append(
            _metrics_row(
                curve=curve,
                data_mode=data_mode,
                scenario_id=scenario_id,
                scenario_label=spec.label,
                short_label=spec.short_label,
                strategy_family=spec.family,
                config=config,
            )
        )
        curve_frames.append(curve)
        policy_frames.append(policy)
    return (
        pd.DataFrame(metric_rows),
        pd.concat(curve_frames, ignore_index=True),
        pd.concat(policy_frames, ignore_index=True),
    )


def build_policy_weights(
    *,
    prices: pd.DataFrame,
    spec: PolicyScenarioSpec,
    products: list[ProductSpec],
    config: DCAPolicyOptimizerConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = prices[products[0].ticker].astype(float)
    empty = pd.Series(np.nan, index=prices.index, dtype="float64")
    trend_value = empty.copy()
    momentum_value = empty.copy()
    volatility_value = empty.copy()
    drawdown = (base / base.cummax() - 1.0).shift(1).fillna(0.0)
    target = pd.Series(1.0, index=prices.index, dtype="float64")
    regime = pd.Series("neutral", index=prices.index, dtype="object")
    reason = pd.Series("default 1x", index=prices.index, dtype="object")

    if spec.kind == "constant":
        target = pd.Series(float(spec.params["target"]), index=prices.index, dtype="float64")
        regime = pd.Series(f"constant_{float(spec.params['target']):.1f}x", index=prices.index)
        reason = pd.Series("constant target leverage", index=prices.index)
    elif spec.kind == "trend_ladder":
        window = int(spec.params["window"])
        trend_value = base.rolling(window).mean().shift(1)
        decision_price = base.shift(1)
        is_risk_on = (decision_price > trend_value).fillna(False)
        target = pd.Series(
            np.where(is_risk_on, float(spec.params["risk_on"]), float(spec.params["risk_off"])),
            index=prices.index,
            dtype="float64",
        )
        regime = pd.Series(np.where(is_risk_on, "trend_on", "trend_off"), index=prices.index)
        reason = pd.Series(f"prior close vs {window}MA", index=prices.index)
    elif spec.kind == "drawdown_ladder":
        guards = sorted(config.drawdown_guards, reverse=True)
        target = pd.Series(float(spec.params["risk_on"]), index=prices.index, dtype="float64")
        target = target.mask(drawdown <= guards[0], min(float(spec.params["risk_on"]), 2.0))
        target = target.mask(drawdown <= guards[1], min(float(spec.params["risk_on"]), 1.5))
        target = target.mask(drawdown <= guards[2], 1.0)
        target = target.mask(drawdown <= guards[3], 0.0)
        regime = pd.Series("drawdown_clear", index=prices.index)
        regime = regime.mask(drawdown <= guards[0], "drawdown_mild")
        regime = regime.mask(drawdown <= guards[1], "drawdown_medium")
        regime = regime.mask(drawdown <= guards[2], "drawdown_severe")
        regime = regime.mask(drawdown <= guards[3], "drawdown_defensive")
        reason = pd.Series("prior QQQ drawdown ladder", index=prices.index)
    elif spec.kind == "vol_target_ladder":
        window = int(spec.params["window"])
        returns = base.pct_change()
        volatility_value = returns.rolling(window).std(ddof=0).shift(1) * np.sqrt(252)
        raw_target = float(spec.params["target_volatility"]) / volatility_value
        target = raw_target.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.0, 3.0)
        regime = pd.Series("vol_target", index=prices.index)
        reason = pd.Series(f"prior {window}D realized volatility", index=prices.index)
    elif spec.kind == "momentum_trend_ladder":
        momentum_window = int(spec.params["momentum_window"])
        trend_window = int(spec.params["trend_window"])
        momentum_value = (base / base.shift(momentum_window) - 1.0).shift(1)
        trend_value = base.rolling(trend_window).mean().shift(1)
        decision_price = base.shift(1)
        is_risk_on = ((momentum_value > 0.0) & (decision_price > trend_value)).fillna(False)
        target = pd.Series(
            np.where(is_risk_on, float(spec.params["risk_on"]), float(spec.params["risk_off"])),
            index=prices.index,
            dtype="float64",
        )
        regime = pd.Series(
            np.where(is_risk_on, "momentum_trend_on", "momentum_trend_off"),
            index=prices.index,
        )
        reason = pd.Series(
            f"prior {momentum_window}D momentum and {trend_window}MA",
            index=prices.index,
        )
    else:
        raise ValueError(f"Unsupported policy scenario kind: {spec.kind}")

    target = target.clip(0.0, 3.0).astype(float)
    weights = _weights_from_target_series(target, products)
    policy = pd.DataFrame(
        {
            "date": prices.index,
            "regime": regime.to_numpy(),
            "reason": reason.to_numpy(),
            "target_effective_leverage": target.to_numpy(),
            "trend_value": trend_value.to_numpy(),
            "momentum_value": momentum_value.to_numpy(),
            "volatility_value": volatility_value.to_numpy(),
            "drawdown": drawdown.to_numpy(),
        }
    )
    for column in weights.columns:
        policy[f"{column}_weight"] = weights[column].to_numpy()
    return weights, policy


def _metrics_row(
    *,
    curve: pd.DataFrame,
    data_mode: str,
    scenario_id: str,
    scenario_label: str,
    short_label: str,
    strategy_family: str,
    config: DCAPolicyOptimizerConfig,
) -> dict[str, Any]:
    returns = curve["investment_return"].astype(float).dropna()
    summary = performance_summary(returns) if not returns.empty else pd.Series(dtype=float)
    total_contributed = float(curve["total_contributed"].iloc[-1])
    ending_equity = float(curve["total_equity"].iloc[-1])
    simple_cash_return = ending_equity / total_contributed - 1.0 if total_contributed else np.nan
    max_drawdown = float(curve["drawdown"].astype(float).min())
    recovery_days = _max_recovery_days(curve["return_index"], curve["date"])
    risk_flag = _risk_flag(max_drawdown=max_drawdown, config=config)
    return {
        "data_mode": data_mode,
        "scenario_id": scenario_id,
        "scenario_label": scenario_label,
        "short_label": short_label,
        "strategy_family": strategy_family,
        "start_date": pd.Timestamp(curve["date"].iloc[0]).date().isoformat(),
        "end_date": pd.Timestamp(curve["date"].iloc[-1]).date().isoformat(),
        "total_contributed": total_contributed,
        "ending_equity": ending_equity,
        "simple_cash_return": simple_cash_return,
        "xirr": _xirr_from_curve(curve),
        "cagr": _summary_value(summary, "cagr"),
        "volatility": _summary_value(summary, "volatility"),
        "sharpe": _summary_value(summary, "sharpe"),
        "sortino": _summary_value(summary, "sortino"),
        "calmar": _summary_value(summary, "calmar"),
        "max_drawdown": max_drawdown,
        "recovery_days": recovery_days,
        "effective_leverage_avg": float(curve["effective_product_leverage"].mean()),
        "effective_leverage_max": float(curve["effective_product_leverage"].max()),
        "risk_flag": risk_flag,
        "risk_failed": risk_flag == "drawdown_limit_breach",
    }


def _clean_prices(prices: pd.DataFrame, products: list[ProductSpec]) -> pd.DataFrame:
    columns = [product.ticker for product in products]
    clean = prices[columns].dropna(how="any").astype(float).sort_index()
    if clean.empty:
        raise ValueError("DCA policy optimizer requires overlapping product prices.")
    return clean


def _weights_from_target_series(
    target: pd.Series,
    products: list[ProductSpec],
) -> pd.DataFrame:
    rows = [target_leverage_to_product_weights(value, products) for value in target]
    columns = [product.ticker for product in products] + [CASH]
    return pd.DataFrame(rows, index=target.index, columns=columns).astype(float)


def _risk_flag(*, max_drawdown: float, config: DCAPolicyOptimizerConfig) -> str:
    if max_drawdown < config.max_drawdown_limit:
        return "drawdown_limit_breach"
    if max_drawdown <= config.high_risk_drawdown_band:
        return "high_drawdown"
    return "ok"


def _policy_score(row: pd.Series) -> float:
    score = 100.0 * _safe(row.get("xirr"))
    score += 0.5 * _safe(row.get("simple_cash_return"))
    score += 0.25 * _safe(row.get("calmar"))
    score += 0.10 * _safe(row.get("sortino"))
    score += _safe(row.get("max_drawdown"))
    score -= min(_safe(row.get("recovery_days")) / 365.25, 25.0) * 0.05
    if row.get("risk_flag") == "high_drawdown":
        score -= 2.0
    if row.get("risk_failed"):
        score -= 1_000.0
    if row.get("validation_status") == VALIDATION_FRAGILE:
        score -= 100.0
    elif row.get("validation_status") == VALIDATION_UNVALIDATED:
        score -= 10.0
    return float(score)


def _validation_summary(
    walk_forward: pd.DataFrame,
    *,
    config: DCAPolicyOptimizerConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario_id, group in walk_forward.groupby("scenario_id", sort=False):
        pass_rate = float(group["passed_fold"].astype(bool).mean())
        worst_drawdown = float(group["test_max_drawdown"].astype(float).min())
        mean_xirr = float(group["test_xirr"].astype(float).mean())
        worst_recovery = float(group["test_recovery_days"].astype(float).max())
        if pass_rate >= 0.75 and worst_drawdown >= config.max_drawdown_limit and mean_xirr > 0:
            status = VALIDATION_STABLE
        elif pass_rate >= 0.50 and worst_drawdown >= config.max_drawdown_limit:
            status = VALIDATION_WATCHLIST
        else:
            status = VALIDATION_FRAGILE
        rows.append(
            {
                "scenario_id": scenario_id,
                "validation_status": status,
                "validation_folds": int(len(group)),
                "walk_forward_pass_rate": pass_rate,
                "mean_test_xirr": mean_xirr,
                "worst_test_drawdown": worst_drawdown,
                "worst_test_recovery_days": worst_recovery,
            }
        )
    return pd.DataFrame(rows)


def _walk_forward_folds(
    index: pd.Index,
    *,
    train_years: int,
    test_years: int,
    step_years: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    dates = pd.DatetimeIndex(index).sort_values()
    if dates.empty:
        return []
    rows: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    train_start = dates[0]
    final_date = dates[-1]
    while True:
        train_end_target = train_start + pd.DateOffset(years=train_years)
        test_end_target = train_end_target + pd.DateOffset(years=test_years)
        train_end = _first_date_on_or_after(dates, train_end_target)
        test_start = train_end
        test_end = _first_date_on_or_after(dates, test_end_target)
        if train_end is None or test_end is None or test_end > final_date:
            break
        rows.append((train_start, train_end, test_start, test_end))
        train_start = _first_date_on_or_after(dates, train_start + pd.DateOffset(years=step_years))
        if train_start is None or train_start >= final_date:
            break
    return rows


def _first_date_on_or_after(
    dates: pd.DatetimeIndex,
    target: pd.Timestamp,
) -> pd.Timestamp | None:
    matches = dates[dates >= target]
    if matches.empty:
        return None
    return pd.Timestamp(matches[0])


def _xirr_from_curve(curve: pd.DataFrame) -> float:
    cash_flows: list[tuple[pd.Timestamp, float]] = []
    for row in curve.itertuples():
        contribution = float(row.contribution)
        if contribution > 0:
            cash_flows.append((pd.Timestamp(row.date), -contribution))
    if curve.empty:
        return np.nan
    cash_flows.append((pd.Timestamp(curve["date"].iloc[-1]), float(curve["total_equity"].iloc[-1])))
    return xirr(cash_flows)


def _max_recovery_days(return_index: pd.Series, dates: pd.Series) -> int:
    values = pd.Series(return_index).astype(float).reset_index(drop=True)
    clean_dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    if values.empty:
        return 0
    peak_value = values.iloc[0]
    peak_date = clean_dates.iloc[0]
    in_drawdown = False
    max_days = 0
    for value, date in zip(values, clean_dates, strict=True):
        if value >= peak_value:
            if in_drawdown:
                max_days = max(max_days, int((date - peak_date).days))
            peak_value = value
            peak_date = date
            in_drawdown = False
        else:
            in_drawdown = True
    if in_drawdown:
        max_days = max(max_days, int((clean_dates.iloc[-1] - peak_date).days))
    return int(max_days)


def _summary_value(summary: pd.Series, key: str) -> float:
    value = summary.get(key, np.nan)
    return float(value) if pd.notna(value) else np.nan


def _safe(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    return float(value)


def _validation_rank(status: str) -> int:
    return {
        VALIDATION_STABLE: 0,
        VALIDATION_WATCHLIST: 1,
        VALIDATION_UNVALIDATED: 2,
        VALIDATION_FRAGILE: 3,
    }.get(str(status), 9)


def _policy_key(scenario_id: str) -> str:
    return str(scenario_id).split("--", maxsplit=1)[-1]


def _leverage_slug(value: float) -> str:
    return f"{value:.1f}".replace(".", "p")


def _json_series(values: Any) -> list[float | None]:
    series = pd.Series(values)
    return [None if pd.isna(value) else float(value) for value in series]


def _render_signal_kpis(current_signal: pd.DataFrame) -> str:
    if current_signal.empty:
        return ""
    row = current_signal.iloc[0]
    return f"""
    <div class="kpis">
      <div class="kpi"><span>as of</span><strong>{escape(str(row["as_of_date"]))}</strong></div>
      <div class="kpi"><span>regime</span><strong>{escape(str(row["regime"]))}</strong></div>
      <div class="kpi">
        <span>effective leverage</span>
        <strong>{_format_number(row["target_effective_leverage"])}x</strong>
      </div>
      <div class="kpi">
        <span>validation</span>
        <strong>{escape(str(row["validation_status"]))}</strong>
      </div>
    </div>
    """


def _render_table(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    if frame.empty:
        return "<p>No data.</p>"
    available = [(key, label) for key, label in columns if key in frame.columns]
    head = "".join(f"<th>{escape(label)}</th>" for _, label in available)
    rows: list[str] = []
    for row in frame.head(120).itertuples(index=False):
        row_dict = row._asdict()
        cells = "".join(
            f"<td>{escape(_format_cell(row_dict.get(key), key))}</td>" for key, _ in available
        )
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _format_cell(value: Any, key: str) -> str:
    if pd.isna(value):
        return ""
    if key in {
        "xirr",
        "simple_cash_return",
        "max_drawdown",
        "walk_forward_pass_rate",
        "mean_test_xirr",
        "worst_test_drawdown",
        "test_xirr",
        "test_max_drawdown",
    }:
        return _format_percent(value)
    if key in {"ending_equity", "total_contributed", "test_ending_equity"}:
        return _format_money(value)
    if key.endswith("_weight"):
        return _format_percent(value)
    if key in {"target_effective_leverage", "effective_leverage_avg", "effective_leverage_max"}:
        return f"{float(value):.2f}x"
    if isinstance(value, float):
        return _format_number(value)
    return str(value)


def _format_percent(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def _format_money(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):,.0f}"


def _format_number(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def _metric_columns() -> list[tuple[str, str]]:
    return [
        ("rank", "Rank"),
        ("scenario_label", "Strategy"),
        ("validation_status", "Validation"),
        ("risk_flag", "Risk"),
        ("xirr", "XIRR"),
        ("ending_equity", "Ending Equity"),
        ("total_contributed", "Contributed"),
        ("simple_cash_return", "Simple Return"),
        ("max_drawdown", "Max DD"),
        ("recovery_days", "Recovery Days"),
        ("effective_leverage_avg", "Avg Lev"),
        ("effective_leverage_max", "Max Lev"),
    ]


def _signal_columns() -> list[tuple[str, str]]:
    return [
        ("rank", "Rank"),
        ("scenario_label", "Strategy"),
        ("as_of_date", "As Of"),
        ("regime", "Regime"),
        ("reason", "Reason"),
        ("target_effective_leverage", "Target Lev"),
        ("QQQ_weight", "QQQ"),
        ("QLD_weight", "QLD"),
        ("TQQQ_weight", "TQQQ"),
        ("CASH_weight", "CASH"),
        ("xirr", "XIRR"),
        ("max_drawdown", "Max DD"),
    ]


def _walk_columns() -> list[tuple[str, str]]:
    return [
        ("data_mode", "Mode"),
        ("fold_index", "Fold"),
        ("scenario_label", "Strategy"),
        ("train_rank", "Train Rank"),
        ("test_start", "Test Start"),
        ("test_end", "Test End"),
        ("test_xirr", "Test XIRR"),
        ("test_ending_equity", "Test Equity"),
        ("test_max_drawdown", "Test DD"),
        ("passed_fold", "Passed"),
    ]


__all__ = [
    "DCAPolicyOptimizerOutputs",
    "DCAPolicyOptimizerReportResult",
    "PolicyScenarioSpec",
    "apply_cross_mode_candidate_filter",
    "apply_policy_validation",
    "build_current_signal",
    "build_dca_policy_optimizer_outputs",
    "build_policy_compare_payload",
    "build_policy_scenario_specs",
    "build_policy_walk_forward_validation",
    "build_policy_weights",
    "policy_config_for_scan_mode",
    "rank_policy_metrics",
    "render_dca_policy_optimizer_html",
    "target_leverage_to_product_weights",
    "write_dca_policy_optimizer_report",
]
