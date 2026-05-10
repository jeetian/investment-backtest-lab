from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from investment_backtest_lab.costs import CostModel
from investment_backtest_lab.dca_policy_optimizer import (
    PolicyScenarioSpec,
    target_leverage_to_product_weights,
)
from investment_backtest_lab.external_signals import (
    EXTERNAL_SIGNAL_FEATURE_COLUMNS,
    EXTERNAL_SIGNAL_FEATURE_METADATA,
    external_signal_columns_for_feature_set,
    load_external_signal_features,
)
from investment_backtest_lab.html_ui import render_html_head
from investment_backtest_lab.leveraged_etf_lab import (
    CASH,
    ProductSpec,
    monthly_rebalance_dates,
    rebalance_dates_for_cadence,
    simulate_weighted_strategy,
    xirr,
)
from investment_backtest_lab.models import AssetSpec, BacktestConfig, StrategySearchConfig
from investment_backtest_lab.reports import performance_summary

OBJECTIVE_PROFILE_PARETO = "pareto"
OBJECTIVE_PROFILE_RETURN_FIRST = "return_first"
SUPPORTED_OBJECTIVE_PROFILES = (OBJECTIVE_PROFILE_PARETO, OBJECTIVE_PROFILE_RETURN_FIRST)
BENCHMARK_BASE_DCA = "base_dca"
BENCHMARK_QQQ_DCA = "qqq_dca"
BENCHMARK_FIXED_1P5X_DCA = "fixed_1p5x_dca"
OFFICIAL_FIXED_1P5X_WIN_COLUMN = "win_rate_vs_fixed_1p5x_dca"
REFERENCE_1X_WIN_COLUMN = "win_rate_vs_0050_dca"
EXECUTION_MONTHLY_CORE_WEEKLY_DELTA = "monthly_core_weekly_delta"

OBJECTIVE_COLUMNS = (
    "expected_xirr",
    "p05_xirr",
    "sharpe",
    "sortino",
    "win_rate_vs_base_dca",
    "p05_drawdown_loss",
    "drawdown_breach_rate",
    "cost_drag_on_contributed",
    "turnover_sum",
)
RETURN_FIRST_OBJECTIVE_COLUMNS = (
    "expected_xirr",
    "win_rate_vs_base_dca",
    "p05_xirr",
)


@dataclass(frozen=True)
class StrategySearchResult:
    study_name: str
    objective_profile: str
    trials: pd.DataFrame
    pareto: pd.DataFrame
    best_candidates: pd.DataFrame
    candidate_triage: pd.DataFrame
    trials_path: Path
    pareto_path: Path
    best_candidates_path: Path
    candidate_triage_path: Path
    html_path: Path
    storage_path: Path
    dynamic_drawdown_limit: float
    study_status: dict[str, Any]
    export_only: bool


@dataclass(frozen=True)
class StrategySearchContext:
    prices: pd.DataFrame
    products: list[ProductSpec]
    product_assets: dict[str, AssetSpec]
    cost_model: CostModel
    config: BacktestConfig
    search_config: StrategySearchConfig
    sentiment: pd.Series | None
    external_signals: pd.DataFrame | None
    external_signal_feature_columns: tuple[str, ...]
    external_signal_common_start: str
    external_signal_common_end: str
    output_dir: Path
    dynamic_drawdown_limit: float
    base_1x_stress_max_drawdown: float
    validation_folds: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]
    holdout: tuple[pd.Timestamp, pd.Timestamp]
    baseline: dict[str, float]


def normalize_objective_profile(profile: str) -> str:
    normalized = str(profile or OBJECTIVE_PROFILE_PARETO).lower().replace("-", "_")
    if normalized not in SUPPORTED_OBJECTIVE_PROFILES:
        supported = ", ".join(SUPPORTED_OBJECTIVE_PROFILES)
        raise ValueError(f"Unsupported objective profile: {profile}. Use one of: {supported}.")
    return normalized


def objective_columns_for_profile(profile: str) -> tuple[str, ...]:
    normalized = normalize_objective_profile(profile)
    if normalized == OBJECTIVE_PROFILE_RETURN_FIRST:
        return RETURN_FIRST_OBJECTIVE_COLUMNS
    return OBJECTIVE_COLUMNS


def objective_directions_for_profile(profile: str) -> list[str]:
    columns = objective_columns_for_profile(profile)
    return [
        "minimize"
        if column
        in {
            "p05_drawdown_loss",
            "drawdown_breach_rate",
            "cost_drag_on_contributed",
            "turnover_sum",
        }
        else "maximize"
        for column in columns
    ]


def validate_study_directions(
    *,
    study: Any,
    objective_columns: tuple[str, ...],
    profile: str,
) -> None:
    if len(study.directions) != len(objective_columns):
        raise ValueError(
            f"Study {study.study_name!r} has {len(study.directions)} objectives, "
            f"but profile {profile!r} expects {len(objective_columns)}. "
            "Use a new study name/storage or rerun with the matching objective profile."
        )


def build_study_fingerprint(
    *,
    context: StrategySearchContext,
    objective_profile: str,
) -> dict[str, Any]:
    payload = {
        "version": 1,
        "family": context.search_config.family,
        "benchmark": context.search_config.benchmark,
        "start_date": pd.Timestamp(context.prices.index.min()).date().isoformat(),
        "end_date": context.config.end_date.isoformat(),
        "price_last_date": pd.Timestamp(context.prices.index.max()).date().isoformat(),
        "objective_profile": normalize_objective_profile(objective_profile),
        "execution_profile": context.search_config.execution_profile,
        "execution_cadence": context.search_config.execution_cadence,
        "contribution_cadence": context.search_config.contribution_cadence,
        "sampler_seed": int(context.search_config.sampler_seed),
        "external_signal_feature_set": context.search_config.external_signal_feature_set,
        "external_signal_columns": list(context.external_signal_feature_columns),
        "external_signal_common_start": context.external_signal_common_start,
        "external_signal_common_end": context.external_signal_common_end,
        "include_external_signals": bool(context.external_signal_feature_columns),
        "include_sentiment": context.sentiment is not None,
        "cost_model": context.config.cost_model,
        "initial_cash": float(context.config.monthly_decision_replay.initial_cash),
        "monthly_contribution": float(context.config.monthly_decision_replay.monthly_contribution),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return payload


def validate_or_set_study_fingerprint(
    *,
    study: Any,
    fingerprint: dict[str, Any],
    export_only: bool,
) -> None:
    existing_json = study.user_attrs.get("study_fingerprint_json")
    fingerprint_json = json.dumps(fingerprint, sort_keys=True, default=str)
    if export_only:
        return
    complete_or_running_trials = [
        trial
        for trial in study.trials
        if getattr(trial.state, "name", "") in {"COMPLETE", "RUNNING", "WAITING", "FAIL"}
    ]
    if existing_json is None and complete_or_running_trials:
        raise ValueError(
            f"Study {study.study_name!r} has existing trials but no study fingerprint. "
            "Use --export-only to inspect it, or start a new study/storage for this config."
        )
    if existing_json is not None:
        try:
            existing = json.loads(str(existing_json))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Study {study.study_name!r} has an unreadable fingerprint. "
                "Use a new study/storage."
            ) from exc
        if existing.get("sha256") != fingerprint.get("sha256"):
            raise ValueError(
                f"Study {study.study_name!r} fingerprint does not match the current config. "
                "Use --export-only to read old results, or start a new study/storage."
            )
        return
    study.set_user_attr("study_fingerprint_json", fingerprint_json)
    study.set_user_attr("study_fingerprint_sha256", fingerprint["sha256"])


def progress_callback(*, started_at: float, interval: int):
    def _callback(study: Any, trial: Any) -> None:
        if interval <= 0:
            return
        completed = sum(1 for item in study.trials if item.state.name == "COMPLETE")
        if completed == 0 or completed % interval != 0:
            return
        elapsed_minutes = (time.perf_counter() - started_at) / 60.0
        attrs = dict(trial.user_attrs)
        print(
            "Optuna progress: "
            f"completed={completed:,}, "
            f"last_trial={trial.number}, "
            f"expected_xirr={_format_percent(attrs.get('expected_xirr', np.nan))}, "
            f"win_rate_vs_benchmark={_format_percent(attrs.get('win_rate_vs_base_dca', np.nan))}, "
            f"elapsed={elapsed_minutes:.1f}m",
            flush=True,
        )

    return _callback


def run_optuna_strategy_search(
    *,
    context: StrategySearchContext,
    study_name: str,
    trials: int,
    storage_path: Path,
    export_only: bool = False,
    objective_profile: str | None = None,
    timeout_hours: float | None = None,
    progress_interval_trials: int = 100,
) -> StrategySearchResult:
    optuna = _require_optuna()
    profile = normalize_objective_profile(
        objective_profile or context.search_config.objective_profile
    )
    objective_columns = objective_columns_for_profile(profile)
    context.output_dir.mkdir(parents=True, exist_ok=True)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_uri = f"sqlite:///{storage_path.as_posix()}"
    prefix = study_name
    trials_path = context.output_dir / f"optuna_{prefix}_trials.csv"
    pareto_path = context.output_dir / f"optuna_{prefix}_pareto.csv"
    best_path = context.output_dir / f"optuna_{prefix}_best_candidates.csv"
    triage_path = context.output_dir / f"optuna_{prefix}_candidate_triage.csv"
    html_path = context.output_dir / f"optuna_{prefix}.html"
    cached_trials = _load_cached_trials(trials_path) if export_only else pd.DataFrame()

    if export_only:
        try:
            study = optuna.load_study(study_name=study_name, storage=storage_uri)
        except KeyError as exc:
            raise ValueError(f"Optuna study not found for export-only: {study_name}") from exc
    else:
        if int(trials) <= 0:
            raise ValueError("Use --export-only to rebuild reports without adding trials.")
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_uri,
            load_if_exists=True,
            directions=objective_directions_for_profile(profile),
            sampler=optuna.samplers.NSGAIISampler(seed=int(context.search_config.sampler_seed)),
        )
    validate_study_directions(study=study, objective_columns=objective_columns, profile=profile)
    fingerprint = build_study_fingerprint(context=context, objective_profile=profile)
    validate_or_set_study_fingerprint(
        study=study,
        fingerprint=fingerprint,
        export_only=export_only,
    )

    def objective(trial: Any) -> tuple[float, ...]:
        params = suggest_strategy_params(
            trial,
            max_leverage=max_product_leverage(context.products),
            include_sentiment=context.sentiment is not None,
            external_feature_columns=context.external_signal_feature_columns,
            execution_cadence=context.search_config.execution_cadence,
        )
        metrics = evaluate_strategy_params(context=context, params=params, use_holdout=False)
        for key, value in metrics.items():
            trial.set_user_attr(key, _jsonable(value))
        trial.set_user_attr("params_json", json.dumps(params, sort_keys=True))
        return tuple(float(metrics[column]) for column in objective_columns)

    if not export_only:
        started_at = time.perf_counter()
        study.optimize(
            objective,
            n_trials=int(trials),
            timeout=None if timeout_hours is None else max(float(timeout_hours), 0.0) * 3600.0,
            gc_after_trial=True,
            callbacks=[
                progress_callback(
                    started_at=started_at,
                    interval=max(int(progress_interval_trials), 0),
                )
            ],
        )
    trial_rows = build_trial_rows(
        context=context,
        study=study,
        include_holdout=True,
        cached_trials=cached_trials,
    )
    trials_df = add_return_first_columns(pd.DataFrame(trial_rows), baseline=context.baseline)
    triage_df = build_candidate_triage(
        trials_df,
        baseline=context.baseline,
        objective_profile=profile,
    )
    pareto_df = pareto_trials_frame(trials_df, study.best_trials)
    best_df = classify_best_candidates(
        pareto_df,
        baseline=context.baseline,
        objective_profile=profile,
        triage=triage_df,
    )
    status = study_status(
        study=study,
        context=context,
        storage_path=storage_path,
        objective_profile=profile,
    )

    trials_df.to_csv(trials_path, index=False)
    pareto_df.to_csv(pareto_path, index=False)
    best_df.to_csv(best_path, index=False)
    triage_df.to_csv(triage_path, index=False)
    html_path.write_text(
        render_strategy_search_html(
            study_name=study_name,
            trials=trials_df,
            pareto=pareto_df,
            best_candidates=best_df,
            candidate_triage=triage_df,
            dynamic_drawdown_limit=context.dynamic_drawdown_limit,
            baseline=context.baseline,
            study_status=status,
            export_only=export_only,
            objective_profile=profile,
        ),
        encoding="utf-8",
    )
    return StrategySearchResult(
        study_name=study_name,
        objective_profile=profile,
        trials=trials_df,
        pareto=pareto_df,
        best_candidates=best_df,
        candidate_triage=triage_df,
        trials_path=trials_path,
        pareto_path=pareto_path,
        best_candidates_path=best_path,
        candidate_triage_path=triage_path,
        html_path=html_path,
        storage_path=storage_path,
        dynamic_drawdown_limit=context.dynamic_drawdown_limit,
        study_status=status,
        export_only=export_only,
    )


def build_strategy_search_context(
    *,
    prices: pd.DataFrame,
    products: list[ProductSpec],
    product_assets: dict[str, AssetSpec],
    cost_model: CostModel,
    config: BacktestConfig,
    output_dir: Path,
    include_sentiment: bool | None = None,
    sentiment_path: Path | None = None,
    include_external_signals: bool | None = None,
    external_signals_path: Path | None = None,
) -> StrategySearchContext:
    search_config = config.strategy_search
    normalized_prices = prices.dropna(how="any").astype(float).sort_index()
    if normalized_prices.empty:
        raise ValueError("Strategy search requires non-empty prices.")
    include = search_config.include_sentiment if include_sentiment is None else include_sentiment
    path = Path(sentiment_path or search_config.sentiment_path)
    sentiment = (
        load_frozen_sentiment(path, trading_index=normalized_prices.index) if include else None
    )
    include_external = (
        search_config.include_external_signals
        if include_external_signals is None
        else include_external_signals
    )
    external_path = Path(external_signals_path or search_config.external_signals_path)
    external_columns = external_signal_columns_for_feature_set(
        search_config.external_signal_feature_set
    )
    external_signals = (
        load_external_signal_features(
            external_path,
            trading_index=normalized_prices.index,
            feature_columns=external_columns,
            require_complete=search_config.external_signal_feature_set == "core",
        )
        if include_external
        else None
    )
    external_start = ""
    external_end = ""
    if external_signals is not None and not external_signals.empty:
        normalized_prices = normalized_prices.reindex(external_signals.index).dropna(how="any")
        external_signals = external_signals.reindex(normalized_prices.index)
        external_start = pd.Timestamp(external_signals.index.min()).date().isoformat()
        external_end = pd.Timestamp(external_signals.index.max()).date().isoformat()
    base_drawdown, dynamic_limit = derive_dynamic_drawdown_limit(
        prices=normalized_prices,
        products=products,
        product_assets=product_assets,
        cost_model=cost_model,
        config=config,
    )
    folds = validation_folds(
        normalized_prices.index,
        holdout_start=pd.Timestamp(search_config.final_holdout_start),
    )
    holdout = (
        pd.Timestamp(search_config.final_holdout_start),
        pd.Timestamp(search_config.final_holdout_end),
    )
    baseline = load_current_baseline(
        output_dir=output_dir,
        family=search_config.family,
        benchmark=official_benchmark_id(config),
    )
    if not baseline:
        baseline = estimate_search_baseline(
            prices=normalized_prices,
            products=products,
            product_assets=product_assets,
            cost_model=cost_model,
            config=config,
            periods=list(folds),
        )
    return StrategySearchContext(
        prices=normalized_prices,
        products=products,
        product_assets=product_assets,
        cost_model=cost_model,
        config=config,
        search_config=search_config,
        sentiment=sentiment,
        external_signals=external_signals,
        external_signal_feature_columns=external_columns if include_external else tuple(),
        external_signal_common_start=external_start,
        external_signal_common_end=external_end,
        output_dir=output_dir,
        dynamic_drawdown_limit=dynamic_limit,
        base_1x_stress_max_drawdown=base_drawdown,
        validation_folds=folds,
        holdout=holdout,
        baseline=baseline,
    )


def load_frozen_sentiment(path: Path, *, trading_index: pd.Index) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen sentiment CSV: {path}. "
            "Create a CSV with columns date,score,rating or run with --no-sentiment."
        )
    data = pd.read_csv(path)
    required = {"date", "score", "rating"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Sentiment CSV missing columns: {sorted(missing)}")
    dates = pd.to_datetime(data["date"], errors="raise")
    if dates.duplicated().any():
        raise ValueError("Sentiment CSV contains duplicate dates.")
    if not dates.is_monotonic_increasing:
        raise ValueError("Sentiment CSV must be sorted by date ascending.")
    scores = pd.to_numeric(data["score"], errors="raise").astype(float)
    if ((scores < 0.0) | (scores > 100.0)).any():
        raise ValueError("Sentiment score must be between 0 and 100.")
    raw = pd.Series(scores.to_numpy(), index=pd.DatetimeIndex(dates), name="sentiment_score")
    aligned = raw.reindex(pd.DatetimeIndex(trading_index).sort_values()).ffill().shift(1)
    return aligned


def suggest_strategy_params(
    trial: Any,
    *,
    max_leverage: float,
    include_sentiment: bool = True,
    external_feature_columns: tuple[str, ...] | None = None,
    execution_cadence: str = "flexible",
) -> dict[str, Any]:
    drawdown_guard = trial.suggest_float("drawdown_guard", -0.35, -0.05)
    severe_drawdown_guard = trial.suggest_float(
        "severe_drawdown_guard", -0.70, drawdown_guard - 0.02
    )
    normalized_execution_cadence = str(execution_cadence or "flexible").lower()
    if normalized_execution_cadence == "weekly":
        rebalance_cadence = "weekly"
        min_holding_days = trial.suggest_categorical("min_holding_days", [0, 5])
    elif normalized_execution_cadence == EXECUTION_MONTHLY_CORE_WEEKLY_DELTA:
        rebalance_cadence = EXECUTION_MONTHLY_CORE_WEEKLY_DELTA
        min_holding_days = trial.suggest_categorical("min_holding_days", [0, 5])
    else:
        rebalance_cadence = trial.suggest_categorical(
            "rebalance_cadence", ["monthly", "quarterly", "signal_only"]
        )
        min_holding_days = trial.suggest_int("min_holding_days", 0, 126)

    params = {
        "trend_window": trial.suggest_categorical("trend_window", [63, 100, 150, 200, 250]),
        "slope_window": trial.suggest_categorical("slope_window", [21, 63, 126]),
        "momentum_window": trial.suggest_categorical("momentum_window", [21, 63, 126, 252]),
        "vol_window": trial.suggest_categorical("vol_window", [21, 63, 126]),
        "vol_percentile_window": trial.suggest_categorical(
            "vol_percentile_window", [252, 504, 756]
        ),
        "vol_target": trial.suggest_float("vol_target", 0.10, 0.35),
        "risk_on_leverage": trial.suggest_float("risk_on_leverage", 1.0, max_leverage),
        "risk_off_leverage": trial.suggest_float("risk_off_leverage", 0.0, 1.0),
        "cash_leverage": trial.suggest_float("cash_leverage", 0.0, 0.50),
        "drawdown_guard": drawdown_guard,
        "severe_drawdown_guard": severe_drawdown_guard,
        "vol_spike_quantile": trial.suggest_float("vol_spike_quantile", 0.70, 0.95),
        "rebalance_cadence": rebalance_cadence,
        "rebalance_threshold": trial.suggest_float("rebalance_threshold", 0.0, 0.25),
        "weekly_delta_threshold": (
            trial.suggest_float("weekly_delta_threshold", 0.15, 0.60)
            if rebalance_cadence == EXECUTION_MONTHLY_CORE_WEEKLY_DELTA
            else None
        ),
        "min_holding_days": min_holding_days,
        "signal_hysteresis": trial.suggest_float("signal_hysteresis", 0.0, 0.05),
        "max_annual_turnover": trial.suggest_float("max_annual_turnover", 2.0, 80.0),
    }
    if include_sentiment:
        params.update(
            {
                "sentiment_mode": trial.suggest_categorical(
                    "sentiment_mode", ["off", "contrarian", "risk_off"]
                ),
                "fear_threshold": trial.suggest_float("fear_threshold", 10.0, 40.0),
                "greed_threshold": trial.suggest_float("greed_threshold", 60.0, 90.0),
                "sentiment_deleverage": trial.suggest_float(
                    "sentiment_deleverage",
                    0.0,
                    0.75,
                ),
            }
        )
    features = (
        EXTERNAL_SIGNAL_FEATURE_COLUMNS
        if external_feature_columns is None
        else tuple(external_feature_columns)
    )
    for feature in features:
        params[f"{feature}_mode"] = trial.suggest_categorical(
            f"{feature}_mode",
            ["off", "risk_off", "contrarian"],
        )
        params[f"{feature}_low_threshold"] = trial.suggest_float(
            f"{feature}_low_threshold",
            5.0,
            40.0,
        )
        params[f"{feature}_high_threshold"] = trial.suggest_float(
            f"{feature}_high_threshold",
            60.0,
            95.0,
        )
        params[f"{feature}_deleverage"] = trial.suggest_float(
            f"{feature}_deleverage",
            0.0,
            0.75,
        )
    return params


def evaluate_strategy_params(
    *,
    context: StrategySearchContext,
    params: dict[str, Any],
    use_holdout: bool,
) -> dict[str, Any]:
    weights = build_rule_strategy_weights(
        prices=context.prices,
        products=context.products,
        params=params,
        sentiment=context.sentiment,
        external_signals=context.external_signals,
    )
    periods = [context.holdout] if use_holdout else list(context.validation_folds)
    rows = []
    for start, end in periods:
        period_prices = _slice_period(context.prices, start, end)
        period_weights = weights.reindex(period_prices.index).ffill()
        if len(period_prices) < 20:
            continue
        curve = simulate_search_curve(
            prices=period_prices,
            weights=period_weights,
            products=context.products,
            product_assets=context.product_assets,
            cost_model=context.cost_model,
            config=context.config,
        )
        base_curve = benchmark_dca_curve(
            prices=period_prices,
            products=context.products,
            product_assets=context.product_assets,
            cost_model=context.cost_model,
            config=context.config,
        )
        reference_1x_curve = baseline_1x_curve(
            prices=period_prices,
            products=context.products,
            product_assets=context.product_assets,
            cost_model=context.cost_model,
            config=context.config,
        )
        metrics = curve_metrics(curve)
        base_metrics = curve_metrics(base_curve)
        reference_1x_metrics = curve_metrics(reference_1x_curve)
        metrics["win_vs_base_dca"] = bool(metrics["xirr"] > base_metrics["xirr"])
        metrics["win_vs_0050_dca"] = bool(metrics["xirr"] > reference_1x_metrics["xirr"])
        metrics["drawdown_breach"] = bool(
            metrics["max_drawdown"] < context.dynamic_drawdown_limit
        )
        metrics["period_start"] = start.date().isoformat()
        metrics["period_end"] = end.date().isoformat()
        rows.append(metrics)
    if not rows:
        return penalized_metrics(reason="no_valid_periods")
    frame = pd.DataFrame(rows)
    turnover_sum = float(frame["turnover_sum"].mean())
    annual_turnover = turnover_sum / max(_period_years(periods), 1e-9)
    gate_passed = bool(
        not frame["drawdown_breach"].astype(bool).any()
        and annual_turnover <= float(params["max_annual_turnover"])
    )
    result = {
        "expected_xirr": float(frame["xirr"].mean()),
        "p05_xirr": float(frame["xirr"].quantile(0.05)),
        "sharpe": float(frame["sharpe"].mean()),
        "sortino": float(frame["sortino"].mean()),
        "win_rate_vs_base_dca": float(frame["win_vs_base_dca"].astype(float).mean()),
        "win_rate_vs_0050_dca": float(frame["win_vs_0050_dca"].astype(float).mean()),
        "p05_max_drawdown": float(frame["max_drawdown"].quantile(0.05)),
        "p05_drawdown_loss": abs(float(min(frame["max_drawdown"].quantile(0.05), 0.0))),
        "drawdown_breach_rate": float(frame["drawdown_breach"].astype(float).mean()),
        "cost_drag_on_contributed": float(frame["cost_drag_on_contributed"].mean()),
        "turnover_sum": turnover_sum,
        "annual_turnover": annual_turnover,
        "cohort_gate_passed": gate_passed,
        "gate_status": "pass" if gate_passed else "fail",
        "gate_reason": "" if gate_passed else "drawdown breach or turnover guard failed",
        "benchmark_id": official_benchmark_id(context.config),
    }
    if official_benchmark_id(context.config) == BENCHMARK_FIXED_1P5X_DCA:
        result[OFFICIAL_FIXED_1P5X_WIN_COLUMN] = result["win_rate_vs_base_dca"]
    if not gate_passed:
        result["expected_xirr"] = min(float(result["expected_xirr"]), -0.50)
        result["p05_xirr"] = min(float(result["p05_xirr"]), -0.50)
        result["drawdown_breach_rate"] = max(float(result["drawdown_breach_rate"]), 1.0)
    return result


def build_rule_strategy_weights(
    *,
    prices: pd.DataFrame,
    products: list[ProductSpec],
    params: dict[str, Any],
    sentiment: pd.Series | None,
    external_signals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    base = prices[products[0].ticker].astype(float)
    returns = base.pct_change()
    trend_window = int(params["trend_window"])
    slope_window = int(params["slope_window"])
    momentum_window = int(params["momentum_window"])
    vol_window = int(params["vol_window"])
    percentile_window = int(params["vol_percentile_window"])
    trend = base.rolling(trend_window).mean().shift(1)
    prior_price = base.shift(1)
    slope = (trend / trend.shift(slope_window) - 1.0).shift(1)
    momentum = (base / base.shift(momentum_window) - 1.0).shift(1)
    volatility = returns.rolling(vol_window).std(ddof=0).shift(1) * np.sqrt(252)
    vol_threshold = volatility.rolling(percentile_window).quantile(
        float(params["vol_spike_quantile"])
    )
    drawdown = (base / base.cummax() - 1.0).shift(1).fillna(0.0)
    risk_on = (
        (prior_price > trend * (1.0 + float(params["signal_hysteresis"])))
        & (momentum > 0.0)
        & (slope > 0.0)
    ).fillna(False)
    target = pd.Series(
        np.where(risk_on, float(params["risk_on_leverage"]), float(params["risk_off_leverage"])),
        index=prices.index,
        dtype="float64",
    )
    vol_cap = float(params["vol_target"]) / volatility.replace(0.0, np.nan)
    target = np.minimum(target, vol_cap.replace([np.inf, -np.inf], np.nan).fillna(1.0))
    target = target.mask(drawdown <= float(params["drawdown_guard"]), np.minimum(target, 1.0))
    target = target.mask(
        drawdown <= float(params["severe_drawdown_guard"]),
        float(params["cash_leverage"]),
    )
    target = target.mask(volatility > vol_threshold, np.minimum(target, 1.0))
    target = apply_sentiment_overlay(target, sentiment=sentiment, params=params)
    target = apply_external_signal_overlay(
        target,
        external_signals=external_signals,
        params=params,
    )
    target = apply_trade_filters(target.clip(0.0, max_product_leverage(products)), params=params)
    return target_series_to_weights(target, products)


def build_optuna_candidate_source_curves(
    *,
    prices: pd.DataFrame,
    products: list[ProductSpec],
    scenarios: pd.DataFrame,
    data_mode: str,
    cost_model: CostModel,
    product_assets: dict[str, AssetSpec],
    initial_cash: float,
    monthly_contribution: float,
    sentiment: pd.Series | None = None,
    external_signals: pd.DataFrame | None = None,
    cost_multiplier: float = 1.0,
) -> tuple[list[PolicyScenarioSpec], pd.DataFrame]:
    if scenarios.empty:
        return [], pd.DataFrame()
    required = {"trial_number", "params_json"}
    missing = required - set(scenarios.columns)
    if missing:
        raise ValueError(f"Optuna scenario CSV missing columns: {sorted(missing)}")
    specs: list[PolicyScenarioSpec] = []
    curves: list[pd.DataFrame] = []
    for row in scenarios.itertuples(index=False):
        trial_number = int(float(row.trial_number))
        params = json.loads(str(row.params_json))
        scenario_name = str(getattr(row, "scenario_name", f"optuna_trial_{trial_number}"))
        scenario_label = str(getattr(row, "scenario_label", f"Optuna V5 #{trial_number}"))
        if scenario_label.startswith(("Optuna V2 #", "Optuna V3 #", "Optuna V4 #")):
            scenario_label = scenario_label.replace("Optuna V2 #", "Optuna V4 #", 1)
            scenario_label = scenario_label.replace("Optuna V3 #", "Optuna V4 #", 1)
            scenario_label = scenario_label.replace("Optuna V4 #", "Optuna V5 #", 1)
        spec = PolicyScenarioSpec(
            name=scenario_name,
            label=scenario_label,
            short_label=f"Optuna #{trial_number}",
            family="optuna_return_first",
            kind="optuna_rule",
            params=params,
        )
        weights = build_rule_strategy_weights(
            prices=prices,
            products=products,
            params=params,
            sentiment=sentiment,
            external_signals=external_signals,
        )
        cadence = str(params.get("rebalance_cadence", "monthly"))
        is_monthly_core_weekly_delta = cadence == EXECUTION_MONTHLY_CORE_WEEKLY_DELTA
        curve, _allocations = simulate_weighted_strategy(
            prices=prices,
            target_weights=weights,
            product_leverages={product.ticker: product.leverage for product in products},
            initial_cash=float(initial_cash),
            rebalance_dates=(
                None
                if is_monthly_core_weekly_delta
                else rebalance_dates_for_cadence(prices.index, cadence)
            ),
            rebalance_on_change=is_monthly_core_weekly_delta,
            rebalance_on_contribution=cadence != "weekly",
            contribution_dates=monthly_rebalance_dates(prices.index),
            contribution_amount=float(monthly_contribution),
            cost_model=cost_model,
            product_assets=product_assets,
            cost_multiplier=cost_multiplier,
        )
        curve.insert(0, "scenario_id", f"{data_mode}--dca-policy-{scenario_name}")
        curve.insert(1, "data_mode", data_mode)
        curve.insert(2, "scenario_label", scenario_label)
        curve.insert(3, "short_label", spec.short_label)
        curve.insert(4, "strategy_family", spec.family)
        specs.append(spec)
        curves.append(curve)
    return specs, pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()


def apply_sentiment_overlay(
    target: pd.Series,
    *,
    sentiment: pd.Series | None,
    params: dict[str, Any],
) -> pd.Series:
    if sentiment is None or params.get("sentiment_mode") == "off":
        return target
    aligned = sentiment.reindex(target.index).ffill()
    adjusted = target.copy()
    deleverage = float(params["sentiment_deleverage"])
    fear = aligned <= float(params["fear_threshold"])
    greed = aligned >= float(params["greed_threshold"])
    if params["sentiment_mode"] == "contrarian":
        adjusted = adjusted.mask(fear, adjusted + deleverage)
        adjusted = adjusted.mask(greed, adjusted - deleverage)
    elif params["sentiment_mode"] == "risk_off":
        adjusted = adjusted.mask(fear | greed, adjusted - deleverage)
    return adjusted


def apply_external_signal_overlay(
    target: pd.Series,
    *,
    external_signals: pd.DataFrame | None,
    params: dict[str, Any],
) -> pd.Series:
    if external_signals is None or external_signals.empty:
        return target
    adjusted = target.copy()
    aligned = external_signals.reindex(target.index).ffill()
    for feature in EXTERNAL_SIGNAL_FEATURE_COLUMNS:
        mode = str(params.get(f"{feature}_mode", "off"))
        if mode == "off" or feature not in aligned.columns:
            continue
        values = pd.to_numeric(aligned[feature], errors="coerce")
        low_threshold = float(params.get(f"{feature}_low_threshold", 20.0))
        high_threshold = float(params.get(f"{feature}_high_threshold", 80.0))
        deleverage = float(params.get(f"{feature}_deleverage", 0.0))
        metadata = EXTERNAL_SIGNAL_FEATURE_METADATA.get(feature, {})
        risk_tail = str(metadata.get("risk_tail", "both"))
        low_event = values <= low_threshold
        high_event = values >= high_threshold
        if risk_tail == "high":
            risk_event = high_event
        elif risk_tail == "low":
            risk_event = low_event
        else:
            risk_event = low_event | high_event
        if mode == "risk_off":
            adjusted = adjusted.mask(risk_event, adjusted - deleverage)
        elif mode == "contrarian":
            adjusted = adjusted.mask(low_event, adjusted + deleverage)
            adjusted = adjusted.mask(high_event, adjusted - deleverage)
    return adjusted


def apply_trade_filters(target: pd.Series, *, params: dict[str, Any]) -> pd.Series:
    cadence = str(params["rebalance_cadence"])
    if cadence == EXECUTION_MONTHLY_CORE_WEEKLY_DELTA:
        return apply_monthly_core_weekly_delta_filter(target, params=params)
    threshold = float(params["rebalance_threshold"])
    min_holding_days = int(params["min_holding_days"])
    allowed_dates = _rebalance_allowed_dates(target.index, cadence)
    filtered = []
    current = float(target.iloc[0])
    last_change_date = pd.Timestamp(target.index[0])
    for date, desired in target.items():
        ts = pd.Timestamp(date)
        days_since_change = int((ts - last_change_date).days)
        can_change = cadence == "signal_only" or ts in allowed_dates
        big_enough = abs(float(desired) - current) >= threshold
        held_long_enough = days_since_change >= min_holding_days
        if can_change and big_enough and held_long_enough:
            current = float(desired)
            last_change_date = ts
        filtered.append(current)
    return pd.Series(filtered, index=target.index, dtype="float64")


def apply_monthly_core_weekly_delta_filter(
    target: pd.Series,
    *,
    params: dict[str, Any],
) -> pd.Series:
    threshold = max(
        float(params.get("rebalance_threshold", 0.0)),
        float(params.get("weekly_delta_threshold", 0.15)),
    )
    min_holding_days = int(params.get("min_holding_days", 0))
    index = pd.DatetimeIndex(target.index).sort_values()
    monthly_dates = monthly_rebalance_dates(index)
    weekly_dates = rebalance_dates_for_cadence(index, "weekly")
    filtered = []
    current = float(target.iloc[0])
    last_change_date = pd.Timestamp(target.index[0])
    for date, desired in target.items():
        ts = pd.Timestamp(date)
        desired_value = float(desired)
        diff = abs(desired_value - current)
        if ts in monthly_dates and diff > 1e-12:
            current = desired_value
            last_change_date = ts
        elif ts in weekly_dates:
            days_since_change = int((ts - last_change_date).days)
            if diff >= threshold and days_since_change >= min_holding_days:
                current = desired_value
                last_change_date = ts
        filtered.append(current)
    return pd.Series(filtered, index=target.index, dtype="float64")


def target_series_to_weights(target: pd.Series, products: list[ProductSpec]) -> pd.DataFrame:
    rows = [target_leverage_to_product_weights(float(value), products) for value in target]
    columns = [product.ticker for product in products] + [CASH]
    return pd.DataFrame(rows, index=target.index, columns=columns).astype(float)


def derive_dynamic_drawdown_limit(
    *,
    prices: pd.DataFrame,
    products: list[ProductSpec],
    product_assets: dict[str, AssetSpec],
    cost_model: CostModel,
    config: BacktestConfig,
) -> tuple[float, float]:
    curve = baseline_1x_curve(
        prices=prices,
        products=products,
        product_assets=product_assets,
        cost_model=cost_model,
        config=config,
    )
    max_drawdown = float(curve["drawdown"].astype(float).min())
    multiplier = float(config.strategy_search.drawdown_limit_multiplier)
    return max_drawdown, -min(abs(max_drawdown) * multiplier, 0.99)


def baseline_1x_curve(
    *,
    prices: pd.DataFrame,
    products: list[ProductSpec],
    product_assets: dict[str, AssetSpec],
    cost_model: CostModel,
    config: BacktestConfig,
) -> pd.DataFrame:
    weights = target_series_to_weights(pd.Series(1.0, index=prices.index), products)
    return simulate_search_curve(
        prices=prices,
        weights=weights,
        products=products,
        product_assets=product_assets,
        cost_model=cost_model,
        config=config,
    )


def benchmark_dca_curve(
    *,
    prices: pd.DataFrame,
    products: list[ProductSpec],
    product_assets: dict[str, AssetSpec],
    cost_model: CostModel,
    config: BacktestConfig,
) -> pd.DataFrame:
    target = benchmark_target_leverage(official_benchmark_id(config))
    weights = target_series_to_weights(pd.Series(target, index=prices.index), products)
    return fixed_target_dca_curve(
        prices=prices,
        weights=weights,
        products=products,
        product_assets=product_assets,
        cost_model=cost_model,
        config=config,
    )


def fixed_target_dca_curve(
    *,
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    products: list[ProductSpec],
    product_assets: dict[str, AssetSpec],
    cost_model: CostModel,
    config: BacktestConfig,
) -> pd.DataFrame:
    contribution_dates = monthly_rebalance_dates(prices.index)
    curve, _allocations = simulate_weighted_strategy(
        prices=prices,
        target_weights=weights,
        product_leverages={product.ticker: product.leverage for product in products},
        initial_cash=float(config.monthly_decision_replay.initial_cash),
        rebalance_dates=contribution_dates,
        contribution_dates=contribution_dates,
        contribution_amount=float(config.monthly_decision_replay.monthly_contribution),
        rebalance_on_change=False,
        rebalance_on_contribution=True,
        cost_model=cost_model,
        product_assets=product_assets,
    )
    return curve


def simulate_search_curve(
    *,
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    products: list[ProductSpec],
    product_assets: dict[str, AssetSpec],
    cost_model: CostModel,
    config: BacktestConfig,
) -> pd.DataFrame:
    contribution_dates = monthly_rebalance_dates(prices.index)
    cadence = _configured_execution_cadence(config)
    is_weekly_execution = cadence == "weekly"
    is_monthly_core_weekly_delta = cadence == EXECUTION_MONTHLY_CORE_WEEKLY_DELTA
    curve, _allocations = simulate_weighted_strategy(
        prices=prices,
        target_weights=weights,
        product_leverages={product.ticker: product.leverage for product in products},
        initial_cash=float(config.monthly_decision_replay.initial_cash),
        rebalance_dates=(
            rebalance_dates_for_cadence(prices.index, cadence) if is_weekly_execution else None
        ),
        contribution_dates=contribution_dates,
        contribution_amount=float(config.monthly_decision_replay.monthly_contribution),
        rebalance_on_change=not is_weekly_execution or is_monthly_core_weekly_delta,
        rebalance_on_contribution=not is_weekly_execution or is_monthly_core_weekly_delta,
        cost_model=cost_model,
        product_assets=product_assets,
    )
    return curve


def curve_metrics(curve: pd.DataFrame) -> dict[str, float]:
    returns = curve["investment_return"].astype(float).dropna()
    summary = performance_summary(returns) if not returns.empty else pd.Series(dtype=float)
    total_contributed = float(curve["total_contributed"].iloc[-1])
    total_trade_cost = float(curve["cumulative_trade_cost"].iloc[-1])
    return {
        "xirr": xirr_from_curve(curve),
        "sharpe": _summary_value(summary, "sharpe"),
        "sortino": _summary_value(summary, "sortino"),
        "max_drawdown": float(curve["drawdown"].astype(float).min()),
        "cost_drag_on_contributed": (
            total_trade_cost / total_contributed if total_contributed else np.nan
        ),
        "turnover_sum": float(curve["turnover"].astype(float).sum()),
    }


def xirr_from_curve(curve: pd.DataFrame) -> float:
    cash_flows: list[tuple[pd.Timestamp, float]] = []
    for row in curve.itertuples():
        contribution = float(row.contribution)
        if contribution > 0:
            cash_flows.append((pd.Timestamp(row.date), -contribution))
    if curve.empty:
        return np.nan
    cash_flows.append((pd.Timestamp(curve["date"].iloc[-1]), float(curve["total_equity"].iloc[-1])))
    return xirr(cash_flows)


def build_trial_rows(
    *,
    context: StrategySearchContext,
    study: Any,
    include_holdout: bool,
    cached_trials: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    holdout_cache = _holdout_cache(cached_trials)
    for trial in study.trials:
        if trial.state.name != "COMPLETE":
            rows.append({"trial_number": trial.number, "state": trial.state.name})
            continue
        attrs = dict(trial.user_attrs)
        params = _trial_params(trial)
        holdout_metrics = {}
        if include_holdout:
            holdout_metrics = holdout_cache.get(trial.number) or evaluate_strategy_params(
                context=context,
                params=params,
                use_holdout=True,
            )
        row = {
            "trial_number": trial.number,
            "state": trial.state.name,
            **{column: attrs.get(column, np.nan) for column in OBJECTIVE_COLUMNS},
            "benchmark_id": attrs.get("benchmark_id", official_benchmark_id(context.config)),
            "win_rate_vs_0050_dca": attrs.get(
                "win_rate_vs_0050_dca",
                attrs.get("win_rate_vs_base_dca", np.nan),
            ),
            OFFICIAL_FIXED_1P5X_WIN_COLUMN: attrs.get(
                OFFICIAL_FIXED_1P5X_WIN_COLUMN,
                attrs.get("win_rate_vs_base_dca", np.nan)
                if official_benchmark_id(context.config) == BENCHMARK_FIXED_1P5X_DCA
                else np.nan,
            ),
            "p05_max_drawdown": attrs.get("p05_max_drawdown", np.nan),
            "annual_turnover": attrs.get("annual_turnover", np.nan),
            "cohort_gate_passed": attrs.get("cohort_gate_passed", False),
            "gate_status": attrs.get("gate_status", ""),
            "gate_reason": attrs.get("gate_reason", ""),
            "holdout_expected_xirr": holdout_metrics.get("expected_xirr", np.nan),
            "holdout_p05_xirr": holdout_metrics.get("p05_xirr", np.nan),
            "holdout_p05_max_drawdown": holdout_metrics.get("p05_max_drawdown", np.nan),
            "holdout_cost_drag_on_contributed": holdout_metrics.get(
                "cost_drag_on_contributed", np.nan
            ),
            "holdout_turnover_sum": holdout_metrics.get("turnover_sum", np.nan),
            "holdout_win_rate_vs_0050_dca": holdout_metrics.get(
                "win_rate_vs_0050_dca", np.nan
            ),
            f"holdout_{OFFICIAL_FIXED_1P5X_WIN_COLUMN}": holdout_metrics.get(
                OFFICIAL_FIXED_1P5X_WIN_COLUMN,
                np.nan,
            ),
            "external_signal_set": external_signal_set(context),
            "external_signal_modes_json": external_signal_modes_json(params),
            "params_json": json.dumps(params, sort_keys=True),
        }
        rows.append(row)
    return rows


def _load_cached_trials(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def _holdout_cache(cached_trials: pd.DataFrame | None) -> dict[int, dict[str, float]]:
    if cached_trials is None or cached_trials.empty or "trial_number" not in cached_trials.columns:
        return {}
    required_columns = {
        "expected_xirr": "holdout_expected_xirr",
        "p05_xirr": "holdout_p05_xirr",
        "p05_max_drawdown": "holdout_p05_max_drawdown",
        "cost_drag_on_contributed": "holdout_cost_drag_on_contributed",
        "turnover_sum": "holdout_turnover_sum",
    }
    optional_columns = {
        "win_rate_vs_0050_dca": "holdout_win_rate_vs_0050_dca",
        OFFICIAL_FIXED_1P5X_WIN_COLUMN: f"holdout_{OFFICIAL_FIXED_1P5X_WIN_COLUMN}",
    }
    if not set(required_columns.values()).issubset(cached_trials.columns):
        return {}
    cache: dict[int, dict[str, float]] = {}
    for row in cached_trials.itertuples(index=False):
        trial_number = _safe_int(getattr(row, "trial_number", None))
        if trial_number is None:
            continue
        values = {
            metric: _safe_float(getattr(row, column, np.nan))
            for metric, column in required_columns.items()
        }
        for metric, column in optional_columns.items():
            if column not in cached_trials.columns:
                continue
            value = _safe_float(getattr(row, column, np.nan))
            if np.isfinite(value):
                values[metric] = value
        if all(np.isfinite(value) for value in values.values()):
            cache[trial_number] = values
    return cache


def _trial_params(trial: Any) -> dict[str, Any]:
    attrs = dict(getattr(trial, "user_attrs", {}) or {})
    params_json = attrs.get("params_json")
    if params_json:
        try:
            return json.loads(str(params_json))
        except json.JSONDecodeError:
            pass
    return dict(getattr(trial, "params", {}) or {})


def add_return_first_columns(frame: pd.DataFrame, *, baseline: dict[str, float]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    if "win_rate_vs_0050_dca" not in result.columns:
        result["win_rate_vs_0050_dca"] = result.get("win_rate_vs_base_dca", np.nan)
    if OFFICIAL_FIXED_1P5X_WIN_COLUMN not in result.columns:
        result[OFFICIAL_FIXED_1P5X_WIN_COLUMN] = np.nan
    if baseline.get("benchmark_id") == BENCHMARK_FIXED_1P5X_DCA:
        result[OFFICIAL_FIXED_1P5X_WIN_COLUMN] = result[
            OFFICIAL_FIXED_1P5X_WIN_COLUMN
        ].fillna(result.get("win_rate_vs_base_dca", np.nan))
    official_win_column = official_win_rate_column(result, baseline=baseline)

    expected = _numeric_series(result, "expected_xirr")
    win_rate = _numeric_series(result, official_win_column)
    p05_xirr = _numeric_series(result, "p05_xirr")
    breach = _numeric_series(result, "drawdown_breach_rate", default=0.0).fillna(1.0)
    cost = _numeric_series(result, "cost_drag_on_contributed", default=0.0)
    turnover = _numeric_series(result, "turnover_sum", default=0.0)
    cohort = result.get("cohort_gate_passed", True)
    cohort_passed = pd.Series(cohort, index=result.index).astype(bool)

    baseline_expected = _safe_float(baseline.get("expected_xirr", np.nan))
    baseline_win = _safe_float(baseline.get("win_rate_vs_base_dca", np.nan))
    baseline_p05 = _safe_float(baseline.get("p05_xirr", np.nan))
    baseline_drawdown = _safe_float(baseline.get("p05_max_drawdown", np.nan))
    baseline_cost = _safe_float(baseline.get("cost_drag_on_contributed", np.nan))
    baseline_turnover = _safe_float(baseline.get("turnover_sum", np.nan))

    result["xirr_delta_vs_baseline"] = expected - baseline_expected
    result["win_rate_delta_vs_baseline"] = win_rate - baseline_win
    result["xirr_delta_vs_fixed_1p5x_dca"] = (
        expected - baseline_expected
        if baseline.get("benchmark_id") == BENCHMARK_FIXED_1P5X_DCA
        else np.nan
    )
    result["win_rate_delta_vs_fixed_1p5x_dca"] = (
        win_rate - baseline_win
        if baseline.get("benchmark_id") == BENCHMARK_FIXED_1P5X_DCA
        else np.nan
    )
    result["p05_xirr_delta_vs_baseline"] = p05_xirr - baseline_p05
    result["p05_drawdown_delta_vs_baseline"] = (
        _numeric_series(result, "p05_max_drawdown") - baseline_drawdown
    )
    result["cost_drag_delta_vs_baseline"] = cost - baseline_cost
    result["turnover_delta_vs_baseline"] = turnover - baseline_turnover
    result["beats_baseline_expected_xirr"] = expected > baseline_expected
    result["beats_baseline_win_rate"] = win_rate > baseline_win
    result["return_first_score"] = (
        expected.fillna(-1.0) * 100.0
        + win_rate.fillna(0.0) * 10.0
        + p05_xirr.fillna(-1.0) * 25.0
        - breach * 100.0
        - np.maximum(
            cost.fillna(10.0) - (baseline_cost if np.isfinite(baseline_cost) else 0.0),
            0.0,
        )
        * 5.0
        - np.maximum(
            turnover.fillna(1_000.0)
            - (baseline_turnover * 2.0 if np.isfinite(baseline_turnover) else 1_000.0),
            0.0,
        )
        * 0.01
        - (~cohort_passed).astype(float) * 1_000.0
    )

    sort_frame = result.copy()
    sort_frame["_cohort_sort"] = cohort_passed.astype(int)
    sort_frame["_expected_sort"] = expected
    sort_frame["_win_sort"] = win_rate
    sort_frame["_p05_sort"] = p05_xirr
    sort_frame["_cost_sort"] = cost
    sort_frame["_turnover_sort"] = turnover
    ordered_index = sort_frame.sort_values(
        [
            "_cohort_sort",
            "_expected_sort",
            "_win_sort",
            "_p05_sort",
            "_cost_sort",
            "_turnover_sort",
        ],
        ascending=[False, False, False, False, True, True],
        na_position="last",
    ).index
    ranks = pd.Series(np.nan, index=result.index, dtype="float64")
    complete_mask = (
        result["state"].astype(str).eq("COMPLETE")
        if "state" in result.columns
        else pd.Series(True, index=result.index)
    )
    rank = 1
    for idx in ordered_index:
        if bool(complete_mask.loc[idx]):
            ranks.loc[idx] = rank
            rank += 1
    result["return_first_rank"] = ranks
    return result


def pareto_trials_frame(trials: pd.DataFrame, best_trials: list[Any]) -> pd.DataFrame:
    if trials.empty:
        return trials.copy()
    best_numbers = {trial.number for trial in best_trials}
    pareto = trials[trials["trial_number"].isin(best_numbers)].copy()
    if pareto.empty:
        return pareto
    sort_columns = [
        column
        for column in [
            "cohort_gate_passed",
            "return_first_rank",
            "expected_xirr",
            "p05_xirr",
            "cost_drag_on_contributed",
        ]
        if column in pareto.columns
    ]
    ascending = []
    for column in sort_columns:
        ascending.append(True if column == "return_first_rank" else False)
    if "cost_drag_on_contributed" in sort_columns:
        ascending[sort_columns.index("cost_drag_on_contributed")] = True
    pareto = pareto.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)
    pareto["pareto_rank"] = pareto.index + 1
    return pareto


def classify_best_candidates(
    pareto: pd.DataFrame,
    *,
    baseline: dict[str, float],
    objective_profile: str = OBJECTIVE_PROFILE_PARETO,
    triage: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if pareto.empty:
        return pareto.copy()
    profile = normalize_objective_profile(objective_profile)
    result = add_return_first_columns(pareto.copy(), baseline=baseline)
    triage_lookup = _triage_lookup(triage)
    baseline_cost = baseline.get("cost_drag_on_contributed", np.nan)
    baseline_p05_xirr = baseline.get("p05_xirr", np.nan)
    baseline_p05_drawdown = baseline.get("p05_max_drawdown", np.nan)
    baseline_turnover = baseline.get("turnover_sum", np.nan)
    stress_only_cost_gate = baseline.get("cost_gate_mode") == "stress_only"
    statuses = []
    reasons = []
    for row in result.itertuples(index=False):
        status = "candidate"
        reason = (
            "Return-first candidate passed search gates."
            if profile == OBJECTIVE_PROFILE_RETURN_FIRST
            else "Pareto candidate passed search gates."
        )
        if not bool(getattr(row, "cohort_gate_passed", False)):
            status = "rejected"
            reason = "Cohort/dynamic drawdown gate failed."
        elif int(row.trial_number) in triage_lookup:
            status, reason = triage_lookup[int(row.trial_number)]
        else:
            cost = float(getattr(row, "cost_drag_on_contributed", np.nan))
            p05_xirr = float(getattr(row, "p05_xirr", np.nan))
            p05_drawdown = float(getattr(row, "p05_max_drawdown", np.nan))
            cost_worse = np.isfinite(baseline_cost) and cost > baseline_cost
            turnover = float(getattr(row, "turnover_sum", np.nan))
            turnover_high = (
                np.isfinite(baseline_turnover)
                and np.isfinite(turnover)
                and turnover > baseline_turnover * 2.0
            )
            xirr_improved = np.isfinite(baseline_p05_xirr) and p05_xirr >= baseline_p05_xirr + 0.01
            dd_improved = (
                np.isfinite(baseline_p05_drawdown)
                and p05_drawdown >= baseline_p05_drawdown + 0.05
            )
            if (
                profile == OBJECTIVE_PROFILE_RETURN_FIRST
                and turnover_high
                and not stress_only_cost_gate
            ):
                status = "watchlist"
                reason = "Turnover is more than 2x baseline."
            elif (
                profile == OBJECTIVE_PROFILE_RETURN_FIRST
                and cost_worse
                and not stress_only_cost_gate
            ):
                status = "watchlist"
                reason = "Higher cost drag than baseline."
            elif cost_worse and not (xirr_improved or dd_improved):
                status = "watchlist"
                reason = "Higher cost drag without clear p05 XIRR or drawdown improvement."
        statuses.append(status)
        reasons.append(reason)
    result.insert(1, "candidate_status", statuses)
    result.insert(2, "candidate_reason", reasons)
    return result


def build_candidate_triage(
    trials: pd.DataFrame,
    *,
    baseline: dict[str, float],
    objective_profile: str = OBJECTIVE_PROFILE_PARETO,
) -> pd.DataFrame:
    if trials.empty or "state" not in trials.columns:
        return pd.DataFrame()
    profile = normalize_objective_profile(objective_profile)
    complete = add_return_first_columns(
        trials[trials["state"].astype(str).eq("COMPLETE")].copy(),
        baseline=baseline,
    )
    if complete.empty:
        return complete
    baseline_cost = float(baseline.get("cost_drag_on_contributed", np.nan))
    baseline_p05_xirr = float(baseline.get("p05_xirr", np.nan))
    baseline_p05_drawdown = float(baseline.get("p05_max_drawdown", np.nan))
    baseline_turnover = float(baseline.get("turnover_sum", np.nan))
    baseline_expected = float(baseline.get("expected_xirr", np.nan))
    baseline_win = float(baseline.get("win_rate_vs_base_dca", np.nan))
    official_win_column = official_win_rate_column(complete, baseline=baseline)
    stress_only_cost_gate = baseline.get("cost_gate_mode") == "stress_only"
    statuses: list[str] = []
    reasons: list[str] = []
    for row in complete.itertuples(index=False):
        status = "candidate"
        reason = (
            "Passed return-first gates."
            if profile == OBJECTIVE_PROFILE_RETURN_FIRST
            else "Passed triage gates."
        )
        cohort_passed = bool(getattr(row, "cohort_gate_passed", False))
        breach_rate = _safe_float(getattr(row, "drawdown_breach_rate", np.nan))
        p05_xirr = _safe_float(getattr(row, "p05_xirr", np.nan))
        p05_drawdown = _safe_float(getattr(row, "p05_max_drawdown", np.nan))
        expected_xirr = _safe_float(getattr(row, "expected_xirr", np.nan))
        official_win_rate = _safe_float(getattr(row, official_win_column, np.nan))
        cost = _safe_float(getattr(row, "cost_drag_on_contributed", np.nan))
        turnover = _safe_float(getattr(row, "turnover_sum", np.nan))
        holdout_xirr = _safe_float(getattr(row, "holdout_expected_xirr", np.nan))
        if not cohort_passed:
            status = "rejected"
            reason = "Cohort or dynamic drawdown gate failed."
        elif np.isfinite(breach_rate) and breach_rate > 0.0:
            status = "rejected"
            reason = "Drawdown breach rate is above zero."
        elif (
            profile != OBJECTIVE_PROFILE_RETURN_FIRST
            and np.isfinite(baseline_p05_xirr)
            and p05_xirr < baseline_p05_xirr
        ):
            status = "rejected"
            reason = "p05 XIRR is worse than baseline."
        elif np.isfinite(holdout_xirr) and holdout_xirr < 0.0:
            status = "rejected"
            reason = "Final holdout expected XIRR is negative."
        elif (
            baseline.get("benchmark_id") == BENCHMARK_FIXED_1P5X_DCA
            and np.isfinite(baseline_expected)
            and np.isfinite(expected_xirr)
            and expected_xirr <= baseline_expected
        ):
            status = "rejected"
            reason = "Expected XIRR does not beat fixed 1.5x DCA baseline."
        elif (
            baseline.get("benchmark_id") == BENCHMARK_FIXED_1P5X_DCA
            and np.isfinite(baseline_win)
            and np.isfinite(official_win_rate)
            and official_win_rate <= baseline_win
        ):
            status = "rejected"
            reason = "Win rate does not beat fixed 1.5x DCA baseline."
        else:
            cost_worse = np.isfinite(baseline_cost) and np.isfinite(cost) and cost > baseline_cost
            xirr_improved = np.isfinite(baseline_p05_xirr) and p05_xirr >= baseline_p05_xirr + 0.01
            dd_improved = (
                np.isfinite(baseline_p05_drawdown)
                and np.isfinite(p05_drawdown)
                and p05_drawdown >= baseline_p05_drawdown + 0.05
            )
            turnover_high = (
                np.isfinite(baseline_turnover)
                and np.isfinite(turnover)
                and turnover > baseline_turnover * 2.0
            )
            if (
                profile == OBJECTIVE_PROFILE_RETURN_FIRST
                and cost_worse
                and not stress_only_cost_gate
            ):
                status = "watchlist"
                reason = "Higher cost drag than baseline."
            elif (
                profile == OBJECTIVE_PROFILE_RETURN_FIRST
                and turnover_high
                and not stress_only_cost_gate
            ):
                status = "watchlist"
                reason = "Turnover is more than 2x baseline."
            elif cost_worse and not (xirr_improved or dd_improved):
                status = "watchlist"
                reason = "Higher cost drag without clear p05 XIRR or drawdown improvement."
            elif turnover_high:
                status = "watchlist"
                reason = "Turnover is more than 2x baseline."
            elif not np.isfinite(holdout_xirr):
                status = "watchlist"
                reason = "Final holdout metrics are missing."
        statuses.append(status)
        reasons.append(reason)
    complete.insert(1, "triage_status", statuses)
    complete.insert(2, "triage_reason", reasons)
    status_order = {"candidate": 0, "watchlist": 1, "rejected": 2}
    complete["_status_order"] = complete["triage_status"].map(status_order).fillna(99)
    if profile == OBJECTIVE_PROFILE_RETURN_FIRST:
        sort_columns = [
            "_status_order",
            "return_first_rank",
            "expected_xirr",
            official_win_column,
            "p05_xirr",
            "cost_drag_on_contributed",
        ]
        sort_columns = [column for column in sort_columns if column in complete.columns]
        ascending = [
            True,
            True,
            False,
            False,
            False,
            True,
        ][: len(sort_columns)]
        return (
            complete.sort_values(sort_columns, ascending=ascending)
            .drop(columns=["_status_order"])
            .reset_index(drop=True)
        )
    return (
        complete.sort_values(
            ["_status_order", "expected_xirr", "p05_xirr", "cost_drag_on_contributed"],
            ascending=[True, False, False, True],
        )
        .drop(columns=["_status_order"])
        .reset_index(drop=True)
    )


def _triage_lookup(triage: pd.DataFrame | None) -> dict[int, tuple[str, str]]:
    if triage is None or triage.empty or "trial_number" not in triage.columns:
        return {}
    result: dict[int, tuple[str, str]] = {}
    for row in triage.itertuples(index=False):
        try:
            trial_number = int(row.trial_number)
        except (TypeError, ValueError):
            continue
        result[trial_number] = (
            str(getattr(row, "triage_status", "")),
            str(getattr(row, "triage_reason", "")),
        )
    return result


def _render_strategy_search_html_legacy(
    *,
    study_name: str,
    trials: pd.DataFrame,
    pareto: pd.DataFrame,
    best_candidates: pd.DataFrame,
    candidate_triage: pd.DataFrame,
    dynamic_drawdown_limit: float,
    baseline: dict[str, float],
    study_status: dict[str, Any],
    export_only: bool,
) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{escape(str(row.trial_number))}</td>"
        f"<td>{escape(str(getattr(row, 'candidate_status', '')))}</td>"
        f"<td>{_format_percent(getattr(row, 'expected_xirr', np.nan))}</td>"
        f"<td>{_format_percent(getattr(row, 'p05_xirr', np.nan))}</td>"
        f"<td>{_format_percent(getattr(row, 'p05_max_drawdown', np.nan))}</td>"
        f"<td>{_format_percent(getattr(row, 'cost_drag_on_contributed', np.nan))}</td>"
        f"<td>{_format_number(getattr(row, 'turnover_sum', np.nan))}</td>"
        f"<td>{escape(str(getattr(row, 'external_signal_set', '')))}</td>"
        f"<td>{escape(str(getattr(row, 'candidate_reason', '')))}</td>"
        "</tr>"
        for row in best_candidates.head(30).itertuples(index=False)
    )
    triage_rows = "\n".join(
        "<tr>"
        f"<td>{escape(str(row.trial_number))}</td>"
        f"<td>{escape(str(getattr(row, 'triage_status', '')))}</td>"
        f"<td>{_format_percent(getattr(row, 'expected_xirr', np.nan))}</td>"
        f"<td>{_format_percent(getattr(row, 'p05_xirr', np.nan))}</td>"
        f"<td>{_format_percent(getattr(row, 'p05_max_drawdown', np.nan))}</td>"
        f"<td>{_format_percent(getattr(row, 'cost_drag_on_contributed', np.nan))}</td>"
        f"<td>{_format_number(getattr(row, 'turnover_sum', np.nan))}</td>"
        f"<td>{_format_percent(getattr(row, 'holdout_expected_xirr', np.nan))}</td>"
        f"<td>{escape(str(getattr(row, 'triage_reason', '')))}</td>"
        "</tr>"
        for row in candidate_triage.head(30).itertuples(index=False)
    )
    completed = int(study_status.get("completed_trials", 0))
    failed = int(study_status.get("failed_trials", 0))
    last_trial = study_status.get("last_trial_number", "")
    external = study_status.get("external_signal_set", "none")
    external_feature_set = study_status.get("external_signal_feature_set", "all")
    external_common_start = study_status.get("external_signal_common_start", "")
    external_common_end = study_status.get("external_signal_common_end", "")
    export_state = "export-only rebuild" if export_only else "fresh after optimization"
    return f"""<!doctype html>
<html lang="zh-Hant">
{render_html_head(title=f"Optuna Strategy Search {study_name}")}
<body>
<main>
  <section class="hero">
    <div class="panel">
      <p class="eyebrow">Optuna-first Strategy Search</p>
      <h1>{escape(study_name)}</h1>
      <p class="lede">
        Pareto 搜尋同時看報酬、尾端風險、交易成本與 turnover；
        結果仍需 replay/comparison/review 驗證。
      </p>
    </div>
    <div class="panel">
      <h2>Search Gate</h2>
      <div class="kpis">
        <div class="kpi"><span>Completed</span><strong>{completed:,}</strong></div>
        <div class="kpi"><span>Failed</span><strong>{failed:,}</strong></div>
        <div class="kpi"><span>Last trial</span><strong>{escape(str(last_trial))}</strong></div>
        <div class="kpi"><span>Pareto</span><strong>{len(pareto):,}</strong></div>
        <div class="kpi">
          <span>Hard DD</span>
          <strong>{_format_percent(dynamic_drawdown_limit)}</strong>
        </div>
        <div class="kpi">
          <span>Baseline cost drag</span>
          <strong>{_format_percent(baseline.get("cost_drag_on_contributed", np.nan))}</strong>
        </div>
      </div>
      <p><strong>External signals:</strong> {escape(str(external))}</p>
      <p>
        <strong>External feature set:</strong> {escape(str(external_feature_set))}
        {escape(str(external_common_start))} to {escape(str(external_common_end))}
      </p>
      <p><strong>Export status:</strong> {escape(export_state)}</p>
    </div>
  </section>
  <section class="panel">
    <h2>Candidate Triage</h2>
    <p>
      Triage filters research results by cohort gate, zero breach, p05 XIRR, cost drag,
      turnover, and final holdout behavior. These rows are still research candidates,
      not actionable defaults.
    </p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Trial</th><th>Status</th><th>Expected XIRR</th><th>p05 XIRR</th>
            <th>p05 DD</th><th>Cost Drag</th><th>Turnover</th><th>Holdout XIRR</th>
            <th>Reason</th>
          </tr>
        </thead>
        <tbody>{triage_rows}</tbody>
      </table>
    </div>
  </section>
  <section class="panel">
    <h2>Best Candidates</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Trial</th><th>Status</th><th>Expected XIRR</th><th>p05 XIRR</th>
            <th>p05 DD</th><th>Cost Drag</th><th>Turnover</th><th>External</th><th>Reason</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
  </section>
</main>
</body>
</html>
"""


def _candidate_view(
    frame: pd.DataFrame,
    *,
    sort_columns: list[str],
    ascending: list[bool],
    limit: int = 30,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    status_column = "triage_status" if "triage_status" in result.columns else "candidate_status"
    if status_column in result.columns:
        result["_status_order"] = result[status_column].map(
            {"candidate": 0, "watchlist": 1, "rejected": 2}
        ).fillna(99)
        sort_columns = ["_status_order", *sort_columns]
        ascending = [True, *ascending]
    present = [column for column in sort_columns if column in result.columns]
    present_ascending = [
        ascending[index] for index, column in enumerate(sort_columns) if column in result.columns
    ]
    if present:
        result = result.sort_values(present, ascending=present_ascending, na_position="last")
    return result.drop(columns=["_status_order"], errors="ignore").head(limit)


def _candidate_table(
    frame: pd.DataFrame,
    *,
    status_column: str = "triage_status",
    reason_column: str = "triage_reason",
) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{_format_number(getattr(row, 'return_first_rank', np.nan))}</td>"
        f"<td>{escape(str(getattr(row, 'trial_number', '')))}</td>"
        f"<td>{escape(str(getattr(row, status_column, '')))}</td>"
        f"<td>{_format_number(getattr(row, 'return_first_score', np.nan))}</td>"
        f"<td>{_format_percent(getattr(row, 'expected_xirr', np.nan))}</td>"
        f"<td>{_format_percent(getattr(row, 'win_rate_vs_0050_dca', np.nan))}</td>"
        f"<td>{_format_percent(getattr(row, 'xirr_delta_vs_baseline', np.nan))}</td>"
        f"<td>{_format_percent(getattr(row, 'p05_xirr', np.nan))}</td>"
        f"<td>{_format_percent(getattr(row, 'p05_max_drawdown', np.nan))}</td>"
        f"<td>{_format_percent(getattr(row, 'cost_drag_on_contributed', np.nan))}</td>"
        f"<td>{_format_number(getattr(row, 'turnover_sum', np.nan))}</td>"
        f"<td>{_format_percent(getattr(row, 'holdout_expected_xirr', np.nan))}</td>"
        f"<td>{escape(str(getattr(row, reason_column, '')))}</td>"
        "</tr>"
        for row in frame.itertuples(index=False)
    )
    return f"""
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Rank</th><th>Trial</th><th>Status</th><th>Score</th>
            <th>Expected XIRR</th><th>Win vs 0050 DCA</th><th>XIRR vs baseline</th>
            <th>p05 XIRR</th><th>p05 DD</th><th>Cost Drag</th><th>Turnover</th>
            <th>Holdout XIRR</th><th>Reason</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
"""


def render_strategy_search_html(
    *,
    study_name: str,
    trials: pd.DataFrame,
    pareto: pd.DataFrame,
    best_candidates: pd.DataFrame,
    candidate_triage: pd.DataFrame,
    dynamic_drawdown_limit: float,
    baseline: dict[str, float],
    study_status: dict[str, Any],
    export_only: bool,
    objective_profile: str,
) -> str:
    profile = normalize_objective_profile(objective_profile)
    completed = int(study_status.get("completed_trials", 0))
    failed = int(study_status.get("failed_trials", 0))
    last_trial = study_status.get("last_trial_number", "")
    external = study_status.get("external_signal_set", "none")
    export_state = "export-only rebuild" if export_only else "fresh after optimization"
    if profile == OBJECTIVE_PROFILE_RETURN_FIRST:
        return_leaders_table = _candidate_table(
            _candidate_view(
                candidate_triage,
                sort_columns=["return_first_rank"],
                ascending=[True],
            )
        )
        high_win_table = _candidate_table(
            _candidate_view(
                candidate_triage,
                sort_columns=[
                    "win_rate_vs_0050_dca",
                    "expected_xirr",
                    "p05_xirr",
                    "cost_drag_on_contributed",
                ],
                ascending=[False, False, False, True],
            )
        )
        balanced_table = _candidate_table(
            _candidate_view(
                candidate_triage,
                sort_columns=[
                    "p05_xirr",
                    "p05_max_drawdown",
                    "expected_xirr",
                    "cost_drag_on_contributed",
                ],
                ascending=[False, False, False, True],
            )
        )
        report_sections = f"""
  <section class="panel">
    <h2>Return Leaders</h2>
    <p>
      Primary view: expected XIRR first, win rate vs 0050 DCA second, p05 XIRR third.
      Raw return leaders that failed final holdout stay rejected and must not be replayed.
    </p>
    {return_leaders_table}
  </section>
  <section class="panel">
    <h2>High Win Rate Leaders</h2>
    <p>Same candidate pool, sorted by win_rate_vs_0050_dca before expected XIRR.</p>
    {high_win_table}
  </section>
  <section class="panel">
    <h2>Balanced Backup</h2>
    <p>Fallback view for candidates with stronger tail XIRR, drawdown, and cost behavior.</p>
    {balanced_table}
  </section>
"""
    else:
        best_table = _candidate_table(
            best_candidates.head(30),
            status_column="candidate_status",
            reason_column="candidate_reason",
        )
        report_sections = f"""
  <section class="panel">
    <h2>Candidate Triage</h2>
    <p>
      Triage filters research results by cohort gate, zero breach, p05 XIRR, cost drag,
      turnover, and final holdout behavior. These rows are still research candidates,
      not actionable defaults.
    </p>
    {_candidate_table(candidate_triage.head(30))}
  </section>
  <section class="panel">
    <h2>Best Candidates</h2>
    {best_table}
  </section>
"""
    return f"""<!doctype html>
<html lang="zh-Hant">
{render_html_head(title=f"Optuna Strategy Search {study_name}")}
<body>
<main>
  <section class="hero">
    <div class="panel">
      <p class="eyebrow">Optuna-first Strategy Search</p>
      <h1>{escape(study_name)}</h1>
      <p class="lede">
        Research search results only. Candidates must still pass TW50 replay,
        comparison, and manual review before becoming actionable defaults.
      </p>
    </div>
    <div class="panel">
      <h2>Search Gate</h2>
      <div class="kpis">
        <div class="kpi"><span>Completed</span><strong>{completed:,}</strong></div>
        <div class="kpi"><span>Failed</span><strong>{failed:,}</strong></div>
        <div class="kpi"><span>Last trial</span><strong>{escape(str(last_trial))}</strong></div>
        <div class="kpi"><span>Pareto</span><strong>{len(pareto):,}</strong></div>
        <div class="kpi"><span>Profile</span><strong>{escape(profile)}</strong></div>
        <div class="kpi">
          <span>Hard DD</span><strong>{_format_percent(dynamic_drawdown_limit)}</strong>
        </div>
        <div class="kpi">
          <span>Baseline XIRR</span>
          <strong>{_format_percent(baseline.get("expected_xirr", np.nan))}</strong>
        </div>
        <div class="kpi">
          <span>Baseline win</span>
          <strong>{_format_percent(baseline.get("win_rate_vs_base_dca", np.nan))}</strong>
        </div>
      </div>
      <p><strong>External signals:</strong> {escape(str(external))}</p>
      <p><strong>Export status:</strong> {escape(export_state)}</p>
      <p>
        <strong>Baseline Vol Target 63D 25%:</strong>
        expected XIRR {_format_percent(baseline.get("expected_xirr", np.nan))},
        win rate vs 0050 DCA {_format_percent(baseline.get("win_rate_vs_base_dca", np.nan))},
        p05 XIRR {_format_percent(baseline.get("p05_xirr", np.nan))},
        p05 DD {_format_percent(baseline.get("p05_max_drawdown", np.nan))}.
      </p>
    </div>
  </section>
  {report_sections}
</main>
</body>
</html>
"""


def validation_folds(
    index: pd.Index,
    *,
    holdout_start: pd.Timestamp,
    fold_count: int = 4,
) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
    dates = pd.DatetimeIndex(index).sort_values()
    train_dates = dates[dates < holdout_start]
    if len(train_dates) < 80:
        raise ValueError("Not enough pre-holdout data for strategy search validation folds.")
    fold_count = max(1, min(int(fold_count), len(train_dates) // 20))
    splits = np.array_split(train_dates, fold_count)
    folds = []
    for split in splits:
        if len(split) >= 20:
            folds.append((pd.Timestamp(split[0]), pd.Timestamp(split[-1])))
    if not folds:
        raise ValueError("No usable validation folds were created.")
    return tuple(folds)


def load_current_baseline(
    *,
    output_dir: Path,
    family: str,
    benchmark: str = BENCHMARK_BASE_DCA,
) -> dict[str, float]:
    ranking_path = output_dir / f"monthly_decision_replay_{family}_ranking.csv"
    if not ranking_path.exists():
        return {}
    ranking = pd.read_csv(ranking_path)
    normalized_benchmark = str(benchmark or BENCHMARK_BASE_DCA).lower()
    if normalized_benchmark == BENCHMARK_FIXED_1P5X_DCA:
        mask = ranking["scenario_id"].astype(str).str.endswith("constant_1p5")
    else:
        mask = ranking["scenario_label"].astype(str).eq("Vol Target 63D 25%")
    if not mask.any():
        return {}
    row = ranking.loc[mask].iloc[0]
    return {
        "benchmark_id": normalized_benchmark,
        "expected_xirr": float(row.get("expected_xirr", np.nan)),
        "win_rate_vs_base_dca": float(
            row.get("win_rate_vs_benchmark", row.get("win_rate_vs_qqq_dca", np.nan))
        ),
        OFFICIAL_FIXED_1P5X_WIN_COLUMN: float(
            row.get(OFFICIAL_FIXED_1P5X_WIN_COLUMN, np.nan)
        ),
        "win_rate_vs_0050_dca": float(row.get("win_rate_vs_0050_dca", np.nan)),
        "p05_xirr": float(row.get("p05_xirr", np.nan)),
        "p05_max_drawdown": float(row.get("p05_max_drawdown", np.nan)),
        "cost_drag_on_contributed": float(row.get("cost_drag_on_contributed", np.nan)),
        "turnover_sum": float(row.get("turnover_sum", np.nan)),
        "cost_gate_mode": (
            "stress_only" if normalized_benchmark == BENCHMARK_FIXED_1P5X_DCA else "baseline"
        ),
    }


def estimate_search_baseline(
    *,
    prices: pd.DataFrame,
    products: list[ProductSpec],
    product_assets: dict[str, AssetSpec],
    cost_model: CostModel,
    config: BacktestConfig,
    periods: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> dict[str, float]:
    rows = []
    for start, end in periods:
        period_prices = _slice_period(prices, start, end)
        if len(period_prices) < 20:
            continue
        rows.append(
            curve_metrics(
                benchmark_dca_curve(
                    prices=period_prices,
                    products=products,
                    product_assets=product_assets,
                    cost_model=cost_model,
                    config=config,
                )
            )
        )
    if not rows:
        return {"benchmark_id": official_benchmark_id(config)}
    frame = pd.DataFrame(rows)
    benchmark_id = official_benchmark_id(config)
    baseline = {
        "benchmark_id": benchmark_id,
        "expected_xirr": float(frame["xirr"].mean()),
        "win_rate_vs_base_dca": 0.50,
        "p05_xirr": float(frame["xirr"].quantile(0.05)),
        "p05_max_drawdown": float(frame["max_drawdown"].quantile(0.05)),
        "cost_drag_on_contributed": float(frame["cost_drag_on_contributed"].mean()),
        "turnover_sum": float(frame["turnover_sum"].mean()),
        "cost_gate_mode": "stress_only" if benchmark_id == BENCHMARK_FIXED_1P5X_DCA else "baseline",
    }
    if benchmark_id == BENCHMARK_FIXED_1P5X_DCA:
        baseline[OFFICIAL_FIXED_1P5X_WIN_COLUMN] = 0.50
    return baseline


def official_benchmark_id(config: BacktestConfig) -> str:
    search_benchmark = str(getattr(config.strategy_search, "benchmark", "") or "").lower()
    if search_benchmark:
        return search_benchmark
    return str(config.monthly_decision_replay.benchmark or BENCHMARK_BASE_DCA).lower()


def benchmark_target_leverage(benchmark: str) -> float:
    normalized = str(benchmark or BENCHMARK_BASE_DCA).lower()
    if normalized == BENCHMARK_FIXED_1P5X_DCA:
        return 1.5
    if normalized in {BENCHMARK_BASE_DCA, BENCHMARK_QQQ_DCA, "0050_dca"}:
        return 1.0
    raise ValueError(f"Unsupported strategy search benchmark: {benchmark!r}.")


def official_win_rate_column(frame: pd.DataFrame, *, baseline: dict[str, float]) -> str:
    if baseline.get("benchmark_id") == BENCHMARK_FIXED_1P5X_DCA:
        return OFFICIAL_FIXED_1P5X_WIN_COLUMN
    if "win_rate_vs_base_dca" in frame.columns:
        return "win_rate_vs_base_dca"
    return REFERENCE_1X_WIN_COLUMN


def _slice_period(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return frame[(frame.index >= start) & (frame.index <= end)].copy()


def _rebalance_allowed_dates(index: pd.Index, cadence: str) -> set[pd.Timestamp]:
    dates = pd.DatetimeIndex(index).sort_values()
    return rebalance_dates_for_cadence(dates, cadence)


def _configured_execution_cadence(config: BacktestConfig) -> str:
    cadence = str(config.strategy_search.execution_cadence or "").lower().replace("-", "_")
    if cadence in {"weekly", "monthly", "quarterly", EXECUTION_MONTHLY_CORE_WEEKLY_DELTA}:
        return cadence
    policy_cadence = str(config.dca_policy_optimizer.rebalance_cadence).lower()
    return policy_cadence if policy_cadence in {"weekly", "monthly", "quarterly"} else "monthly"


def _period_years(periods: list[tuple[pd.Timestamp, pd.Timestamp]]) -> float:
    return sum(max((end - start).days / 365.25, 0.0) for start, end in periods)


def max_product_leverage(products: list[ProductSpec]) -> float:
    return max(float(product.leverage) for product in products)


def penalized_metrics(*, reason: str) -> dict[str, Any]:
    return {
        "expected_xirr": -1.0,
        "p05_xirr": -1.0,
        "sharpe": -10.0,
        "sortino": -10.0,
        "win_rate_vs_base_dca": 0.0,
        "win_rate_vs_0050_dca": 0.0,
        OFFICIAL_FIXED_1P5X_WIN_COLUMN: 0.0,
        "p05_max_drawdown": -1.0,
        "p05_drawdown_loss": 1.0,
        "drawdown_breach_rate": 1.0,
        "cost_drag_on_contributed": 10.0,
        "turnover_sum": 1_000.0,
        "annual_turnover": 1_000.0,
        "cohort_gate_passed": False,
        "gate_status": "fail",
        "gate_reason": reason,
    }


def _summary_value(summary: pd.Series, key: str) -> float:
    value = summary.get(key, np.nan)
    return float(value) if pd.notna(value) else np.nan


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _format_percent(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def _format_number(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):,.2f}"


def external_signal_set(context: StrategySearchContext) -> str:
    if context.external_signals is None or context.external_signals.empty:
        return "none"
    columns = [
        column
        for column in context.external_signal_feature_columns
        if column in context.external_signals.columns
        and pd.to_numeric(context.external_signals[column], errors="coerce").notna().any()
    ]
    return ",".join(columns) if columns else "none"


def external_signal_modes_json(params: dict[str, Any]) -> str:
    modes = {
        feature: {
            "mode": params.get(f"{feature}_mode", "off"),
            "low_threshold": params.get(f"{feature}_low_threshold"),
            "high_threshold": params.get(f"{feature}_high_threshold"),
            "deleverage": params.get(f"{feature}_deleverage"),
        }
        for feature in EXTERNAL_SIGNAL_FEATURE_COLUMNS
        if params.get(f"{feature}_mode", "off") != "off"
    }
    return json.dumps(modes, sort_keys=True)


def study_status(
    *,
    study: Any,
    context: StrategySearchContext,
    storage_path: Path,
    objective_profile: str,
) -> dict[str, Any]:
    states = [trial.state.name for trial in study.trials]
    return {
        "completed_trials": states.count("COMPLETE"),
        "failed_trials": states.count("FAIL"),
        "running_trials": states.count("RUNNING"),
        "waiting_trials": states.count("WAITING"),
        "last_trial_number": max((trial.number for trial in study.trials), default=""),
        "external_signal_set": external_signal_set(context),
        "external_signal_feature_set": context.search_config.external_signal_feature_set,
        "external_signal_common_start": context.external_signal_common_start,
        "external_signal_common_end": context.external_signal_common_end,
        "objective_profile": objective_profile,
        "execution_profile": context.search_config.execution_profile,
        "execution_cadence": context.search_config.execution_cadence,
        "contribution_cadence": context.search_config.contribution_cadence,
        "study_fingerprint_sha256": study.user_attrs.get("study_fingerprint_sha256", ""),
        "storage_path": str(storage_path),
    }


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _numeric_series(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(default, index=frame.index, dtype="float64")


def _require_optuna() -> Any:
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - exercised before dependency install
        raise RuntimeError("Optuna is required for strategy search. Run `uv sync`.") from exc
    return optuna


__all__ = [
    "OBJECTIVE_COLUMNS",
    "OBJECTIVE_PROFILE_PARETO",
    "OBJECTIVE_PROFILE_RETURN_FIRST",
    "RETURN_FIRST_OBJECTIVE_COLUMNS",
    "SUPPORTED_OBJECTIVE_PROFILES",
    "StrategySearchContext",
    "StrategySearchResult",
    "add_return_first_columns",
    "build_optuna_candidate_source_curves",
    "apply_sentiment_overlay",
    "apply_external_signal_overlay",
    "apply_trade_filters",
    "build_rule_strategy_weights",
    "build_strategy_search_context",
    "build_candidate_triage",
    "classify_best_candidates",
    "derive_dynamic_drawdown_limit",
    "evaluate_strategy_params",
    "load_frozen_sentiment",
    "normalize_objective_profile",
    "run_optuna_strategy_search",
    "validation_folds",
]
