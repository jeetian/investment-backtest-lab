from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from investment_backtest_lab.dca_policy_optimizer import (
    DATA_MODE_ACTUAL,
    DATA_MODE_SYNTHETIC,
    _cohort_source_curves,
    build_policy_scenario_specs,
    build_rolling_cohort_validation,
)
from investment_backtest_lab.html_ui import render_html_head
from investment_backtest_lab.leveraged_etf_lab import ProductSpec, monthly_rebalance_dates
from investment_backtest_lab.models import DCAPolicyOptimizerConfig, MonthlyDecisionReplayConfig

SELECTOR_ACTUAL_PRIMARY = "actual_primary"
SELECTOR_SYNTHETIC_PRIMARY = "synthetic_primary"
SELECTOR_HYBRID_PRIMARY = "hybrid_primary"
DATA_MODE_HYBRID = "hybrid_primary"
BENCHMARK_QQQ_DCA = "qqq_dca"


@dataclass(frozen=True)
class HybridPriceResult:
    prices: pd.DataFrame
    coverage: pd.DataFrame


@dataclass(frozen=True)
class MonthlyDecisionReplayOutputs:
    cohorts: pd.DataFrame
    ranking: pd.DataFrame
    decisions: pd.DataFrame
    equity: pd.DataFrame
    mc_trials: pd.DataFrame
    source_coverage: pd.DataFrame
    selector: str
    ranking_method: str


@dataclass(frozen=True)
class MonthlyDecisionReplayReportResult:
    cohorts: pd.DataFrame
    ranking: pd.DataFrame
    decisions: pd.DataFrame
    equity: pd.DataFrame
    mc_trials: pd.DataFrame
    mc_summary: pd.DataFrame
    source_coverage: pd.DataFrame
    html_path: Path
    decisions_path: Path
    cohorts_path: Path
    ranking_path: Path
    equity_path: Path
    mc_trials_path: Path
    mc_summary_path: Path
    source_coverage_path: Path


def build_hybrid_actual_preferred_prices(
    *,
    actual_prices: pd.DataFrame,
    synthetic_prices: pd.DataFrame,
    products: list[ProductSpec],
) -> HybridPriceResult:
    """Use actual ETF prices when available and synthetic backfill before listing."""

    if actual_prices.empty and synthetic_prices.empty:
        raise ValueError("hybrid_primary requires actual or synthetic prices.")

    columns = [product.ticker for product in products]
    combined_columns: dict[str, pd.Series] = {}
    coverage_rows: list[dict[str, Any]] = []

    for ticker in columns:
        actual = _clean_series(actual_prices.get(ticker))
        synthetic = _clean_series(synthetic_prices.get(ticker))
        if actual.empty and synthetic.empty:
            raise ValueError(f"hybrid_primary has no actual or synthetic prices for {ticker}.")

        if actual.empty:
            combined = synthetic.copy()
            splice_date = ""
            splice_scale = np.nan
            backfill = synthetic.copy()
        elif synthetic.empty:
            combined = actual.copy()
            splice_date = actual.index.min().date().isoformat()
            splice_scale = np.nan
            backfill = pd.Series(dtype="float64")
        else:
            first_actual = pd.Timestamp(actual.index.min())
            synthetic_anchor = synthetic.loc[synthetic.index <= first_actual]
            if synthetic_anchor.empty:
                backfill = pd.Series(dtype="float64")
                splice_scale = np.nan
            else:
                anchor_value = float(synthetic_anchor.iloc[-1])
                splice_scale = float(actual.iloc[0]) / anchor_value if anchor_value else np.nan
                backfill = synthetic.loc[synthetic.index < first_actual] * splice_scale
            combined = pd.concat([backfill, actual]).sort_index()
            splice_date = first_actual.date().isoformat()

        combined_columns[ticker] = combined
        coverage_rows.append(
            {
                "ticker": ticker,
                "replay_start": _date_or_blank(
                    combined.index.min() if not combined.empty else None
                ),
                "replay_end": _date_or_blank(combined.index.max() if not combined.empty else None),
                "actual_start": _date_or_blank(actual.index.min() if not actual.empty else None),
                "actual_end": _date_or_blank(actual.index.max() if not actual.empty else None),
                "synthetic_backfill_start": _date_or_blank(
                    backfill.index.min() if not backfill.empty else None
                ),
                "synthetic_backfill_end": _date_or_blank(
                    backfill.index.max() if not backfill.empty else None
                ),
                "splice_date": splice_date,
                "splice_scale": splice_scale,
            }
        )

    prices = pd.DataFrame(combined_columns).dropna(how="any").sort_index()
    if prices.empty:
        raise ValueError("hybrid_primary prices are empty after source alignment.")
    return HybridPriceResult(
        prices=prices.astype(float),
        coverage=pd.DataFrame(coverage_rows),
    )


def build_monthly_decision_replay_outputs(
    *,
    mode_prices: dict[str, pd.DataFrame],
    products: list[ProductSpec],
    optimizer_config: DCAPolicyOptimizerConfig,
    replay_config: MonthlyDecisionReplayConfig,
    scan_mode: str,
    selector: str | None = None,
    source_coverage: pd.DataFrame | None = None,
    progress: Callable[[str], None] | None = None,
) -> MonthlyDecisionReplayOutputs:
    progress = progress or (lambda _message: None)
    normalized_selector = (selector or replay_config.selector).lower()
    source_mode = _source_mode_for_selector(normalized_selector)
    prices = mode_prices.get(source_mode)
    if prices is None or prices.empty:
        raise ValueError(f"{normalized_selector} requires {source_mode} prices.")

    progress("Preparing replay optimizer configuration.")
    replay_optimizer_config = _optimizer_config_for_replay(
        optimizer_config=optimizer_config,
        replay_config=replay_config,
    )
    progress("Building replay policy scenario grid.")
    specs = build_policy_scenario_specs(
        config=replay_optimizer_config,
        products=products,
        scan_mode=scan_mode,
    )
    product_leverages = {product.ticker: product.leverage for product in products}
    coverage = (
        source_coverage.copy()
        if source_coverage is not None and not source_coverage.empty
        else _source_coverage_from_prices(prices, products)
    )

    progress(f"Building replay source curves ({len(specs)} scenarios).")
    source_curves = _cohort_source_curves(
        mode_prices={source_mode: prices},
        specs=specs,
        products=products,
        product_leverages=product_leverages,
        config=replay_optimizer_config,
        base_curves=None,
    )
    progress(
        "Running deterministic rolling cohort gate "
        f"({len(specs)} scenarios, horizons={replay_optimizer_config.cohort_horizons_years})."
    )
    cohorts = build_rolling_cohort_validation(
        mode_prices={source_mode: prices},
        specs=specs,
        products=products,
        product_leverages=product_leverages,
        config=replay_optimizer_config,
        base_curves=source_curves,
    )
    progress(f"Rolling cohort gate produced {len(cohorts):,} rows.")
    if cohorts.empty:
        return MonthlyDecisionReplayOutputs(
            cohorts=_empty_replay_cohorts_frame(),
            ranking=_empty_replay_ranking_frame(),
            decisions=_empty_replay_decisions_frame(),
            equity=_empty_replay_equity_frame(),
            mc_trials=_empty_mc_trials_frame(),
            source_coverage=coverage,
            selector=normalized_selector,
            ranking_method="none",
        )

    progress("Attaching QQQ DCA benchmark to deterministic cohorts.")
    enriched = _attach_benchmark(
        cohorts,
        selector=normalized_selector,
        benchmark=replay_config.benchmark,
    )

    ranking_method = "rolling_cohort"
    mc_trials = _empty_mc_trials_frame()
    if replay_config.monte_carlo_enabled:
        samples = _monte_carlo_samples_for_scan_mode(replay_config, scan_mode)
        progress(
            "Running Monte Carlo replay "
            f"(block_lengths={replay_config.monte_carlo_block_lengths_days}, "
            f"samples_per_scale={samples}, seed={replay_config.monte_carlo_seed})."
        )
        mc_trials = build_monte_carlo_replay_trials(
            source_curves=source_curves,
            selector=normalized_selector,
            data_mode=source_mode,
            replay_config=replay_config,
            samples_per_scale=samples,
            progress=progress,
        )
        progress(f"Monte Carlo replay produced {len(mc_trials):,} scenario trial rows.")
        ranking = build_monte_carlo_replay_ranking(
            mc_trials,
            cohorts=enriched,
            selector=normalized_selector,
            config=replay_config,
        )
        ranking_method = "monte_carlo"
    else:
        progress("Ranking replay candidates from deterministic cohorts.")
        ranking = build_replay_ranking(
            enriched,
            selector=normalized_selector,
            config=replay_config,
        )

    progress("Building replay decision and equity audit tables.")
    decisions = build_replay_decisions(
        enriched,
        ranking=ranking,
        selector=normalized_selector,
    )
    equity = build_replay_equity(decisions)
    return MonthlyDecisionReplayOutputs(
        cohorts=enriched,
        ranking=ranking,
        decisions=decisions,
        equity=equity,
        mc_trials=mc_trials,
        source_coverage=coverage,
        selector=normalized_selector,
        ranking_method=ranking_method,
    )


def build_monte_carlo_replay_trials(
    *,
    source_curves: pd.DataFrame,
    selector: str,
    data_mode: str,
    replay_config: MonthlyDecisionReplayConfig,
    samples_per_scale: int,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    if source_curves.empty:
        return _empty_mc_trials_frame()
    progress = progress or (lambda _message: None)
    returns, metadata = _source_return_matrix(source_curves)
    if returns.empty:
        return _empty_mc_trials_frame()
    benchmark_id = _benchmark_scenario_id(returns.columns)
    scenario_ids = list(returns.columns)
    benchmark_position = scenario_ids.index(benchmark_id)
    rng = np.random.default_rng(int(replay_config.monte_carlo_seed))
    rows: list[pd.DataFrame] = []
    returns_matrix = returns.to_numpy(dtype=float)

    for horizon in replay_config.horizons_years:
        target_days = max(int(round(int(horizon) * 252)), 2)
        path_dates = _mc_path_dates(target_days)
        for block_length in replay_config.monte_carlo_block_lengths_days:
            progress(
                f"Monte Carlo horizon={horizon}y block={block_length}d "
                f"samples={samples_per_scale}."
            )
            for sample_index in range(samples_per_scale):
                indices = _bootstrap_indices(
                    source_length=len(returns),
                    target_length=target_days,
                    block_length=int(block_length),
                    rng=rng,
                )
                sampled = returns_matrix[indices, :]
                metrics = _dca_metrics_from_return_matrix(
                    sampled_returns=sampled,
                    dates=path_dates,
                    config=replay_config,
                )
                benchmark_xirr = float(metrics["xirr"][benchmark_position])
                benchmark_ending = float(metrics["ending_equity"][benchmark_position])
                xirr_values = metrics["xirr"]
                trial = pd.DataFrame(
                    {
                        "selector": selector,
                        "data_mode": data_mode,
                        "mc_path_id": (
                            f"h{int(horizon)}_b{int(block_length)}_s{sample_index + 1:04d}"
                        ),
                        "horizon_years": int(horizon),
                        "block_length_days": int(block_length),
                        "sample_index": int(sample_index + 1),
                        "scenario_id": scenario_ids,
                        "scenario_label": [
                            metadata.loc[scenario_id, "scenario_label"]
                            for scenario_id in scenario_ids
                        ],
                        "strategy_family": [
                            metadata.loc[scenario_id, "strategy_family"]
                            for scenario_id in scenario_ids
                        ],
                        "total_contributed": metrics["total_contributed"],
                        "ending_equity": metrics["ending_equity"],
                        "simple_cash_return": metrics["simple_cash_return"],
                        "xirr": xirr_values,
                        "max_drawdown": metrics["max_drawdown"],
                        "drawdown_breach": metrics["max_drawdown"]
                        < replay_config.max_drawdown_limit,
                        "benchmark_scenario_id": benchmark_id,
                        "benchmark_xirr": benchmark_xirr,
                        "benchmark_ending_equity": benchmark_ending,
                        "win_vs_benchmark": xirr_values > benchmark_xirr,
                        "tie_vs_benchmark": np.isclose(
                            xirr_values,
                            benchmark_xirr,
                            atol=1e-10,
                            equal_nan=False,
                        ),
                    }
                )
                rows.append(trial)
    if not rows:
        return _empty_mc_trials_frame()
    return pd.concat(rows, ignore_index=True)


def build_monte_carlo_replay_ranking(
    mc_trials: pd.DataFrame,
    *,
    cohorts: pd.DataFrame,
    selector: str,
    config: MonthlyDecisionReplayConfig,
) -> pd.DataFrame:
    if mc_trials.empty:
        return _empty_replay_ranking_frame()
    cohort_gate = _cohort_gate_by_scenario(cohorts, config=config)
    rows: list[dict[str, Any]] = []
    for scenario_id, group in mc_trials.groupby("scenario_id", sort=False):
        xirr = group["xirr"].astype(float)
        max_drawdowns = group["max_drawdown"].astype(float)
        wins = group["win_vs_benchmark"].astype(float)
        ties = group["tie_vs_benchmark"].astype(float)
        trial_count = int(len(group))
        win_rate = float((wins.sum() + 0.5 * ties.sum()) / trial_count)
        mc_breach_rate = float(group["drawdown_breach"].astype(bool).mean())
        gate = cohort_gate.get(
            scenario_id,
            {
                "cohort_gate_passed": False,
                "cohort_drawdown_breach_rate": np.nan,
                "cohort_worst_max_drawdown": np.nan,
                "cohort_count": 0,
            },
        )
        expected_xirr = float(xirr.mean())
        eligible = bool(
            gate["cohort_gate_passed"]
            and win_rate >= config.min_win_rate
            and np.isfinite(expected_xirr)
        )
        rows.append(
            {
                "selector": selector,
                "ranking_method": "monte_carlo",
                "scenario_id": scenario_id,
                "scenario_label": group["scenario_label"].iloc[0],
                "strategy_family": group["strategy_family"].iloc[0],
                "mc_trial_count": trial_count,
                "cohort_count": int(gate["cohort_count"]),
                "win_rate_vs_qqq_dca": win_rate,
                "expected_xirr": expected_xirr,
                "median_xirr": float(xirr.median()),
                "p05_xirr": float(xirr.quantile(0.05)),
                "worst_xirr": float(xirr.min()),
                "expected_ending_equity": float(group["ending_equity"].astype(float).mean()),
                "expected_max_drawdown": float(max_drawdowns.mean()),
                "p05_max_drawdown": float(max_drawdowns.quantile(0.05)),
                "median_max_drawdown": float(max_drawdowns.median()),
                "worst_max_drawdown": float(max_drawdowns.min()),
                "drawdown_breach_rate": mc_breach_rate,
                "cohort_gate_passed": bool(gate["cohort_gate_passed"]),
                "cohort_drawdown_breach_rate": float(gate["cohort_drawdown_breach_rate"]),
                "cohort_worst_max_drawdown": float(gate["cohort_worst_max_drawdown"]),
                "manual_review_required": bool(mc_breach_rate > 0.0 or not eligible),
                "eligible_for_monthly_signal": eligible,
            }
        )
    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values(
        [
            "eligible_for_monthly_signal",
            "drawdown_breach_rate",
            "win_rate_vs_qqq_dca",
            "expected_xirr",
            "p05_xirr",
            "expected_max_drawdown",
        ],
        ascending=[False, True, False, False, False, False],
    ).reset_index(drop=True)
    ranking["replay_rank"] = ranking.index + 1
    return ranking


def build_replay_ranking(
    cohorts: pd.DataFrame,
    *,
    selector: str,
    config: MonthlyDecisionReplayConfig,
) -> pd.DataFrame:
    if cohorts.empty:
        return _empty_replay_ranking_frame()
    rows: list[dict[str, Any]] = []
    for scenario_id, group in cohorts.groupby("scenario_id", sort=False):
        cohort_count = int(len(group))
        wins = group["win_vs_benchmark"].astype(float)
        ties = group["tie_vs_benchmark"].astype(float)
        win_rate = float((wins.sum() + 0.5 * ties.sum()) / cohort_count)
        max_drawdowns = group["max_drawdown"].astype(float)
        xirr = group["xirr"].astype(float)
        breach_rate = float(group["drawdown_breach"].astype(bool).mean())
        worst_drawdown = float(max_drawdowns.min())
        worst_xirr = float(xirr.min())
        median_xirr = float(xirr.median())
        median_drawdown = float(max_drawdowns.median())
        median_rank = float(group["cohort_rank"].astype(float).median())
        eligible = bool(
            win_rate >= config.min_win_rate
            and breach_rate == 0.0
            and worst_drawdown >= config.max_drawdown_limit
        )
        rows.append(
            {
                "selector": selector,
                "ranking_method": "rolling_cohort",
                "scenario_id": scenario_id,
                "scenario_label": group["scenario_label"].iloc[0],
                "strategy_family": group["strategy_family"].iloc[0],
                "cohort_count": cohort_count,
                "win_rate_vs_qqq_dca": win_rate,
                "expected_xirr": median_xirr,
                "median_xirr": median_xirr,
                "p05_xirr": float(xirr.quantile(0.05)),
                "worst_xirr": worst_xirr,
                "expected_max_drawdown": median_drawdown,
                "p05_max_drawdown": float(max_drawdowns.quantile(0.05)),
                "median_max_drawdown": median_drawdown,
                "worst_max_drawdown": worst_drawdown,
                "drawdown_breach_rate": breach_rate,
                "median_cohort_rank": median_rank,
                "cohort_gate_passed": eligible,
                "cohort_drawdown_breach_rate": breach_rate,
                "cohort_worst_max_drawdown": worst_drawdown,
                "replay_score": _replay_score(
                    win_rate=win_rate,
                    median_xirr=median_xirr,
                    worst_xirr=worst_xirr,
                    worst_drawdown=worst_drawdown,
                    breach_rate=breach_rate,
                    median_rank=median_rank,
                ),
                "manual_review_required": not eligible,
                "eligible_for_monthly_signal": eligible,
            }
        )
    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values(
        [
            "eligible_for_monthly_signal",
            "drawdown_breach_rate",
            "win_rate_vs_qqq_dca",
            "worst_max_drawdown",
            "replay_score",
        ],
        ascending=[False, True, False, False, False],
    ).reset_index(drop=True)
    ranking["replay_rank"] = ranking.index + 1
    return ranking


def build_replay_decisions(
    cohorts: pd.DataFrame,
    *,
    ranking: pd.DataFrame,
    selector: str,
) -> pd.DataFrame:
    if cohorts.empty or ranking.empty:
        return _empty_replay_decisions_frame()
    eligible_ids = set(
        ranking.loc[ranking["eligible_for_monthly_signal"].astype(bool), "scenario_id"]
        .astype(str)
        .tolist()
    )
    source = cohorts[cohorts["scenario_id"].astype(str).isin(eligible_ids)].copy()
    if source.empty:
        source = cohorts.copy()
    rank_order = {
        str(row.scenario_id): int(row.replay_rank)
        for row in ranking[["scenario_id", "replay_rank"]].itertuples(index=False)
    }
    source["_replay_rank"] = source["scenario_id"].map(rank_order).fillna(999_999).astype(int)
    rows: list[dict[str, Any]] = []
    group_columns = ["data_mode", "horizon_years", "cohort_start", "cohort_end"]
    for keys, group in source.groupby(group_columns, sort=False):
        selected = group.sort_values(
            ["drawdown_breach", "_replay_rank", "xirr", "max_drawdown"],
            ascending=[True, True, False, False],
        ).iloc[0]
        rows.append(
            {
                "selector": selector,
                "data_mode": keys[0],
                "horizon_years": int(keys[1]),
                "cohort_start": keys[2],
                "cohort_end": keys[3],
                "scenario_id": selected["scenario_id"],
                "scenario_label": selected["scenario_label"],
                "cohort_rank": int(selected["cohort_rank"]),
                "replay_rank": int(selected["_replay_rank"]),
                "total_contributed": float(selected["total_contributed"]),
                "ending_equity": float(selected["ending_equity"]),
                "simple_cash_return": float(selected["simple_cash_return"]),
                "xirr": float(selected["xirr"]),
                "max_drawdown": float(selected["max_drawdown"]),
                "drawdown_breach": bool(selected["drawdown_breach"]),
                "benchmark_scenario_id": selected["benchmark_scenario_id"],
                "benchmark_xirr": float(selected["benchmark_xirr"]),
                "benchmark_ending_equity": float(selected["benchmark_ending_equity"]),
                "win_vs_qqq_dca": bool(selected["win_vs_benchmark"]),
            }
        )
    return pd.DataFrame(rows)


def build_replay_equity(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return _empty_replay_equity_frame()
    columns = [
        "selector",
        "data_mode",
        "horizon_years",
        "cohort_start",
        "cohort_end",
        "scenario_id",
        "scenario_label",
        "total_contributed",
        "ending_equity",
        "benchmark_ending_equity",
        "xirr",
        "benchmark_xirr",
        "max_drawdown",
        "win_vs_qqq_dca",
    ]
    return decisions[columns].copy()


def write_monthly_decision_replay_report(
    *,
    outputs: MonthlyDecisionReplayOutputs,
    output_dir: Path,
    family: str,
    actual_primary_signal: pd.DataFrame | None = None,
) -> MonthlyDecisionReplayReportResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"monthly_decision_replay_{family.lower()}"
    html_path = output_dir / f"{prefix}.html"
    decisions_path = output_dir / f"{prefix}_decisions.csv"
    cohorts_path = output_dir / f"{prefix}_cohorts.csv"
    ranking_path = output_dir / f"{prefix}_ranking.csv"
    equity_path = output_dir / f"{prefix}_equity.csv"
    mc_summary_path = output_dir / f"{prefix}_mc_summary.csv"
    mc_trials_path = output_dir / f"{prefix}_mc_trials.csv.gz"
    source_coverage_path = output_dir / f"{prefix}_source_coverage.csv"
    mc_summary = build_monte_carlo_trial_summary(outputs.mc_trials)
    outputs.decisions.to_csv(decisions_path, index=False)
    outputs.cohorts.to_csv(cohorts_path, index=False)
    outputs.ranking.to_csv(ranking_path, index=False)
    outputs.equity.to_csv(equity_path, index=False)
    mc_summary.to_csv(mc_summary_path, index=False)
    outputs.mc_trials.to_csv(mc_trials_path, index=False, compression="gzip")
    outputs.source_coverage.to_csv(source_coverage_path, index=False)
    html_path.write_text(
        render_monthly_decision_replay_html(
            outputs=outputs,
            actual_primary_signal=actual_primary_signal,
            decisions_path=decisions_path,
            cohorts_path=cohorts_path,
            ranking_path=ranking_path,
            equity_path=equity_path,
            mc_trials_path=mc_trials_path,
            mc_summary_path=mc_summary_path,
            source_coverage_path=source_coverage_path,
        ),
        encoding="utf-8",
    )
    return MonthlyDecisionReplayReportResult(
        cohorts=outputs.cohorts,
        ranking=outputs.ranking,
        decisions=outputs.decisions,
        equity=outputs.equity,
        mc_trials=outputs.mc_trials,
        mc_summary=mc_summary,
        source_coverage=outputs.source_coverage,
        html_path=html_path,
        decisions_path=decisions_path,
        cohorts_path=cohorts_path,
        ranking_path=ranking_path,
        equity_path=equity_path,
        mc_trials_path=mc_trials_path,
        mc_summary_path=mc_summary_path,
        source_coverage_path=source_coverage_path,
    )


def render_monthly_decision_replay_html(
    *,
    outputs: MonthlyDecisionReplayOutputs,
    actual_primary_signal: pd.DataFrame | None,
    decisions_path: Path,
    cohorts_path: Path,
    ranking_path: Path,
    equity_path: Path,
    mc_trials_path: Path,
    mc_summary_path: Path,
    source_coverage_path: Path,
) -> str:
    best = outputs.ranking.iloc[0] if not outputs.ranking.empty else None
    actual_label = _actual_primary_label(actual_primary_signal)
    return f"""<!doctype html>
<html lang="zh-Hant">
{render_html_head(title="Monthly Decision Replay")}
<body>
<main>
  <section class="hero">
    <div class="panel">
      <p class="eyebrow">Monthly Replay</p>
      <h1>Hybrid-Primary Monte Carlo Replay</h1>
      <p class="lede">
        This report ranks strategies with Monte Carlo block bootstrap when enabled, while
        deterministic rolling cohorts remain the audit gate. The hybrid-primary selector uses
        actual ETF prices when available and synthetic backfill before ETF listing.
      </p>
      <div class="note">
        Selector: <strong>{escape(outputs.selector)}</strong>.
        Ranking method: <strong>{escape(outputs.ranking_method)}</strong>.
        Current actual-primary monthly signal: <strong>{escape(actual_label)}</strong>.
      </div>
    </div>
    <div class="panel">
      <p class="eyebrow">Best Candidate</p>
      {_render_best_candidate(best)}
    </div>
  </section>

  <section class="panel">
    <h2>Source Coverage</h2>
    <p>
      Hybrid-primary maximizes the evaluation window by using actual ETF data after listing and
      scaled synthetic backfill before listing.
    </p>
    <div class="table-wrap">{_render_table(outputs.source_coverage, _coverage_columns())}</div>
  </section>

  <section class="panel">
    <h2>Ranking</h2>
    <p>
      Monte Carlo ranking uses expected XIRR as the primary objective. Win rate, p05 XIRR,
      drawdown breach rate, and deterministic cohort gate are shown for risk review.
    </p>
    <div class="table-wrap">{_render_table(outputs.ranking.head(20), _ranking_columns())}</div>
  </section>

  <section class="panel">
    <h2>Deterministic Cohort Decisions</h2>
    <p>
      These rows show which eligible replay-ranked candidate is selected for each historical
      cohort. They are an audit trail, not the Monte Carlo sampling table.
    </p>
    <div class="table-wrap">{_render_table(outputs.decisions.head(80), _decision_columns())}</div>
  </section>

  <section class="panel links">
    <h2>Audit Files</h2>
    <a href="{decisions_path.name}">decisions CSV</a>
    <a href="{cohorts_path.name}">cohorts CSV</a>
    <a href="{ranking_path.name}">ranking CSV</a>
    <a href="{equity_path.name}">equity CSV</a>
    <a href="{mc_summary_path.name}">MC summary CSV</a>
    <a href="{mc_trials_path.name}">compressed MC trials CSV</a>
    <a href="{source_coverage_path.name}">source coverage CSV</a>
  </section>
</main>
</body>
</html>
"""


def build_monte_carlo_trial_summary(mc_trials: pd.DataFrame) -> pd.DataFrame:
    if mc_trials.empty:
        return _empty_mc_summary_frame()
    rows: list[dict[str, Any]] = []
    group_columns = [
        "selector",
        "data_mode",
        "scenario_id",
        "scenario_label",
        "strategy_family",
        "horizon_years",
        "block_length_days",
    ]
    for keys, group in mc_trials.groupby(group_columns, sort=False):
        xirr = group["xirr"].astype(float)
        max_drawdown = group["max_drawdown"].astype(float)
        wins = group["win_vs_benchmark"].astype(float)
        ties = group["tie_vs_benchmark"].astype(float)
        trial_count = int(len(group))
        rows.append(
            {
                "selector": keys[0],
                "data_mode": keys[1],
                "scenario_id": keys[2],
                "scenario_label": keys[3],
                "strategy_family": keys[4],
                "horizon_years": int(keys[5]),
                "block_length_days": int(keys[6]),
                "mc_trial_count": trial_count,
                "win_rate_vs_qqq_dca": float((wins.sum() + 0.5 * ties.sum()) / trial_count),
                "expected_xirr": float(xirr.mean()),
                "median_xirr": float(xirr.median()),
                "p05_xirr": float(xirr.quantile(0.05)),
                "expected_ending_equity": float(group["ending_equity"].astype(float).mean()),
                "expected_max_drawdown": float(max_drawdown.mean()),
                "p05_max_drawdown": float(max_drawdown.quantile(0.05)),
                "drawdown_breach_rate": float(group["drawdown_breach"].astype(bool).mean()),
            }
        )
    return pd.DataFrame(rows)


def _optimizer_config_for_replay(
    *,
    optimizer_config: DCAPolicyOptimizerConfig,
    replay_config: MonthlyDecisionReplayConfig,
) -> DCAPolicyOptimizerConfig:
    return DCAPolicyOptimizerConfig(
        enabled=optimizer_config.enabled,
        family=optimizer_config.family,
        objective=optimizer_config.objective,
        max_drawdown_limit=replay_config.max_drawdown_limit,
        high_risk_drawdown_band=optimizer_config.high_risk_drawdown_band,
        target_leverage_grid=optimizer_config.target_leverage_grid,
        trend_windows=optimizer_config.trend_windows,
        momentum_windows=optimizer_config.momentum_windows,
        volatility_windows=optimizer_config.volatility_windows,
        volatility_targets=optimizer_config.volatility_targets,
        drawdown_guards=optimizer_config.drawdown_guards,
        dca_initial_cash=replay_config.initial_cash,
        dca_contribution=replay_config.monthly_contribution,
        top_n=optimizer_config.top_n,
        fast_top_n=optimizer_config.fast_top_n,
        walk_forward_top_n=optimizer_config.walk_forward_top_n,
        cohort_validation_enabled=True,
        cohort_horizons_years=replay_config.horizons_years,
        rebalance_cadence=optimizer_config.rebalance_cadence,
        monitor_cadence=optimizer_config.monitor_cadence,
        optuna_enabled=optimizer_config.optuna_enabled,
    )


def _source_mode_for_selector(selector: str) -> str:
    if selector == SELECTOR_SYNTHETIC_PRIMARY:
        return DATA_MODE_SYNTHETIC
    if selector == SELECTOR_ACTUAL_PRIMARY:
        return DATA_MODE_ACTUAL
    if selector == SELECTOR_HYBRID_PRIMARY:
        return DATA_MODE_HYBRID
    raise ValueError(
        "selector must be actual_primary, synthetic_primary, or hybrid_primary; "
        f"got {selector!r}."
    )


def _attach_benchmark(
    cohorts: pd.DataFrame,
    *,
    selector: str,
    benchmark: str,
) -> pd.DataFrame:
    if benchmark != BENCHMARK_QQQ_DCA:
        raise ValueError(f"Unsupported replay benchmark: {benchmark!r}.")
    result = cohorts.copy()
    baseline = result[result["scenario_id"].astype(str).str.endswith("constant_1p0")].copy()
    if baseline.empty:
        raise ValueError("Replay requires Constant 1.0x / QQQ DCA baseline cohorts.")
    key_columns = ["data_mode", "horizon_years", "cohort_start", "cohort_end"]
    baseline = baseline[
        [
            *key_columns,
            "scenario_id",
            "scenario_label",
            "xirr",
            "ending_equity",
            "max_drawdown",
        ]
    ].rename(
        columns={
            "scenario_id": "benchmark_scenario_id",
            "scenario_label": "benchmark_scenario_label",
            "xirr": "benchmark_xirr",
            "ending_equity": "benchmark_ending_equity",
            "max_drawdown": "benchmark_max_drawdown",
        }
    )
    result = result.merge(baseline, on=key_columns, how="left")
    if result["benchmark_xirr"].isna().any():
        raise ValueError("Replay benchmark is missing for at least one cohort.")
    xirr = result["xirr"].astype(float)
    benchmark_xirr = result["benchmark_xirr"].astype(float)
    result["win_vs_benchmark"] = xirr > benchmark_xirr
    result["tie_vs_benchmark"] = np.isclose(xirr, benchmark_xirr, atol=1e-10)
    result["selector"] = selector
    return result


def _source_return_matrix(source_curves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = source_curves.copy()
    source["date"] = pd.to_datetime(source["date"])
    returns = (
        source.pivot(index="date", columns="scenario_id", values="investment_return")
        .sort_index()
        .dropna(how="any")
    )
    metadata = (
        source.sort_values("date")
        .drop_duplicates("scenario_id")
        .set_index("scenario_id")[["scenario_label", "strategy_family"]]
    )
    return returns, metadata


def _benchmark_scenario_id(scenario_ids: pd.Index | list[str]) -> str:
    matches = [str(value) for value in scenario_ids if str(value).endswith("constant_1p0")]
    if not matches:
        raise ValueError("Monte Carlo replay requires Constant 1.0x / QQQ DCA baseline.")
    return matches[0]


def _bootstrap_indices(
    *,
    source_length: int,
    target_length: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if source_length <= 0:
        raise ValueError("Cannot bootstrap from an empty source.")
    effective_block = max(1, min(int(block_length), int(source_length)))
    starts = rng.integers(
        0,
        source_length - effective_block + 1,
        size=_block_count(target_length, effective_block),
    )
    chunks = [np.arange(start, start + effective_block) for start in starts]
    return np.concatenate(chunks)[:target_length]


def _block_count(target_length: int, block_length: int) -> int:
    return int(np.ceil(float(target_length) / float(block_length)))


def _mc_path_dates(target_days: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2000-01-03", periods=target_days)


def _dca_metrics_from_return_matrix(
    *,
    sampled_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    config: MonthlyDecisionReplayConfig,
) -> dict[str, np.ndarray | float]:
    returns = np.asarray(sampled_returns, dtype=float).copy()
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)
    returns[0, :] = 0.0
    factors = np.maximum(1.0 + returns, 1e-12)
    return_index = np.cumprod(factors, axis=0)
    peaks = np.maximum.accumulate(return_index, axis=0)
    max_drawdown = np.nanmin(return_index / peaks - 1.0, axis=0)

    contributions = _mc_contribution_schedule(
        dates,
        initial_cash=config.initial_cash,
        monthly_contribution=config.monthly_contribution,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        contribution_units = np.divide(
            contributions[:, None],
            return_index,
            out=np.zeros_like(return_index),
            where=return_index != 0.0,
        )
    units = np.cumsum(contribution_units, axis=0)
    equity = units * return_index
    ending_equity = equity[-1, :]
    total_contributed = float(contributions.sum())
    simple_cash_return = ending_equity / total_contributed - 1.0
    xirr_values = _vectorized_xirr(dates, contributions, ending_equity)
    return {
        "total_contributed": total_contributed,
        "ending_equity": ending_equity,
        "simple_cash_return": simple_cash_return,
        "xirr": xirr_values,
        "max_drawdown": max_drawdown,
    }


def _mc_contribution_schedule(
    dates: pd.DatetimeIndex,
    *,
    initial_cash: float,
    monthly_contribution: float,
) -> np.ndarray:
    contribution_dates = monthly_rebalance_dates(dates)
    contributions = np.array(
        [
            monthly_contribution if pd.Timestamp(date) in contribution_dates else 0.0
            for date in dates
        ],
        dtype=float,
    )
    contributions[0] += float(initial_cash)
    return contributions


def _vectorized_xirr(
    dates: pd.DatetimeIndex,
    contributions: np.ndarray,
    final_equity: np.ndarray,
) -> np.ndarray:
    final = np.asarray(final_equity, dtype=float)
    result = np.full(final.shape, np.nan, dtype=float)
    if len(dates) == 0 or final.size == 0:
        return result
    cash_mask = contributions > 0.0
    cash_dates = dates[cash_mask]
    cash_amounts = -contributions[cash_mask]
    if len(cash_dates) == 0:
        return result
    start = pd.Timestamp(cash_dates[0])
    cash_years = np.array([(pd.Timestamp(date) - start).days / 365.25 for date in cash_dates])
    final_years = max((pd.Timestamp(dates[-1]) - start).days / 365.25, 1e-9)

    low = np.full(final.shape, -0.9999, dtype=float)
    high = np.full(final.shape, 10.0, dtype=float)
    low_value = _common_cash_npv(low, cash_amounts, cash_years, final, final_years)
    high_value = _common_cash_npv(high, cash_amounts, cash_years, final, final_years)
    for _ in range(8):
        mask = low_value * high_value > 0.0
        if not mask.any():
            break
        high[mask] *= 2.0
        high_value[mask] = _common_cash_npv(
            high[mask],
            cash_amounts,
            cash_years,
            final[mask],
            final_years,
        )
    valid = np.isfinite(final) & (low_value * high_value <= 0.0)
    if not valid.any():
        return result

    lo = low[valid]
    hi = high[valid]
    lo_value = low_value[valid]
    final_valid = final[valid]
    for _ in range(80):
        middle = (lo + hi) / 2.0
        middle_value = _common_cash_npv(
            middle,
            cash_amounts,
            cash_years,
            final_valid,
            final_years,
        )
        use_left = lo_value * middle_value <= 0.0
        hi[use_left] = middle[use_left]
        lo[~use_left] = middle[~use_left]
        lo_value[~use_left] = middle_value[~use_left]
    result[valid] = (lo + hi) / 2.0
    return result


def _common_cash_npv(
    rates: np.ndarray,
    cash_amounts: np.ndarray,
    cash_years: np.ndarray,
    final: np.ndarray,
    final_years: float,
) -> np.ndarray:
    rates = np.asarray(rates, dtype=float)
    base = 1.0 + rates
    contribution_npv = np.zeros_like(rates, dtype=float)
    for amount, years in zip(cash_amounts, cash_years, strict=True):
        contribution_npv += amount / (base**years)
    return contribution_npv + final / (base**final_years)


def _cohort_gate_by_scenario(
    cohorts: pd.DataFrame,
    *,
    config: MonthlyDecisionReplayConfig,
) -> dict[str, dict[str, Any]]:
    if cohorts.empty:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for scenario_id, group in cohorts.groupby("scenario_id", sort=False):
        breach_rate = float(group["drawdown_breach"].astype(bool).mean())
        worst_drawdown = float(group["max_drawdown"].astype(float).min())
        result[str(scenario_id)] = {
            "cohort_gate_passed": bool(
                breach_rate == 0.0 and worst_drawdown >= config.max_drawdown_limit
            ),
            "cohort_drawdown_breach_rate": breach_rate,
            "cohort_worst_max_drawdown": worst_drawdown,
            "cohort_count": int(len(group)),
        }
    return result


def _monte_carlo_samples_for_scan_mode(
    replay_config: MonthlyDecisionReplayConfig,
    scan_mode: str,
) -> int:
    if scan_mode == "full":
        return int(replay_config.monte_carlo_full_samples_per_scale)
    return int(replay_config.monte_carlo_fast_samples_per_scale)


def _replay_score(
    *,
    win_rate: float,
    median_xirr: float,
    worst_xirr: float,
    worst_drawdown: float,
    breach_rate: float,
    median_rank: float,
) -> float:
    return float(
        100.0 * win_rate
        + 50.0 * median_xirr
        + 80.0 * worst_xirr
        + 10.0 * worst_drawdown
        - 100.0 * breach_rate
        - 0.5 * median_rank
    )


def _render_best_candidate(best: pd.Series | None) -> str:
    if best is None:
        return "<p>No replay candidate.</p>"
    expected_xirr = _format_percent(best.get("expected_xirr"))
    win_rate = _format_percent(best.get("win_rate_vs_qqq_dca"))
    p05_xirr = _format_percent(best.get("p05_xirr"))
    breach_rate = _format_percent(best.get("drawdown_breach_rate"))
    cohort_gate = escape(str(best.get("cohort_gate_passed", "")))
    return f"""
    <h2>{escape(str(best["scenario_label"]))}</h2>
    <div class="kpis">
      <div class="kpi"><span>expected XIRR</span><strong>{expected_xirr}</strong></div>
      <div class="kpi"><span>win rate vs QQQ DCA</span><strong>{win_rate}</strong></div>
      <div class="kpi"><span>p05 XIRR</span><strong>{p05_xirr}</strong></div>
      <div class="kpi"><span>drawdown breach</span><strong>{breach_rate}</strong></div>
    </div>
    <p><strong>Cohort gate:</strong> {cohort_gate}</p>
    """


def _actual_primary_label(frame: pd.DataFrame | None) -> str:
    if frame is None or frame.empty or "scenario_label" not in frame.columns:
        return "not available"
    row = frame.iloc[0]
    return f"{row.get('scenario_label', '')} ({row.get('data_mode', '')})"


def _render_table(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    if frame.empty:
        return "<p>No data.</p>"
    available = [(key, label) for key, label in columns if key in frame.columns]
    header = "".join(f"<th>{escape(label)}</th>" for _, label in available)
    rows: list[str] = []
    for row in frame.itertuples(index=False):
        row_dict = row._asdict()
        cells = "".join(
            f"<td>{escape(_format_cell(row_dict.get(key), key))}</td>" for key, _ in available
        )
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _coverage_columns() -> list[tuple[str, str]]:
    return [
        ("ticker", "Ticker"),
        ("replay_start", "Replay Start"),
        ("replay_end", "Replay End"),
        ("actual_start", "Actual Start"),
        ("actual_end", "Actual End"),
        ("synthetic_backfill_start", "Backfill Start"),
        ("synthetic_backfill_end", "Backfill End"),
        ("splice_date", "Splice Date"),
        ("splice_scale", "Splice Scale"),
    ]


def _ranking_columns() -> list[tuple[str, str]]:
    return [
        ("replay_rank", "Rank"),
        ("scenario_label", "Strategy"),
        ("eligible_for_monthly_signal", "Eligible"),
        ("cohort_gate_passed", "Cohort Gate"),
        ("win_rate_vs_qqq_dca", "Win Rate"),
        ("expected_xirr", "Expected XIRR"),
        ("median_xirr", "Median XIRR"),
        ("p05_xirr", "P05 XIRR"),
        ("expected_max_drawdown", "Expected DD"),
        ("p05_max_drawdown", "P05 DD"),
        ("drawdown_breach_rate", "Breach Rate"),
    ]


def _decision_columns() -> list[tuple[str, str]]:
    return [
        ("horizon_years", "Years"),
        ("cohort_start", "Start"),
        ("cohort_end", "End"),
        ("scenario_label", "Selected"),
        ("replay_rank", "Replay Rank"),
        ("xirr", "XIRR"),
        ("benchmark_xirr", "QQQ DCA XIRR"),
        ("win_vs_qqq_dca", "Win"),
        ("max_drawdown", "Max DD"),
    ]


def _format_cell(value: Any, key: str) -> str:
    if pd.isna(value):
        return ""
    if key in {
        "win_rate_vs_qqq_dca",
        "expected_xirr",
        "median_xirr",
        "p05_xirr",
        "worst_xirr",
        "expected_max_drawdown",
        "p05_max_drawdown",
        "median_max_drawdown",
        "worst_max_drawdown",
        "drawdown_breach_rate",
        "cohort_drawdown_breach_rate",
        "cohort_worst_max_drawdown",
        "xirr",
        "benchmark_xirr",
        "max_drawdown",
    }:
        return _format_percent(value)
    if key in {
        "ending_equity",
        "benchmark_ending_equity",
        "total_contributed",
        "expected_ending_equity",
    }:
        return f"{float(value):,.0f}"
    if key == "splice_scale":
        return f"{float(value):.6f}"
    return str(value)


def _format_percent(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def _clean_series(series: Any) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    clean = pd.Series(series).dropna().astype(float)
    clean.index = pd.to_datetime(clean.index)
    return clean.sort_index()


def _date_or_blank(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).date().isoformat()


def _source_coverage_from_prices(prices: pd.DataFrame, products: list[ProductSpec]) -> pd.DataFrame:
    rows = []
    for product in products:
        series = _clean_series(prices.get(product.ticker))
        rows.append(
            {
                "ticker": product.ticker,
                "replay_start": _date_or_blank(series.index.min() if not series.empty else None),
                "replay_end": _date_or_blank(series.index.max() if not series.empty else None),
                "actual_start": "",
                "actual_end": "",
                "synthetic_backfill_start": "",
                "synthetic_backfill_end": "",
                "splice_date": "",
                "splice_scale": np.nan,
            }
        )
    return pd.DataFrame(rows)


def _empty_replay_cohorts_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "selector",
            "data_mode",
            "horizon_years",
            "cohort_start",
            "cohort_end",
            "scenario_id",
            "scenario_label",
            "strategy_family",
            "cohort_rank",
            "total_contributed",
            "ending_equity",
            "simple_cash_return",
            "xirr",
            "max_drawdown",
            "drawdown_breach",
            "benchmark_scenario_id",
            "benchmark_xirr",
            "benchmark_ending_equity",
            "win_vs_benchmark",
            "tie_vs_benchmark",
        ]
    )


def _empty_replay_ranking_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "selector",
            "ranking_method",
            "scenario_id",
            "scenario_label",
            "strategy_family",
            "mc_trial_count",
            "cohort_count",
            "win_rate_vs_qqq_dca",
            "expected_xirr",
            "median_xirr",
            "p05_xirr",
            "worst_xirr",
            "expected_ending_equity",
            "expected_max_drawdown",
            "p05_max_drawdown",
            "median_max_drawdown",
            "worst_max_drawdown",
            "drawdown_breach_rate",
            "cohort_gate_passed",
            "cohort_drawdown_breach_rate",
            "cohort_worst_max_drawdown",
            "manual_review_required",
            "eligible_for_monthly_signal",
            "replay_rank",
        ]
    )


def _empty_replay_decisions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "selector",
            "data_mode",
            "horizon_years",
            "cohort_start",
            "cohort_end",
            "scenario_id",
            "scenario_label",
            "cohort_rank",
            "replay_rank",
            "total_contributed",
            "ending_equity",
            "simple_cash_return",
            "xirr",
            "max_drawdown",
            "drawdown_breach",
            "benchmark_scenario_id",
            "benchmark_xirr",
            "benchmark_ending_equity",
            "win_vs_qqq_dca",
        ]
    )


def _empty_replay_equity_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "selector",
            "data_mode",
            "horizon_years",
            "cohort_start",
            "cohort_end",
            "scenario_id",
            "scenario_label",
            "total_contributed",
            "ending_equity",
            "benchmark_ending_equity",
            "xirr",
            "benchmark_xirr",
            "max_drawdown",
            "win_vs_qqq_dca",
        ]
    )


def _empty_mc_trials_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "selector",
            "data_mode",
            "mc_path_id",
            "horizon_years",
            "block_length_days",
            "sample_index",
            "scenario_id",
            "scenario_label",
            "strategy_family",
            "total_contributed",
            "ending_equity",
            "simple_cash_return",
            "xirr",
            "max_drawdown",
            "drawdown_breach",
            "benchmark_scenario_id",
            "benchmark_xirr",
            "benchmark_ending_equity",
            "win_vs_benchmark",
            "tie_vs_benchmark",
        ]
    )


def _empty_mc_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "selector",
            "data_mode",
            "scenario_id",
            "scenario_label",
            "strategy_family",
            "horizon_years",
            "block_length_days",
            "mc_trial_count",
            "win_rate_vs_qqq_dca",
            "expected_xirr",
            "median_xirr",
            "p05_xirr",
            "expected_ending_equity",
            "expected_max_drawdown",
            "p05_max_drawdown",
            "drawdown_breach_rate",
        ]
    )


__all__ = [
    "BENCHMARK_QQQ_DCA",
    "DATA_MODE_HYBRID",
    "HybridPriceResult",
    "MonthlyDecisionReplayOutputs",
    "MonthlyDecisionReplayReportResult",
    "SELECTOR_ACTUAL_PRIMARY",
    "SELECTOR_HYBRID_PRIMARY",
    "SELECTOR_SYNTHETIC_PRIMARY",
    "build_hybrid_actual_preferred_prices",
    "build_monthly_decision_replay_outputs",
    "build_monte_carlo_replay_ranking",
    "build_monte_carlo_replay_trials",
    "build_monte_carlo_trial_summary",
    "build_replay_decisions",
    "build_replay_equity",
    "build_replay_ranking",
    "render_monthly_decision_replay_html",
    "write_monthly_decision_replay_report",
]
