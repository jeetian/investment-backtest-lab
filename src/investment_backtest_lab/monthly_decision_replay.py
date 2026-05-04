from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from investment_backtest_lab.dca_policy_optimizer import (
    DATA_MODE_ACTUAL,
    DATA_MODE_SYNTHETIC,
    build_policy_scenario_specs,
    build_rolling_cohort_validation,
)
from investment_backtest_lab.html_ui import render_html_head
from investment_backtest_lab.leveraged_etf_lab import ProductSpec
from investment_backtest_lab.models import DCAPolicyOptimizerConfig, MonthlyDecisionReplayConfig

SELECTOR_ACTUAL_PRIMARY = "actual_primary"
SELECTOR_SYNTHETIC_PRIMARY = "synthetic_primary"
BENCHMARK_QQQ_DCA = "qqq_dca"


@dataclass(frozen=True)
class MonthlyDecisionReplayOutputs:
    cohorts: pd.DataFrame
    ranking: pd.DataFrame
    decisions: pd.DataFrame
    equity: pd.DataFrame
    selector: str


@dataclass(frozen=True)
class MonthlyDecisionReplayReportResult:
    cohorts: pd.DataFrame
    ranking: pd.DataFrame
    decisions: pd.DataFrame
    equity: pd.DataFrame
    html_path: Path
    decisions_path: Path
    cohorts_path: Path
    ranking_path: Path
    equity_path: Path


def build_monthly_decision_replay_outputs(
    *,
    mode_prices: dict[str, pd.DataFrame],
    products: list[ProductSpec],
    optimizer_config: DCAPolicyOptimizerConfig,
    replay_config: MonthlyDecisionReplayConfig,
    scan_mode: str,
    selector: str | None = None,
) -> MonthlyDecisionReplayOutputs:
    normalized_selector = (selector or replay_config.selector).lower()
    source_mode = _source_mode_for_selector(normalized_selector)
    prices = mode_prices.get(source_mode)
    if prices is None or prices.empty:
        raise ValueError(f"{normalized_selector} requires {source_mode} prices.")

    replay_optimizer_config = _optimizer_config_for_replay(
        optimizer_config=optimizer_config,
        replay_config=replay_config,
    )
    specs = build_policy_scenario_specs(
        config=replay_optimizer_config,
        products=products,
        scan_mode=scan_mode,
    )
    product_leverages = {product.ticker: product.leverage for product in products}
    cohorts = build_rolling_cohort_validation(
        mode_prices={source_mode: prices},
        specs=specs,
        products=products,
        product_leverages=product_leverages,
        config=replay_optimizer_config,
    )
    if cohorts.empty:
        return MonthlyDecisionReplayOutputs(
            cohorts=_empty_replay_cohorts_frame(),
            ranking=_empty_replay_ranking_frame(),
            decisions=_empty_replay_decisions_frame(),
            equity=_empty_replay_equity_frame(),
            selector=normalized_selector,
        )

    enriched = _attach_benchmark(
        cohorts,
        selector=normalized_selector,
        benchmark=replay_config.benchmark,
    )
    ranking = build_replay_ranking(
        enriched,
        selector=normalized_selector,
        config=replay_config,
    )
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
        selector=normalized_selector,
    )


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
                "scenario_id": scenario_id,
                "scenario_label": group["scenario_label"].iloc[0],
                "strategy_family": group["strategy_family"].iloc[0],
                "cohort_count": cohort_count,
                "win_rate_vs_qqq_dca": win_rate,
                "median_xirr": median_xirr,
                "worst_xirr": worst_xirr,
                "median_max_drawdown": median_drawdown,
                "worst_max_drawdown": worst_drawdown,
                "drawdown_breach_rate": breach_rate,
                "median_cohort_rank": median_rank,
                "replay_score": _replay_score(
                    win_rate=win_rate,
                    median_xirr=median_xirr,
                    worst_xirr=worst_xirr,
                    worst_drawdown=worst_drawdown,
                    breach_rate=breach_rate,
                    median_rank=median_rank,
                ),
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
    rows: list[dict[str, Any]] = []
    group_columns = ["data_mode", "horizon_years", "cohort_start", "cohort_end"]
    for keys, group in source.groupby(group_columns, sort=False):
        selected = group.sort_values(
            ["drawdown_breach", "xirr", "max_drawdown"],
            ascending=[True, False, False],
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
    outputs.decisions.to_csv(decisions_path, index=False)
    outputs.cohorts.to_csv(cohorts_path, index=False)
    outputs.ranking.to_csv(ranking_path, index=False)
    outputs.equity.to_csv(equity_path, index=False)
    html_path.write_text(
        render_monthly_decision_replay_html(
            outputs=outputs,
            actual_primary_signal=actual_primary_signal,
            decisions_path=decisions_path,
            cohorts_path=cohorts_path,
            ranking_path=ranking_path,
            equity_path=equity_path,
        ),
        encoding="utf-8",
    )
    return MonthlyDecisionReplayReportResult(
        cohorts=outputs.cohorts,
        ranking=outputs.ranking,
        decisions=outputs.decisions,
        equity=outputs.equity,
        html_path=html_path,
        decisions_path=decisions_path,
        cohorts_path=cohorts_path,
        ranking_path=ranking_path,
        equity_path=equity_path,
    )


def render_monthly_decision_replay_html(
    *,
    outputs: MonthlyDecisionReplayOutputs,
    actual_primary_signal: pd.DataFrame | None,
    decisions_path: Path,
    cohorts_path: Path,
    ranking_path: Path,
    equity_path: Path,
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
      <p class="eyebrow">Synthetic Primary Replay</p>
      <h1>壓測優先的月度決策回放</h1>
      <p class="lede">
        這份報表不直接採用 actual ETF 全期間排名，而是用 synthetic stress
        作為主要排序來源，並用不同起點與終點的 DCA cohort replay
        檢查每月決策流程是否穩定。
      </p>
      <div class="note">
        這是研究訊號，不是投資建議。v1 先輸出獨立 replay 報表，不覆蓋既有
        Monthly Decision Pack；你可以把兩者放在一起比較，觀察 actual-primary
        和 synthetic-primary 會不會選出不同策略。
      </div>
    </div>
    <div class="panel">
      <p class="eyebrow">Best Candidate</p>
      {_render_best_candidate(best, actual_label)}
    </div>
  </section>

  <section class="panel">
    <h2>Ranking</h2>
    <p>
      排名優先看 win rate vs QQQ DCA、drawdown breach rate、worst drawdown 與
      replay score；不用 ending equity 或單一全期間 XIRR 當主要依據。
    </p>
    <div class="table-wrap">{_render_table(outputs.ranking.head(20), _ranking_columns())}</div>
  </section>

  <section class="panel">
    <h2>Replay Decisions</h2>
    <p>
      每一列代表一段 cohort 期間。每段都重新從 10,000 USD 初始投入與每月
      1,000 USD DCA 開始，並和同起點、同終點、同投入金額的 QQQ DCA benchmark 比較。
    </p>
    <div class="table-wrap">{_render_table(outputs.decisions.head(80), _decision_columns())}</div>
  </section>

  <section class="panel links">
    <h2>Audit Files</h2>
    <a href="{decisions_path.name}">decisions CSV</a>
    <a href="{cohorts_path.name}">cohorts CSV</a>
    <a href="{ranking_path.name}">ranking CSV</a>
    <a href="{equity_path.name}">equity CSV</a>
  </section>
</main>
</body>
</html>
"""


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
    raise ValueError(f"selector must be actual_primary or synthetic_primary, got {selector!r}.")


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


def _render_best_candidate(best: pd.Series | None, actual_label: str) -> str:
    if best is None:
        return "<p>No replay candidate.</p>"
    return f"""
    <h2>{escape(str(best["scenario_label"]))}</h2>
    <div class="kpis">
      <div class="kpi">
        <span>win rate vs QQQ DCA</span>
        <strong>{_format_percent(best["win_rate_vs_qqq_dca"])}</strong>
      </div>
      <div class="kpi">
        <span>worst drawdown</span>
        <strong>{_format_percent(best["worst_max_drawdown"])}</strong>
      </div>
      <div class="kpi">
        <span>drawdown breach rate</span>
        <strong>{_format_percent(best["drawdown_breach_rate"])}</strong>
      </div>
      <div class="kpi">
        <span>eligible</span>
        <strong>{escape(str(best["eligible_for_monthly_signal"]))}</strong>
      </div>
    </div>
    <p><strong>Current actual-primary monthly signal:</strong> {escape(actual_label)}</p>
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
            f"<td>{escape(_format_cell(row_dict.get(key), key))}</td>"
            for key, _ in available
        )
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _ranking_columns() -> list[tuple[str, str]]:
    return [
        ("replay_rank", "Rank"),
        ("scenario_label", "Strategy"),
        ("win_rate_vs_qqq_dca", "Win Rate"),
        ("median_xirr", "Median XIRR"),
        ("worst_xirr", "Worst XIRR"),
        ("worst_max_drawdown", "Worst DD"),
        ("drawdown_breach_rate", "Breach Rate"),
        ("eligible_for_monthly_signal", "Eligible"),
    ]


def _decision_columns() -> list[tuple[str, str]]:
    return [
        ("horizon_years", "Years"),
        ("cohort_start", "Start"),
        ("cohort_end", "End"),
        ("scenario_label", "Selected"),
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
        "median_xirr",
        "worst_xirr",
        "median_max_drawdown",
        "worst_max_drawdown",
        "drawdown_breach_rate",
        "xirr",
        "benchmark_xirr",
        "max_drawdown",
    }:
        return _format_percent(value)
    if key in {"ending_equity", "benchmark_ending_equity", "total_contributed"}:
        return f"{float(value):,.0f}"
    return str(value)


def _format_percent(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2%}"


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
            "scenario_id",
            "scenario_label",
            "strategy_family",
            "cohort_count",
            "win_rate_vs_qqq_dca",
            "median_xirr",
            "worst_xirr",
            "median_max_drawdown",
            "worst_max_drawdown",
            "drawdown_breach_rate",
            "median_cohort_rank",
            "replay_score",
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


__all__ = [
    "BENCHMARK_QQQ_DCA",
    "MonthlyDecisionReplayOutputs",
    "MonthlyDecisionReplayReportResult",
    "SELECTOR_ACTUAL_PRIMARY",
    "SELECTOR_SYNTHETIC_PRIMARY",
    "build_monthly_decision_replay_outputs",
    "build_replay_decisions",
    "build_replay_equity",
    "build_replay_ranking",
    "render_monthly_decision_replay_html",
    "write_monthly_decision_replay_report",
]
