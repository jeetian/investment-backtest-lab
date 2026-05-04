from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from investment_backtest_lab.dca_policy_optimizer import DATA_MODE_ACTUAL
from investment_backtest_lab.html_ui import render_html_head, risk_badge

WEIGHT_COLUMNS = ["QQQ_weight", "QLD_weight", "TQQQ_weight", "CASH_weight"]
COMPARISON_COLUMNS = [
    "generated_at",
    "decision_authority",
    "actual_primary_as_of_date",
    "actual_primary_scenario_id",
    "actual_primary_scenario_label",
    "actual_primary_regime",
    "actual_primary_reason",
    "actual_primary_target_effective_leverage",
    "actual_primary_QQQ_weight",
    "actual_primary_QLD_weight",
    "actual_primary_TQQQ_weight",
    "actual_primary_CASH_weight",
    "actual_primary_manual_review_required",
    "actual_primary_review_reasons",
    "replay_primary_selector",
    "replay_primary_ranking_method",
    "replay_primary_scenario_id",
    "replay_primary_scenario_label",
    "replay_primary_replay_rank",
    "replay_primary_eligible_for_monthly_signal",
    "replay_primary_cohort_gate_passed",
    "replay_primary_win_rate_vs_qqq_dca",
    "replay_primary_expected_xirr",
    "replay_primary_median_xirr",
    "replay_primary_p05_xirr",
    "replay_primary_expected_max_drawdown",
    "replay_primary_p05_max_drawdown",
    "replay_primary_drawdown_breach_rate",
    "recommended_policy_source",
    "recommended_as_of_date",
    "recommended_scenario_id",
    "recommended_scenario_label",
    "recommended_regime",
    "recommended_reason",
    "recommended_target_effective_leverage",
    "recommended_QQQ_weight",
    "recommended_QLD_weight",
    "recommended_TQQQ_weight",
    "recommended_CASH_weight",
    "recommended_weight_sum",
    "actual_disagrees_with_authority",
    "strategies_differ",
    "manual_review_required",
    "review_reasons",
]


@dataclass(frozen=True)
class MonthlyDecisionComparisonInputs:
    monthly_decision: pd.DataFrame
    replay_ranking: pd.DataFrame
    actual_policy: pd.DataFrame
    source_coverage: pd.DataFrame


@dataclass(frozen=True)
class MonthlyDecisionComparisonOutputs:
    comparison: pd.DataFrame
    top_candidates: pd.DataFrame
    source_coverage: pd.DataFrame


@dataclass(frozen=True)
class MonthlyDecisionComparisonReportResult:
    comparison: pd.DataFrame
    top_candidates: pd.DataFrame
    source_coverage: pd.DataFrame
    html_path: Path
    csv_path: Path
    top_candidates_path: Path


def load_monthly_decision_comparison_inputs(
    *,
    output_dir: Path,
    family: str,
    require_outputs: bool = True,
) -> MonthlyDecisionComparisonInputs:
    prefix = family.lower()
    pack_path = output_dir / f"monthly_decision_pack_{prefix}.csv"
    replay_ranking_path = output_dir / f"monthly_decision_replay_{prefix}_ranking.csv"
    actual_policy_path = output_dir / f"dca_policy_optimizer_{prefix}_policy.csv"
    source_coverage_path = output_dir / f"monthly_decision_replay_{prefix}_source_coverage.csv"
    missing = [
        path for path in [pack_path, replay_ranking_path, actual_policy_path] if not path.exists()
    ]
    if missing:
        if require_outputs:
            missing_list = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(
                "Monthly decision comparison requires existing monthly pack, replay ranking, "
                f"and actual ETF policy outputs.\nMissing files:\n{missing_list}\n"
                "Run first:\n"
                "python -m uv run python scripts\\analyze_dca_policy_optimizer.py "
                "--config configs\\mvp_example.yaml --family qqq --scan-mode fast "
                "--cohort-validation\n"
                "python -m uv run python scripts\\analyze_monthly_decision_pack.py "
                "--config configs\\mvp_example.yaml --family qqq\n"
                "python -m uv run python scripts\\analyze_monthly_decision_replay.py "
                "--config configs\\mvp_example.yaml --family qqq --selector hybrid_primary"
            )
        return MonthlyDecisionComparisonInputs(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
    return MonthlyDecisionComparisonInputs(
        monthly_decision=pd.read_csv(pack_path),
        replay_ranking=pd.read_csv(replay_ranking_path),
        actual_policy=pd.read_csv(actual_policy_path),
        source_coverage=(
            pd.read_csv(source_coverage_path)
            if source_coverage_path.exists()
            else pd.DataFrame()
        ),
    )


def build_monthly_decision_comparison(
    *,
    monthly_decision: pd.DataFrame,
    replay_ranking: pd.DataFrame,
    actual_policy: pd.DataFrame | None = None,
    source_coverage: pd.DataFrame | None = None,
    generated_at: str | None = None,
    top_n: int = 8,
) -> MonthlyDecisionComparisonOutputs:
    if monthly_decision.empty:
        raise ValueError("Monthly decision comparison requires a non-empty monthly pack CSV.")
    if replay_ranking.empty:
        raise ValueError("Monthly decision comparison requires a non-empty replay ranking CSV.")

    actual_policy = actual_policy if actual_policy is not None else pd.DataFrame()
    actual = monthly_decision.iloc[0]
    ranking = _sorted_replay_ranking(replay_ranking)
    authority = ranking.iloc[0]
    recommended = _latest_actual_policy_for_authority(actual_policy, authority)

    actual_disagrees = _policy_key(actual.get("scenario_id")) != _policy_key(
        authority.get("scenario_id")
    )
    recommended_row = _recommended_columns(recommended)
    reasons = _review_reasons(
        actual=actual,
        authority=authority,
        recommended=recommended,
    )
    row = {
        "generated_at": generated_at or datetime.now(UTC).replace(microsecond=0).isoformat(),
        "decision_authority": _decision_authority(authority),
        "actual_primary_as_of_date": actual.get("as_of_date", ""),
        "actual_primary_scenario_id": actual.get("scenario_id", ""),
        "actual_primary_scenario_label": actual.get("scenario_label", ""),
        "actual_primary_regime": actual.get("regime", ""),
        "actual_primary_reason": actual.get("reason", ""),
        "actual_primary_target_effective_leverage": _safe_float(
            actual.get("target_effective_leverage")
        ),
        "actual_primary_QQQ_weight": _safe_float(actual.get("QQQ_weight")),
        "actual_primary_QLD_weight": _safe_float(actual.get("QLD_weight")),
        "actual_primary_TQQQ_weight": _safe_float(actual.get("TQQQ_weight")),
        "actual_primary_CASH_weight": _safe_float(actual.get("CASH_weight")),
        "actual_primary_manual_review_required": _safe_bool(
            actual.get("manual_review_required")
        ),
        "actual_primary_review_reasons": actual.get("review_reasons", ""),
        "replay_primary_selector": authority.get("selector", ""),
        "replay_primary_ranking_method": authority.get("ranking_method", ""),
        "replay_primary_scenario_id": authority.get("scenario_id", ""),
        "replay_primary_scenario_label": authority.get("scenario_label", ""),
        "replay_primary_replay_rank": int(_safe_float(authority.get("replay_rank"), 0.0)),
        "replay_primary_eligible_for_monthly_signal": _safe_bool(
            authority.get("eligible_for_monthly_signal")
        ),
        "replay_primary_cohort_gate_passed": _safe_bool(authority.get("cohort_gate_passed")),
        "replay_primary_win_rate_vs_qqq_dca": _safe_float(
            authority.get("win_rate_vs_qqq_dca")
        ),
        "replay_primary_expected_xirr": _safe_float(authority.get("expected_xirr")),
        "replay_primary_median_xirr": _safe_float(authority.get("median_xirr")),
        "replay_primary_p05_xirr": _safe_float(authority.get("p05_xirr")),
        "replay_primary_expected_max_drawdown": _safe_float(
            authority.get("expected_max_drawdown")
        ),
        "replay_primary_p05_max_drawdown": _safe_float(authority.get("p05_max_drawdown")),
        "replay_primary_drawdown_breach_rate": _safe_float(
            authority.get("drawdown_breach_rate")
        ),
        **recommended_row,
        "actual_disagrees_with_authority": actual_disagrees,
        "strategies_differ": actual_disagrees,
        "manual_review_required": bool(reasons),
        "review_reasons": "; ".join(reasons),
    }
    comparison = pd.DataFrame([row], columns=COMPARISON_COLUMNS)
    top_candidates = _top_candidates(ranking, top_n=top_n)
    return MonthlyDecisionComparisonOutputs(
        comparison=comparison,
        top_candidates=top_candidates,
        source_coverage=source_coverage if source_coverage is not None else pd.DataFrame(),
    )


def write_monthly_decision_comparison_report(
    *,
    output_dir: Path,
    family: str,
    top_n: int = 8,
) -> MonthlyDecisionComparisonReportResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_monthly_decision_comparison_inputs(output_dir=output_dir, family=family)
    outputs = build_monthly_decision_comparison(
        monthly_decision=inputs.monthly_decision,
        replay_ranking=inputs.replay_ranking,
        actual_policy=inputs.actual_policy,
        source_coverage=inputs.source_coverage,
        top_n=top_n,
    )
    prefix = f"monthly_decision_comparison_{family.lower()}"
    html_path = output_dir / f"{prefix}.html"
    csv_path = output_dir / f"{prefix}.csv"
    top_candidates_path = output_dir / f"{prefix}_top_candidates.csv"
    outputs.comparison.to_csv(csv_path, index=False)
    outputs.top_candidates.to_csv(top_candidates_path, index=False)
    html_path.write_text(
        render_monthly_decision_comparison_html(
            comparison=outputs.comparison,
            top_candidates=outputs.top_candidates,
            source_coverage=outputs.source_coverage,
            csv_path=csv_path,
            top_candidates_path=top_candidates_path,
        ),
        encoding="utf-8",
    )
    return MonthlyDecisionComparisonReportResult(
        comparison=outputs.comparison,
        top_candidates=outputs.top_candidates,
        source_coverage=outputs.source_coverage,
        html_path=html_path,
        csv_path=csv_path,
        top_candidates_path=top_candidates_path,
    )


def render_monthly_decision_comparison_html(
    *,
    comparison: pd.DataFrame,
    top_candidates: pd.DataFrame,
    source_coverage: pd.DataFrame,
    csv_path: Path,
    top_candidates_path: Path,
) -> str:
    row = comparison.iloc[0]
    review = bool(row["manual_review_required"])
    disagree = bool(row["actual_disagrees_with_authority"])
    review_label = "Manual Review Required" if review else "Ready For Monthly Review"
    recommended_target = _format_leverage(row["recommended_target_effective_leverage"])
    expected_xirr = _format_percent(row["replay_primary_expected_xirr"])
    win_rate = _format_percent(row["replay_primary_win_rate_vs_qqq_dca"])
    p05_xirr = _format_percent(row["replay_primary_p05_xirr"])
    expected_drawdown = _format_percent(row["replay_primary_expected_max_drawdown"])
    p05_drawdown = _format_percent(row["replay_primary_p05_max_drawdown"])
    breach_rate = _format_percent(row["replay_primary_drawdown_breach_rate"])
    audit_files = _replay_audit_file_names(csv_path)
    eligible_badge = risk_badge(
        str(row["replay_primary_eligible_for_monthly_signal"]),
        bool(row["replay_primary_eligible_for_monthly_signal"]),
    )
    gate_badge = risk_badge(
        str(row["replay_primary_cohort_gate_passed"]),
        bool(row["replay_primary_cohort_gate_passed"]),
    )
    return f"""<!doctype html>
<html lang="zh-Hant">
{render_html_head(title="Monthly Decision Comparison")}
<body>
<main>
  <section class="hero">
    <div class="panel">
      <p class="eyebrow">Monthly Decision Pack</p>
      <h1>{escape(str(row["recommended_scenario_label"]) or "No Recommendation")}</h1>
      <p class="lede">
        Official recommendation from hybrid-primary Monte Carlo ranking. Tradable weights use
        the latest actual ETF policy state for the same policy key.
      </p>
      <span class="pill {'danger' if review else 'ok'}">{escape(review_label)}</span>
      <p><strong>Review reasons:</strong> {escape(str(row["review_reasons"]) or "None")}</p>
    </div>
    <div class="panel">
      <p class="eyebrow">Tradable Weights</p>
      <h2>{escape(str(row["recommended_as_of_date"]))}</h2>
      <p><strong>As of:</strong> {escape(str(row["recommended_as_of_date"]))}</p>
      <p><strong>Regime:</strong> {escape(str(row["recommended_regime"]))}</p>
      <p><strong>Reason:</strong> {escape(str(row["recommended_reason"]))}</p>
      <div class="kpis">
        <div class="kpi"><span>Target</span><strong>{recommended_target}</strong></div>
        <div class="kpi"><span>Expected XIRR</span><strong>{expected_xirr}</strong></div>
        <div class="kpi"><span>Win Rate</span><strong>{win_rate}</strong></div>
        <div class="kpi"><span>P05 XIRR</span><strong>{p05_xirr}</strong></div>
      </div>
      {_render_recommended_weight_cards(row)}
    </div>
  </section>

  <section class="two-col">
    <div class="panel">
      <p class="eyebrow">Hybrid-Primary Authority</p>
      <h2>{escape(str(row["replay_primary_scenario_label"]))}</h2>
      <p><strong>Authority:</strong> {escape(str(row["decision_authority"]))}</p>
      <p><strong>Eligible:</strong> {eligible_badge}</p>
      <p><strong>Cohort gate:</strong> {gate_badge}</p>
      <p><strong>Expected XIRR:</strong> {expected_xirr}</p>
      <p><strong>P05 XIRR:</strong> {p05_xirr}</p>
      <p><strong>Expected drawdown:</strong> {expected_drawdown}</p>
      <p><strong>P05 drawdown:</strong> {p05_drawdown}</p>
      <p><strong>Drawdown breach rate:</strong> {breach_rate}</p>
    </div>
    <div class="panel">
      <p class="eyebrow">Actual Primary Reference</p>
      <h2>{escape(str(row["actual_primary_scenario_label"]))}</h2>
      <p><strong>Actual disagrees with authority:</strong> {escape(str(disagree))}</p>
      <p>
        Difference is a reference signal divergence, not a failure. The official monthly
        strategy remains the hybrid-primary replay authority unless review reasons say otherwise.
      </p>
      <p><strong>Regime:</strong> {escape(str(row["actual_primary_regime"]))}</p>
      <p><strong>Reason:</strong> {escape(str(row["actual_primary_reason"]))}</p>
      <p>
        <strong>Pack review:</strong>
        {escape(str(row["actual_primary_manual_review_required"]))}
      </p>
    </div>
  </section>

  <section class="panel">
    <h2>Source Coverage</h2>
    <p>
      Hybrid-primary uses actual ETF data after listing and scaled synthetic backfill before
      listing. This table shows which period came from which source.
    </p>
    <div class="table-wrap">{_render_table(source_coverage, _source_coverage_columns())}</div>
  </section>

  <section class="panel">
    <h2>Replay-Primary Top Candidates</h2>
    <div class="table-wrap">{_render_table(top_candidates, _top_candidate_columns())}</div>
  </section>

  <section class="panel links">
    <h2>Audit Files</h2>
    <a href="{csv_path.name}">comparison CSV</a>
    <a href="{top_candidates_path.name}">top candidates CSV</a>
    <a href="{audit_files["ranking"]}">replay ranking CSV</a>
    <a href="{audit_files["mc_summary"]}">MC summary CSV</a>
    <a href="{audit_files["mc_trials"]}">compressed MC trials CSV</a>
    <a href="{audit_files["source_coverage"]}">source coverage CSV</a>
    <span>
      Daily use: this comparison page, comparison CSV, top candidates, and MC summary.
      Deep audit only: compressed MC trials and deterministic cohort artifacts.
    </span>
  </section>
</main>
</body>
</html>
"""


def _sorted_replay_ranking(replay_ranking: pd.DataFrame) -> pd.DataFrame:
    if "replay_rank" not in replay_ranking.columns:
        raise ValueError("Replay ranking CSV must contain replay_rank.")
    return replay_ranking.sort_values("replay_rank").reset_index(drop=True)


def _latest_actual_policy_for_authority(
    actual_policy: pd.DataFrame,
    authority: pd.Series,
) -> pd.Series | None:
    if actual_policy.empty or "scenario_id" not in actual_policy.columns:
        return None
    policy_key = _policy_key(authority.get("scenario_id"))
    source = actual_policy.copy()
    source["_policy_key"] = source["scenario_id"].map(_policy_key)
    if "data_mode" in source.columns:
        source = source[source["data_mode"].astype(str) == DATA_MODE_ACTUAL]
    matches = source[source["_policy_key"] == policy_key].copy()
    if matches.empty:
        return None
    matches["date"] = pd.to_datetime(matches["date"])
    return matches.sort_values("date").iloc[-1]


def _recommended_columns(recommended: pd.Series | None) -> dict[str, Any]:
    if recommended is None:
        return {
            "recommended_policy_source": "missing_actual_etf_policy_state",
            "recommended_as_of_date": "",
            "recommended_scenario_id": "",
            "recommended_scenario_label": "",
            "recommended_regime": "",
            "recommended_reason": "",
            "recommended_target_effective_leverage": np.nan,
            "recommended_QQQ_weight": np.nan,
            "recommended_QLD_weight": np.nan,
            "recommended_TQQQ_weight": np.nan,
            "recommended_CASH_weight": np.nan,
            "recommended_weight_sum": np.nan,
        }
    weights = [_safe_float(recommended.get(column)) for column in WEIGHT_COLUMNS]
    return {
        "recommended_policy_source": "latest_actual_etf_policy_state",
        "recommended_as_of_date": _date_or_blank(recommended.get("date")),
        "recommended_scenario_id": recommended.get("scenario_id", ""),
        "recommended_scenario_label": recommended.get("scenario_label", ""),
        "recommended_regime": recommended.get("regime", ""),
        "recommended_reason": recommended.get("reason", ""),
        "recommended_target_effective_leverage": _safe_float(
            recommended.get("target_effective_leverage")
        ),
        "recommended_QQQ_weight": weights[0],
        "recommended_QLD_weight": weights[1],
        "recommended_TQQQ_weight": weights[2],
        "recommended_CASH_weight": weights[3],
        "recommended_weight_sum": float(np.nansum(weights)),
    }


def _decision_authority(authority: pd.Series) -> str:
    selector = str(authority.get("selector", "replay_primary") or "replay_primary")
    method = str(authority.get("ranking_method", "ranking") or "ranking")
    return f"{selector}_{method}"


def _top_candidates(replay_ranking: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    columns = [
        "replay_rank",
        "selector",
        "ranking_method",
        "scenario_id",
        "scenario_label",
        "eligible_for_monthly_signal",
        "cohort_gate_passed",
        "win_rate_vs_qqq_dca",
        "expected_xirr",
        "median_xirr",
        "p05_xirr",
        "expected_max_drawdown",
        "p05_max_drawdown",
        "drawdown_breach_rate",
    ]
    available = [column for column in columns if column in replay_ranking.columns]
    return replay_ranking.head(top_n)[available].copy()


def _review_reasons(
    *,
    actual: pd.Series,
    authority: pd.Series,
    recommended: pd.Series | None,
) -> list[str]:
    reasons: list[str] = []
    if _safe_bool(actual.get("manual_review_required")):
        reasons.append("Actual-primary reference pack requires review")
    if not _safe_bool(authority.get("eligible_for_monthly_signal")):
        reasons.append("Replay-primary top candidate is not eligible")
    if not _safe_bool(authority.get("cohort_gate_passed")):
        reasons.append("Deterministic rolling cohort gate failed")
    if _safe_float(authority.get("drawdown_breach_rate"), default=1.0) > 0.0:
        reasons.append("Monte Carlo replay has drawdown breach paths")
    if recommended is None:
        reasons.append("Recommended policy has no latest actual ETF tradable weight row")
    else:
        weights = [_safe_float(recommended.get(column)) for column in WEIGHT_COLUMNS]
        if not np.isclose(np.nansum(weights), 1.0, atol=1e-6):
            reasons.append("Recommended actual ETF weights do not sum to 100%")
        if _review_now_from_policy(recommended):
            reasons.append("Recommended actual ETF policy state is defensive/off and needs review")
    return reasons


def _review_now_from_policy(row: pd.Series) -> bool:
    regime = str(row.get("regime", ""))
    return any(token in regime.lower() for token in ["off", "defensive", "severe"])


def _policy_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    if "--" in text:
        return text.split("--", 1)[1]
    return text


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _render_recommended_weight_cards(row: pd.Series) -> str:
    cards = []
    for ticker in ["QQQ", "QLD", "TQQQ", "CASH"]:
        value = row.get(f"recommended_{ticker}_weight", 0.0)
        cards.append(
            f"""
      <div class="card">
        <span>{escape(ticker)}</span>
        <strong>{_format_percent(value, digits=0)}</strong>
      </div>"""
        )
    return f'<div class="cards">{"".join(cards)}</div>'


def _render_table(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    if frame.empty:
        return "<p>No candidates.</p>"
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


def _top_candidate_columns() -> list[tuple[str, str]]:
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


def _source_coverage_columns() -> list[tuple[str, str]]:
    return [
        ("ticker", "Ticker"),
        ("replay_start", "Replay Start"),
        ("replay_end", "Replay End"),
        ("actual_start", "Actual Start"),
        ("actual_end", "Actual End"),
        ("synthetic_backfill_start", "Backfill Start"),
        ("synthetic_backfill_end", "Backfill End"),
        ("splice_date", "Splice Date"),
    ]


def _replay_audit_file_names(csv_path: Path) -> dict[str, str]:
    family = csv_path.stem.replace("monthly_decision_comparison_", "")
    replay_prefix = f"monthly_decision_replay_{family}"
    return {
        "ranking": f"{replay_prefix}_ranking.csv",
        "mc_summary": f"{replay_prefix}_mc_summary.csv",
        "mc_trials": f"{replay_prefix}_mc_trials.csv.gz",
        "source_coverage": f"{replay_prefix}_source_coverage.csv",
    }


def _format_cell(value: Any, key: str) -> str:
    if pd.isna(value):
        return ""
    if key in {
        "win_rate_vs_qqq_dca",
        "expected_xirr",
        "median_xirr",
        "p05_xirr",
        "expected_max_drawdown",
        "p05_max_drawdown",
        "drawdown_breach_rate",
    }:
        return _format_percent(value)
    return str(value)


def _format_percent(value: Any, *, digits: int = 2) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}%}"


def _format_leverage(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2f}x"


def _date_or_blank(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).date().isoformat()


__all__ = [
    "COMPARISON_COLUMNS",
    "MonthlyDecisionComparisonInputs",
    "MonthlyDecisionComparisonOutputs",
    "MonthlyDecisionComparisonReportResult",
    "build_monthly_decision_comparison",
    "load_monthly_decision_comparison_inputs",
    "render_monthly_decision_comparison_html",
    "write_monthly_decision_comparison_report",
]
