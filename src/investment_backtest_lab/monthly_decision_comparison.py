from __future__ import annotations

import json
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
    "actual_primary_weights_json",
    "actual_primary_manual_review_required",
    "actual_primary_review_reasons",
    "replay_primary_selector",
    "replay_primary_ranking_method",
    "replay_primary_scenario_id",
    "replay_primary_scenario_label",
    "replay_primary_replay_rank",
    "replay_primary_eligible_for_monthly_signal",
    "replay_primary_cohort_gate_passed",
    "replay_primary_win_rate_vs_benchmark",
    "replay_primary_win_rate_vs_qqq_dca",
    "replay_primary_win_rate_vs_0050_dca",
    "replay_primary_win_rate_vs_fixed_1p5x_dca",
    "replay_primary_expected_xirr",
    "replay_primary_median_xirr",
    "replay_primary_p05_xirr",
    "replay_primary_expected_max_drawdown",
    "replay_primary_p05_max_drawdown",
    "replay_primary_drawdown_breach_rate",
    "replay_primary_cost_mode",
    "replay_primary_total_trade_cost",
    "replay_primary_cost_drag_on_contributed",
    "replay_primary_turnover_sum",
    "replay_primary_manual_review_required",
    "actionable_default_scenario_id",
    "actionable_default_scenario_label",
    "actionable_default_replay_rank",
    "actionable_default_eligible_for_monthly_signal",
    "actionable_default_cohort_gate_passed",
    "actionable_default_win_rate_vs_benchmark",
    "actionable_default_win_rate_vs_qqq_dca",
    "actionable_default_win_rate_vs_0050_dca",
    "actionable_default_win_rate_vs_fixed_1p5x_dca",
    "actionable_default_expected_xirr",
    "actionable_default_p05_xirr",
    "actionable_default_drawdown_breach_rate",
    "actionable_default_cost_mode",
    "actionable_default_total_trade_cost",
    "actionable_default_cost_drag_on_contributed",
    "actionable_default_turnover_sum",
    "actionable_default_manual_review_required",
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
    "recommended_weights_json",
    "recommended_weight_sum",
    "recommendation_layers_differ",
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
    compare_payload: dict[str, Any]


@dataclass(frozen=True)
class MonthlyDecisionComparisonOutputs:
    comparison: pd.DataFrame
    top_candidates: pd.DataFrame
    source_coverage: pd.DataFrame
    compare_payload: dict[str, Any]


@dataclass(frozen=True)
class MonthlyDecisionComparisonReportResult:
    comparison: pd.DataFrame
    top_candidates: pd.DataFrame
    source_coverage: pd.DataFrame
    compare_payload: dict[str, Any]
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
    replay_compare_payload_path = (
        output_dir / f"monthly_decision_replay_{prefix}_compare_payload.json"
    )
    optimizer_compare_payload_path = (
        output_dir / f"dca_policy_optimizer_{prefix}_compare_payload.json"
    )
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
            {},
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
        compare_payload=_load_comparison_payload(
            replay_path=replay_compare_payload_path,
            optimizer_path=optimizer_compare_payload_path,
        ),
    )


def _load_comparison_payload(*, replay_path: Path, optimizer_path: Path) -> dict[str, Any]:
    replay_payload = (
        json.loads(replay_path.read_text(encoding="utf-8")) if replay_path.exists() else {}
    )
    optimizer_payload = (
        json.loads(optimizer_path.read_text(encoding="utf-8"))
        if optimizer_path.exists()
        else {}
    )
    if not replay_payload:
        return optimizer_payload
    merged = json.loads(json.dumps(replay_payload))
    existing = {str(item.get("key", "")) for item in merged.get("scenarios", [])}
    for item in optimizer_payload.get("scenarios", []):
        key = str(item.get("key", ""))
        if item.get("data_mode") == DATA_MODE_ACTUAL and key not in existing:
            merged.setdefault("scenarios", []).append(item)
            existing.add(key)
    merged["optimizer_payload_available"] = bool(optimizer_payload)
    return merged


def build_monthly_decision_comparison(
    *,
    monthly_decision: pd.DataFrame,
    replay_ranking: pd.DataFrame,
    actual_policy: pd.DataFrame | None = None,
    source_coverage: pd.DataFrame | None = None,
    compare_payload: dict[str, Any] | None = None,
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
    actionable = _actionable_default_candidate(ranking)
    recommended = (
        _latest_actual_policy_for_authority(actual_policy, actionable)
        if actionable is not None
        else None
    )

    actual_disagrees = _policy_key(actual.get("scenario_id")) != _policy_key(
        authority.get("scenario_id")
    )
    layers_differ = (
        actionable is None
        or _policy_key(authority.get("scenario_id")) != _policy_key(actionable.get("scenario_id"))
    )
    recommended_row = _recommended_columns(recommended)
    if not recommended_row["recommended_as_of_date"]:
        recommended_row["recommended_as_of_date"] = actual.get("as_of_date", "")
    reasons = _review_reasons(
        actual=actual,
        authority=authority,
        actionable=actionable,
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
        "actual_primary_weights_json": _weights_json(actual),
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
        "replay_primary_win_rate_vs_benchmark": _safe_float(
            authority.get("win_rate_vs_benchmark", authority.get("win_rate_vs_qqq_dca"))
        ),
        "replay_primary_win_rate_vs_qqq_dca": _safe_float(
            authority.get("win_rate_vs_qqq_dca")
        ),
        "replay_primary_win_rate_vs_0050_dca": _safe_float(
            authority.get("win_rate_vs_0050_dca", authority.get("win_rate_vs_qqq_dca"))
        ),
        "replay_primary_win_rate_vs_fixed_1p5x_dca": _safe_float(
            authority.get("win_rate_vs_fixed_1p5x_dca")
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
        "replay_primary_cost_mode": authority.get("cost_mode", "gross_no_cost_model"),
        "replay_primary_total_trade_cost": _safe_float(authority.get("total_trade_cost")),
        "replay_primary_cost_drag_on_contributed": _safe_float(
            authority.get("cost_drag_on_contributed")
        ),
        "replay_primary_turnover_sum": _safe_float(authority.get("turnover_sum")),
        "replay_primary_manual_review_required": _safe_bool(
            authority.get("manual_review_required")
        ),
        **_actionable_default_columns(actionable),
        **recommended_row,
        "recommendation_layers_differ": layers_differ,
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
        compare_payload=compare_payload or {},
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
        compare_payload=inputs.compare_payload,
        top_n=top_n,
    )
    prefix = f"monthly_decision_comparison_{family.lower()}"
    html_path = output_dir / f"{prefix}.html"
    csv_path = output_dir / f"{prefix}.csv"
    top_candidates_path = output_dir / f"{prefix}_top_candidates.csv"
    outputs.comparison.to_csv(csv_path, index=False)
    outputs.top_candidates.to_csv(top_candidates_path, index=False)
    html_path.write_text(
        render_monthly_decision_comparison_html_clean(
            comparison=outputs.comparison,
            top_candidates=outputs.top_candidates,
            source_coverage=outputs.source_coverage,
            compare_payload=outputs.compare_payload,
            csv_path=csv_path,
            top_candidates_path=top_candidates_path,
        ),
        encoding="utf-8",
    )
    return MonthlyDecisionComparisonReportResult(
        comparison=outputs.comparison,
        top_candidates=outputs.top_candidates,
        source_coverage=outputs.source_coverage,
        compare_payload=outputs.compare_payload,
        html_path=html_path,
        csv_path=csv_path,
        top_candidates_path=top_candidates_path,
    )


def validate_monthly_decision_comparison_as_of(
    comparison: pd.DataFrame,
    *,
    expected_as_of: str,
) -> None:
    if comparison.empty:
        raise ValueError("Monthly decision comparison as-of check requires a non-empty frame.")
    actual_as_of = str(comparison.iloc[0].get("recommended_as_of_date", ""))
    if actual_as_of != expected_as_of:
        raise ValueError(
            "Monthly decision comparison is not fresh enough for the configured cutoff. "
            f"Expected recommended_as_of_date {expected_as_of}, got {actual_as_of or 'blank'}. "
            "Check yfinance availability and cache coverage before using this report."
        )


def render_monthly_decision_comparison_html(
    *,
    comparison: pd.DataFrame,
    top_candidates: pd.DataFrame,
    source_coverage: pd.DataFrame,
    compare_payload: dict[str, Any] | None = None,
    csv_path: Path,
    top_candidates_path: Path,
) -> str:
    row = comparison.iloc[0]
    review = bool(row["manual_review_required"])
    disagree = bool(row["actual_disagrees_with_authority"])
    review_label = "Manual Review Required" if review else "Ready For Monthly Review"
    recommended_target = _format_leverage(row["recommended_target_effective_leverage"])
    research_expected_xirr = _format_percent(row["replay_primary_expected_xirr"])
    research_win_rate = _format_percent(row["replay_primary_win_rate_vs_benchmark"])
    research_p05_xirr = _format_percent(row["replay_primary_p05_xirr"])
    research_drawdown = _format_percent(row["replay_primary_expected_max_drawdown"])
    research_p05_drawdown = _format_percent(row["replay_primary_p05_max_drawdown"])
    research_breach_rate = _format_percent(row["replay_primary_drawdown_breach_rate"])
    research_cost = _format_number(row["replay_primary_total_trade_cost"])
    research_cost_drag = _format_percent(row["replay_primary_cost_drag_on_contributed"])
    research_turnover = _format_number(row["replay_primary_turnover_sum"])
    research_review_required = escape(str(row["replay_primary_manual_review_required"]))
    actionable_expected_xirr = _format_percent(row["actionable_default_expected_xirr"])
    actionable_p05_xirr = _format_percent(row["actionable_default_p05_xirr"])
    actionable_breach_rate = _format_percent(row["actionable_default_drawdown_breach_rate"])
    actionable_cost = _format_number(row["actionable_default_total_trade_cost"])
    actionable_cost_drag = _format_percent(row["actionable_default_cost_drag_on_contributed"])
    actionable_turnover = _format_number(row["actionable_default_turnover_sum"])
    chart_payload = _comparison_chart_payload(compare_payload or {}, row)
    decision_title = (
        str(row["recommended_scenario_label"])
        or "需要人工 Review，暫無正式可交易建議"
    )
    actionable_label = (
        str(row["actionable_default_scenario_label"])
        or "目前沒有 zero-breach actionable default"
    )
    if not str(row["recommended_scenario_label"]):
        decision_title = "Manual review required: no automatic tradable recommendation"
    if not str(row["actionable_default_scenario_label"]):
        actionable_label = "No zero-breach actionable default"
    audit_files = _replay_audit_file_names(csv_path)
    coverage_warning = _coverage_warning(source_coverage)
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
{render_html_head(title="Weekly Execution Decision Pack", plotly=bool(chart_payload))}
<body>
<main>
  <section class="hero">
    <div class="panel">
      <p class="eyebrow">Monthly Decision Pack</p>
      <h1>{escape(decision_title)}</h1>
      <p class="lede">
        這頁先回答「現在能不能行動、參考權重是什麼、主要風險和成本在哪裡」。
        Research top 是收益研究候選；Actionable default 才是未人工 override 前的正式可行動層。
      </p>
      <span class="pill {'danger' if review else 'ok'}">{escape(review_label)}</span>
      <p><strong>Research top:</strong> {escape(str(row["replay_primary_scenario_label"]))}</p>
      <p><strong>Actionable default:</strong> {escape(actionable_label)}</p>
      <p><strong>Review reasons:</strong> {escape(str(row["review_reasons"]) or "None")}</p>
    </div>
    <div class="panel">
      <p class="eyebrow">本次回測設定</p>
      {_render_backtest_settings_clean(row, source_coverage, chart_payload)}
    </div>
  </section>

  <section class="two-col">
    <div class="panel">
      <p class="eyebrow">Actual Reference Weights</p>
      <h2>人工 review 參考，不是自動下單建議</h2>
      <p><strong>As of:</strong> {escape(str(row["actual_primary_as_of_date"]))}</p>
      <p><strong>Strategy:</strong> {escape(str(row["actual_primary_scenario_label"]))}</p>
      <p><strong>Regime:</strong> {escape(str(row["actual_primary_regime"]))}</p>
      {_render_actual_reference_weight_cards_clean(row)}
    </div>
    <div class="panel">
      <p class="eyebrow">Actionable Default</p>
      <h2>{escape(actionable_label)}</h2>
      <p>
        正式可行動層必須同時通過 cohort gate、monthly eligibility，且 Monte Carlo
        drawdown breach rate 為 0。現在沒有符合條件的候選，因此報表只提供人工 review 參考。
      </p>
      <p><strong>Target:</strong> {recommended_target}</p>
      <p><strong>Expected XIRR:</strong> {actionable_expected_xirr}</p>
      <p><strong>P05 XIRR:</strong> {actionable_p05_xirr}</p>
      <p><strong>MC Breach:</strong> {actionable_breach_rate}</p>
      <p><strong>Total trade cost:</strong> {actionable_cost}</p>
      <p><strong>Cost drag:</strong> {actionable_cost_drag}</p>
      <p><strong>Turnover:</strong> {actionable_turnover}</p>
      <p class="eyebrow">Actionable Default Weights</p>
      {_render_recommended_weight_cards_clean(row)}
    </div>
  </section>

  <section class="panel">
    <h2>這些數字代表什麼</h2>
    {_render_plain_language_metric_cards_clean(row)}
  </section>

  <section class="two-col">
    <div class="panel">
      <p class="eyebrow">Research Authority</p>
      <h2>{escape(str(row["replay_primary_scenario_label"]))}</h2>
      <p><strong>Authority:</strong> {escape(str(row["decision_authority"]))}</p>
      <p><strong>Eligible:</strong> {eligible_badge}</p>
      <p><strong>Cohort gate:</strong> {gate_badge}</p>
      <p><strong>Expected XIRR:</strong> {research_expected_xirr}</p>
      <p><strong>Win rate:</strong> {research_win_rate}</p>
      <p><strong>P05 XIRR:</strong> {research_p05_xirr}</p>
      <p><strong>Expected drawdown:</strong> {research_drawdown}</p>
      <p><strong>P05 drawdown:</strong> {research_p05_drawdown}</p>
      <p><strong>Drawdown breach rate:</strong> {research_breach_rate}</p>
      <p><strong>Cost mode:</strong> {escape(str(row["replay_primary_cost_mode"]))}</p>
      <p><strong>Total trade cost:</strong> {research_cost}</p>
      <p><strong>Cost drag:</strong> {research_cost_drag}</p>
      <p><strong>Turnover:</strong> {research_turnover}</p>
      <p><strong>Requires review:</strong> {research_review_required}</p>
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

  {_render_compare_chart_section_clean(chart_payload)}

  <section class="panel">
    <h2>Source Coverage</h2>
    <p>
      Hybrid-primary uses actual ETF data after listing and scaled synthetic backfill before
      listing. This table shows which period came from which source.
    </p>
    {coverage_warning}
    <div class="table-wrap">{_render_table(source_coverage, _source_coverage_columns())}</div>
  </section>

  <section class="panel">
    <h2>Replay-Primary Top Candidates</h2>
    <div class="table-wrap">{_render_table(top_candidates, _top_candidate_columns())}</div>
  </section>

  {_render_external_signal_audit_section(csv_path)}

  {_render_optuna_candidate_section(csv_path)}

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


def render_monthly_decision_comparison_html_clean(
    *,
    comparison: pd.DataFrame,
    top_candidates: pd.DataFrame,
    source_coverage: pd.DataFrame,
    compare_payload: dict[str, Any] | None = None,
    csv_path: Path,
    top_candidates_path: Path,
) -> str:
    row = comparison.iloc[0]
    chart_payload = _comparison_chart_payload(compare_payload or {}, row)
    review = bool(row["manual_review_required"])
    pack_title = (
        "Weekly Execution Decision Pack"
        if _is_tw50(row, source_coverage)
        else "Monthly Decision Pack"
    )
    pack_lede = (
        "TW50 now uses weekly trade checks with monthly DCA contributions. Research authority, "
        "actionable default, and actual ETF reference are shown separately."
        if _is_tw50(row, source_coverage)
        else (
            "Research authority, actionable default, and actual ETF reference are shown "
            "separately. A candidate with Monte Carlo breach paths remains a research idea "
            "until manual review."
        )
    )
    decision_title = (
        str(row["recommended_scenario_label"])
        or "Manual review required: no automatic tradable recommendation"
    )
    actionable_label = str(row["actionable_default_scenario_label"]) or (
        "No zero-breach actionable default"
    )
    audit_files = _replay_audit_file_names(csv_path)
    eligible_badge = risk_badge(
        str(row["replay_primary_eligible_for_monthly_signal"]),
        bool(row["replay_primary_eligible_for_monthly_signal"]),
    )
    gate_badge = risk_badge(
        str(row["replay_primary_cohort_gate_passed"]),
        bool(row["replay_primary_cohort_gate_passed"]),
    )
    actionable_leverage = _format_leverage(
        row["recommended_target_effective_leverage"]
    )
    actionable_expected_xirr = _format_percent(
        row["actionable_default_expected_xirr"]
    )
    actionable_p05_xirr = _format_percent(row["actionable_default_p05_xirr"])
    actionable_breach = _format_percent(
        row["actionable_default_drawdown_breach_rate"]
    )
    actionable_cost_drag = _format_percent(
        row["actionable_default_cost_drag_on_contributed"]
    )
    actionable_turnover = _format_number(
        row["actionable_default_turnover_sum"]
    )
    research_expected_xirr = _format_percent(row["replay_primary_expected_xirr"])
    research_win_rate = _format_percent(
        row["replay_primary_win_rate_vs_benchmark"]
    )
    research_p05_xirr = _format_percent(row["replay_primary_p05_xirr"])
    research_expected_dd = _format_percent(
        row["replay_primary_expected_max_drawdown"]
    )
    research_p05_dd = _format_percent(row["replay_primary_p05_max_drawdown"])
    research_breach = _format_percent(
        row["replay_primary_drawdown_breach_rate"]
    )
    research_cost_mode = escape(str(row["replay_primary_cost_mode"]))
    research_trade_cost = _format_number(row["replay_primary_total_trade_cost"])
    research_cost_drag = _format_percent(
        row["replay_primary_cost_drag_on_contributed"]
    )
    research_turnover = _format_number(row["replay_primary_turnover_sum"])
    actual_disagrees = escape(str(row["actual_disagrees_with_authority"]))
    ranking_file = audit_files["ranking"]
    mc_summary_file = audit_files["mc_summary"]
    mc_trials_file = audit_files["mc_trials"]
    source_coverage_file = audit_files["source_coverage"]
    compare_payload_file = audit_files["compare_payload"]
    return f"""<!doctype html>
<html lang="zh-Hant">
{render_html_head(title=pack_title, plotly=bool(chart_payload))}
<body>
<main>
  <section class="hero">
    <div class="panel">
      <p class="eyebrow">{escape(pack_title)}</p>
      <h1>{escape(decision_title)}</h1>
      <p class="lede">{escape(pack_lede)}</p>
      <span class="pill {'danger' if review else 'ok'}">
        {escape("Manual Review Required" if review else "Ready For Monthly Review")}
      </span>
      <p><strong>Research top:</strong> {escape(str(row["replay_primary_scenario_label"]))}</p>
      <p><strong>Actionable default:</strong> {escape(actionable_label)}</p>
      <p><strong>Review reasons:</strong> {escape(str(row["review_reasons"]) or "None")}</p>
    </div>
    <div class="panel">
      <p class="eyebrow">Backtest Settings</p>
      {_render_backtest_settings_clean(row, source_coverage, chart_payload)}
    </div>
  </section>

  <section class="two-col">
    <div class="panel">
      <p class="eyebrow">Actual Reference Weights</p>
      <h2>Manual review reference, not an automatic order</h2>
      <p><strong>As of:</strong> {escape(str(row["actual_primary_as_of_date"]))}</p>
      <p><strong>Strategy:</strong> {escape(str(row["actual_primary_scenario_label"]))}</p>
      <p><strong>Regime:</strong> {escape(str(row["actual_primary_regime"]))}</p>
      {_render_actual_reference_weight_cards_clean(row)}
    </div>
    <div class="panel">
      <p class="eyebrow">Actionable Default</p>
      <h2>{escape(actionable_label)}</h2>
      <p>
        The actionable default must be eligible, pass deterministic cohorts, and have zero
        Monte Carlo breach paths. If none exists, tradable weights are withheld.
      </p>
      <p><strong>Target leverage:</strong> {actionable_leverage}</p>
      <p><strong>Expected XIRR:</strong> {actionable_expected_xirr}</p>
      <p><strong>p05 XIRR:</strong> {actionable_p05_xirr}</p>
      <p><strong>MC breach:</strong> {actionable_breach}</p>
      <p><strong>Cost drag:</strong> {actionable_cost_drag}</p>
      <p><strong>Turnover:</strong> {actionable_turnover}</p>
      <p class="eyebrow">Actionable Default Weights</p>
      {_render_recommended_weight_cards_clean(row)}
    </div>
  </section>

  <section class="panel">
    <h2>Plain-Language Risk Metrics</h2>
    {_render_plain_language_metric_cards_clean(row)}
  </section>

  <section class="panel">
    <h2>Why Costs Are High</h2>
    {_render_cost_diagnostics_clean(top_candidates)}
  </section>

  <section class="two-col">
    <div class="panel">
      <p class="eyebrow">Research Authority</p>
      <h2>{escape(str(row["replay_primary_scenario_label"]))}</h2>
      <p><strong>Authority:</strong> {escape(str(row["decision_authority"]))}</p>
      <p><strong>Eligible:</strong> {eligible_badge}</p>
      <p><strong>Cohort gate:</strong> {gate_badge}</p>
      <p><strong>Expected XIRR:</strong> {research_expected_xirr}</p>
      <p><strong>Win rate vs official benchmark:</strong> {research_win_rate}</p>
      <p><strong>p05 XIRR:</strong> {research_p05_xirr}</p>
      <p><strong>Expected drawdown:</strong> {research_expected_dd}</p>
      <p><strong>p05 drawdown:</strong> {research_p05_dd}</p>
      <p><strong>Drawdown breach rate:</strong> {research_breach}</p>
      <p><strong>Cost mode:</strong> {research_cost_mode}</p>
      <p><strong>Total trade cost:</strong> {research_trade_cost}</p>
      <p><strong>Cost drag:</strong> {research_cost_drag}</p>
      <p><strong>Turnover:</strong> {research_turnover}</p>
    </div>
    <div class="panel">
      <p class="eyebrow">Actual Primary Reference</p>
      <h2>{escape(str(row["actual_primary_scenario_label"]))}</h2>
      <p><strong>Actual disagrees with authority:</strong> {actual_disagrees}</p>
      <p>
        Different signals are not a data error. They mean the actual ETF reference and
        replay research authority currently disagree and need review.
      </p>
      <p><strong>Reason:</strong> {escape(str(row["actual_primary_reason"]))}</p>
    </div>
  </section>

  {_render_compare_chart_section_clean(chart_payload)}

  <section class="panel">
    <h2>Source Coverage</h2>
    <p>
      For weekly core external replay, the official window is restricted to dates where all core
      external signals are available after shift. Earlier 1999 data is reserved for long-stress
      appendices, not official external scoring.
    </p>
    {_coverage_warning(source_coverage)}
    <div class="table-wrap">{_render_table(source_coverage, _source_coverage_columns())}</div>
  </section>

  <section class="panel">
    <h2>Replay Top Candidates</h2>
    <div class="table-wrap">{_render_table(top_candidates, _top_candidate_columns())}</div>
  </section>

  {_render_external_signal_audit_section(csv_path)}
  {_render_optuna_candidate_section(csv_path)}

  <section class="panel links">
    <h2>Audit Files</h2>
    <a href="{csv_path.name}">comparison CSV</a>
    <a href="{top_candidates_path.name}">top candidates CSV</a>
    <a href="{ranking_file}">replay ranking CSV</a>
    <a href="{mc_summary_file}">MC summary CSV</a>
    <a href="{mc_trials_file}">compressed MC trials CSV</a>
    <a href="{source_coverage_file}">source coverage CSV</a>
    <a href="{compare_payload_file}">replay compare payload JSON</a>
  </section>
</main>
</body>
</html>
"""


def _render_backtest_settings_clean(
    row: pd.Series,
    source_coverage: pd.DataFrame,
    compare_payload: dict[str, Any],
) -> str:
    actual_summary = _payload_mode_summary(compare_payload, DATA_MODE_ACTUAL)
    replay_summary = _payload_mode_summary(compare_payload, "non_actual")
    is_tw50 = _is_tw50(row, source_coverage)
    family_label = _family_label(row, source_coverage)
    contribution_note = (
        "Initial 100,000; monthly contribution 10,000; "
        "trade cadence monthly core + weekly delta trigger; contribution cadence monthly"
        if is_tw50
        else "Configured initial cash and monthly contribution"
    )
    cost_note = (
        "TW ETF commission 0.1425% * 28%, minimum 1 TWD, "
        "ETF sell transaction tax 0.1%, slippage 0.01%."
        if is_tw50
        else "Net-of-cost settings from config.cost_model."
    )
    cadence_kpis = (
        """
        <div class="kpi"><span>Trade cadence</span><strong>monthly + weekly trigger</strong></div>
        <div class="kpi"><span>Contribution cadence</span><strong>monthly</strong></div>
        <div class="kpi"><span>Official benchmark</span><strong>Fixed 1.5x DCA</strong></div>
        """
        if is_tw50
        else ""
    )
    coverage_note = _coverage_window_note(source_coverage)
    cutoff = escape(str(row["recommended_as_of_date"]))
    escaped_contribution = escape(contribution_note)
    cost_mode = escape(str(row["replay_primary_cost_mode"]))
    actual_summary_text = _format_mode_summary_clean(actual_summary)
    replay_summary_text = _format_mode_summary_clean(replay_summary)
    return f"""
      <div class="kpis">
        <div class="kpi"><span>Family</span><strong>{escape(family_label)}</strong></div>
        <div class="kpi"><span>Cutoff</span><strong>{cutoff}</strong></div>
        <div class="kpi"><span>Contribution plan</span><strong>{escaped_contribution}</strong></div>
        {cadence_kpis}
        <div class="kpi"><span>Cost mode</span><strong>{cost_mode}</strong></div>
      </div>
      <p><strong>Actual ETF optimizer:</strong> {actual_summary_text}</p>
      <p><strong>Official replay / compare lab:</strong> {replay_summary_text}</p>
      <p><strong>Official window:</strong> {escape(coverage_note)}</p>
      <p><strong>Cost model:</strong> {escape(cost_note)}</p>
    """


def _render_actual_reference_weight_cards_clean(row: pd.Series) -> str:
    weights = _display_weights_from_json_field(row, "actual_primary_weights_json")
    leverage = _format_leverage(row.get("actual_primary_target_effective_leverage"))
    if not weights:
        return "<p>No actual reference weights.</p>"
    cards = "".join(
        f"""
      <div class="card">
        <span>{escape(ticker)}</span>
        <strong>{_format_percent(value, digits=1)}</strong>
      </div>"""
        for ticker, value in weights.items()
    )
    return f"""
    <p class="note">
      These are latest actual ETF reference weights for manual review. They are not an
      automatic order unless an actionable default is available and accepted.
    </p>
    <p><strong>Effective leverage:</strong> {leverage}</p>
    <div class="cards">{cards}</div>
    """


def _render_recommended_weight_cards_clean(row: pd.Series) -> str:
    weights = _recommended_weights(row)
    if not weights:
        return "<p>No tradable weights. Manual review is required.</p>"
    cards = "".join(
        f"""
      <div class="card">
        <span>{escape(ticker)}</span>
        <strong>{_format_percent(value, digits=0)}</strong>
      </div>"""
        for ticker, value in weights.items()
    )
    return f'<div class="cards">{cards}</div>'


def _render_plain_language_metric_cards_clean(row: pd.Series) -> str:
    cards = [
        (
            "p05 XIRR",
            _format_percent(row["replay_primary_p05_xirr"]),
            "The bad 5% Monte Carlo annualized return. Higher is better.",
        ),
        (
            "p05 drawdown",
            _format_percent(row["replay_primary_p05_max_drawdown"]),
            "The bad 5% maximum drawdown. Less negative is better.",
        ),
        (
            "cost drag",
            _format_percent(row["replay_primary_cost_drag_on_contributed"]),
            "Total trade cost divided by total contributed capital.",
        ),
        (
            "turnover",
            _format_number(row["replay_primary_turnover_sum"]),
            "Cumulative trading intensity. Higher turnover usually raises cost and review risk.",
        ),
        (
            "MC breach",
            _format_percent(row["replay_primary_drawdown_breach_rate"]),
            "Share of Monte Carlo paths breaching the hard drawdown rule.",
        ),
    ]
    body = "".join(
        f"""
      <div class="kpi">
        <span>{escape(label)}</span>
        <strong>{escape(value)}</strong>
        <p>{escape(note)}</p>
      </div>"""
        for label, value, note in cards
    )
    return f'<div class="kpis">{body}</div>'


def _render_cost_diagnostics_clean(top_candidates: pd.DataFrame) -> str:
    if top_candidates.empty:
        return "<p>No trade cost diagnostics are available yet.</p>"
    rows = top_candidates.head(5).copy()
    table = _render_table(
        rows,
        [
            ("replay_rank", "Rank"),
            ("scenario_label", "Strategy"),
            ("total_trade_cost", "Trade Cost"),
            ("cost_drag_on_contributed", "Cost / Contributed"),
            ("cost_to_final_equity", "Cost / Final Equity"),
            ("turnover_sum", "Turnover"),
            ("trade_days", "Trade Days"),
            ("max_single_day_trade_cost", "Max Daily Cost"),
        ],
    )
    return f"""
    <p>
      TW50 buying cost is roughly 0.0399% commission plus 0.01% slippage
      with a 1 TWD minimum commission.
      Selling adds the 0.1% ETF transaction tax. High costs usually come from
      large weekly/monthly switches after the portfolio has grown, not from the fee
      schedule itself.
    </p>
    <p>
      Cost / contributed can look alarming because the denominator is only cash
      invested. Cost / final equity is shown beside it to judge whether the cost
      actually overwhelms the accumulated portfolio.
    </p>
    <div class="table-wrap">{table}</div>
    """


def _render_compare_chart_section_clean(compare_payload: dict[str, Any]) -> str:
    if not compare_payload or not compare_payload.get("scenarios"):
        return """
  <section class="panel">
    <h2>Replay Compare Lab</h2>
    <p class="note">
      No replay compare payload is available. Rerun monthly decision replay to generate
      the interactive candidate, baseline, and 0050 DCA chart.
    </p>
  </section>
"""
    payload_json = json.dumps(compare_payload, ensure_ascii=False)
    return f"""
  <section class="panel">
    <h2>Replay Compare Lab</h2>
    <p>
      Compare research top, Vol Target baseline, 0050 base DCA, and actual reference.
      Use the metric buttons to switch equity, drawdown, effective leverage, cumulative
      contribution, cumulative trade cost, and turnover.
    </p>
    <div class="compare-grid">
      <div>
        <div class="metric-buttons" id="comparison-metric-buttons"></div>
        <div class="controls" id="comparison-scenario-controls"></div>
      </div>
      <div id="compare-chart"></div>
    </div>
  </section>
  <script id="comparison-compare-payload" type="application/json">
{payload_json}
  </script>
  <script>
const comparisonPayload = JSON.parse(
  document.getElementById('comparison-compare-payload').textContent
);
let comparisonActiveMetric = comparisonPayload.default_metric || 'normalized_equity';
const comparisonControls = document.getElementById('comparison-scenario-controls');
const comparisonMetricButtons = document.getElementById('comparison-metric-buttons');
function comparisonScenarioLabel(item) {{
  return item.short + ' | ' + item.data_mode + ' | ' + item.validation_status;
}}
Object.entries(comparisonPayload.metrics).forEach(([key, config]) => {{
  const button = document.createElement('button');
  button.textContent = config.label;
  button.dataset.metric = key;
  if (key === comparisonActiveMetric) button.classList.add('active');
  button.onclick = () => {{
    comparisonActiveMetric = key;
    document
      .querySelectorAll('#comparison-metric-buttons button')
      .forEach((node) => node.classList.remove('active'));
    button.classList.add('active');
    drawComparisonChart();
  }};
  comparisonMetricButtons.appendChild(button);
}});
comparisonPayload.scenarios.forEach((item) => {{
  const label = document.createElement('label');
  label.className = 'check';
  const input = document.createElement('input');
  input.type = 'checkbox';
  input.checked = Boolean(item.comparison_default);
  input.dataset.key = item.key;
  input.onchange = drawComparisonChart;
  label.appendChild(input);
  label.append(' ' + comparisonScenarioLabel(item));
  comparisonControls.appendChild(label);
}});
function selectedComparisonScenarios() {{
  const selected = new Set(
    Array.from(document.querySelectorAll('#comparison-scenario-controls input:checked'))
      .map((node) => node.dataset.key)
  );
  return comparisonPayload.scenarios.filter((item) => selected.has(item.key));
}}
function drawComparisonChart() {{
  if (!window.Plotly) return;
  const metric = comparisonPayload.metrics[comparisonActiveMetric];
  const traces = selectedComparisonScenarios().map((item) => ({{
    x: item.dates,
    y: item.series[comparisonActiveMetric],
    mode: 'lines',
    name: item.short + ' | ' + item.data_mode,
    hovertemplate: item.full + '<br>%{{x}}<br>%{{y:.3f}}<extra></extra>'
  }}));
  Plotly.react('compare-chart', traces, {{
    margin: {{l: 64, r: 20, t: 20, b: 48}},
    paper_bgcolor: '#fffffc',
    plot_bgcolor: '#fffffc',
    yaxis: {{title: metric.axis, zeroline: false}},
    xaxis: {{title: ''}},
    legend: {{orientation: 'h', y: -0.18}},
  }}, {{responsive: true, displaylogo: false}});
}}
drawComparisonChart();
  </script>
"""


def _localized_chart_metrics_clean(metrics: dict[str, Any]) -> dict[str, Any]:
    labels = {
        "total_equity": ("Net equity", "TWD", "money"),
        "normalized_equity": ("Normalized equity", "Index", "money"),
        "drawdown": ("Drawdown", "Drawdown", "percent"),
        "effective_leverage": ("Effective leverage", "Leverage", "number"),
        "total_contributed": ("Cumulative contributed", "TWD", "money"),
        "cumulative_trade_cost": ("Cumulative trade cost", "TWD", "money"),
        "trade_cost": ("Period trade cost", "TWD", "money"),
        "commission": ("Commission", "TWD", "money"),
        "transaction_tax": ("Transaction tax", "TWD", "money"),
        "slippage": ("Slippage", "TWD", "money"),
        "turnover": ("Turnover", "Turnover", "number"),
    }
    ordered = [
        "total_equity",
        "normalized_equity",
        "drawdown",
        "effective_leverage",
        "total_contributed",
        "cumulative_trade_cost",
        "trade_cost",
        "commission",
        "transaction_tax",
        "slippage",
        "turnover",
    ]
    result: dict[str, Any] = {}
    for key in ordered:
        if key not in metrics:
            continue
        label, axis, fmt = labels[key]
        result[key] = {**metrics.get(key, {}), "label": label, "axis": axis, "format": fmt}
    for key, config in metrics.items():
        if key not in result:
            result[key] = config
    return result


def _format_mode_summary_clean(summary: dict[str, Any]) -> str:
    start = summary.get("start") or "unknown"
    end = summary.get("end") or "unknown"
    contributed = _format_number(summary.get("total_contributed"))
    if contributed:
        return f"{escape(str(start))} ~ {escape(str(end))}; total contributed {contributed}"
    return f"{escape(str(start))} ~ {escape(str(end))}"


def _coverage_window_note(source_coverage: pd.DataFrame) -> str:
    if source_coverage.empty:
        return "unknown"
    starts = [
        str(value)
        for value in source_coverage.get("replay_start", pd.Series(dtype=str)).dropna()
        if str(value)
    ]
    ends = [
        str(value)
        for value in source_coverage.get("replay_end", pd.Series(dtype=str)).dropna()
        if str(value)
    ]
    if not starts or not ends:
        return "unknown"
    start = min(starts)
    end = max(ends)
    notes = " ".join(
        str(value)
        for value in source_coverage.get("source_notes", pd.Series(dtype=str)).dropna()
    )
    if "official_core_external_replay_window" in notes:
        return f"{start} ~ {end} (core external feature-complete official window)"
    return f"{start} ~ {end}"


def _sorted_replay_ranking(replay_ranking: pd.DataFrame) -> pd.DataFrame:
    if "replay_rank" not in replay_ranking.columns:
        raise ValueError("Replay ranking CSV must contain replay_rank.")
    return replay_ranking.sort_values("replay_rank").reset_index(drop=True)


def _actionable_default_candidate(ranking: pd.DataFrame) -> pd.Series | None:
    source = ranking[
        ranking["eligible_for_monthly_signal"].map(_safe_bool)
        & ranking["cohort_gate_passed"].map(_safe_bool)
        & np.isclose(
            ranking["drawdown_breach_rate"].map(_safe_float),
            0.0,
            atol=1e-12,
        )
    ]
    if source.empty:
        return None
    return source.iloc[0]


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
            "recommended_weights_json": "{}",
            "recommended_weight_sum": np.nan,
        }
    weights = [_safe_float(recommended.get(column)) for column in WEIGHT_COLUMNS]
    dynamic_weights = _weights_dict(recommended)
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
        "recommended_weights_json": json.dumps(
            dynamic_weights,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "recommended_weight_sum": float(np.nansum(list(dynamic_weights.values()))),
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
        "win_rate_vs_benchmark",
        "win_rate_vs_qqq_dca",
        "expected_xirr",
        "median_xirr",
        "p05_xirr",
        "expected_max_drawdown",
        "p05_max_drawdown",
        "drawdown_breach_rate",
        "manual_review_required",
        "cost_mode",
        "total_trade_cost",
        "cost_drag_on_contributed",
        "cost_to_final_equity",
        "turnover_sum",
        "trade_days",
        "max_single_day_trade_cost",
        "effective_leverage_avg",
        "effective_leverage_max",
        "effective_leverage_latest",
    ]
    available = [column for column in columns if column in replay_ranking.columns]
    return replay_ranking.head(top_n)[available].copy()


def _review_reasons(
    *,
    actual: pd.Series,
    authority: pd.Series,
    actionable: pd.Series | None,
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
        reasons.append("Research authority has Monte Carlo drawdown breach paths")
    if actionable is None:
        reasons.append("No zero-breach actionable default is available")
    if recommended is None:
        reasons.append("Actionable default has no latest actual ETF tradable weight row")
    else:
        weights = list(_weights_dict(recommended).values())
        if not np.isclose(np.nansum(weights), 1.0, atol=1e-6):
            reasons.append("Recommended actual ETF weights do not sum to 100%")
        if _review_now_from_policy(recommended):
            reasons.append("Recommended actual ETF policy state is defensive/off and needs review")
    return reasons


def _actionable_default_columns(actionable: pd.Series | None) -> dict[str, Any]:
    if actionable is None:
        return {
            "actionable_default_scenario_id": "",
            "actionable_default_scenario_label": "",
            "actionable_default_replay_rank": np.nan,
            "actionable_default_eligible_for_monthly_signal": False,
            "actionable_default_cohort_gate_passed": False,
            "actionable_default_win_rate_vs_benchmark": np.nan,
            "actionable_default_win_rate_vs_qqq_dca": np.nan,
            "actionable_default_win_rate_vs_0050_dca": np.nan,
            "actionable_default_win_rate_vs_fixed_1p5x_dca": np.nan,
            "actionable_default_expected_xirr": np.nan,
            "actionable_default_p05_xirr": np.nan,
            "actionable_default_drawdown_breach_rate": np.nan,
            "actionable_default_cost_mode": "",
            "actionable_default_total_trade_cost": np.nan,
            "actionable_default_cost_drag_on_contributed": np.nan,
            "actionable_default_turnover_sum": np.nan,
            "actionable_default_manual_review_required": True,
        }
    return {
        "actionable_default_scenario_id": actionable.get("scenario_id", ""),
        "actionable_default_scenario_label": actionable.get("scenario_label", ""),
        "actionable_default_replay_rank": int(_safe_float(actionable.get("replay_rank"), 0.0)),
        "actionable_default_eligible_for_monthly_signal": _safe_bool(
            actionable.get("eligible_for_monthly_signal")
        ),
        "actionable_default_cohort_gate_passed": _safe_bool(
            actionable.get("cohort_gate_passed")
        ),
        "actionable_default_win_rate_vs_benchmark": _safe_float(
            actionable.get("win_rate_vs_benchmark", actionable.get("win_rate_vs_qqq_dca"))
        ),
        "actionable_default_win_rate_vs_qqq_dca": _safe_float(
            actionable.get("win_rate_vs_qqq_dca")
        ),
        "actionable_default_win_rate_vs_0050_dca": _safe_float(
            actionable.get("win_rate_vs_0050_dca", actionable.get("win_rate_vs_qqq_dca"))
        ),
        "actionable_default_win_rate_vs_fixed_1p5x_dca": _safe_float(
            actionable.get("win_rate_vs_fixed_1p5x_dca")
        ),
        "actionable_default_expected_xirr": _safe_float(actionable.get("expected_xirr")),
        "actionable_default_p05_xirr": _safe_float(actionable.get("p05_xirr")),
        "actionable_default_drawdown_breach_rate": _safe_float(
            actionable.get("drawdown_breach_rate")
        ),
        "actionable_default_cost_mode": actionable.get("cost_mode", "gross_no_cost_model"),
        "actionable_default_total_trade_cost": _safe_float(
            actionable.get("total_trade_cost")
        ),
        "actionable_default_cost_drag_on_contributed": _safe_float(
            actionable.get("cost_drag_on_contributed")
        ),
        "actionable_default_turnover_sum": _safe_float(actionable.get("turnover_sum")),
        "actionable_default_manual_review_required": _safe_bool(
            actionable.get("manual_review_required")
        ),
    }


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


def _weight_columns(row: pd.Series) -> list[str]:
    return [
        str(column)
        for column in row.index
        if str(column).endswith("_weight")
        and str(column) != "cash_weight"
        and not str(column).startswith(
            (
                "previous_",
                "actual_primary_",
                "actionable_default_",
                "recommended_",
                "replay_primary_",
            )
        )
    ]


def _weights_dict(row: pd.Series) -> dict[str, float]:
    return {
        column.removesuffix("_weight"): _safe_float(row.get(column), default=0.0)
        for column in _weight_columns(row)
    }


def _weights_json(row: pd.Series) -> str:
    return json.dumps(_weights_dict(row), ensure_ascii=False, sort_keys=True)


def _recommended_weights(row: pd.Series) -> dict[str, float]:
    raw = row.get("recommended_weights_json", "")
    if raw and not pd.isna(raw):
        try:
            parsed = json.loads(str(raw))
            if isinstance(parsed, dict):
                return {str(key): _safe_float(value, default=0.0) for key, value in parsed.items()}
        except json.JSONDecodeError:
            pass
    return {
        ticker: _safe_float(row.get(f"recommended_{ticker}_weight"), default=0.0)
        for ticker in ["QQQ", "QLD", "TQQQ", "CASH"]
        if not pd.isna(row.get(f"recommended_{ticker}_weight", np.nan))
    }


def _render_recommended_weight_cards(row: pd.Series) -> str:
    cards = []
    for ticker, value in _recommended_weights(row).items():
        cards.append(
            f"""
      <div class="card">
        <span>{escape(ticker)}</span>
        <strong>{_format_percent(value, digits=0)}</strong>
      </div>"""
        )
    return f'<div class="cards">{"".join(cards)}</div>'


def _render_actual_reference_weight_cards(row: pd.Series) -> str:
    weights = _display_weights_from_json_field(row, "actual_primary_weights_json")
    if not weights:
        return "<p>No actual reference weights.</p>"
    cards = []
    for ticker, value in weights.items():
        cards.append(
            f"""
      <div class="card">
        <span>{escape(ticker)}</span>
        <strong>{_format_percent(value, digits=1)}</strong>
      </div>"""
        )
    return f"""
    <p class="note">
      這組權重來自 latest actual ETF policy state，只供人工 review 參考；
      目前沒有 zero-breach actionable default，所以它不是自動下單建議。
    </p>
    <div class="cards">{"".join(cards)}</div>
    """


def _render_backtest_settings(
    row: pd.Series,
    source_coverage: pd.DataFrame,
    compare_payload: dict[str, Any],
) -> str:
    actual_summary = _payload_mode_summary(compare_payload, DATA_MODE_ACTUAL)
    replay_summary = _payload_mode_summary(compare_payload, "non_actual")
    family_label = _family_label(row, source_coverage)
    contribution_note = (
        "初始 100,000；每月 10,000"
        if _is_tw50(row, source_coverage)
        else "依 config 的 initial cash / monthly contribution 設定"
    )
    cost_note = (
        "台股 ETF 手續費 0.1425% * 28%，每筆最低 20；ETF 賣出證交稅 0.1%；"
        "slippage 0.01%。"
        if _is_tw50(row, source_coverage)
        else "使用 config.cost_model 的 net-of-cost 設定。"
    )
    cutoff = escape(str(row["recommended_as_of_date"]))
    cost_mode = escape(str(row["replay_primary_cost_mode"]))
    return f"""
      <div class="kpis">
        <div class="kpi"><span>Family</span><strong>{escape(family_label)}</strong></div>
        <div class="kpi"><span>Cutoff</span><strong>{cutoff}</strong></div>
        <div class="kpi"><span>投入設定</span><strong>{escape(contribution_note)}</strong></div>
        <div class="kpi"><span>Cost mode</span><strong>{cost_mode}</strong></div>
      </div>
      <p><strong>Actual ETF optimizer:</strong> {_format_mode_summary(actual_summary)}</p>
      <p><strong>Hybrid replay:</strong> {_format_mode_summary(replay_summary)}</p>
      <p><strong>成本模型:</strong> {escape(cost_note)}</p>
    """


def _render_plain_language_metric_cards(row: pd.Series) -> str:
    cards = [
        (
            "p05 XIRR",
            _format_percent(row["replay_primary_p05_xirr"]),
            "偏壞 5% 情境的年化報酬；約 5% Monte Carlo 路徑會比這個更差。",
        ),
        (
            "p05 drawdown",
            _format_percent(row["replay_primary_p05_max_drawdown"]),
            "偏壞 5% 情境的最大回撤；用來看左尾壓力，不是平均情境。",
        ),
        (
            "cost drag",
            _format_percent(row["replay_primary_cost_drag_on_contributed"]),
            "總交易成本 / 總投入；不是年化值，代表整段 replay 的成本負擔。",
        ),
        (
            "turnover",
            _format_number(row["replay_primary_turnover_sum"]),
            "累積換倉強度；數字越高，代表策略越常大幅調整權重。",
        ),
        (
            "MC breach",
            _format_percent(row["replay_primary_drawdown_breach_rate"]),
            "Monte Carlo 路徑中跌破硬性回撤門檻的比例；大於 0 就需要人工 review。",
        ),
    ]
    body = "".join(
        f"""
      <div class="kpi">
        <span>{escape(label)}</span>
        <strong>{escape(value)}</strong>
        <p>{escape(note)}</p>
      </div>"""
        for label, value, note in cards
    )
    return f'<div class="kpis">{body}</div>'


def _render_compare_chart_section(compare_payload: dict[str, Any]) -> str:
    if not compare_payload or not compare_payload.get("scenarios"):
        return """
  <section class="panel">
    <h2>候選策略趨勢圖</h2>
    <p class="note">
      找不到 optimizer compare payload。請先重跑
      analyze_dca_policy_optimizer.py，再重新產生 comparison report。
    </p>
  </section>
"""
    payload_json = json.dumps(compare_payload, ensure_ascii=False)
    return f"""
  <section class="panel">
    <h2>候選策略趨勢圖</h2>
    <p>
      可勾選完整候選策略，並切換淨資產、回撤、有效槓桿、累積投入、
      累積交易成本與 turnover。預設勾選 research top、actual reference 與 base 1x。
    </p>
    <div class="compare-grid">
      <div>
        <div class="metric-buttons" id="comparison-metric-buttons"></div>
        <div class="controls" id="comparison-scenario-controls"></div>
      </div>
      <div id="compare-chart"></div>
    </div>
  </section>
  <script id="comparison-compare-payload" type="application/json">
{payload_json}
  </script>
  <script>
const comparisonPayload = JSON.parse(
  document.getElementById('comparison-compare-payload').textContent
);
let comparisonActiveMetric = comparisonPayload.default_metric || 'total_equity';
const comparisonControls = document.getElementById('comparison-scenario-controls');
const comparisonMetricButtons = document.getElementById('comparison-metric-buttons');
function comparisonScenarioLabel(item) {{
  return item.short + ' · ' + item.data_mode + ' · ' + item.validation_status;
}}
function comparisonFormatValue(value, format) {{
  if (value === null || Number.isNaN(value)) return '';
  if (format === 'percent') return (value * 100).toFixed(2) + '%';
  if (format === 'money') {{
    return Number(value).toLocaleString(undefined, {{maximumFractionDigits: 0}});
  }}
  return Number(value).toFixed(3);
}}
Object.entries(comparisonPayload.metrics).forEach(([key, config]) => {{
  const button = document.createElement('button');
  button.textContent = config.label;
  button.dataset.metric = key;
  if (key === comparisonActiveMetric) button.classList.add('active');
  button.onclick = () => {{
    comparisonActiveMetric = key;
    document
      .querySelectorAll('#comparison-metric-buttons button')
      .forEach((node) => node.classList.remove('active'));
    button.classList.add('active');
    drawComparisonChart();
  }};
  comparisonMetricButtons.appendChild(button);
}});
comparisonPayload.scenarios.forEach((item) => {{
  const label = document.createElement('label');
  label.className = 'check';
  const input = document.createElement('input');
  input.type = 'checkbox';
  input.checked = Boolean(item.comparison_default);
  input.dataset.key = item.key;
  input.onchange = drawComparisonChart;
  label.appendChild(input);
  label.append(' ' + comparisonScenarioLabel(item));
  comparisonControls.appendChild(label);
}});
function selectedComparisonScenarios() {{
  const selected = new Set(
    Array.from(document.querySelectorAll('#comparison-scenario-controls input:checked'))
      .map((node) => node.dataset.key)
  );
  return comparisonPayload.scenarios.filter((item) => selected.has(item.key));
}}
function drawComparisonChart() {{
  if (!window.Plotly) return;
  const metric = comparisonPayload.metrics[comparisonActiveMetric];
  const traces = selectedComparisonScenarios().map((item) => ({{
    x: item.dates,
    y: item.series[comparisonActiveMetric],
    mode: 'lines',
    name: item.short + ' · ' + item.data_mode,
    hovertemplate: item.full + '<br>%{{x}}<br>%{{y:.3f}}<extra></extra>'
  }}));
  Plotly.react('compare-chart', traces, {{
    margin: {{l: 64, r: 20, t: 20, b: 48}},
    paper_bgcolor: '#fffffc',
    plot_bgcolor: '#fffffc',
    yaxis: {{title: metric.axis, zeroline: false}},
    xaxis: {{title: ''}},
    legend: {{orientation: 'h', y: -0.18}},
  }}, {{responsive: true, displaylogo: false}});
}}
drawComparisonChart();
  </script>
"""


def _comparison_chart_payload(
    compare_payload: dict[str, Any],
    row: pd.Series,
) -> dict[str, Any]:
    if not compare_payload or not compare_payload.get("scenarios"):
        return {}
    payload = json.loads(json.dumps(compare_payload))
    payload["default_metric"] = "normalized_equity"
    payload["metrics"] = _localized_chart_metrics_clean(payload.get("metrics", {}))
    research_key = _policy_key(row.get("replay_primary_scenario_id"))
    actual_key = _policy_key(row.get("actual_primary_scenario_id"))
    actual_weights = _display_weights_from_json_field(row, "actual_primary_weights_json")
    base_label = "0050 1x DCA" if "0050" in actual_weights else "Base 1x DCA"
    any_default = False
    for item in payload.get("scenarios", []):
        key = _policy_key(item.get("key"))
        data_mode = str(item.get("data_mode", ""))
        is_research = key == research_key and data_mode != DATA_MODE_ACTUAL
        is_actual = key == actual_key and data_mode == DATA_MODE_ACTUAL
        is_base = key.endswith("constant_1p0")
        is_fixed_1p5 = key.endswith("constant_1p5")
        is_2x = key.endswith("constant_2p0")
        if is_base:
            item["short"] = base_label
            item["full"] = f"{base_label} ({item.get('full', item.get('key', ''))})"
        if is_fixed_1p5:
            item["short"] = "Fixed 1.5x DCA"
            item["full"] = f"Fixed 1.5x DCA ({item.get('full', item.get('key', ''))})"
        if is_2x:
            item["short"] = "00631L 2x DCA"
            item["full"] = f"00631L 2x DCA ({item.get('full', item.get('key', ''))})"
        item["comparison_default"] = bool(
            is_research or is_actual or is_base or is_fixed_1p5 or is_2x
        )
        any_default = any_default or item["comparison_default"]
    if not any_default:
        for item in payload.get("scenarios", []):
            item["comparison_default"] = bool(item.get("default"))
    return payload


def _localized_chart_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    labels = {
        "total_equity": ("淨資產", "Equity", "money"),
        "normalized_equity": ("標準化淨值", "Index", "money"),
        "drawdown": ("回撤", "Drawdown", "percent"),
        "effective_leverage": ("有效槓桿", "Leverage", "number"),
        "total_contributed": ("累積投入", "Contribution", "money"),
        "cumulative_trade_cost": ("累積交易成本", "Cost", "money"),
        "trade_cost": ("單期交易成本", "Cost", "money"),
        "turnover": ("換倉強度", "Turnover", "number"),
    }
    result: dict[str, Any] = {}
    ordered = [
        "total_equity",
        "normalized_equity",
        "drawdown",
        "effective_leverage",
        "total_contributed",
        "cumulative_trade_cost",
        "trade_cost",
        "turnover",
    ]
    for key in ordered:
        if key not in metrics:
            continue
        label, axis, fmt = labels[key]
        result[key] = {**metrics.get(key, {}), "label": label, "axis": axis, "format": fmt}
    for key, config in metrics.items():
        if key not in result:
            result[key] = config
    return result


def _payload_mode_summary(compare_payload: dict[str, Any], data_mode: str) -> dict[str, Any]:
    scenarios = compare_payload.get("scenarios", []) if compare_payload else []
    if data_mode == "non_actual":
        selected = [item for item in scenarios if item.get("data_mode") != DATA_MODE_ACTUAL]
    else:
        selected = [item for item in scenarios if item.get("data_mode") == data_mode]
    starts: list[str] = []
    ends: list[str] = []
    contributed: list[float] = []
    for item in selected:
        dates = item.get("dates") or []
        if dates:
            starts.append(str(dates[0]))
            ends.append(str(dates[-1]))
        series = item.get("series", {}).get("total_contributed") or []
        values = [value for value in series if value is not None]
        if values:
            contributed.append(float(values[-1]))
    return {
        "start": min(starts) if starts else "",
        "end": max(ends) if ends else "",
        "total_contributed": max(contributed) if contributed else np.nan,
    }


def _format_mode_summary(summary: dict[str, Any]) -> str:
    start = summary.get("start") or "unknown"
    end = summary.get("end") or "unknown"
    contributed = _format_number(summary.get("total_contributed"))
    if contributed:
        return f"{escape(str(start))} ~ {escape(str(end))}；總投入 {contributed}"
    return f"{escape(str(start))} ~ {escape(str(end))}"


def _family_label(row: pd.Series, source_coverage: pd.DataFrame) -> str:
    if _is_tw50(row, source_coverage):
        return "0050 / 00631L / CASH"
    weights = _weights_from_json_field(row, "actual_primary_weights_json")
    if weights:
        return " / ".join(weights)
    return "QQQ / QLD / TQQQ / CASH"


def _is_tw50(row: pd.Series, source_coverage: pd.DataFrame) -> bool:
    if not source_coverage.empty and "ticker" in source_coverage.columns:
        tickers = {str(value).upper() for value in source_coverage["ticker"].dropna()}
        if {"0050", "00631L"} & tickers:
            return True
    weights = set(_weights_from_json_field(row, "actual_primary_weights_json"))
    return bool({"0050", "00631L"} & weights)


def _weights_from_json_field(row: pd.Series, field: str) -> dict[str, float]:
    raw = row.get(field, "")
    if not raw or pd.isna(raw):
        return {}
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): _safe_float(value, default=0.0) for key, value in parsed.items()}


def _display_weights_from_json_field(row: pd.Series, field: str) -> dict[str, float]:
    weights = _weights_from_json_field(row, field)
    if {"0050", "00631L"} & set(weights):
        return {ticker: weights.get(ticker, 0.0) for ticker in ["0050", "00631L", "CASH"]}
    if {"QQQ", "QLD", "TQQQ"} & set(weights):
        return {ticker: weights.get(ticker, 0.0) for ticker in ["QQQ", "QLD", "TQQQ", "CASH"]}
    return weights


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
        ("win_rate_vs_benchmark", "Win vs Official"),
        ("win_rate_vs_fixed_1p5x_dca", "Win vs 1.5x"),
        ("win_rate_vs_0050_dca", "Win vs 1x"),
        ("expected_xirr", "Expected XIRR"),
        ("median_xirr", "Median XIRR"),
        ("p05_xirr", "P05 XIRR"),
        ("expected_max_drawdown", "Expected DD"),
        ("p05_max_drawdown", "P05 DD"),
        ("drawdown_breach_rate", "Breach Rate"),
        ("cost_mode", "Cost Mode"),
        ("total_trade_cost", "Trade Cost"),
        ("cost_drag_on_contributed", "Cost Drag"),
        ("cost_to_final_equity", "Cost / Final Equity"),
        ("turnover_sum", "Turnover"),
        ("trade_days", "Trade Days"),
        ("max_single_day_trade_cost", "Max Daily Cost"),
        ("effective_leverage_avg", "Avg Lev"),
        ("effective_leverage_max", "Max Lev"),
        ("effective_leverage_latest", "Latest Lev"),
        ("manual_review_required", "Review"),
    ]


def _render_optuna_candidate_section(csv_path: Path) -> str:
    family = _family_from_comparison_path(csv_path)
    if family != "tw50":
        return ""
    candidates_path = csv_path.parent / "optuna_tw50_v1_best_candidates.csv"
    html_path = csv_path.parent / "optuna_tw50_v1.html"
    if not candidates_path.exists():
        return """
  <section class="panel">
    <h2>Optuna Research Candidates</h2>
    <p>
      尚未產生 <code>optuna_tw50_v1_best_candidates.csv</code>。
      這裡之後只顯示研究候選；候選策略仍必須重跑 replay、comparison 與 review record，
      不能直接升格為 actionable default。
    </p>
  </section>
"""
    candidates = pd.read_csv(candidates_path)
    link = (
        f'<p><a href="{escape(html_path.name)}">Open Optuna HTML report</a></p>'
        if html_path.exists()
        else ""
    )
    return f"""
  <section class="panel">
    <h2>Optuna Research Candidates</h2>
    <p>
      這是策略搜尋候選，不是交易建議。高報酬但高成本或高 turnover 的策略會保留在
      watchlist；top candidates 仍要回到月度 replay、comparison 與 review record。
    </p>
    <div class="table-wrap">
      {_render_table(candidates.head(12), _optuna_candidate_columns())}
    </div>
    {link}
  </section>
"""


def _render_external_signal_audit_section(csv_path: Path) -> str:
    family = _family_from_comparison_path(csv_path)
    if family != "tw50":
        return ""
    audit_path = csv_path.parent / "external_signal_audit_tw50.csv"
    html_path = csv_path.parent / "external_signal_audit_tw50.html"
    if not audit_path.exists():
        return """
  <section class="panel">
    <h2>External Regime Signals</h2>
    <p>
      尚未產生 <code>external_signal_audit_tw50.csv</code>。Optuna 若要使用
      Fear & Greed、VIX、USD/TWD 或台股籌碼特徵，請先跑 external signal fetch 與 audit。
    </p>
  </section>
"""
    audit = pd.read_csv(audit_path)
    status = _external_audit_status(audit)
    link = (
        f'<p><a href="{escape(html_path.name)}">Open external signal audit</a></p>'
        if html_path.exists()
        else ""
    )
    return f"""
  <section class="panel">
    <h2>External Regime Signals</h2>
    <p><strong>Audit status:</strong> {escape(status)}</p>
    <p>
      Formal TW50 core external searches exclude CNN Fear & Greed and use only
      VIX, USD/TWD, Taiwan margin balance, and institutional flow features. Signals
      are frozen local data, aligned to trading dates and shifted to t-1.
    </p>
    <p>
      Optuna external signals 是 frozen local data，交易日對齊後使用 t-1。
      這些指標只影響研究搜尋，不會直接改變本月 actionable default。
    </p>
    <div class="table-wrap">{_render_table(audit, _external_audit_columns())}</div>
    {link}
  </section>
"""


def _family_from_comparison_path(csv_path: Path) -> str:
    prefix = "monthly_decision_comparison_"
    stem = csv_path.stem
    return stem[len(prefix) :].lower() if stem.startswith(prefix) else ""


def _optuna_candidate_columns() -> list[tuple[str, str]]:
    return [
        ("pareto_rank", "Pareto rank"),
        ("trial_number", "Trial"),
        ("candidate_status", "Status"),
        ("expected_xirr", "Expected XIRR"),
        ("p05_xirr", "p05 XIRR"),
        ("p05_max_drawdown", "p05 Drawdown"),
        ("cost_drag_on_contributed", "Cost Drag"),
        ("turnover_sum", "Turnover"),
        ("candidate_reason", "Reason"),
    ]


def _external_audit_status(audit: pd.DataFrame) -> str:
    statuses = set(audit.get("status", pd.Series(dtype=str)).astype(str))
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"


def _external_audit_columns() -> list[tuple[str, str]]:
    return [
        ("category", "Category"),
        ("check_id", "Check"),
        ("status", "Status"),
        ("summary", "Summary"),
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
        ("proxy_start", "Proxy Start"),
        ("proxy_end", "Proxy End"),
        ("total_return_proxy_start", "TR Proxy Start"),
        ("total_return_proxy_end", "TR Proxy End"),
        ("splice_date", "Splice Date"),
        ("source_notes", "Source Notes"),
    ]


def _coverage_warning(source_coverage: pd.DataFrame) -> str:
    if source_coverage.empty or "source_notes" not in source_coverage.columns:
        return ""
    notes = " ".join(str(value) for value in source_coverage["source_notes"].dropna())
    if "price_proxy_not_total_return" not in notes:
        return ""
    return """
    <p class="note">
      TW50 warning: 1999-2002 uses ^TWII price proxy, not full total return. It is included
      only to align the long replay start; official total-return proxy begins in 2003.
    </p>
    """


def _replay_audit_file_names(csv_path: Path) -> dict[str, str]:
    family = csv_path.stem.replace("monthly_decision_comparison_", "")
    replay_prefix = f"monthly_decision_replay_{family}"
    return {
        "ranking": f"{replay_prefix}_ranking.csv",
        "mc_summary": f"{replay_prefix}_mc_summary.csv",
        "mc_trials": f"{replay_prefix}_mc_trials.csv.gz",
        "source_coverage": f"{replay_prefix}_source_coverage.csv",
        "compare_payload": f"{replay_prefix}_compare_payload.json",
    }


def _format_cell(value: Any, key: str) -> str:
    if pd.isna(value):
        return ""
    if key in {
        "win_rate_vs_qqq_dca",
        "win_rate_vs_0050_dca",
        "win_rate_vs_fixed_1p5x_dca",
        "win_rate_vs_benchmark",
        "expected_xirr",
        "median_xirr",
        "p05_xirr",
        "expected_max_drawdown",
        "p05_max_drawdown",
        "drawdown_breach_rate",
        "cost_drag_on_contributed",
        "cost_to_final_equity",
    }:
        return _format_percent(value)
    if key in {"effective_leverage_avg", "effective_leverage_max", "effective_leverage_latest"}:
        return _format_leverage(value)
    if key in {"total_trade_cost", "turnover_sum", "max_single_day_trade_cost"}:
        return _format_number(value)
    return str(value)


def _format_percent(value: Any, *, digits: int = 2) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}%}"


def _format_leverage(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2f}x"


def _format_number(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):,.2f}"


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
    "validate_monthly_decision_comparison_as_of",
    "write_monthly_decision_comparison_report",
]
