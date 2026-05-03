from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from investment_backtest_lab.models import MonthlyDecisionPackConfig

WEIGHT_COLUMNS = ["QQQ_weight", "QLD_weight", "TQQQ_weight", "CASH_weight"]
SIGNAL_HISTORY_COLUMNS = [
    "generated_at",
    "first_run",
    "manual_review_required",
    "review_reasons",
    "weight_change_summary",
    "allocation_changed",
    "previous_as_of_date",
    "previous_scenario_id",
    "previous_scenario_label",
    "previous_regime",
    "previous_target_effective_leverage",
    "previous_QQQ_weight",
    "previous_QLD_weight",
    "previous_TQQQ_weight",
    "previous_CASH_weight",
]


@dataclass(frozen=True)
class OptimizerOutputPaths:
    metrics: Path
    policy: Path
    walk_forward: Path
    cohorts: Path
    cohort_summary: Path
    allocation_signal: Path


@dataclass(frozen=True)
class MonthlyDecisionPackResult:
    decision: pd.DataFrame
    history: pd.DataFrame
    html_path: Path
    csv_path: Path
    history_path: Path


def optimizer_output_paths(output_dir: Path, family: str) -> OptimizerOutputPaths:
    prefix = f"dca_policy_optimizer_{family.lower()}"
    return OptimizerOutputPaths(
        metrics=output_dir / f"{prefix}_metrics.csv",
        policy=output_dir / f"{prefix}_policy.csv",
        walk_forward=output_dir / f"{prefix}_walk_forward.csv",
        cohorts=output_dir / f"{prefix}_cohorts.csv",
        cohort_summary=output_dir / f"{prefix}_cohort_summary.csv",
        allocation_signal=output_dir / f"{prefix}_allocation_signal.csv",
    )


def load_optimizer_outputs(
    *,
    output_dir: Path,
    family: str,
    require_outputs: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = optimizer_output_paths(output_dir, family)
    missing = [path for path in vars(paths).values() if not path.exists()]
    if missing:
        if require_outputs:
            missing_list = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(
                "Monthly Decision Pack requires existing DCA optimizer outputs.\n"
                f"Missing files:\n{missing_list}\n"
                "Run first:\n"
                "uv run python scripts\\analyze_dca_policy_optimizer.py "
                "--config configs\\mvp_example.yaml --family qqq "
                "--scan-mode fast --cohort-validation"
            )
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    return (
        pd.read_csv(paths.metrics),
        pd.read_csv(paths.walk_forward),
        pd.read_csv(paths.cohort_summary),
        pd.read_csv(paths.allocation_signal),
    )


def build_monthly_decision_pack(
    *,
    allocation_signal: pd.DataFrame,
    metrics: pd.DataFrame,
    cohort_summary: pd.DataFrame,
    history: pd.DataFrame,
    config: MonthlyDecisionPackConfig,
    generated_at: str | None = None,
) -> pd.DataFrame:
    if allocation_signal.empty:
        raise ValueError("allocation signal CSV is empty; rerun the DCA policy optimizer.")
    current = allocation_signal.iloc[0].copy()
    scenario_id = str(current["scenario_id"])
    current_metric = _metric_row(metrics, scenario_id)
    synthetic_metric = _synthetic_counterpart(metrics, scenario_id)
    cohort_row = _metric_row(cohort_summary, scenario_id)
    previous = _previous_row(history)

    result = current.to_dict()
    result["generated_at"] = generated_at or datetime.now(UTC).replace(microsecond=0).isoformat()
    result["first_run"] = previous is None
    result.update(_previous_columns(previous))
    result["allocation_changed"] = _allocation_changed(current, previous)
    result["weight_change_summary"] = _weight_change_summary(current, previous)
    result.update(_checklist_columns(current, current_metric, synthetic_metric, cohort_row, config))
    result["review_reasons"] = "; ".join(_review_reasons(result))
    result["manual_review_required"] = bool(result["review_reasons"])
    return pd.DataFrame([result])


def write_monthly_decision_pack_report(
    *,
    output_dir: Path,
    family: str,
    config: MonthlyDecisionPackConfig,
) -> MonthlyDecisionPackResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, _walk_forward, cohort_summary, allocation_signal = load_optimizer_outputs(
        output_dir=output_dir,
        family=family,
        require_outputs=config.require_optimizer_outputs,
    )
    prefix = f"monthly_decision_pack_{family.lower()}"
    html_path = output_dir / f"{prefix}.html"
    csv_path = output_dir / f"{prefix}.csv"
    history_path = output_dir / f"{prefix}_signal_history.csv"
    history = _read_history(history_path)
    decision = build_monthly_decision_pack(
        allocation_signal=allocation_signal,
        metrics=metrics,
        cohort_summary=cohort_summary,
        history=history,
        config=config,
    )
    decision.to_csv(csv_path, index=False)
    _append_history(history_path, decision)
    updated_history = _read_history(history_path)
    html_path.write_text(
        render_monthly_decision_pack_html(
            decision=decision,
            history=updated_history,
            csv_path=csv_path,
            history_path=history_path,
        ),
        encoding="utf-8",
    )
    return MonthlyDecisionPackResult(
        decision=decision,
        history=updated_history,
        html_path=html_path,
        csv_path=csv_path,
        history_path=history_path,
    )


def render_monthly_decision_pack_html(
    *,
    decision: pd.DataFrame,
    history: pd.DataFrame,
    csv_path: Path,
    history_path: Path,
) -> str:
    row = decision.iloc[0]
    review = bool(row["manual_review_required"])
    review_label = "需要人工 review" if review else "未觸發額外 review"
    review_class = "danger" if review else "ok"
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Monthly Decision Pack</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link
    href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&family=Noto+Sans+JP:wght@400;500;700&display=swap"
    rel="stylesheet">
  <style>
    :root {{
      --paper: #f7f5ef;
      --surface: #fffffc;
      --ink: #232520;
      --muted: #677069;
      --line: #d9d6cb;
      --indigo: #526a83;
      --sage: #6d856f;
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
    main {{ max-width: 1280px; margin: 0 auto; padding: 28px; }}
    h1, h2, h3 {{ margin: 0; letter-spacing: 0; }}
    h1 {{ font-size: 30px; }}
    h2 {{ font-size: 21px; margin-bottom: 12px; }}
    h3 {{ font-size: 16px; }}
    .panel {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      box-shadow: 0 10px 22px rgba(47, 47, 43, 0.04);
      margin-bottom: 16px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 16px;
      align-items: stretch;
    }}
    .eyebrow {{
      text-transform: uppercase;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 6px;
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
    .pill {{
      display: inline-block;
      border-radius: 999px;
      padding: 5px 10px;
      margin-top: 10px;
      font-weight: 700;
      font-size: 13px;
      border: 1px solid var(--line);
    }}
    .pill.ok {{ color: var(--sage); background: #f3f8f1; }}
    .pill.danger {{ color: var(--danger); background: #fff5f2; border-color: #d9b89c; }}
    .weights {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }}
    .weight {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fbfaf6;
    }}
    .weight strong {{ font-size: 24px; }}
    .note {{
      border-left: 4px solid var(--copper);
      background: #fff8f3;
      padding: 12px 14px;
      border-radius: 6px;
      color: #47372e;
    }}
    .checklist {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .check {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfaf6;
    }}
    .check strong {{ display: block; }}
    .check.ok strong {{ color: var(--sage); }}
    .check.warn strong {{ color: var(--danger); }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; }}
    th {{ color: var(--muted); font-weight: 700; background: #fbfaf6; }}
    a {{ color: var(--indigo); font-weight: 700; text-decoration: none; margin-right: 14px; }}
    @media (max-width: 900px) {{
      main {{ padding: 16px; }}
      .hero, .kpis, .weights, .checklist {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<main>
  <section class="hero">
    <div class="panel">
      <p class="eyebrow">Monthly Decision Pack</p>
      <h1>QQQ Family 月度配置決策包</h1>
      <p>
        這頁把 DCA Policy Optimizer 的最新研究訊號整理成每月人工決策入口。
        主口徑為 USD；這是研究訊號，不是投資建議，也不會自動下單。
      </p>
      <span class="pill {review_class}">{review_label}</span>
      <p><strong>manual_review_required:</strong> {escape(str(review))}</p>
      <div class="kpis">
        <div class="kpi"><span>As of</span><strong>{escape(str(row["as_of_date"]))}</strong></div>
        <div class="kpi">
          <span>Target Leverage</span>
          <strong>{_format_leverage(row["target_effective_leverage"])}</strong>
        </div>
        <div class="kpi">
          <span>Next Rebalance</span>
          <strong>{escape(str(row["next_rebalance_date"]))}</strong>
        </div>
        <div class="kpi">
          <span>Next Monitor</span>
          <strong>{escape(str(row["next_monitor_date"]))}</strong>
        </div>
      </div>
    </div>
    <div class="panel">
      <p class="eyebrow">Why</p>
      <h2>{escape(str(row["scenario_label"]))}</h2>
      <p><strong>Regime:</strong> {escape(str(row["regime"]))}</p>
      <p><strong>Reason:</strong> {escape(str(row["reason"]))}</p>
      <p><strong>Change:</strong> {escape(str(row["weight_change_summary"]))}</p>
    </div>
  </section>

  <section class="panel">
    <h2>目前研究配置</h2>
    <div class="weights">
      {_render_weight_card("QQQ", row["QQQ_weight"])}
      {_render_weight_card("QLD", row["QLD_weight"])}
      {_render_weight_card("TQQQ", row["TQQQ_weight"])}
      {_render_weight_card("CASH", row["CASH_weight"])}
    </div>
  </section>

  <section class="panel">
    <h2>上期 vs 本期</h2>
    {_render_previous_summary(row)}
  </section>

  <section class="panel">
    <h2>Decision Checklist</h2>
    <div class="checklist">
      {_render_check("Weight sum", row["weight_sum_ok"], row["weight_sum_note"])}
      {_render_check("Actual ETF stable", row["actual_stable"], row["actual_note"])}
      {_render_check("Synthetic stress", row["synthetic_stress_ok"], row["synthetic_note"])}
      {_render_check("Walk-forward", row["walk_forward_ok"], row["walk_forward_note"])}
      {_render_check("Cohort robustness", row["cohort_ok"], row["cohort_note"])}
      {_render_check("Drawdown hard line", row["drawdown_limit_ok"], row["drawdown_note"])}
      {_render_check("Weekly review flag", not bool(row["review_now"]), row["review_now_note"])}
    </div>
    <div class="note">
      <strong>Review reasons:</strong> {escape(str(row["review_reasons"]) or "無")}
    </div>
  </section>

  <section class="panel">
    <h2>Decision Row</h2>
    <div class="table-wrap">{_render_table(decision, _decision_columns())}</div>
  </section>

  <section class="panel">
    <h2>Signal History</h2>
    <div class="table-wrap">{_render_table(history.tail(20), _history_columns())}</div>
    <p>
      <a href="{csv_path.name}">decision CSV</a>
      <a href="{history_path.name}">signal history CSV</a>
    </p>
  </section>
</main>
</body>
</html>
"""


def _metric_row(frame: pd.DataFrame, scenario_id: str) -> pd.Series | None:
    if frame.empty or "scenario_id" not in frame.columns:
        return None
    matches = frame[frame["scenario_id"].astype(str) == scenario_id]
    if matches.empty:
        return None
    return matches.iloc[0]


def _synthetic_counterpart(metrics: pd.DataFrame, scenario_id: str) -> pd.Series | None:
    if metrics.empty or "policy_key" not in metrics.columns:
        return None
    policy_key = str(scenario_id).split("--", maxsplit=1)[-1]
    matches = metrics[
        (metrics["data_mode"].astype(str) == "synthetic_stress")
        & (metrics["policy_key"].astype(str) == policy_key)
    ]
    if matches.empty:
        return None
    return matches.iloc[0]


def _previous_row(history: pd.DataFrame) -> pd.Series | None:
    if history.empty:
        return None
    return history.iloc[-1]


def _previous_columns(previous: pd.Series | None) -> dict[str, Any]:
    if previous is None:
        return {column: "" for column in SIGNAL_HISTORY_COLUMNS if column.startswith("previous_")}
    return {
        "previous_as_of_date": previous.get("as_of_date", ""),
        "previous_scenario_id": previous.get("scenario_id", ""),
        "previous_scenario_label": previous.get("scenario_label", ""),
        "previous_regime": previous.get("regime", ""),
        "previous_target_effective_leverage": previous.get("target_effective_leverage", np.nan),
        "previous_QQQ_weight": previous.get("QQQ_weight", np.nan),
        "previous_QLD_weight": previous.get("QLD_weight", np.nan),
        "previous_TQQQ_weight": previous.get("TQQQ_weight", np.nan),
        "previous_CASH_weight": previous.get("CASH_weight", np.nan),
    }


def _allocation_changed(current: pd.Series, previous: pd.Series | None) -> bool:
    if previous is None:
        return False
    if str(current.get("scenario_id", "")) != str(previous.get("scenario_id", "")):
        return True
    if str(current.get("regime", "")) != str(previous.get("regime", "")):
        return True
    leverage_delta = abs(
        _safe_float(current.get("target_effective_leverage"))
        - _safe_float(previous.get("target_effective_leverage"))
    )
    weight_delta = max(
        abs(_safe_float(current.get(column)) - _safe_float(previous.get(column)))
        for column in WEIGHT_COLUMNS
    )
    return bool(leverage_delta > 1e-6 or weight_delta > 1e-6)


def _weight_change_summary(current: pd.Series, previous: pd.Series | None) -> str:
    if previous is None:
        return "first run: no previous signal history"
    parts = []
    for ticker, column in zip(["QQQ", "QLD", "TQQQ", "CASH"], WEIGHT_COLUMNS, strict=True):
        delta = _safe_float(current.get(column)) - _safe_float(previous.get(column))
        parts.append(f"{ticker} {delta:+.1%}pt")
    leverage_delta = (
        _safe_float(current.get("target_effective_leverage"))
        - _safe_float(previous.get("target_effective_leverage"))
    )
    parts.append(f"leverage {leverage_delta:+.2f}x")
    return ", ".join(parts)


def _checklist_columns(
    current: pd.Series,
    metric: pd.Series | None,
    synthetic: pd.Series | None,
    cohort: pd.Series | None,
    config: MonthlyDecisionPackConfig,
) -> dict[str, Any]:
    actual_stable = metric is not None and str(metric.get("validation_status")) == "stable"
    synthetic_ok = (
        synthetic is not None
        and str(synthetic.get("risk_flag", "")) == "ok"
        and _safe_float(synthetic.get("max_drawdown")) >= config.high_risk_drawdown_band
    )
    walk_forward_ok = (
        metric is not None
        and str(metric.get("validation_status")) == "stable"
        and _safe_float(metric.get("walk_forward_pass_rate"), default=0.0) >= 0.75
    )
    cohort_ok = (
        cohort is not None
        and _safe_float(cohort.get("drawdown_breach_rate"), default=1.0) == 0.0
        and _safe_float(cohort.get("worst_cohort_max_drawdown"), default=-1.0)
        >= config.max_drawdown_limit
    )
    weight_sum_ok = np.isclose(_safe_float(current.get("weight_sum")), 1.0, atol=1e-6)
    actual_max_drawdown = _safe_float(metric.get("max_drawdown") if metric is not None else np.nan)
    synthetic_max_drawdown = _safe_float(
        synthetic.get("max_drawdown") if synthetic is not None else np.nan
    )
    drawdown_values = [
        value
        for value in [actual_max_drawdown, synthetic_max_drawdown]
        if not np.isnan(value)
    ]
    worst_drawdown = min(drawdown_values) if drawdown_values else np.nan
    drawdown_limit_ok = bool(
        not np.isnan(worst_drawdown) and worst_drawdown >= config.max_drawdown_limit
    )
    near_drawdown_limit = bool(
        not np.isnan(worst_drawdown) and worst_drawdown <= config.max_drawdown_limit + 0.05
    )
    return {
        "weight_sum_ok": weight_sum_ok,
        "weight_sum_note": f"weight sum={_format_percent(current.get('weight_sum'))}",
        "actual_stable": actual_stable,
        "actual_note": _actual_note(metric),
        "synthetic_stress_ok": synthetic_ok,
        "synthetic_note": _synthetic_note(synthetic),
        "walk_forward_ok": walk_forward_ok,
        "walk_forward_note": _walk_forward_note(metric),
        "cohort_ok": cohort_ok,
        "cohort_note": _cohort_note(cohort),
        "drawdown_limit_ok": drawdown_limit_ok,
        "near_drawdown_limit": near_drawdown_limit,
        "drawdown_note": f"worst checked drawdown {_format_percent(worst_drawdown)}",
        "review_now_note": (
            "review_now=true，週度監控要求人工檢查"
            if bool(current.get("review_now", False))
            else "review_now=false，維持月度調整節奏"
        ),
    }


def _review_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    checks = [
        ("actual_stable", "Actual ETF ranking is not stable"),
        ("weight_sum_ok", "Allocation weights do not sum to 100%"),
        ("synthetic_stress_ok", "Synthetic stress is high risk or missing"),
        ("walk_forward_ok", "Walk-forward validation is not strong enough"),
        ("cohort_ok", "Cohort robustness has drawdown breach or missing data"),
        ("drawdown_limit_ok", "Max drawdown breached the hard line"),
    ]
    for key, reason in checks:
        if not bool(row.get(key, False)):
            reasons.append(reason)
    if bool(row.get("near_drawdown_limit", False)):
        reasons.append("Max drawdown is close to the -95% hard line")
    if bool(row.get("review_now", False)):
        reasons.append("Weekly monitor review_now=true")
    return reasons


def _actual_note(metric: pd.Series | None) -> str:
    if metric is None:
        return "missing actual ETF metric row"
    return (
        f"validation={metric.get('validation_status')}, "
        f"rank={metric.get('rank')}, XIRR={_format_percent(metric.get('xirr'))}"
    )


def _synthetic_note(metric: pd.Series | None) -> str:
    if metric is None:
        return "missing synthetic stress counterpart"
    return (
        f"risk={metric.get('risk_flag')}, "
        f"max drawdown={_format_percent(metric.get('max_drawdown'))}"
    )


def _walk_forward_note(metric: pd.Series | None) -> str:
    if metric is None:
        return "missing walk-forward metrics"
    return (
        f"pass rate={_format_percent(metric.get('walk_forward_pass_rate'))}, "
        f"folds={_format_int(metric.get('validation_folds'))}"
    )


def _cohort_note(cohort: pd.Series | None) -> str:
    if cohort is None:
        return "missing cohort summary"
    return (
        f"breach rate={_format_percent(cohort.get('drawdown_breach_rate'))}, "
        f"worst DD={_format_percent(cohort.get('worst_cohort_max_drawdown'))}"
    )


def _read_history(history_path: Path) -> pd.DataFrame:
    if not history_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(history_path)
    except pd.errors.ParserError:
        return pd.read_csv(history_path, engine="python", on_bad_lines="skip")


def _append_history(history_path: Path, decision: pd.DataFrame) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if not history_path.exists():
        decision.to_csv(history_path, index=False)
        return
    try:
        existing = pd.read_csv(history_path)
    except pd.errors.ParserError:
        existing = _read_history(history_path)
        columns = list(existing.columns) if not existing.empty else list(decision.columns)
        repaired = pd.concat(
            [existing.reindex(columns=columns), decision.reindex(columns=columns)],
            ignore_index=True,
        )
        repaired.to_csv(history_path, index=False)
        return
    columns = list(existing.columns)
    decision.reindex(columns=columns).to_csv(
        history_path,
        mode="a",
        header=False,
        index=False,
    )


def _render_weight_card(label: str, value: Any) -> str:
    return f"""
    <div class="weight">
      <span>{escape(label)}</span><br>
      <strong>{_format_percent(value)}</strong>
    </div>
    """


def _render_previous_summary(row: pd.Series) -> str:
    if bool(row["first_run"]):
        return "<p>first run：目前沒有上一期 signal history，可從本次開始累積。</p>"
    return f"""
    <div class="kpis">
      <div class="kpi">
        <span>Previous As Of</span>
        <strong>{escape(str(row["previous_as_of_date"]))}</strong>
      </div>
      <div class="kpi">
        <span>Previous Strategy</span>
        <strong>{escape(str(row["previous_scenario_label"]))}</strong>
      </div>
      <div class="kpi">
        <span>Previous Regime</span>
        <strong>{escape(str(row["previous_regime"]))}</strong>
      </div>
      <div class="kpi">
        <span>Allocation Changed</span>
        <strong>{escape(str(row["allocation_changed"]))}</strong>
      </div>
    </div>
    <p>{escape(str(row["weight_change_summary"]))}</p>
    """


def _render_check(label: str, passed: Any, note: Any) -> str:
    ok = bool(passed)
    state = "ok" if ok else "warn"
    text = "OK" if ok else "Review"
    return f"""
    <div class="check {state}">
      <strong>{escape(label)}: {text}</strong>
      <span>{escape(str(note))}</span>
    </div>
    """


def _render_table(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    if frame.empty:
        return "<p>No data.</p>"
    available = [(key, label) for key, label in columns if key in frame.columns]
    head = "".join(f"<th>{escape(label)}</th>" for _, label in available)
    rows: list[str] = []
    for row in frame.tail(120).itertuples(index=False):
        row_dict = row._asdict()
        cells = "".join(
            f"<td>{escape(_format_cell(row_dict.get(key), key))}</td>" for key, _ in available
        )
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _decision_columns() -> list[tuple[str, str]]:
    return [
        ("as_of_date", "As Of"),
        ("scenario_label", "Strategy"),
        ("regime", "Regime"),
        ("target_effective_leverage", "Target Lev"),
        ("QQQ_weight", "QQQ"),
        ("QLD_weight", "QLD"),
        ("TQQQ_weight", "TQQQ"),
        ("CASH_weight", "CASH"),
        ("manual_review_required", "Manual Review"),
        ("review_reasons", "Review Reasons"),
    ]


def _history_columns() -> list[tuple[str, str]]:
    return [
        ("generated_at", "Generated"),
        ("as_of_date", "As Of"),
        ("scenario_label", "Strategy"),
        ("regime", "Regime"),
        ("target_effective_leverage", "Lev"),
        ("QQQ_weight", "QQQ"),
        ("QLD_weight", "QLD"),
        ("TQQQ_weight", "TQQQ"),
        ("CASH_weight", "CASH"),
        ("manual_review_required", "Review"),
        ("weight_change_summary", "Change"),
    ]


def _format_cell(value: Any, key: str) -> str:
    if pd.isna(value):
        return ""
    if key.endswith("_weight"):
        return _format_percent(value)
    if key == "target_effective_leverage":
        return _format_leverage(value)
    return str(value)


def _format_percent(value: Any) -> str:
    number = _safe_float(value)
    if np.isnan(number):
        return ""
    return f"{number:.2%}"


def _format_leverage(value: Any) -> str:
    number = _safe_float(value)
    if np.isnan(number):
        return ""
    return f"{number:.2f}x"


def _format_int(value: Any) -> str:
    number = _safe_float(value)
    if np.isnan(number):
        return ""
    return str(int(number))


def _safe_float(value: Any, *, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "MonthlyDecisionPackResult",
    "OptimizerOutputPaths",
    "build_monthly_decision_pack",
    "load_optimizer_outputs",
    "optimizer_output_paths",
    "render_monthly_decision_pack_html",
    "write_monthly_decision_pack_report",
]
