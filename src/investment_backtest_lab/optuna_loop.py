from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from investment_backtest_lab.html_ui import render_html_head
from investment_backtest_lab.optuna_shortlist import params_signature, render_optuna_shortlist_html

BASELINE_SCENARIO_LABEL = "Vol Target 63D 25%"
OFFICIAL_FIXED_1P5X_WIN_COLUMN = "win_rate_vs_fixed_1p5x_dca"
LOOP_STATUS_PROMISING = "promising"
LOOP_STATUS_WATCHLIST = "watchlist"
LOOP_STATUS_WEAK = "weak"
LOOP_STATUS_FAILED = "failed"


@dataclass(frozen=True)
class LoopPaths:
    output_dir: Path
    study_name: str
    family: str

    @property
    def summary_csv(self) -> Path:
        return self.output_dir / f"optuna_{self.study_name}_loop_summary.csv"

    @property
    def summary_md(self) -> Path:
        return self.output_dir / f"optuna_{self.study_name}_loop_summary.md"

    @property
    def summary_html(self) -> Path:
        return self.output_dir / f"optuna_{self.study_name}_loop_summary.html"

    @property
    def triage_csv(self) -> Path:
        return self.output_dir / f"optuna_{self.study_name}_candidate_triage.csv"

    @property
    def shortlist_csv(self) -> Path:
        return self.output_dir / f"optuna_{self.study_name}_shortlist.csv"

    @property
    def final_shortlist_csv(self) -> Path:
        return self.output_dir / f"optuna_{self.study_name}_final_shortlist.csv"

    @property
    def final_shortlist_html(self) -> Path:
        return self.output_dir / f"optuna_{self.study_name}_final_shortlist.html"

    @property
    def replay_ranking_csv(self) -> Path:
        return self.output_dir / f"monthly_decision_replay_{self.family}_ranking.csv"

    @property
    def stress_replay_ranking_csv(self) -> Path:
        return self.output_dir / f"monthly_decision_replay_{self.family}_stress_2x_ranking.csv"

    @property
    def snapshot_root(self) -> Path:
        return self.output_dir / "loop_snapshots" / self.study_name

    @property
    def log_root(self) -> Path:
        return self.output_dir / "loop_logs" / self.study_name


def build_loop_summary_row(
    *,
    cycle: int,
    study_name: str,
    started_at: datetime,
    ended_at: datetime,
    triage: pd.DataFrame,
    shortlist: pd.DataFrame,
    replay_ranking: pd.DataFrame,
    stress_ranking: pd.DataFrame | None = None,
) -> dict[str, Any]:
    completed = int((triage.get("state", pd.Series(dtype=str)).astype(str) == "COMPLETE").sum())
    candidate_count = int(
        (triage.get("triage_status", pd.Series(dtype=str)).astype(str) == "candidate").sum()
    )
    rejected_count = int(
        (triage.get("triage_status", pd.Series(dtype=str)).astype(str) == "rejected").sum()
    )
    failed_count = int((triage.get("state", pd.Series(dtype=str)).astype(str) == "FAIL").sum())
    top_shortlist = shortlist.iloc[0].to_dict() if not shortlist.empty else {}
    ranking = replay_ranking.copy()
    if not ranking.empty and "replay_rank" in ranking.columns:
        ranking["_rank"] = pd.to_numeric(ranking["replay_rank"], errors="coerce")
        ranking = ranking.sort_values("_rank", na_position="last")
    optuna_mask = ranking.get("scenario_id", pd.Series(dtype=str)).astype(str).str.contains(
        "optuna_trial_",
        na=False,
    )
    optuna_ranking = ranking[optuna_mask].copy()
    replay_top = optuna_ranking.iloc[0].to_dict() if not optuna_ranking.empty else {}
    baseline = _baseline_row(ranking)
    stress_top = _stress_row_for_replay_top(stress_ranking, replay_top)
    stress_baseline = _baseline_row(
        stress_ranking if stress_ranking is not None else pd.DataFrame()
    )
    row = {
        "cycle": int(cycle),
        "study_name": study_name,
        "started_at": _iso(started_at),
        "ended_at": _iso(ended_at),
        "elapsed_minutes": (ended_at - started_at).total_seconds() / 60.0,
        "completed_trials": completed,
        "failed_trials": failed_count,
        "candidate_count": candidate_count,
        "rejected_count": rejected_count,
        "shortlist_top_trial_number": _safe_int(top_shortlist.get("trial_number")),
        "shortlist_top_label": top_shortlist.get("scenario_label", ""),
        "shortlist_top_expected_xirr": _safe_float(top_shortlist.get("expected_xirr")),
        "shortlist_top_win_rate": _safe_float(
            top_shortlist.get(
                OFFICIAL_FIXED_1P5X_WIN_COLUMN,
                top_shortlist.get("win_rate_vs_0050_dca"),
            )
        ),
        "shortlist_top_p05_xirr": _safe_float(top_shortlist.get("p05_xirr")),
        "replay_top_trial_number": _trial_number_from_replay_row(replay_top),
        "replay_top_scenario_label": replay_top.get("scenario_label", ""),
        "replay_expected_xirr": _safe_float(replay_top.get("expected_xirr")),
        "replay_win_rate": _safe_float(
            replay_top.get("win_rate_vs_benchmark", replay_top.get("win_rate_vs_qqq_dca"))
        ),
        "replay_p05_xirr": _safe_float(replay_top.get("p05_xirr")),
        "replay_p05_max_drawdown": _safe_float(replay_top.get("p05_max_drawdown")),
        "replay_drawdown_breach_rate": _safe_float(replay_top.get("drawdown_breach_rate")),
        "replay_cost_drag": _safe_float(replay_top.get("cost_drag_on_contributed")),
        "replay_turnover": _safe_float(replay_top.get("turnover_sum")),
        "baseline_expected_xirr": _safe_float(baseline.get("expected_xirr")),
        "baseline_win_rate": _safe_float(
            baseline.get("win_rate_vs_benchmark", baseline.get("win_rate_vs_qqq_dca"))
        ),
        "baseline_p05_xirr": _safe_float(baseline.get("p05_xirr")),
        "baseline_drawdown_breach_rate": _safe_float(baseline.get("drawdown_breach_rate")),
        "baseline_cost_drag": _safe_float(baseline.get("cost_drag_on_contributed")),
    }
    row.update(_stress_summary_columns(stress_top, stress_baseline))
    status, reason = classify_loop_row(row)
    row["loop_status"] = status
    row["status_reason"] = reason
    row["beats_baseline_xirr"] = bool(row["replay_expected_xirr"] > row["baseline_expected_xirr"])
    row["beats_baseline_win_rate"] = bool(row["replay_win_rate"] > row["baseline_win_rate"])
    row["zero_breach"] = bool(row["replay_drawdown_breach_rate"] == 0.0)
    return row


def build_failed_loop_summary_row(
    *,
    cycle: int,
    study_name: str,
    started_at: datetime,
    ended_at: datetime,
    step: str,
    error: str,
) -> dict[str, Any]:
    return {
        "cycle": int(cycle),
        "study_name": study_name,
        "started_at": _iso(started_at),
        "ended_at": _iso(ended_at),
        "elapsed_minutes": (ended_at - started_at).total_seconds() / 60.0,
        "loop_status": LOOP_STATUS_FAILED,
        "status_reason": f"{step} failed: {error}",
    }


def classify_loop_row(row: dict[str, Any]) -> tuple[str, str]:
    expected = _safe_float(row.get("replay_expected_xirr"))
    win_rate = _safe_float(row.get("replay_win_rate"))
    p05_xirr = _safe_float(row.get("replay_p05_xirr"))
    breach = _safe_float(row.get("replay_drawdown_breach_rate"))
    cost_drag = _safe_float(row.get("replay_cost_drag"))
    baseline_expected = _safe_float(row.get("baseline_expected_xirr"))
    baseline_win = _safe_float(row.get("baseline_win_rate"))
    baseline_p05 = _safe_float(row.get("baseline_p05_xirr"))
    baseline_breach = _safe_float(row.get("baseline_drawdown_breach_rate"))
    baseline_cost = _safe_float(row.get("baseline_cost_drag"))
    if pd.isna(expected) or pd.isna(win_rate):
        return LOOP_STATUS_FAILED, "Replay top Optuna metrics are missing."
    xirr_win = expected > baseline_expected
    win_rate_win = win_rate > baseline_win
    p05_ok = pd.isna(baseline_p05) or p05_xirr >= baseline_p05 - 0.01
    breach_ok = pd.isna(baseline_breach) or breach <= baseline_breach
    cost_ok = pd.isna(baseline_cost) or cost_drag <= baseline_cost * 1.25
    stress_pass = row.get("stress_2x_pass")
    if xirr_win and win_rate_win and p05_ok and breach_ok and cost_ok:
        if stress_pass is False:
            return LOOP_STATUS_WATCHLIST, "Replay improved, but 2x cost stress failed."
        return LOOP_STATUS_PROMISING, "Replay improves XIRR and win rate with acceptable risk/cost."
    if xirr_win and win_rate_win and p05_ok:
        return LOOP_STATUS_WATCHLIST, "Replay return improved, but breach or cost needs review."
    if xirr_win or win_rate_win:
        return LOOP_STATUS_WATCHLIST, "Partial improvement; keep as research watchlist."
    return LOOP_STATUS_WEAK, "Replay did not clearly beat baseline."


def append_loop_summary(paths: LoopPaths, row: dict[str, Any]) -> pd.DataFrame:
    existing = pd.read_csv(paths.summary_csv) if paths.summary_csv.exists() else pd.DataFrame()
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    write_loop_summary(paths, updated)
    return updated


def write_loop_summary(paths: LoopPaths, summary: pd.DataFrame) -> None:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(paths.summary_csv, index=False)
    paths.summary_md.write_text(render_loop_summary_markdown(summary), encoding="utf-8")
    paths.summary_html.write_text(render_loop_summary_html(summary), encoding="utf-8")


def render_loop_summary_markdown(summary: pd.DataFrame) -> str:
    lines = [
        "# Optuna Replay Loop Summary",
        "",
        "| Cycle | Status | Trials | Replay Top | XIRR | Win Rate | p05 XIRR | "
        "Breach | Cost Drag | 2x Stress | Reason |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            "| "
            f"{getattr(row, 'cycle', '')} | {getattr(row, 'loop_status', '')} | "
            f"{_format_int(getattr(row, 'completed_trials', None))} | "
            f"{getattr(row, 'replay_top_scenario_label', '')} | "
            f"{_format_percent(getattr(row, 'replay_expected_xirr', None))} | "
            f"{_format_percent(getattr(row, 'replay_win_rate', None))} | "
            f"{_format_percent(getattr(row, 'replay_p05_xirr', None))} | "
            f"{_format_percent(getattr(row, 'replay_drawdown_breach_rate', None))} | "
            f"{_format_percent(getattr(row, 'replay_cost_drag', None))} | "
            f"{_format_stress_pass(getattr(row, 'stress_2x_pass', None))} | "
            f"{getattr(row, 'status_reason', '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_loop_summary_html(summary: pd.DataFrame) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{escape(str(getattr(row, 'cycle', '')))}</td>"
        f"<td>{escape(str(getattr(row, 'loop_status', '')))}</td>"
        f"<td>{_format_int(getattr(row, 'completed_trials', None))}</td>"
        f"<td>{escape(str(getattr(row, 'replay_top_scenario_label', '')))}</td>"
        f"<td>{_format_percent(getattr(row, 'replay_expected_xirr', None))}</td>"
        f"<td>{_format_percent(getattr(row, 'replay_win_rate', None))}</td>"
        f"<td>{_format_percent(getattr(row, 'replay_p05_xirr', None))}</td>"
        f"<td>{_format_percent(getattr(row, 'replay_drawdown_breach_rate', None))}</td>"
        f"<td>{_format_percent(getattr(row, 'replay_cost_drag', None))}</td>"
        f"<td>{_format_stress_pass(getattr(row, 'stress_2x_pass', None))}</td>"
        f"<td>{escape(str(getattr(row, 'status_reason', '')))}</td>"
        "</tr>"
        for row in summary.itertuples(index=False)
    )
    return f"""<!doctype html>
<html lang="zh-Hant">
{render_html_head(title="Optuna Replay Loop Summary")}
<body>
<main>
  <section class="hero">
    <div class="panel">
      <p class="eyebrow">TW50 V4 Weekly Core External</p>
      <h1>Optuna Replay Loop Summary</h1>
      <p class="lede">
        Each cycle searches first, then rebuilds reports, shortlists candidates,
        runs fast replay, and snapshots the artifacts.
      </p>
    </div>
  </section>
  <section class="panel">
    <h2>Cycles</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Cycle</th><th>Status</th><th>Completed Trials</th><th>Replay Top</th>
            <th>XIRR</th><th>Win Rate</th><th>p05 XIRR</th><th>MC Breach</th>
            <th>Cost Drag</th><th>2x Stress</th><th>Reason</th>
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


def copy_cycle_snapshot(paths: LoopPaths, *, cycle: int) -> Path:
    snapshot = next_snapshot_dir(paths.snapshot_root, cycle)
    snapshot.mkdir(parents=True, exist_ok=False)
    for artifact in snapshot_artifacts(paths):
        if artifact.exists():
            shutil.copy2(artifact, snapshot / artifact.name)
    return snapshot


def copy_final_snapshot(paths: LoopPaths) -> Path:
    snapshot = next_snapshot_dir(paths.snapshot_root, 0)
    snapshot.mkdir(parents=True, exist_ok=False)
    for artifact in snapshot_artifacts(paths):
        if artifact.exists():
            shutil.copy2(artifact, snapshot / artifact.name)
    for artifact in [paths.final_shortlist_csv, paths.final_shortlist_html]:
        if artifact.exists():
            shutil.copy2(artifact, snapshot / artifact.name)
    return snapshot


def next_snapshot_dir(root: Path, cycle: int) -> Path:
    base = root / ("final" if cycle == 0 else f"cycle_{cycle:04d}")
    if not base.exists():
        return base
    suffix = 2
    while True:
        candidate = root / (
            f"final_{suffix}" if cycle == 0 else f"cycle_{cycle:04d}_{suffix}"
        )
        if not candidate.exists():
            return candidate
        suffix += 1


def snapshot_artifacts(paths: LoopPaths) -> list[Path]:
    output = paths.output_dir
    study = paths.study_name
    family = paths.family
    names = [
        f"optuna_{study}.html",
        f"optuna_{study}_trials.csv",
        f"optuna_{study}_pareto.csv",
        f"optuna_{study}_best_candidates.csv",
        f"optuna_{study}_candidate_triage.csv",
        f"optuna_{study}_shortlist.csv",
        f"optuna_{study}_shortlist.html",
        f"monthly_decision_replay_{family}.html",
        f"monthly_decision_replay_{family}_ranking.csv",
        f"monthly_decision_replay_{family}_mc_summary.csv",
        f"monthly_decision_replay_{family}_compare_payload.json",
        f"monthly_decision_replay_{family}_trade_audit.csv",
        f"monthly_decision_replay_{family}_trade_audit.html",
        f"monthly_decision_replay_{family}_stress_2x.html",
        f"monthly_decision_replay_{family}_stress_2x_ranking.csv",
        f"monthly_decision_replay_{family}_stress_2x_mc_summary.csv",
        f"monthly_decision_replay_{family}_stress_2x_compare_payload.json",
        f"monthly_decision_replay_{family}_stress_2x_trade_audit.csv",
        f"monthly_decision_replay_{family}_stress_2x_trade_audit.html",
        f"monthly_decision_comparison_{family}.html",
        f"monthly_decision_comparison_{family}.csv",
        f"monthly_decision_comparison_{family}_top_candidates.csv",
        f"monthly_decision_review_{family}.md",
        f"monthly_decision_review_{family}.csv",
    ]
    return [output / name for name in names]


def write_snapshot_index(
    *,
    snapshot: Path,
    study_name: str,
    family: str,
    title: str,
    cycle: int | None = None,
) -> Path:
    prefix = family.lower()
    ranking = read_frame(snapshot / f"monthly_decision_replay_{prefix}_ranking.csv")
    stress_ranking = read_frame(
        snapshot / f"monthly_decision_replay_{prefix}_stress_2x_ranking.csv"
    )
    comparison = read_frame(snapshot / f"monthly_decision_comparison_{prefix}.csv")
    top = _first_row(ranking)
    baseline = _baseline_row(ranking)
    stress_top = _stress_row_for_replay_top(stress_ranking, top)
    stress_baseline = _baseline_row(stress_ranking)
    stress_summary = _stress_summary_columns(stress_top, stress_baseline)
    comparison_row = _first_row(comparison)
    review_reasons = comparison_row.get("review_reasons", "")
    actual_label = comparison_row.get("actual_primary_scenario_label", "")
    actual_leverage = _format_leverage(
        _safe_float(comparison_row.get("actual_primary_target_effective_leverage"))
    )
    comparison_link = _snapshot_link(
        snapshot,
        f"monthly_decision_comparison_{prefix}.html",
        "Monthly decision comparison + compare lab",
    )
    replay_link = _snapshot_link(
        snapshot, f"monthly_decision_replay_{prefix}.html", "Replay report"
    )
    review_link = _snapshot_link(
        snapshot, f"monthly_decision_review_{prefix}.md", "Review record"
    )
    optuna_link = _snapshot_link(snapshot, f"optuna_{study_name}.html", "Optuna report")
    shortlist_link = _snapshot_link(
        snapshot, f"optuna_{study_name}_shortlist.html", "Shortlist"
    )
    ranking_link = _snapshot_link(
        snapshot,
        f"monthly_decision_replay_{prefix}_ranking.csv",
        "Replay ranking CSV",
    )
    trade_audit_link = _snapshot_link(
        snapshot,
        f"monthly_decision_replay_{prefix}_trade_audit.html",
        "Trade cost audit",
    )
    stress_replay_link = _snapshot_link(
        snapshot,
        f"monthly_decision_replay_{prefix}_stress_2x.html",
        "2x cost stress replay",
    )
    stress_audit_link = _snapshot_link(
        snapshot,
        f"monthly_decision_replay_{prefix}_stress_2x_trade_audit.html",
        "2x cost stress trade audit",
    )
    html = f"""<!doctype html>
<html lang="zh-Hant">
{render_html_head(title=title)}
<body>
<main>
  <section class="hero">
    <div class="panel">
      <p class="eyebrow">{escape(study_name)}</p>
      <h1>{escape(title)}</h1>
      <p class="lede">
        Read this page first. It links the Optuna search, replay ranking, comparison chart,
        and manual review record for this snapshot.
      </p>
      <p><strong>Cycle:</strong> {escape(str(cycle if cycle is not None else "final"))}</p>
      <p><strong>Family:</strong> {escape(family.upper())}</p>
      <p><strong>Review reasons:</strong> {escape(str(review_reasons) or "None")}</p>
    </div>
    <div class="panel">
      <p class="eyebrow">Current actual reference</p>
      <h2>{escape(str(actual_label) or "No actual reference")}</h2>
      <p><strong>Effective leverage:</strong> {actual_leverage}</p>
      <p>
        Official weekly external replay uses the core-feature-complete window. Earlier
        1999 data is reserved for long-stress checks, not official external scoring.
      </p>
    </div>
  </section>
  <section class="two-col">
    <div class="panel">
      <p class="eyebrow">Research top</p>
      {_render_snapshot_candidate(top)}
    </div>
    <div class="panel">
      <p class="eyebrow">Baseline</p>
      {_render_snapshot_candidate(baseline)}
    </div>
  </section>
  <section class="panel">
    <p class="eyebrow">Cost stress</p>
    <h2>2x Transaction Cost Check</h2>
    {_render_snapshot_stress_summary(stress_summary)}
  </section>
  <section class="panel links">
    <h2>Open Reports</h2>
    {comparison_link}
    {replay_link}
    {trade_audit_link}
    {stress_replay_link}
    {stress_audit_link}
    {review_link}
    {optuna_link}
    {shortlist_link}
    {ranking_link}
  </section>
</main>
</body>
</html>
"""
    path = snapshot / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def build_final_shortlist(
    *,
    summary: pd.DataFrame,
    triage: pd.DataFrame,
    study_name: str,
    max_candidates: int,
) -> pd.DataFrame:
    if triage.empty:
        return pd.DataFrame()
    triage_by_trial = {
        int(float(row.trial_number)): row._asdict()
        for row in triage.itertuples(index=False)
        if _safe_int(getattr(row, "trial_number", None)) is not None
    }
    selected: list[dict[str, Any]] = []
    used_signatures: set[str] = set()
    if not summary.empty:
        ranked = summary.copy()
        for column in [
            "replay_expected_xirr",
            "replay_win_rate",
            "replay_p05_xirr",
            "replay_drawdown_breach_rate",
            "replay_cost_drag",
        ]:
            ranked[column] = pd.to_numeric(ranked.get(column), errors="coerce")
        ranked["_status_order"] = ranked.get("loop_status", "").map(
            {
                LOOP_STATUS_PROMISING: 0,
                LOOP_STATUS_WATCHLIST: 1,
                LOOP_STATUS_WEAK: 2,
            }
        ).fillna(99)
        ranked = ranked.sort_values(
            [
                "_status_order",
                "beats_baseline_xirr",
                "beats_baseline_win_rate",
                "replay_drawdown_breach_rate",
                "replay_expected_xirr",
                "replay_win_rate",
                "replay_p05_xirr",
                "replay_cost_drag",
            ],
            ascending=[True, False, False, True, False, False, False, True],
            na_position="last",
        )
        for row in ranked.itertuples(index=False):
            trial_number = _safe_int(getattr(row, "replay_top_trial_number", None))
            if trial_number is None or trial_number not in triage_by_trial:
                continue
            candidate = dict(triage_by_trial[trial_number])
            signature = params_signature(str(candidate.get("params_json", "")))
            if signature in used_signatures:
                continue
            candidate["shortlist_reason"] = f"loop_cycle_{int(row.cycle)}"
            selected.append(candidate)
            used_signatures.add(signature)
            if len(selected) >= int(max_candidates):
                break
    if len(selected) < int(max_candidates):
        fallback = triage[triage["triage_status"].astype(str).eq("candidate")].copy()
        win_column = (
            OFFICIAL_FIXED_1P5X_WIN_COLUMN
            if OFFICIAL_FIXED_1P5X_WIN_COLUMN in fallback.columns
            else "win_rate_vs_0050_dca"
        )
        fallback = fallback.sort_values(
            ["expected_xirr", win_column, "p05_xirr"],
            ascending=[False, False, False],
        )
        for row in fallback.itertuples(index=False):
            candidate = row._asdict()
            signature = params_signature(str(candidate.get("params_json", "")))
            if signature in used_signatures:
                continue
            candidate["shortlist_reason"] = "triage_fallback"
            selected.append(candidate)
            used_signatures.add(signature)
            if len(selected) >= int(max_candidates):
                break
    if not selected:
        return pd.DataFrame()
    result = pd.DataFrame(selected).reset_index(drop=True)
    result.insert(0, "shortlist_rank", result.index + 1)
    result["scenario_name"] = result["trial_number"].map(
        lambda value: f"optuna_trial_{int(float(value))}"
    )
    result["scenario_label"] = result.apply(_scenario_label, axis=1)
    return result


def write_final_shortlist(
    *,
    paths: LoopPaths,
    summary: pd.DataFrame,
    triage: pd.DataFrame,
    max_candidates: int,
) -> pd.DataFrame:
    final = build_final_shortlist(
        summary=summary,
        triage=triage,
        study_name=paths.study_name,
        max_candidates=max_candidates,
    )
    final.to_csv(paths.final_shortlist_csv, index=False)
    paths.final_shortlist_html.write_text(
        render_optuna_shortlist_html(shortlist=final, study_name=f"{paths.study_name} final"),
        encoding="utf-8",
    )
    return final


def run_command(command: list[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(command)}")


def command_search(args: Any) -> list[str]:
    command = _base_command("scripts/search_strategy_optuna.py", args)
    command.extend(
        [
            "--study-name",
            args.study_name,
            "--storage",
            args.storage,
            "--trials",
            str(args.trials),
            "--include-external-signals",
            "--objective-profile",
            args.objective_profile,
            "--timeout-hours",
            str(args.search_hours),
            "--progress-interval",
            str(args.progress_interval),
        ]
    )
    return command


def command_export(args: Any) -> list[str]:
    command = _base_command("scripts/search_strategy_optuna.py", args)
    command.extend(
        [
            "--study-name",
            args.study_name,
            "--storage",
            args.storage,
            "--export-only",
            "--include-external-signals",
            "--objective-profile",
            args.objective_profile,
        ]
    )
    return command


def command_shortlist(args: Any) -> list[str]:
    return [
        sys.executable,
        "scripts/select_optuna_candidates.py",
        "--study-name",
        args.study_name,
        "--output-dir",
        args.output_dir,
        "--max-candidates",
        str(args.max_candidates),
    ]


def command_replay(
    args: Any,
    *,
    shortlist_path: Path,
    full: bool = False,
    cost_multiplier: float = 1.0,
    report_suffix: str = "",
) -> list[str]:
    command = _base_command("scripts/analyze_monthly_decision_replay.py", args)
    command.extend(
        [
            "--selector",
            args.selector,
            "--full" if full else "--fast",
            "--optuna-scenarios",
            str(shortlist_path),
        ]
    )
    if float(cost_multiplier) != 1.0:
        command.extend(["--cost-stress-multiplier", str(float(cost_multiplier))])
    if report_suffix:
        command.extend(["--report-suffix", str(report_suffix)])
    return command


def command_comparison(args: Any) -> list[str]:
    return _base_command("scripts/analyze_monthly_decision_comparison.py", args)


def command_review(args: Any) -> list[str]:
    command = [
        sys.executable,
        "scripts/record_monthly_decision_review.py",
        "--family",
        args.family,
        "--output-dir",
        args.output_dir,
        "--status",
        "pending_review",
        "--reviewer",
        args.reviewer,
    ]
    return command


def _base_command(script: str, args: Any) -> list[str]:
    return [
        sys.executable,
        script,
        "--config",
        args.config,
        "--family",
        args.family,
        "--output-dir",
        args.output_dir,
    ]


def read_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def next_cycle_number(summary_path: Path) -> int:
    if not summary_path.exists():
        return 1
    summary = pd.read_csv(summary_path)
    if summary.empty or "cycle" not in summary.columns:
        return 1
    return int(pd.to_numeric(summary["cycle"], errors="coerce").max()) + 1


def _baseline_row(ranking: pd.DataFrame) -> dict[str, Any]:
    if ranking.empty or "scenario_label" not in ranking.columns:
        return {}
    mask = ranking["scenario_id"].astype(str).str.endswith("constant_1p5")
    if not mask.any():
        mask = ranking["scenario_label"].astype(str).eq(BASELINE_SCENARIO_LABEL)
    if not mask.any():
        return {}
    return ranking.loc[mask].iloc[0].to_dict()


def _stress_row_for_replay_top(
    stress_ranking: pd.DataFrame | None,
    replay_top: dict[str, Any],
) -> dict[str, Any]:
    if stress_ranking is None or stress_ranking.empty or not replay_top:
        return {}
    scenario_id = str(replay_top.get("scenario_id", ""))
    if scenario_id and "scenario_id" in stress_ranking.columns:
        mask = stress_ranking["scenario_id"].astype(str).eq(scenario_id)
        if mask.any():
            return stress_ranking.loc[mask].iloc[0].to_dict()
    trial_number = _trial_number_from_replay_row(replay_top)
    if trial_number is None or "scenario_id" not in stress_ranking.columns:
        return {}
    mask = stress_ranking["scenario_id"].astype(str).str.contains(
        f"optuna_trial_{trial_number}",
        na=False,
    )
    if mask.any():
        return stress_ranking.loc[mask].iloc[0].to_dict()
    return {}


def _stress_summary_columns(
    stress: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    if not stress:
        return {
            "stress_2x_scenario_label": "",
            "stress_2x_expected_xirr": float("nan"),
            "stress_2x_win_rate_vs_fixed_1p5x_dca": float("nan"),
            "stress_2x_win_rate_vs_0050_dca": float("nan"),
            "stress_2x_p05_xirr": float("nan"),
            "stress_2x_drawdown_breach_rate": float("nan"),
            "stress_2x_cost_drag": float("nan"),
            "stress_2x_turnover": float("nan"),
            "stress_2x_pass": None,
        }
    expected = _safe_float(stress.get("expected_xirr"))
    win_rate = _safe_float(
        stress.get(
            OFFICIAL_FIXED_1P5X_WIN_COLUMN,
            stress.get("win_rate_vs_benchmark", stress.get("win_rate_vs_qqq_dca")),
        )
    )
    p05_xirr = _safe_float(stress.get("p05_xirr"))
    breach = _safe_float(stress.get("drawdown_breach_rate"))
    baseline_expected = _safe_float(baseline.get("expected_xirr"))
    baseline_win = _safe_float(
        baseline.get(
            OFFICIAL_FIXED_1P5X_WIN_COLUMN,
            baseline.get("win_rate_vs_benchmark", baseline.get("win_rate_vs_qqq_dca")),
        )
    )
    baseline_p05 = _safe_float(baseline.get("p05_xirr"))
    baseline_breach = _safe_float(baseline.get("drawdown_breach_rate"))
    pass_check = (
        not pd.isna(expected)
        and not pd.isna(win_rate)
        and not pd.isna(baseline_expected)
        and not pd.isna(baseline_win)
        and expected > baseline_expected
        and win_rate >= baseline_win
        and (pd.isna(baseline_p05) or p05_xirr >= baseline_p05 - 0.01)
        and (pd.isna(baseline_breach) or breach <= baseline_breach)
    )
    return {
        "stress_2x_scenario_label": stress.get("scenario_label", ""),
        "stress_2x_expected_xirr": expected,
        "stress_2x_win_rate_vs_fixed_1p5x_dca": win_rate,
        "stress_2x_win_rate_vs_0050_dca": win_rate,
        "stress_2x_p05_xirr": p05_xirr,
        "stress_2x_drawdown_breach_rate": breach,
        "stress_2x_cost_drag": _safe_float(stress.get("cost_drag_on_contributed")),
        "stress_2x_turnover": _safe_float(stress.get("turnover_sum")),
        "stress_2x_pass": bool(pass_check),
    }


def _first_row(frame: pd.DataFrame) -> dict[str, Any]:
    return frame.iloc[0].to_dict() if frame is not None and not frame.empty else {}


def _render_snapshot_candidate(row: dict[str, Any]) -> str:
    if not row:
        return "<p>No candidate available.</p>"
    win_rate = _format_percent(
        row.get(OFFICIAL_FIXED_1P5X_WIN_COLUMN, row.get("win_rate_vs_benchmark"))
    )
    average_leverage = _format_leverage(row.get("effective_leverage_avg"))
    return f"""
      <h2>{escape(str(row.get("scenario_label", "")))}</h2>
      <p><strong>Expected XIRR:</strong> {_format_percent(row.get("expected_xirr"))}</p>
      <p><strong>Win rate vs fixed 1.5x DCA:</strong> {win_rate}</p>
      <p><strong>p05 XIRR:</strong> {_format_percent(row.get("p05_xirr"))}</p>
      <p><strong>MC breach:</strong> {_format_percent(row.get("drawdown_breach_rate"))}</p>
      <p><strong>Cost drag:</strong> {_format_percent(row.get("cost_drag_on_contributed"))}</p>
      <p><strong>Turnover:</strong> {_format_number(row.get("turnover_sum"))}</p>
      <p><strong>Avg effective leverage:</strong> {average_leverage}</p>
    """


def _render_snapshot_stress_summary(row: dict[str, Any]) -> str:
    if row.get("stress_2x_pass") is None:
        return "<p>No 2x cost stress replay was available for this snapshot.</p>"
    status = _format_stress_pass(row.get("stress_2x_pass"))
    win_rate = _format_percent(
        row.get("stress_2x_win_rate_vs_fixed_1p5x_dca", row.get("stress_2x_win_rate_vs_0050_dca"))
    )
    breach = _format_percent(row.get("stress_2x_drawdown_breach_rate"))
    return f"""
      <p><strong>Status:</strong> {status}</p>
      <p><strong>Scenario:</strong> {escape(str(row.get("stress_2x_scenario_label", "")))}</p>
      <p><strong>Expected XIRR:</strong> {_format_percent(row.get("stress_2x_expected_xirr"))}</p>
      <p><strong>Win rate vs fixed 1.5x DCA:</strong> {win_rate}</p>
      <p><strong>p05 XIRR:</strong> {_format_percent(row.get("stress_2x_p05_xirr"))}</p>
      <p><strong>MC breach:</strong> {breach}</p>
      <p><strong>Cost drag:</strong> {_format_percent(row.get("stress_2x_cost_drag"))}</p>
      <p><strong>Turnover:</strong> {_format_number(row.get("stress_2x_turnover"))}</p>
    """


def _snapshot_link(snapshot: Path, filename: str, label: str) -> str:
    if not (snapshot / filename).exists():
        return ""
    return f'<a href="{escape(filename)}">{escape(label)}</a>'


def _trial_number_from_replay_row(row: dict[str, Any]) -> int | None:
    text = " ".join(str(row.get(key, "")) for key in ["scenario_id", "scenario_label"])
    match = re.search(r"optuna_trial_(\d+)", text)
    if not match:
        match = re.search(r"#(\d+)", text)
    return int(match.group(1)) if match else None


def _scenario_label(row: pd.Series) -> str:
    trial_number = int(float(row["trial_number"]))
    xirr = _format_percent(row.get("expected_xirr"))
    win = _format_percent(row.get(OFFICIAL_FIXED_1P5X_WIN_COLUMN, row.get("win_rate_vs_0050_dca")))
    return f"Optuna V5 #{trial_number} ({xirr} XIRR, {win} win)"


def _safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.replace(microsecond=0).isoformat()


def _format_percent(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return ""


def _format_int(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return ""


def _format_number(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return ""


def _format_leverage(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.2f}x"
    except (TypeError, ValueError):
        return ""


def _format_stress_pass(value: Any) -> str:
    if value is None:
        return "not run"
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "nan", "none"}:
            return "not run"
        return "pass" if text in {"1", "true", "pass", "passed"} else "fail"
    try:
        if pd.isna(value):
            return "not run"
    except TypeError:
        pass
    return "pass" if bool(value) else "fail"


__all__ = [
    "LOOP_STATUS_FAILED",
    "LOOP_STATUS_PROMISING",
    "LOOP_STATUS_WATCHLIST",
    "LOOP_STATUS_WEAK",
    "LoopPaths",
    "append_loop_summary",
    "build_failed_loop_summary_row",
    "build_final_shortlist",
    "build_loop_summary_row",
    "classify_loop_row",
    "copy_cycle_snapshot",
    "copy_final_snapshot",
    "next_cycle_number",
    "next_snapshot_dir",
    "read_frame",
    "render_loop_summary_html",
    "render_loop_summary_markdown",
    "run_command",
    "snapshot_artifacts",
    "write_final_shortlist",
    "write_snapshot_index",
    "write_loop_summary",
]
