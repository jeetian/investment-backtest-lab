from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from investment_backtest_lab.html_ui import render_html_head

SHORTLIST_REASON_ORDER = (
    "highest_expected_xirr",
    "highest_p05_xirr",
    "highest_holdout_xirr",
    "lowest_p05_drawdown",
    "lowest_cost_high_return",
)
OFFICIAL_FIXED_1P5X_WIN_COLUMN = "win_rate_vs_fixed_1p5x_dca"
REFERENCE_1X_WIN_COLUMN = "win_rate_vs_0050_dca"


@dataclass(frozen=True)
class OptunaShortlistResult:
    shortlist: pd.DataFrame
    csv_path: Path
    html_path: Path


def build_optuna_shortlist(
    triage: pd.DataFrame,
    *,
    max_candidates: int = 5,
) -> pd.DataFrame:
    if triage.empty:
        return _empty_shortlist()
    required = {"trial_number", "triage_status", "params_json"}
    missing = required - set(triage.columns)
    if missing:
        raise ValueError(f"Optuna triage CSV missing columns: {sorted(missing)}")

    candidates = triage[triage["triage_status"].astype(str).eq("candidate")].copy()
    if candidates.empty:
        return _empty_shortlist()
    candidates["_params_signature"] = candidates["params_json"].map(params_signature)
    win_column = _official_win_column(candidates)
    selected: list[pd.Series] = []
    used_signatures: set[str] = set()

    best_expected = _numeric(candidates, "expected_xirr").max()
    selectors = [
        (
            "highest_expected_xirr",
            candidates.sort_values(
                ["expected_xirr", win_column, "p05_xirr"],
                ascending=[False, False, False],
            ),
        ),
        (
            "highest_p05_xirr",
            candidates.sort_values(
                ["p05_xirr", "expected_xirr", win_column],
                ascending=[False, False, False],
            ),
        ),
        (
            "highest_holdout_xirr",
            candidates.sort_values(
                ["holdout_expected_xirr", "expected_xirr"],
                ascending=[False, False],
            ),
        ),
        (
            "lowest_p05_drawdown",
            candidates.sort_values(
                ["p05_max_drawdown", "expected_xirr"],
                ascending=[False, False],
            ),
        ),
        (
            "lowest_cost_high_return",
            candidates[
                (_numeric(candidates, "expected_xirr") >= float(best_expected) - 0.001)
                & (_numeric(candidates, win_column) >= 1.0)
            ].sort_values(
                ["expected_xirr", "cost_drag_on_contributed"],
                ascending=[False, True],
            ),
        ),
    ]
    for reason, frame in selectors:
        row = _first_unused(frame, used_signatures)
        if row is None:
            continue
        row = row.copy()
        row["shortlist_reason"] = reason
        selected.append(row)
        used_signatures.add(str(row["_params_signature"]))
        if len(selected) >= int(max_candidates):
            break

    if len(selected) < int(max_candidates):
        fallback = candidates.sort_values(
            ["return_first_rank", "expected_xirr", win_column],
            ascending=[True, False, False],
        )
        for _idx, row in fallback.iterrows():
            signature = str(row["_params_signature"])
            if signature in used_signatures:
                continue
            row = row.copy()
            row["shortlist_reason"] = "return_first_next_best"
            selected.append(row)
            used_signatures.add(signature)
            if len(selected) >= int(max_candidates):
                break

    if not selected:
        return _empty_shortlist()
    shortlist = pd.DataFrame(selected).drop(columns=["_params_signature"], errors="ignore")
    shortlist = shortlist.reset_index(drop=True)
    shortlist.insert(0, "shortlist_rank", shortlist.index + 1)
    shortlist["scenario_name"] = shortlist["trial_number"].map(
        lambda value: f"optuna_trial_{int(float(value))}"
    )
    shortlist["scenario_label"] = shortlist.apply(_scenario_label, axis=1)
    return shortlist


def write_optuna_shortlist(
    *,
    triage_path: Path,
    output_dir: Path,
    study_name: str,
    max_candidates: int = 5,
) -> OptunaShortlistResult:
    if not triage_path.exists():
        raise FileNotFoundError(f"Missing Optuna candidate triage CSV: {triage_path}")
    triage = pd.read_csv(triage_path)
    shortlist = build_optuna_shortlist(triage, max_candidates=max_candidates)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"optuna_{study_name}_shortlist"
    csv_path = output_dir / f"{prefix}.csv"
    html_path = output_dir / f"{prefix}.html"
    shortlist.to_csv(csv_path, index=False)
    html_path.write_text(
        render_optuna_shortlist_html(shortlist=shortlist, study_name=study_name),
        encoding="utf-8",
    )
    return OptunaShortlistResult(shortlist=shortlist, csv_path=csv_path, html_path=html_path)


def render_optuna_shortlist_html(*, shortlist: pd.DataFrame, study_name: str) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{escape(str(row.shortlist_rank))}</td>"
        f"<td>{escape(str(row.trial_number))}</td>"
        f"<td>{escape(str(row.shortlist_reason))}</td>"
        f"<td>{_format_percent(getattr(row, 'expected_xirr', None))}</td>"
        f"<td>{_format_percent(_row_value(row, _official_win_column(shortlist)))}</td>"
        f"<td>{_format_percent(getattr(row, 'p05_xirr', None))}</td>"
        f"<td>{_format_percent(getattr(row, 'p05_max_drawdown', None))}</td>"
        f"<td>{_format_percent(getattr(row, 'cost_drag_on_contributed', None))}</td>"
        f"<td>{_format_number(getattr(row, 'turnover_sum', None))}</td>"
        f"<td>{_format_percent(getattr(row, 'holdout_expected_xirr', None))}</td>"
        f"<td>{escape(str(row.scenario_label))}</td>"
        "</tr>"
        for row in shortlist.itertuples(index=False)
    )
    return f"""<!doctype html>
<html lang="zh-Hant">
{render_html_head(title=f"Optuna Shortlist {study_name}")}
<body>
<main>
  <section class="hero">
    <div class="panel">
      <p class="eyebrow">Optuna V2 Shortlist</p>
      <h1>{escape(study_name)}</h1>
      <p class="lede">
        Only triage candidates are eligible here. Rejected raw return leaders are excluded
        before replay validation.
      </p>
    </div>
  </section>
  <section class="panel">
    <h2>Replay Candidates</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Rank</th><th>Trial</th><th>Reason</th><th>Expected XIRR</th>
            <th>Win vs official baseline</th><th>p05 XIRR</th><th>p05 DD</th>
            <th>Cost Drag</th><th>Turnover</th><th>Holdout XIRR</th><th>Scenario</th>
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


def params_signature(params_json: str) -> str:
    try:
        params = json.loads(params_json)
    except json.JSONDecodeError:
        params = {"raw": params_json}
    return sha256(json.dumps(_normalize_params(params), sort_keys=True).encode()).hexdigest()


def _normalize_params(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_params(val) for key, val in sorted(value.items())}
    if isinstance(value, list):
        return [_normalize_params(item) for item in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def _first_unused(frame: pd.DataFrame, used_signatures: set[str]) -> pd.Series | None:
    for _idx, row in frame.iterrows():
        if str(row["_params_signature"]) not in used_signatures:
            return row
    return None


def _scenario_label(row: pd.Series) -> str:
    trial_number = int(float(row["trial_number"]))
    xirr = _format_percent(row.get("expected_xirr"))
    win = _format_percent(row.get(_official_win_column(pd.DataFrame([row]))))
    return f"Optuna V5 #{trial_number} ({xirr} XIRR, {win} win)"


def _official_win_column(frame: pd.DataFrame) -> str:
    if (
        OFFICIAL_FIXED_1P5X_WIN_COLUMN in frame.columns
        and pd.to_numeric(frame[OFFICIAL_FIXED_1P5X_WIN_COLUMN], errors="coerce").notna().any()
    ):
        return OFFICIAL_FIXED_1P5X_WIN_COLUMN
    return REFERENCE_1X_WIN_COLUMN


def _row_value(row: Any, column: str) -> Any:
    return getattr(row, column, None)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column), errors="coerce")


def _format_percent(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def _format_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):,.2f}"


def _empty_shortlist() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "shortlist_rank",
            "shortlist_reason",
            "trial_number",
            "scenario_name",
            "scenario_label",
            "params_json",
        ]
    )


__all__ = [
    "OptunaShortlistResult",
    "build_optuna_shortlist",
    "params_signature",
    "render_optuna_shortlist_html",
    "write_optuna_shortlist",
]
