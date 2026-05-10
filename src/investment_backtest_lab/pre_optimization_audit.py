from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from investment_backtest_lab.html_ui import render_html_head
from investment_backtest_lab.models import BacktestConfig
from investment_backtest_lab.tw_total_return import TW50_BASE_TICKER, TW50_LEVERAGED_TICKER

AUDIT_PASS = "pass"
AUDIT_WARN = "warn"
AUDIT_FAIL = "fail"
AUDIT_STATUSES = (AUDIT_PASS, AUDIT_WARN, AUDIT_FAIL)


@dataclass(frozen=True)
class PreOptimizationAuditResult:
    family: str
    overall_status: str
    checks: pd.DataFrame
    markdown_path: Path
    csv_path: Path
    html_path: Path
    dynamic_drawdown_limit: float
    base_1x_stress_max_drawdown: float


def write_pre_optimization_audit(
    *,
    config: BacktestConfig,
    config_path: Path,
    family: str,
    output_dir: Path,
) -> PreOptimizationAuditResult:
    family = family.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    checks = build_pre_optimization_audit_checks(
        config=config,
        config_path=config_path,
        family=family,
        output_dir=output_dir,
    )
    overall = overall_audit_status(checks)
    base_drawdown, dynamic_limit = dynamic_drawdown_limit_from_checks(checks)
    prefix = f"pre_optimization_audit_{family}"
    csv_path = output_dir / f"{prefix}.csv"
    markdown_path = output_dir / f"{prefix}.md"
    html_path = output_dir / f"{prefix}.html"
    checks.to_csv(csv_path, index=False)
    markdown_path.write_text(
        render_pre_optimization_audit_markdown(
            family=family,
            overall_status=overall,
            checks=checks,
            dynamic_drawdown_limit=dynamic_limit,
            base_1x_stress_max_drawdown=base_drawdown,
        ),
        encoding="utf-8",
    )
    html_path.write_text(
        render_pre_optimization_audit_html(
            family=family,
            overall_status=overall,
            checks=checks,
            dynamic_drawdown_limit=dynamic_limit,
            base_1x_stress_max_drawdown=base_drawdown,
        ),
        encoding="utf-8",
    )
    return PreOptimizationAuditResult(
        family=family,
        overall_status=overall,
        checks=checks,
        markdown_path=markdown_path,
        csv_path=csv_path,
        html_path=html_path,
        dynamic_drawdown_limit=dynamic_limit,
        base_1x_stress_max_drawdown=base_drawdown,
    )


def build_pre_optimization_audit_checks(
    *,
    config: BacktestConfig,
    config_path: Path,
    family: str,
    output_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    family = family.lower()
    required_artifacts = _required_artifacts(family)
    for check_id, relative_path, required_columns in required_artifacts:
        path = output_dir / relative_path
        if not path.exists():
            rows.append(
                _row(
                    "artifact",
                    check_id,
                    AUDIT_FAIL,
                    f"Missing required artifact: {relative_path}",
                    str(path),
                )
            )
            continue
        rows.append(
            _row(
                "artifact",
                check_id,
                AUDIT_PASS,
                f"Found {relative_path}",
                f"{path.stat().st_size:,} bytes",
            )
        )
        if path.suffix.lower() == ".csv" and required_columns:
            rows.append(_schema_check(check_id, path, required_columns))
        rows.append(_freshness_check(check_id, path, config_path))

    metrics_path = output_dir / f"dca_policy_optimizer_{family}_metrics.csv"
    rows.extend(_base_drawdown_checks(metrics_path, config))
    rows.extend(_cost_model_checks(config))
    rows.extend(_tw50_source_checks(output_dir) if family == "tw50" else [])
    rows.extend(_no_lookahead_checks())
    rows.extend(_strategy_search_config_checks(config))
    return pd.DataFrame(rows, columns=_audit_columns())


def overall_audit_status(checks: pd.DataFrame) -> str:
    statuses = set(checks["status"].astype(str)) if not checks.empty else {AUDIT_FAIL}
    if AUDIT_FAIL in statuses:
        return AUDIT_FAIL
    if AUDIT_WARN in statuses:
        return AUDIT_WARN
    return AUDIT_PASS


def dynamic_drawdown_limit_from_checks(checks: pd.DataFrame) -> tuple[float, float]:
    base = _numeric_detail(checks, "base_1x_stress_max_drawdown", default=np.nan)
    limit = _numeric_detail(checks, "dynamic_drawdown_limit", default=np.nan)
    return base, limit


def render_pre_optimization_audit_markdown(
    *,
    family: str,
    overall_status: str,
    checks: pd.DataFrame,
    dynamic_drawdown_limit: float,
    base_1x_stress_max_drawdown: float,
) -> str:
    lines = [
        f"# Pre-Optimization Audit: {family}",
        "",
        f"- Overall status: `{overall_status}`",
        f"- Base 1x stress max drawdown: `{_format_percent(base_1x_stress_max_drawdown)}`",
        f"- Dynamic hard drawdown limit: `{_format_percent(dynamic_drawdown_limit)}`",
        "",
        "## Checks",
        "",
        "| Category | Check | Status | Summary |",
        "|---|---|---:|---|",
    ]
    for row in checks.itertuples(index=False):
        lines.append(
            f"| {row.category} | `{row.check_id}` | `{row.status}` | "
            f"{str(row.summary).replace('|', '/')} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_pre_optimization_audit_html(
    *,
    family: str,
    overall_status: str,
    checks: pd.DataFrame,
    dynamic_drawdown_limit: float,
    base_1x_stress_max_drawdown: float,
) -> str:
    status_class = "ok" if overall_status == AUDIT_PASS else "danger"
    table_rows = "\n".join(
        "<tr>"
        f"<td>{escape(str(row.category))}</td>"
        f"<td>{escape(str(row.check_id))}</td>"
        f"<td><span class=\"badge {escape(_status_class(row.status))}\">"
        f"{escape(str(row.status))}</span></td>"
        f"<td>{escape(str(row.summary))}</td>"
        f"<td>{escape(str(row.details))}</td>"
        "</tr>"
        for row in checks.itertuples(index=False)
    )
    return f"""<!doctype html>
<html lang="zh-Hant">
{render_html_head(title=f"Pre-Optimization Audit {family}")}
<body>
<main>
  <section class="hero">
    <div class="panel">
      <p class="eyebrow">Pre-Optimization Audit</p>
      <h1>{escape(family.upper())} 搜尋前可信度檢查</h1>
      <p class="lede">
        這份報表確認資料、含息、成本、no-lookahead、ranking 與月度報表主鏈路
        是否足以進入 Optuna 搜尋。
      </p>
      <span class="badge {status_class}">overall {escape(overall_status)}</span>
    </div>
    <div class="panel">
      <h2>Dynamic Risk Gate</h2>
      <div class="kpis">
        <div class="kpi">
          <span>1x stress drawdown</span>
          <strong>{_format_percent(base_1x_stress_max_drawdown)}</strong>
        </div>
        <div class="kpi">
          <span>hard limit</span>
          <strong>{_format_percent(dynamic_drawdown_limit)}</strong>
        </div>
      </div>
    </div>
  </section>
  <section class="panel">
    <h2>Audit Checks</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Category</th><th>Check</th><th>Status</th><th>Summary</th><th>Details</th></tr></thead>
        <tbody>{table_rows}</tbody>
      </table>
    </div>
  </section>
</main>
</body>
</html>
"""


def _required_artifacts(family: str) -> list[tuple[str, str, tuple[str, ...]]]:
    artifacts = [
        (
            "optimizer_metrics",
            f"dca_policy_optimizer_{family}_metrics.csv",
            (
                "data_mode",
                "scenario_id",
                "scenario_label",
                "xirr",
                "max_drawdown",
                "cost_drag_on_contributed",
                "turnover_sum",
            ),
        ),
        (
            "optimizer_policy",
            f"dca_policy_optimizer_{family}_policy.csv",
            ("date", "scenario_id", "target_effective_leverage"),
        ),
        (
            "replay_ranking",
            f"monthly_decision_replay_{family}_ranking.csv",
            (
                "scenario_id",
                "expected_xirr",
                "p05_xirr",
                "p05_max_drawdown",
                "drawdown_breach_rate",
                "cohort_gate_passed",
            ),
        ),
        (
            "comparison",
            f"monthly_decision_comparison_{family}.csv",
            ("decision_authority", "manual_review_required", "recommended_as_of_date"),
        ),
    ]
    if family == "tw50":
        artifacts.extend(
            [
                (
                    "tw50_source_coverage",
                    "tw50_total_return_sources.csv",
                    ("ticker", "source_notes", "splice_date"),
                ),
                (
                    "tw50_dividend_audit",
                    "tw50_dividend_audit.csv",
                    ("ticker", "ex_dividend_date", "split_adjusted_cash_dividend"),
                ),
            ]
        )
    return artifacts


def _schema_check(check_id: str, path: Path, required_columns: tuple[str, ...]) -> dict[str, Any]:
    try:
        columns = set(pd.read_csv(path, nrows=0).columns)
    except Exception as exc:
        return _row("schema", f"{check_id}_schema", AUDIT_FAIL, "Cannot read CSV schema", exc)
    missing = sorted(set(required_columns) - columns)
    if missing:
        return _row(
            "schema",
            f"{check_id}_schema",
            AUDIT_FAIL,
            f"Missing columns: {', '.join(missing)}",
            str(path),
        )
    return _row("schema", f"{check_id}_schema", AUDIT_PASS, "Schema contains required columns", "")


def _freshness_check(check_id: str, path: Path, config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return _row("freshness", f"{check_id}_freshness", AUDIT_WARN, "Config path missing", "")
    artifact_mtime = path.stat().st_mtime
    config_mtime = config_path.stat().st_mtime
    if artifact_mtime + 1 < config_mtime:
        return _row(
            "freshness",
            f"{check_id}_freshness",
            AUDIT_WARN,
            "Artifact is older than config file",
            f"artifact={path.name}",
        )
    return _row("freshness", f"{check_id}_freshness", AUDIT_PASS, "Artifact is fresh enough", "")


def _base_drawdown_checks(metrics_path: Path, config: BacktestConfig) -> list[dict[str, Any]]:
    if not metrics_path.exists():
        return [
            _row(
                "risk_gate",
                "base_1x_stress_max_drawdown",
                AUDIT_FAIL,
                "Cannot derive base 1x stress drawdown without optimizer metrics",
                str(metrics_path),
            )
        ]
    metrics = pd.read_csv(metrics_path)
    mask = (
        metrics["data_mode"].astype(str).eq("synthetic_stress")
        & metrics["scenario_id"].astype(str).str.contains("constant_1p0", regex=False)
    )
    if not mask.any():
        return [
            _row(
                "risk_gate",
                "base_1x_stress_max_drawdown",
                AUDIT_FAIL,
                "Missing synthetic stress Constant 1.0x baseline",
                str(metrics_path),
            )
        ]
    base_drawdown = float(metrics.loc[mask, "max_drawdown"].astype(float).iloc[0])
    multiplier = float(config.strategy_search.drawdown_limit_multiplier)
    dynamic_limit = -min(abs(base_drawdown) * multiplier, 0.99)
    return [
        _row(
            "risk_gate",
            "base_1x_stress_max_drawdown",
            AUDIT_PASS,
            "Derived base 1x stress max drawdown",
            base_drawdown,
        ),
        _row(
            "risk_gate",
            "dynamic_drawdown_limit",
            AUDIT_PASS,
            "Derived dynamic hard drawdown limit",
            dynamic_limit,
        ),
    ]


def _cost_model_checks(config: BacktestConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tw = config.cost_model.get("tw", {}) if isinstance(config.cost_model, dict) else {}
    required = [
        "commission_rate",
        "commission_discount",
        "min_commission",
        "etf_transaction_tax_rate",
        "slippage_bps",
    ]
    missing = [key for key in required if key not in tw]
    rows.append(
        _row(
            "cost_model",
            "tw_cost_model",
            AUDIT_FAIL if missing else AUDIT_PASS,
            (
                "TW ETF cost model is configured"
                if not missing
                else f"Missing TW cost keys: {missing}"
            ),
            tw,
        )
    )
    return rows


def _tw50_source_checks(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_path = output_dir / "tw50_total_return_sources.csv"
    dividend_path = output_dir / "tw50_dividend_audit.csv"
    if source_path.exists():
        source = pd.read_csv(source_path, dtype={"ticker": str})
        tickers = set(source.get("ticker", pd.Series(dtype=str)).astype(str))
        missing = sorted({TW50_BASE_TICKER, TW50_LEVERAGED_TICKER} - tickers)
        rows.append(
            _row(
                "tw50_sources",
                "tw50_ticker_coverage",
                AUDIT_FAIL if missing else AUDIT_PASS,
                "TW50 source coverage includes 0050 and 00631L"
                if not missing
                else f"Missing source coverage tickers: {missing}",
                "",
            )
        )
        notes = " ".join(source.get("source_notes", pd.Series(dtype=str)).astype(str))
        proxy_status = AUDIT_PASS if "price_proxy_not_total_return" in notes else AUDIT_WARN
        rows.append(
            _row(
                "tw50_sources",
                "tw50_proxy_disclosure",
                proxy_status,
                "1999-2002 price proxy disclosure is present"
                if proxy_status == AUDIT_PASS
                else "1999-2002 price proxy disclosure not found",
                "",
            )
        )
    if dividend_path.exists():
        dividends = pd.read_csv(dividend_path, dtype={"ticker": str})
        has_base = dividends.get("ticker", pd.Series(dtype=str)).astype(str).eq(TW50_BASE_TICKER)
        rows.append(
            _row(
                "tw50_sources",
                "tw50_dividend_rows",
                AUDIT_PASS if bool(has_base.any()) else AUDIT_FAIL,
                "0050 dividend audit rows exist"
                if bool(has_base.any())
                else "0050 dividend audit rows are missing",
                "",
            )
        )
    return rows


def _no_lookahead_checks() -> list[dict[str, Any]]:
    optimizer_path = Path("src/investment_backtest_lab/dca_policy_optimizer.py")
    test_path = Path("tests/test_dca_policy_optimizer.py")
    optimizer_text = optimizer_path.read_text(encoding="utf-8") if optimizer_path.exists() else ""
    test_text = test_path.read_text(encoding="utf-8") if test_path.exists() else ""
    shifted = all(
        token in optimizer_text
        for token in ["shift(1)", "decision_price = base.shift(1)"]
    )
    tested = "test_trend_policy_uses_shifted_signal_to_avoid_lookahead" in test_text
    return [
        _row(
            "no_lookahead",
            "shifted_indicator_code",
            AUDIT_PASS if shifted else AUDIT_FAIL,
            "Optimizer indicators use shifted prior data"
            if shifted
            else "Expected shifted indicator logic not found",
            str(optimizer_path),
        ),
        _row(
            "no_lookahead",
            "shifted_indicator_test",
            AUDIT_PASS if tested else AUDIT_FAIL,
            "No-lookahead regression test is present"
            if tested
            else "No-lookahead regression test not found",
            str(test_path),
        ),
    ]


def _strategy_search_config_checks(config: BacktestConfig) -> list[dict[str, Any]]:
    search = config.strategy_search
    status = (
        AUDIT_PASS
        if search.engine == "optuna" and search.objective_mode == "pareto"
        else AUDIT_WARN
    )
    return [
        _row(
            "strategy_search",
            "strategy_search_config",
            status,
            "Strategy search config is Optuna Pareto"
            if status == AUDIT_PASS
            else "Strategy search config differs from Optuna Pareto default",
            {
                "engine": search.engine,
                "objective_mode": search.objective_mode,
                "n_trials": search.n_trials,
            },
        )
    ]


def _row(
    category: str,
    check_id: str,
    status: str,
    summary: str,
    details: Any,
) -> dict[str, Any]:
    if status not in AUDIT_STATUSES:
        raise ValueError(f"Unexpected audit status: {status}")
    return {
        "category": category,
        "check_id": check_id,
        "status": status,
        "summary": summary,
        "details": "" if details is None else str(details),
    }


def _audit_columns() -> list[str]:
    return ["category", "check_id", "status", "summary", "details"]


def _numeric_detail(checks: pd.DataFrame, check_id: str, *, default: float) -> float:
    if checks.empty:
        return default
    row = checks[checks["check_id"].astype(str).eq(check_id)]
    if row.empty:
        return default
    try:
        return float(row["details"].iloc[0])
    except (TypeError, ValueError):
        return default


def _status_class(status: str) -> str:
    if status == AUDIT_PASS:
        return "ok"
    return "danger"


def _format_percent(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{float(value):.2%}"


__all__ = [
    "AUDIT_FAIL",
    "AUDIT_PASS",
    "AUDIT_WARN",
    "PreOptimizationAuditResult",
    "build_pre_optimization_audit_checks",
    "dynamic_drawdown_limit_from_checks",
    "overall_audit_status",
    "write_pre_optimization_audit",
]
