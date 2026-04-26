from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MasterDashboardResult:
    scenarios: pd.DataFrame
    payload: dict[str, Any]
    html_path: Path
    scenarios_path: Path
    payload_path: Path


def write_master_dashboard_from_reports(
    *,
    output_dir: Path,
    slug: str,
    config_path: Path,
) -> MasterDashboardResult:
    output_dir = Path(output_dir)
    ledger_metrics = _read_required_csv(output_dir / f"ledger_{slug}_metrics.csv")
    ledger_equity = _read_required_csv(output_dir / f"ledger_{slug}_equity.csv")
    ledger_cash_flows = _read_optional_csv(output_dir / f"ledger_{slug}_cash_flows.csv")
    leverage_metrics = _read_required_csv(output_dir / f"leverage_{slug}_metrics.csv")
    leverage_curve = _read_required_csv(output_dir / f"leverage_{slug}_curve.csv")
    leverage_cash_flows = _read_optional_csv(output_dir / f"leverage_{slug}_cash_flows.csv")

    scenarios, payload = build_master_dashboard_data(
        ledger_metrics=ledger_metrics,
        ledger_equity=ledger_equity,
        ledger_cash_flows=ledger_cash_flows,
        leverage_metrics=leverage_metrics,
        leverage_curve=leverage_curve,
        leverage_cash_flows=leverage_cash_flows,
    )

    html_path = output_dir / f"master_{slug}.html"
    scenarios_path = output_dir / f"master_{slug}_scenarios.csv"
    payload_path = output_dir / f"master_{slug}_compare_payload.json"

    scenarios.to_csv(scenarios_path, index=False, encoding="utf-8")
    payload_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    html_path.write_text(
        render_master_dashboard_html(
            scenarios=scenarios,
            payload=payload,
            config_path=config_path,
            ledger_report_path=output_dir / f"ledger_{slug}.html",
            leverage_report_path=output_dir / f"leverage_{slug}.html",
        ),
        encoding="utf-8",
    )
    return MasterDashboardResult(
        scenarios=scenarios,
        payload=payload,
        html_path=html_path,
        scenarios_path=scenarios_path,
        payload_path=payload_path,
    )


def build_master_dashboard_data(
    *,
    ledger_metrics: pd.DataFrame,
    ledger_equity: pd.DataFrame,
    ledger_cash_flows: pd.DataFrame,
    leverage_metrics: pd.DataFrame,
    leverage_curve: pd.DataFrame,
    leverage_cash_flows: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scenario_rows: list[dict[str, Any]] = []
    payload_scenarios: list[dict[str, Any]] = []

    ledger_usd = ledger_metrics[
        (ledger_metrics["strategy"] == "ledger_dca")
        & (ledger_metrics["basis"].astype(str).str.upper() == "USD")
    ].copy()
    for row in ledger_usd.itertuples():
        curve = _scenario_frame(
            ledger_equity,
            ticker=str(row.ticker),
            strategy=str(row.strategy),
            dividend_mode=str(row.dividend_mode),
        )
        flows = _scenario_frame(
            ledger_cash_flows,
            ticker=str(row.ticker),
            strategy=str(row.strategy),
            dividend_mode=str(row.dividend_mode),
        )
        row_data = _ledger_scenario_row(row)
        scenario_rows.append(row_data)
        payload_scenarios.append(
            _payload_scenario(
                row_data,
                curve,
                cash_flows=flows,
                equity_column="total_equity",
                debt_column=None,
                actual_leverage_column=None,
                safety_buffer_column=None,
            )
        )

    leverage_dca = leverage_metrics[
        leverage_metrics["strategy"].isin(["dca_leveraged", "dynamic_dca_leveraged"])
    ].copy()
    for row in leverage_dca.itertuples():
        curve = _scenario_frame(
            leverage_curve,
            ticker=str(row.ticker),
            strategy=str(row.strategy),
            dividend_mode=str(row.dividend_mode),
        )
        flows = _scenario_frame(
            leverage_cash_flows,
            ticker=str(row.ticker),
            strategy=str(row.strategy),
            dividend_mode=str(row.dividend_mode),
        )
        row_data = _leverage_scenario_row(row)
        scenario_rows.append(row_data)
        payload_scenarios.append(
            _payload_scenario(
                row_data,
                curve,
                cash_flows=flows,
                equity_column="total_equity",
                debt_column="debt",
                actual_leverage_column="actual_leverage",
                safety_buffer_column="safety_buffer",
            )
        )

    scenarios = pd.DataFrame(scenario_rows)
    if not scenarios.empty:
        scenarios = scenarios.sort_values(
            ["ticker", "dividend_mode", "sort_order", "strategy"],
            kind="stable",
        ).reset_index(drop=True)
    payload_scenarios = sorted(
        payload_scenarios,
        key=lambda item: (item["ticker"], item["dividend_mode"], item["sort_order"]),
    )
    _mark_default_payload_scenarios(payload_scenarios)
    return scenarios, _compare_payload(payload_scenarios)


def render_master_dashboard_html(
    *,
    scenarios: pd.DataFrame,
    payload: dict[str, Any],
    config_path: Path,
    ledger_report_path: Path,
    leverage_report_path: Path,
) -> str:
    style = _master_dashboard_css()
    payload_json = _json_for_script(payload)
    table_html = _decision_table(scenarios)
    kpi_html = _summary_kpis(scenarios)
    checkboxes = _compare_checkboxes(payload["scenarios"])
    metric_buttons = _metric_buttons(payload["metrics"], default_metric="net_equity")
    generated_at = pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M:%S %Z")
    font_url = (
        "https://fonts.googleapis.com/css2?"
        "family=Noto+Sans+JP:wght@400;500;600;700&"
        "family=Noto+Sans+TC:wght@400;500;600;700&display=swap"
    )
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Master Dashboard | DCA vs 槓桿 DCA</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="{font_url}" rel="stylesheet">
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>{style}</style>
</head>
<body>
  <main class="dashboard-shell">
    <header class="dashboard-header">
      <div>
        <p class="eyebrow">Investment Master Dashboard</p>
        <h1>DCA vs 槓桿 DCA</h1>
        <p class="header-copy">
          主比較口徑固定為 USD。這頁只做決策總覽；交易、股息、利息、負債與
          安全緩衝明細請回到 audit 報表。
        </p>
      </div>
      <div class="header-meta">
        <span>主口徑</span>
        <strong>USD</strong>
        <small>generated {escape(generated_at)}</small>
      </div>
    </header>

    <section class="section-block">
      <div class="section-heading">
        <div>
          <p class="eyebrow">Decision Board</p>
          <h2>DCA 決策看板</h2>
        </div>
        <p>同時比較一般 DCA、固定槓桿 DCA、動態槓桿 DCA。
          槓桿 ending equity 是扣除 debt 後的淨資產，不是總曝險。</p>
      </div>
      <div class="notice-card warning-notice">
        <strong>閱讀規則</strong>
        <p>不要只看期末淨資產。請同時看 simple cash return、max drawdown、
          利息、期末負債、worst safety buffer 與 margin call count。</p>
      </div>
      {kpi_html}
      {table_html}
    </section>

    <section class="section-block compare-lab" data-master-compare-lab>
      <div class="section-heading">
        <div>
          <p class="eyebrow">Compare Lab</p>
          <h2>自選疊圖比較</h2>
        </div>
        <p>自由勾選一般 DCA 與槓桿 DCA，切換比較指標。
          建議最多 6 條線，超過仍可比較但可讀性會下降。</p>
      </div>
      <div class="notice-card warning-notice">
        <strong>normalized curve 不作本金報酬排名</strong>
        <p>DCA 有外部現金流，normalized equity 只能看路徑形狀。
          實際回報請以累計投入、期末淨資產與 simple cash return 為主。</p>
      </div>
      <div class="compare-layout">
        <aside class="compare-control">
          <h3>可比較情境</h3>
          <div class="compare-checkbox-list">{checkboxes}</div>
        </aside>
        <div class="compare-main">
          <div class="compare-toolbar">{metric_buttons}</div>
          <div class="compare-selection" data-compare-selection>尚未選取情境。</div>
          <div class="compare-warning" data-compare-warning></div>
          <div class="compare-chart" data-compare-chart></div>
        </div>
      </div>
      <script type="application/json" id="master-compare-payload">{payload_json}</script>
    </section>

    <section class="section-block">
      <div class="section-heading">
        <div>
          <p class="eyebrow">Audit Links</p>
          <h2>明細報表</h2>
        </div>
        <p>Master Dashboard 只整合決策口徑；底層明細仍以原始 audit 報表為準。</p>
      </div>
      <div class="audit-links">
        <a class="audit-link" href="{escape(ledger_report_path.name)}">Ledger audit report</a>
        <a class="audit-link" href="{escape(leverage_report_path.name)}">Leverage audit report</a>
        <span class="audit-note">config: {escape(str(config_path))}</span>
      </div>
    </section>
  </main>
  {_master_compare_script()}
</body>
</html>
"""


def _ledger_scenario_row(row: Any) -> dict[str, Any]:
    return {
        "source": "ledger",
        "ticker": str(row.ticker),
        "strategy": str(row.strategy),
        "strategy_label": "一般 DCA",
        "dividend_mode": str(row.dividend_mode),
        "basis": "USD",
        "leverage_type": "none",
        "sort_order": 10,
        "total_contributed_usd": float(row.total_contributed),
        "ending_equity_usd": float(row.ending_equity),
        "simple_cash_return": float(row.simple_cash_return),
        "max_drawdown": _float_or_nan(getattr(row, "max_drawdown", np.nan)),
        "interest_paid": 0.0,
        "final_debt": 0.0,
        "max_actual_leverage": 1.0,
        "worst_safety_buffer": np.nan,
        "margin_call_count": 0,
    }


def _leverage_scenario_row(row: Any) -> dict[str, Any]:
    strategy = str(row.strategy)
    return {
        "source": "leverage",
        "ticker": str(row.ticker),
        "strategy": strategy,
        "strategy_label": _strategy_label(strategy),
        "dividend_mode": str(row.dividend_mode),
        "basis": "USD",
        "leverage_type": "dynamic" if strategy.startswith("dynamic_") else "fixed",
        "sort_order": 30 if strategy.startswith("dynamic_") else 20,
        "total_contributed_usd": float(row.total_contributed_usd),
        "ending_equity_usd": float(row.ending_equity_usd),
        "simple_cash_return": float(row.simple_cash_return),
        "max_drawdown": _float_or_nan(getattr(row, "max_drawdown", np.nan)),
        "interest_paid": _float_or_zero(getattr(row, "interest_paid", 0.0)),
        "final_debt": _float_or_zero(getattr(row, "final_debt", 0.0)),
        "max_actual_leverage": _float_or_nan(getattr(row, "max_actual_leverage", np.nan)),
        "worst_safety_buffer": _float_or_nan(getattr(row, "worst_safety_buffer", np.nan)),
        "margin_call_count": int(getattr(row, "margin_call_count", 0)),
    }


def _payload_scenario(
    row_data: dict[str, Any],
    curve: pd.DataFrame,
    *,
    cash_flows: pd.DataFrame,
    equity_column: str,
    debt_column: str | None,
    actual_leverage_column: str | None,
    safety_buffer_column: str | None,
) -> dict[str, Any]:
    curve = curve.copy()
    curve["date"] = pd.to_datetime(curve["date"])
    curve = curve.sort_values("date")
    equity = _numeric_series(curve, equity_column)
    dates = [pd.Timestamp(date).date().isoformat() for date in curve["date"]]
    contributed = _cumulative_contributed(curve["date"], cash_flows)
    key = _scenario_key(
        row_data["source"],
        row_data["ticker"],
        row_data["strategy"],
        row_data["dividend_mode"],
    )
    short = (
        f"{row_data['ticker']} {row_data['strategy_label']} "
        f"{_mode_label(row_data['dividend_mode'])}"
    )
    return {
        "key": key,
        "short": short,
        "full": (
            f"{row_data['ticker']} · {row_data['strategy_label']} "
            f"({row_data['strategy']}) · {_mode_label(row_data['dividend_mode'])}"
        ),
        "source": row_data["source"],
        "ticker": row_data["ticker"],
        "strategy": row_data["strategy"],
        "dividend_mode": row_data["dividend_mode"],
        "sort_order": row_data["sort_order"],
        "default": False,
        "dates": dates,
        "series": {
            "net_equity": _json_series(equity),
            "normalized_equity": _json_series(_normalized_series(equity)),
            "drawdown": _json_series(_drawdown_series(equity)),
            "cumulative_contributed": _json_series(contributed),
            "debt": _json_series(
                _numeric_series(curve, debt_column)
                if debt_column
                else _constant_series(curve, 0.0)
            ),
            "actual_leverage": _json_series(
                _numeric_series(curve, actual_leverage_column)
                if actual_leverage_column
                else _constant_series(curve, 1.0)
            ),
            "safety_buffer": _json_series(
                _numeric_series(curve, safety_buffer_column)
                if safety_buffer_column
                else _constant_series(curve, np.nan)
            ),
        },
    }


def _compare_payload(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "metrics": {
            "net_equity": {"label": "期末淨資產路徑", "axis": "USD", "format": "money"},
            "normalized_equity": {
                "label": "標準化 10,000",
                "axis": "Normalized USD",
                "format": "money",
            },
            "drawdown": {"label": "回撤", "axis": "Drawdown", "format": "percent"},
            "cumulative_contributed": {
                "label": "累計投入",
                "axis": "USD",
                "format": "money",
            },
            "debt": {"label": "負債", "axis": "USD", "format": "money"},
            "actual_leverage": {"label": "實際槓桿", "axis": "Leverage", "format": "number"},
            "safety_buffer": {
                "label": "安全緩衝",
                "axis": "Safety Buffer",
                "format": "percent",
            },
        },
        "scenarios": scenarios,
    }


def _mark_default_payload_scenarios(scenarios: list[dict[str, Any]]) -> None:
    defaults = {
        _scenario_key("ledger", "SPY", "ledger_dca", "cash"),
        _scenario_key("leverage", "SPY", "dca_leveraged", "cash"),
        _scenario_key("leverage", "SPY", "dynamic_dca_leveraged", "cash"),
    }
    for scenario in scenarios:
        scenario["default"] = scenario["key"] in defaults
    if not any(scenario["default"] for scenario in scenarios):
        for scenario in scenarios[:3]:
            scenario["default"] = True


def _decision_table(scenarios: pd.DataFrame) -> str:
    if scenarios.empty:
        return '<p class="empty-state">沒有可比較的 DCA 情境。</p>'
    display = scenarios.copy()
    display["scenario"] = display.apply(
        lambda row: (
            f"{row['ticker']} · {row['strategy_label']} · "
            f"{_mode_label(str(row['dividend_mode']))}"
        ),
        axis=1,
    )
    columns = [
        ("scenario", "情境"),
        ("total_contributed_usd", "累計投入 USD"),
        ("ending_equity_usd", "期末淨資產 USD"),
        ("simple_cash_return", "Simple Return"),
        ("max_drawdown", "Max Drawdown"),
        ("interest_paid", "利息"),
        ("final_debt", "期末負債"),
        ("max_actual_leverage", "最高槓桿"),
        ("worst_safety_buffer", "最低安全緩衝"),
        ("margin_call_count", "Margin Call"),
    ]
    rows: list[str] = []
    for _, row in display.iterrows():
        cells = []
        for column, _ in columns:
            value = row[column]
            if column in {"simple_cash_return", "max_drawdown", "worst_safety_buffer"}:
                text = _format_percent(value)
            elif column in {
                "total_contributed_usd",
                "ending_equity_usd",
                "interest_paid",
                "final_debt",
            }:
                text = _format_money(value)
            elif column == "max_actual_leverage":
                text = _format_leverage(value)
            else:
                text = str(value)
            cells.append(f"<td>{escape(text)}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    headers = "".join(f"<th>{escape(label)}</th>" for _, label in columns)
    return f"""<div class="table-wrap">
  <table class="decision-table">
    <thead><tr>{headers}</tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</div>"""


def _summary_kpis(scenarios: pd.DataFrame) -> str:
    if scenarios.empty:
        return ""
    best_return = scenarios.loc[scenarios["simple_cash_return"].astype(float).idxmax()]
    safest = scenarios.copy()
    safest["risk_sort"] = safest["worst_safety_buffer"].fillna(1.0)
    safest = safest.sort_values(
        ["margin_call_count", "risk_sort", "max_drawdown"],
        ascending=[True, False, False],
    ).iloc[0]
    cards = [
        ("比較口徑", "USD", "TWD 保留在 CSV，不作主 UI 比較"),
        (
            "最高 Simple Return",
            _format_percent(best_return["simple_cash_return"]),
            f"{best_return['ticker']} {best_return['strategy_label']}",
        ),
        (
            "安全優先候選",
            f"{safest['ticker']} {safest['strategy_label']}",
            f"margin call={int(safest['margin_call_count'])}",
        ),
        (
            "槓桿閱讀",
            "Net Equity",
            "已扣除 debt；不是 gross exposure",
        ),
    ]
    return "<div class=\"kpi-grid\">" + "".join(
        f"""<article class="kpi-card">
  <span>{escape(label)}</span>
  <strong>{escape(value)}</strong>
  <small>{escape(note)}</small>
</article>"""
        for label, value, note in cards
    ) + "</div>"


def _compare_checkboxes(scenarios: list[dict[str, Any]]) -> str:
    if not scenarios:
        return '<p class="empty-state">沒有可比較情境。</p>'
    labels: list[str] = []
    for scenario in scenarios:
        checked = " checked" if scenario.get("default") else ""
        labels.append(
            f"""<label class="compare-option">
  <input type="checkbox"
    data-compare-checkbox
    value="{escape(str(scenario["key"]))}"{checked}>
  <span>
    <strong>{escape(str(scenario["short"]))}</strong>
    <small>{escape(str(scenario["full"]))}</small>
  </span>
</label>"""
        )
    return "\n".join(labels)


def _metric_buttons(metrics: dict[str, dict[str, str]], *, default_metric: str) -> str:
    buttons: list[str] = []
    for metric, config in metrics.items():
        active = " is-active" if metric == default_metric else ""
        buttons.append(
            f"""<button class="metric-button{active}" type="button"
    data-compare-metric="{escape(metric)}">
  {escape(config["label"])}
</button>"""
        )
    return "\n".join(buttons)


def _master_compare_script() -> str:
    return """<script>
const payloadElement = document.getElementById("master-compare-payload");
const chart = document.querySelector("[data-compare-chart]");
const checkboxes = Array.from(document.querySelectorAll("[data-compare-checkbox]"));
const metricButtons = Array.from(document.querySelectorAll("[data-compare-metric]"));
const selectedText = document.querySelector("[data-compare-selection]");
const warning = document.querySelector("[data-compare-warning]");
const payload = payloadElement
  ? JSON.parse(payloadElement.textContent)
  : { metrics: {}, scenarios: [] };
const scenarioMap = new Map(payload.scenarios.map((scenario) => [scenario.key, scenario]));
let activeMetric = "net_equity";

function redrawCompareChart() {
  if (!window.Plotly || !chart) return;
  const selectedKeys = checkboxes
    .filter((checkbox) => checkbox.checked)
    .map((checkbox) => checkbox.value);
  const metricConfig = payload.metrics[activeMetric] || {};
  const traces = selectedKeys
    .map((key) => scenarioMap.get(key))
    .filter(Boolean)
    .map((scenario) => ({
      x: scenario.dates,
      y: scenario.series[activeMetric],
      mode: "lines",
      type: "scatter",
      name: scenario.short,
      hovertemplate: `${scenario.full}<br>%{x}<br>${metricConfig.label}: %{y}<extra></extra>`,
    }));
  const layout = {
    template: "plotly_white",
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    height: 440,
    margin: { l: 64, r: 28, t: 24, b: 54 },
    hovermode: "x unified",
    showlegend: true,
    legend: { orientation: "h", y: 1.14, x: 0, font: { size: 11 } },
    font: { family: "Noto Sans TC, Noto Sans JP, Segoe UI, sans-serif", color: "#202521" },
    xaxis: { showgrid: false, zeroline: false },
    yaxis: { title: metricConfig.axis || activeMetric, gridcolor: "#e6e8e1", zeroline: false },
  };
  if (metricConfig.format === "percent") {
    layout.yaxis.tickformat = ".0%";
  }
  Plotly.react(chart, traces, layout, { displaylogo: false, responsive: true });
  if (selectedText) {
    selectedText.textContent = selectedKeys.length
      ? `已選 ${selectedKeys.length} 個情境：${
          selectedKeys.map((key) => scenarioMap.get(key)?.short).join("、")
        }`
      : "尚未選取情境。";
  }
  if (warning) {
    warning.textContent = selectedKeys.length > 6
      ? "已超過建議最多 6 條線，可繼續比較，但建議縮小範圍。"
      : "";
  }
}

checkboxes.forEach((checkbox) => checkbox.addEventListener("change", redrawCompareChart));
metricButtons.forEach((button) => {
  button.addEventListener("click", () => {
    activeMetric = button.dataset.compareMetric;
    metricButtons.forEach((item) => item.classList.toggle("is-active", item === button));
    redrawCompareChart();
  });
});
redrawCompareChart();
</script>"""


def _master_dashboard_css() -> str:
    return """
:root {
  --bg: #f7f6f1;
  --surface: #ffffff;
  --surface-soft: #eeefe8;
  --ink: #202521;
  --muted: #66706a;
  --line: #d8ddd3;
  --indigo: #3f5f73;
  --sage: #6f8375;
  --copper: #b66f52;
  --shadow: 0 14px 36px rgba(32, 37, 33, 0.07);
}

* { box-sizing: border-box; }

html {
  background: var(--bg);
  color: var(--ink);
  font-family: "Noto Sans TC", "Noto Sans JP", "Segoe UI", "Microsoft JhengHei", sans-serif;
  letter-spacing: 0;
}

body {
  margin: 0;
  background:
    linear-gradient(180deg, rgba(63, 95, 115, 0.08), rgba(247, 246, 241, 0) 420px),
    var(--bg);
}

.dashboard-shell {
  width: min(1440px, calc(100% - 32px));
  margin: 0 auto;
  padding: 28px 0 56px;
}

.dashboard-header,
.section-block {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.88);
  box-shadow: var(--shadow);
}

.dashboard-header {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 24px;
  align-items: end;
  padding: 28px;
}

.dashboard-header h1 {
  margin: 6px 0 10px;
  font-size: clamp(2rem, 4vw, 3.2rem);
  line-height: 1.05;
}

.header-copy,
.section-heading p,
.notice-card p,
.compare-option small,
.kpi-card small,
.compare-selection {
  color: var(--muted);
  line-height: 1.6;
}

.header-meta {
  min-width: 190px;
  padding: 16px;
  border-left: 3px solid var(--sage);
  border-radius: 8px;
  background: var(--surface-soft);
}

.header-meta span,
.header-meta strong,
.header-meta small,
.eyebrow,
.kpi-card span {
  display: block;
}

.header-meta small {
  margin-top: 8px;
  color: var(--muted);
}

.eyebrow {
  margin: 0;
  color: var(--indigo);
  font-size: 0.78rem;
  font-weight: 700;
}

.section-block {
  margin-top: 18px;
  padding: 24px;
}

.section-heading {
  display: flex;
  justify-content: space-between;
  gap: 24px;
  align-items: end;
  margin-bottom: 18px;
}

.section-heading h2 {
  margin: 4px 0 0;
  font-size: 1.25rem;
}

.section-heading p {
  max-width: 680px;
  margin: 0;
}

.notice-card,
.kpi-card,
.compare-control,
.compare-main {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfbf7;
}

.notice-card {
  padding: 15px;
  margin-bottom: 14px;
}

.warning-notice {
  border-color: rgba(182, 111, 82, 0.45);
  background: #fff8f3;
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin-bottom: 14px;
}

.kpi-card {
  padding: 16px;
  background: linear-gradient(180deg, #ffffff, #fafaf6);
}

.kpi-card strong {
  display: block;
  margin-top: 8px;
  font-size: clamp(1.08rem, 2vw, 1.55rem);
}

.table-wrap {
  width: 100%;
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88rem;
}

th,
td {
  padding: 10px 11px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  white-space: nowrap;
}

th {
  color: var(--muted);
  background: #f4f5ee;
}

td:first-child {
  min-width: 260px;
  white-space: normal;
}

.compare-layout {
  display: grid;
  grid-template-columns: minmax(270px, 0.32fr) minmax(0, 1fr);
  gap: 14px;
  align-items: start;
}

.compare-control {
  max-height: 620px;
  overflow: auto;
  padding: 14px;
}

.compare-checkbox-list {
  display: grid;
  gap: 8px;
}

.compare-option {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: 9px;
  align-items: start;
  padding: 10px;
  border: 1px solid transparent;
  border-radius: 8px;
  background: var(--surface);
  cursor: pointer;
}

.compare-option:hover {
  border-color: var(--indigo);
}

.compare-option input {
  margin-top: 3px;
  accent-color: var(--indigo);
}

.compare-option strong,
.compare-option small {
  display: block;
}

.compare-main {
  min-width: 0;
  padding: 14px;
}

.compare-toolbar,
.audit-links {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 10px;
}

.metric-button,
.audit-link,
.audit-note {
  min-height: 34px;
  padding: 7px 11px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface);
  color: var(--ink);
  font: inherit;
  font-size: 0.88rem;
  text-decoration: none;
}

.metric-button {
  cursor: pointer;
}

.metric-button:hover,
.metric-button.is-active,
.audit-link:hover {
  border-color: var(--indigo);
  background: #f1f5f2;
  color: var(--indigo);
}

.compare-warning {
  min-height: 1.4em;
  margin-top: 4px;
  color: var(--copper);
  font-weight: 700;
}

.compare-chart {
  min-height: 440px;
  margin-top: 8px;
}

.empty-state {
  margin: 0;
  color: var(--muted);
}

@media (max-width: 980px) {
  .dashboard-header,
  .section-heading {
    display: block;
  }

  .header-meta,
  .section-heading p {
    margin-top: 14px;
  }

  .kpi-grid,
  .compare-layout {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 640px) {
  .dashboard-shell {
    width: min(100% - 20px, 1440px);
    padding-top: 10px;
  }

  .dashboard-header,
  .section-block {
    padding: 18px;
  }

  .kpi-grid,
  .compare-layout {
    grid-template-columns: 1fr;
  }
}
"""


def _scenario_frame(
    frame: pd.DataFrame,
    *,
    ticker: str,
    strategy: str,
    dividend_mode: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    mask = (
        (frame["ticker"].astype(str) == ticker)
        & (frame["strategy"].astype(str) == strategy)
        & (frame["dividend_mode"].astype(str) == dividend_mode)
    )
    return frame[mask].copy()


def _cumulative_contributed(dates: pd.Series, cash_flows: pd.DataFrame) -> pd.Series:
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    if cash_flows.empty:
        return pd.Series(0.0, index=index)
    flows = cash_flows.copy()
    flows["date"] = pd.to_datetime(flows["date"])
    if "kind" in flows.columns:
        flows = flows[flows["kind"].astype(str) == "deposit"]
    flow_series = flows.groupby("date")["amount"].sum().sort_index().cumsum()
    return flow_series.reindex(index, method="ffill").fillna(0.0).astype(float)


def _numeric_series(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None or frame.empty or column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _constant_series(frame: pd.DataFrame, value: float) -> pd.Series:
    return pd.Series(value, index=frame.index, dtype="float64")


def _normalized_series(values: Any) -> pd.Series:
    values = pd.Series(values).astype(float).replace([np.inf, -np.inf], np.nan)
    valid = values.dropna()
    if valid.empty or valid.iloc[0] == 0:
        return pd.Series(np.nan, index=values.index)
    return values / valid.iloc[0] * 10_000.0


def _drawdown_series(values: Any) -> pd.Series:
    values = pd.Series(values).astype(float).replace([np.inf, -np.inf], np.nan)
    if values.dropna().empty:
        return pd.Series(np.nan, index=values.index)
    return values / values.cummax() - 1.0


def _json_series(values: Any) -> list[float | None]:
    series = pd.Series(values)
    output: list[float | None] = []
    for value in series:
        if pd.isna(value) or value in (np.inf, -np.inf):
            output.append(None)
        else:
            output.append(round(float(value), 6))
    return output


def _json_for_script(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def _scenario_key(source: str, ticker: str, strategy: str, mode: str) -> str:
    raw = f"{source}__{ticker}__{strategy}__{mode}"
    return "".join(ch if ch.isalnum() else "-" for ch in raw).strip("-")


def _strategy_label(strategy: str) -> str:
    return {
        "ledger_dca": "一般 DCA",
        "dca_leveraged": "固定槓桿 DCA",
        "dynamic_dca_leveraged": "動態槓桿 DCA",
    }.get(strategy, strategy)


def _mode_label(mode: str) -> str:
    return {"cash": "現金股息", "reinvest": "股息再投入"}.get(mode, mode)


def _format_money(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):,.2f}"


def _format_percent(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def _format_leverage(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2f}x"


def _float_or_nan(value: Any) -> float:
    return np.nan if pd.isna(value) else float(value)


def _float_or_zero(value: Any) -> float:
    return 0.0 if pd.isna(value) else float(value)


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required report input: {path}. "
            "Run analyze_ledger.py and analyze_leverage.py first."
        )
    return pd.read_csv(path)


def _read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


__all__ = [
    "MasterDashboardResult",
    "build_master_dashboard_data",
    "render_master_dashboard_html",
    "write_master_dashboard_from_reports",
]
