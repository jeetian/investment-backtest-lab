from pathlib import Path

import pandas as pd
import pytest

from investment_backtest_lab.ledger_reports import _preferred_basis_metrics
from investment_backtest_lab.master_dashboard import (
    build_master_dashboard_data,
    write_master_dashboard_from_reports,
)


def test_master_dashboard_combines_ledger_and_leverage_dca_in_usd():
    scenarios, payload = build_master_dashboard_data(
        ledger_metrics=ledger_metrics(),
        ledger_equity=ledger_equity(),
        ledger_cash_flows=ledger_cash_flows(),
        leverage_metrics=leverage_metrics(),
        leverage_curve=leverage_curve(),
        leverage_cash_flows=leverage_cash_flows(),
    )

    assert set(scenarios["source"]) == {"ledger", "leverage"}
    assert set(scenarios["basis"]) == {"USD"}
    assert set(scenarios["strategy"]) == {
        "ledger_dca",
        "dca_leveraged",
        "dynamic_dca_leveraged",
    }
    fixed = scenarios[scenarios["strategy"] == "dca_leveraged"].iloc[0]
    assert fixed["ending_equity_usd"] == pytest.approx(1_250)
    assert fixed["total_contributed_usd"] == pytest.approx(1_000)
    assert fixed["simple_cash_return"] == pytest.approx(0.25)
    assert fixed["final_debt"] == pytest.approx(300)

    payload_keys = {scenario["strategy"] for scenario in payload["scenarios"]}
    assert {"ledger_dca", "dca_leveraged", "dynamic_dca_leveraged"}.issubset(payload_keys)
    assert {"net_equity", "debt", "actual_leverage", "safety_buffer"}.issubset(
        payload["metrics"]
    )
    assert sum(1 for scenario in payload["scenarios"] if scenario["default"]) <= 4


def test_write_master_dashboard_outputs_html_csv_and_payload(tmp_path):
    slug = "spy_qqq"
    ledger_metrics().to_csv(tmp_path / f"ledger_{slug}_metrics.csv", index=False)
    ledger_equity().to_csv(tmp_path / f"ledger_{slug}_equity.csv", index=False)
    ledger_cash_flows().to_csv(tmp_path / f"ledger_{slug}_cash_flows.csv", index=False)
    leverage_metrics().to_csv(tmp_path / f"leverage_{slug}_metrics.csv", index=False)
    leverage_curve().to_csv(tmp_path / f"leverage_{slug}_curve.csv", index=False)
    leverage_cash_flows().to_csv(tmp_path / f"leverage_{slug}_cash_flows.csv", index=False)

    result = write_master_dashboard_from_reports(
        output_dir=tmp_path,
        slug=slug,
        config_path=Path("configs/mvp_example.yaml"),
    )

    assert result.html_path.exists()
    assert result.scenarios_path.exists()
    assert result.payload_path.exists()
    html = result.html_path.read_text(encoding="utf-8")
    assert "DCA 決策看板" in html
    assert "主比較口徑固定為 USD" in html
    assert "槓桿 ending equity 是扣除 debt 後的淨資產" in html
    assert "normalized curve 不作本金報酬排名" in html
    assert "ledger_spy_qqq.html" in html
    assert "leverage_spy_qqq.html" in html
    assert "data-compare-checkbox" in html
    assert "dca_leveraged" in html


def test_ledger_dashboard_prefers_usd_for_first_view():
    metrics = pd.DataFrame(
        [
            {"basis": "TWD", "ending_equity": 30_000},
            {"basis": "USD", "ending_equity": 1_000},
        ]
    )

    preferred = _preferred_basis_metrics(metrics)

    assert set(preferred["basis"]) == {"USD"}


def ledger_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "SPY",
                "strategy": "ledger_dca",
                "dividend_mode": "cash",
                "basis": "USD",
                "ending_equity": 1_100.0,
                "total_contributed": 1_000.0,
                "simple_cash_return": 0.10,
                "max_drawdown": -0.05,
            },
            {
                "ticker": "SPY",
                "strategy": "ledger_dca",
                "dividend_mode": "cash",
                "basis": "TWD",
                "ending_equity": 33_000.0,
                "total_contributed": 30_000.0,
                "simple_cash_return": 0.10,
                "max_drawdown": -0.05,
            },
        ]
    )


def leverage_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            leverage_metric_row("dca_leveraged", 1_250.0, 0.25, 300.0),
            leverage_metric_row("dynamic_dca_leveraged", 1_180.0, 0.18, 120.0),
        ]
    )


def leverage_metric_row(
    strategy: str,
    ending_equity: float,
    simple_return: float,
    debt: float,
) -> dict[str, object]:
    return {
        "ticker": "SPY",
        "strategy": strategy,
        "dividend_mode": "cash",
        "total_contributed_usd": 1_000.0,
        "ending_equity_usd": ending_equity,
        "simple_cash_return": simple_return,
        "max_drawdown": -0.10,
        "interest_paid": 12.0,
        "final_debt": debt,
        "max_actual_leverage": 1.3,
        "worst_safety_buffer": 0.40,
        "margin_call_count": 0,
    }


def ledger_equity() -> pd.DataFrame:
    return pd.DataFrame(
        [
            equity_row("ledger_dca", "2024-01-02", 1_000.0, 0.0, 1.0, None),
            equity_row("ledger_dca", "2024-01-03", 1_100.0, 0.0, 1.0, None),
        ]
    )


def leverage_curve() -> pd.DataFrame:
    rows = []
    for strategy, ending, debt in [
        ("dca_leveraged", 1_250.0, 300.0),
        ("dynamic_dca_leveraged", 1_180.0, 120.0),
    ]:
        rows.extend(
            [
                equity_row(strategy, "2024-01-02", 999.0, 300.0, 1.3, 0.42),
                equity_row(strategy, "2024-01-03", ending, debt, 1.2, 0.44),
            ]
        )
    return pd.DataFrame(rows)


def equity_row(
    strategy: str,
    date: str,
    total_equity: float,
    debt: float,
    leverage: float,
    safety: float | None,
) -> dict[str, object]:
    row = {
        "ticker": "SPY",
        "strategy": strategy,
        "dividend_mode": "cash",
        "date": date,
        "cash": 0.0,
        "market_value": total_equity + debt,
        "total_equity": total_equity,
    }
    if strategy != "ledger_dca":
        row.update(
            {
                "debt": debt,
                "actual_leverage": leverage,
                "safety_buffer": safety,
            }
        )
    return row


def ledger_cash_flows() -> pd.DataFrame:
    return cash_flows("ledger_dca")


def leverage_cash_flows() -> pd.DataFrame:
    return pd.concat(
        [
            cash_flows("dca_leveraged"),
            cash_flows("dynamic_dca_leveraged"),
        ],
        ignore_index=True,
    )


def cash_flows(strategy: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "SPY",
                "strategy": strategy,
                "dividend_mode": "cash",
                "date": "2024-01-02",
                "asset": "SPY",
                "amount": 1_000.0,
                "currency": "USD",
                "kind": "deposit",
                "note": "test contribution",
            }
        ]
    )
