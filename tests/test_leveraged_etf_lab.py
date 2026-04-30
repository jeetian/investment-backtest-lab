from pathlib import Path

import pandas as pd
import pytest

from investment_backtest_lab.leveraged_etf_lab import (
    CASH,
    ProductSpec,
    build_leveraged_etf_lab_outputs,
    drawdown_guard_weights,
    monthly_rebalance_dates,
    rank_metrics,
    simulate_weighted_strategy,
    static_weight_grid,
    synthetic_daily_reset_prices,
    trend_guard_weights,
    write_leveraged_etf_lab_report,
)
from investment_backtest_lab.models import LeveragedETFLabConfig


def test_synthetic_daily_reset_prices_apply_daily_leverage():
    base = pd.Series(
        [100.0, 110.0, 99.0],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        name="QQQ",
    )
    products = [
        ProductSpec("QQQ", 1.0, "QQQ 1x"),
        ProductSpec("QLD", 2.0, "QLD 2x"),
        ProductSpec("TQQQ", 3.0, "TQQQ 3x"),
    ]

    synthetic = synthetic_daily_reset_prices(base, products)

    assert synthetic.loc["2024-01-02", "QLD"] == pytest.approx(100.0)
    assert synthetic.loc["2024-01-03", "QLD"] == pytest.approx(120.0)
    assert synthetic.loc["2024-01-04", "QLD"] == pytest.approx(96.0)
    assert synthetic.loc["2024-01-03", "TQQQ"] == pytest.approx(130.0)
    assert synthetic.loc["2024-01-04", "TQQQ"] == pytest.approx(91.0)


def test_trend_guard_uses_shifted_signal_to_avoid_lookahead():
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    prices = pd.DataFrame(
        {
            "QQQ": [100.0, 100.0, 120.0, 130.0],
            "TQQQ": [100.0, 100.0, 130.0, 140.0],
        },
        index=dates,
    )

    weights = trend_guard_weights(
        prices,
        base_ticker="QQQ",
        risk_on_ticker="TQQQ",
        defensive_ticker=CASH,
        window=2,
    )

    assert weights.iloc[2]["TQQQ"] == 0.0
    assert weights.iloc[2][CASH] == 1.0
    assert weights.iloc[3]["TQQQ"] == 1.0


def test_static_weight_grid_sums_to_one():
    grid = static_weight_grid(["QQQ", "QLD", "TQQQ", CASH], step=0.5)

    assert len(grid) == 10
    assert all(sum(weights.values()) == pytest.approx(1.0) for weights in grid)


def test_simulate_static_mix_records_monthly_rebalance_allocations():
    dates = pd.date_range("2024-01-02", periods=45, freq="B")
    prices = pd.DataFrame(
        {
            "QQQ": pd.Series(range(100, 145), index=dates, dtype="float64"),
            "QLD": pd.Series(range(100, 145), index=dates, dtype="float64"),
        }
    )
    weights = pd.DataFrame(
        [{"QQQ": 0.5, "QLD": 0.0, CASH: 0.5} for _ in dates],
        index=dates,
    )

    curve, allocations = simulate_weighted_strategy(
        prices=prices,
        target_weights=weights,
        product_leverages={"QQQ": 1.0, "QLD": 2.0},
        initial_cash=10_000.0,
        rebalance_dates=monthly_rebalance_dates(dates),
    )

    assert curve.iloc[0]["total_equity"] == pytest.approx(10_000.0)
    assert "initial allocation" in set(allocations["reason"])
    assert "scheduled rebalance" in set(allocations["reason"])


def test_drawdown_guard_reduces_exposure_after_prior_drawdown():
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    prices = pd.DataFrame(
        {
            "QQQ": [100.0, 90.0, 75.0, 80.0, 85.0],
            "QLD": [100.0, 80.0, 55.0, 60.0, 70.0],
            "TQQQ": [100.0, 70.0, 35.0, 45.0, 60.0],
        },
        index=dates,
    )

    weights = drawdown_guard_weights(
        prices,
        base_ticker="QQQ",
        risk_on_ticker="TQQQ",
        middle_ticker="QLD",
        severe_ticker=CASH,
        mild_guard=-0.10,
        severe_guard=-0.20,
    )

    assert weights.iloc[1]["TQQQ"] == 1.0
    assert weights.iloc[2]["QLD"] == 1.0
    assert weights.iloc[3][CASH] == 1.0


def test_ranking_penalizes_stress_failure_instead_of_only_cagr():
    metrics = pd.DataFrame(
        [
            metric_row("safe", "ok", cagr=0.12, calmar=1.5, sortino=1.0, sharpe=0.8),
            metric_row(
                "risky",
                "synthetic_stress_failed",
                cagr=0.60,
                calmar=10.0,
                sortino=4.0,
                sharpe=2.0,
            ),
        ]
    )

    ranked = rank_metrics(metrics)

    assert ranked.iloc[0]["scenario_id"] == "safe"
    assert ranked.iloc[0]["cagr"] < ranked.iloc[1]["cagr"]


def test_leveraged_etf_lab_outputs_report_html_csv_and_payload(tmp_path):
    prices = sample_prices()
    products = [
        ProductSpec("QQQ", 1.0, "QQQ 1x"),
        ProductSpec("QLD", 2.0, "QLD 2x"),
        ProductSpec("TQQQ", 3.0, "TQQQ 3x"),
    ]
    config = LeveragedETFLabConfig(
        grid_step=0.5,
        top_n=4,
        trend_windows=(2,),
        drawdown_guards=(-0.10, -0.20),
    )

    outputs = build_leveraged_etf_lab_outputs(
        actual_prices=prices,
        synthetic_prices=prices,
        products=products,
        lab_config=config,
    )
    result = write_leveraged_etf_lab_report(
        outputs=outputs,
        output_dir=tmp_path,
        family="qqq",
        config_path=Path("configs/mvp_example.yaml"),
    )

    assert result.html_path.exists()
    assert result.metrics_path.exists()
    assert result.payload_path.exists()
    assert result.curves_path.exists()
    assert result.allocations_path.exists()
    html = result.html_path.read_text(encoding="utf-8")
    assert "Leveraged ETF Product Lab" in html
    assert "Actual ETF" in html
    assert "Synthetic stress" in html
    assert "data-compare-checkbox" in html
    assert "只做 2000/2008 類壓力測試" in html
    assert {"actual_etf", "synthetic_stress"} == set(outputs.metrics["data_mode"])
    assert {"rank_score", "risk_flag", "max_recovery_days"}.issubset(outputs.metrics.columns)


def metric_row(
    scenario_id: str,
    risk_flag: str,
    *,
    cagr: float,
    calmar: float,
    sortino: float,
    sharpe: float,
) -> dict[str, object]:
    return {
        "data_mode": "synthetic_stress",
        "scenario_id": scenario_id,
        "scenario_label": scenario_id,
        "short_label": scenario_id,
        "strategy_family": "test",
        "weights_summary": "{}",
        "start_date": "2024-01-02",
        "end_date": "2024-01-03",
        "ending_equity": 1.0,
        "total_return": 0.0,
        "cagr": cagr,
        "volatility": 0.0,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": -0.1,
        "max_recovery_days": 0,
        "recovered": True,
        "risk_flag": risk_flag,
        "risk_failed": risk_flag != "ok",
        "default_selected": False,
    }


def sample_prices() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=80, freq="B")
    qqq = pd.Series(100.0, index=dates)
    qqq = qqq + pd.Series(range(len(dates)), index=dates) * 0.2
    return pd.DataFrame(
        {
            "QQQ": qqq,
            "QLD": qqq * 1.2,
            "TQQQ": qqq * 1.5,
        },
        index=dates,
    )
