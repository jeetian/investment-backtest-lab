from pathlib import Path

import pandas as pd
import pytest

from investment_backtest_lab.leveraged_etf_lab import (
    CASH,
    CASH_FLOW_DCA,
    CASH_FLOW_LUMP_SUM,
    ProductSpec,
    build_leveraged_etf_lab_outputs,
    drawdown_guard_weights,
    lab_config_for_scan_mode,
    monthly_rebalance_dates,
    rank_metrics,
    resolve_cash_flow_modes,
    resolve_scan_mode,
    simulate_weighted_strategy,
    static_weight_grid,
    synthetic_daily_reset_prices,
    trend_guard_weights,
    write_leveraged_etf_lab_report,
    xirr,
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


def test_scan_mode_config_uses_fast_by_default_and_full_for_complete_grid():
    config = LeveragedETFLabConfig(
        grid_step=0.10,
        fast_grid_step=0.50,
        full_grid_step=0.10,
        top_n=24,
        fast_top_n=4,
    )

    fast = lab_config_for_scan_mode(config, "fast")
    full = lab_config_for_scan_mode(config, "full")

    assert fast.grid_step == 0.50
    assert fast.top_n == 4
    assert full.grid_step == 0.10
    assert full.top_n == 24
    assert resolve_scan_mode() == "fast"
    assert resolve_scan_mode(full=True) == "full"
    assert resolve_cash_flow_modes("both") == [CASH_FLOW_LUMP_SUM, CASH_FLOW_DCA]
    assert resolve_cash_flow_modes("dca") == [CASH_FLOW_DCA]


def test_fast_mode_generates_fewer_scenarios_than_full_mode():
    prices = sample_prices()
    products = sample_products()
    base_config = LeveragedETFLabConfig(
        grid_step=0.25,
        fast_grid_step=0.5,
        full_grid_step=0.25,
        top_n=12,
        fast_top_n=4,
        trend_windows=(2,),
        drawdown_guards=(-0.10, -0.20),
    )

    fast = build_leveraged_etf_lab_outputs(
        actual_prices=prices,
        synthetic_prices=prices,
        products=products,
        lab_config=lab_config_for_scan_mode(base_config, "fast"),
        scan_mode="fast",
    )
    full = build_leveraged_etf_lab_outputs(
        actual_prices=prices,
        synthetic_prices=prices,
        products=products,
        lab_config=lab_config_for_scan_mode(base_config, "full"),
        scan_mode="full",
    )

    assert len(fast.metrics) < len(full.metrics)
    assert fast.scan_mode == "fast"
    assert fast.payload["scan_mode"] == "fast"
    assert full.scan_mode == "full"


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


def test_dca_contributions_follow_monthly_first_trading_day():
    dates = pd.date_range("2024-01-02", periods=45, freq="B")
    prices = pd.DataFrame({"QQQ": [100.0] * len(dates)}, index=dates)
    weights = pd.DataFrame([{ "QQQ": 1.0, CASH: 0.0 } for _ in dates], index=dates)

    curve, allocations = simulate_weighted_strategy(
        prices=prices,
        target_weights=weights,
        product_leverages={"QQQ": 1.0},
        initial_cash=10_000.0,
        contribution_dates=monthly_rebalance_dates(dates),
        contribution_amount=1_000.0,
    )

    contribution_rows = curve[curve["contribution"] > 0]
    assert list(contribution_rows["date"]) == sorted(monthly_rebalance_dates(dates))
    assert curve.iloc[-1]["total_contributed"] == pytest.approx(13_000.0)
    assert curve.iloc[-1]["total_equity"] == pytest.approx(13_000.0)
    assert set(allocations["reason"]) == {"initial allocation", "contribution rebalance"}


def test_dca_static_mix_allocates_new_cash_to_target_weights():
    dates = pd.date_range("2024-01-02", periods=45, freq="B")
    prices = pd.DataFrame(
        {
            "QQQ": [100.0] * len(dates),
            "QLD": [100.0] * len(dates),
        },
        index=dates,
    )
    weights = pd.DataFrame(
        [{"QQQ": 0.5, "QLD": 0.0, CASH: 0.5} for _ in dates],
        index=dates,
    )

    curve, _allocations = simulate_weighted_strategy(
        prices=prices,
        target_weights=weights,
        product_leverages={"QQQ": 1.0, "QLD": 2.0},
        initial_cash=10_000.0,
        contribution_dates=monthly_rebalance_dates(dates),
        contribution_amount=1_000.0,
    )

    assert curve.iloc[-1]["total_contributed"] == pytest.approx(13_000.0)
    assert curve.iloc[-1]["total_equity"] == pytest.approx(13_000.0)
    assert curve.iloc[-1]["cash_weight"] == pytest.approx(0.5)
    assert curve.iloc[-1]["QQQ_weight"] == pytest.approx(0.5)


def test_xirr_matches_simple_annual_cash_flow():
    result = xirr(
        [
            (pd.Timestamp("2023-01-01"), -1_000.0),
            (pd.Timestamp("2024-01-01"), 1_100.0),
        ]
    )

    assert result == pytest.approx(0.10, abs=0.001)


def test_dca_normalized_payload_uses_time_weighted_path_not_contribution_jumps():
    dates = pd.date_range("2024-01-02", periods=45, freq="B")
    prices = pd.DataFrame(
        {
            "QQQ": [100.0] * len(dates),
            "QLD": [100.0] * len(dates),
            "TQQQ": [100.0] * len(dates),
        },
        index=dates,
    )
    outputs = build_leveraged_etf_lab_outputs(
        actual_prices=prices,
        synthetic_prices=pd.DataFrame(),
        products=sample_products(),
        lab_config=LeveragedETFLabConfig(
            grid_step=0.5,
            top_n=4,
            trend_windows=(2,),
            drawdown_guards=(-0.10, -0.20),
        ),
        scan_mode="fast",
        cash_flow_mode=CASH_FLOW_DCA,
    )

    scenario = next(
        item
        for item in outputs.payload["scenarios"]
        if item["cash_flow_mode"] == CASH_FLOW_DCA and item["short"] == "DCA B&H QQQ"
    )

    assert scenario["series"]["total_equity"][-1] > scenario["series"]["total_equity"][0]
    assert max(scenario["series"]["normalized_equity"]) == pytest.approx(10_000.0)
    assert min(scenario["series"]["normalized_equity"]) == pytest.approx(10_000.0)


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


def test_ranking_keeps_lump_sum_and_dca_separate():
    metrics = pd.DataFrame(
        [
            metric_row("lump_safe", "ok", cagr=0.10, calmar=1.0, sortino=1.0, sharpe=1.0),
            metric_row(
                "lump_better",
                "ok",
                cagr=0.20,
                calmar=2.0,
                sortino=1.0,
                sharpe=1.0,
            ),
            metric_row(
                "dca_safe",
                "ok",
                cash_flow_mode=CASH_FLOW_DCA,
                cagr=0.05,
                calmar=0.5,
                sortino=1.0,
                sharpe=1.0,
            ),
            metric_row(
                "dca_better",
                "ok",
                cash_flow_mode=CASH_FLOW_DCA,
                cagr=0.08,
                calmar=0.8,
                sortino=1.0,
                sharpe=1.0,
            ),
        ]
    )

    ranked = rank_metrics(metrics)

    top_by_mode = ranked[ranked["rank"] == 1].set_index("cash_flow_mode")["scenario_id"]
    assert top_by_mode[CASH_FLOW_LUMP_SUM] == "lump_better"
    assert top_by_mode[CASH_FLOW_DCA] == "dca_better"


def test_leveraged_etf_lab_outputs_report_html_csv_and_payload(tmp_path):
    prices = sample_prices()
    products = sample_products()
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
        scan_mode="fast",
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
    assert "三步閱讀法" in html
    assert "Trend Guard" in html
    assert "Drawdown Guard" in html
    assert "Calmar" in html
    assert "Max Drawdown" in html
    assert "DCA Decision Board" in html
    assert "Lump Sum 一次投入" in html
    assert "Robust Ranking 穩健排名" in html
    assert "DCA 不用 CAGR 當主要判斷" in html
    assert "這不是投資建議" in html
    assert "策略候選" in html
    assert "目前掃描模式：fast" in html
    assert "data-compare-checkbox" in html
    assert "只做 2000/2008 類壓力測試" in html
    assert "瘛刻" not in html
    assert outputs.payload["metrics"]["total_equity"]["label"] == "淨資產"
    assert any(
        scenario["cash_flow_mode"] == CASH_FLOW_DCA
        for scenario in outputs.payload["scenarios"]
    )
    assert {"actual_etf", "synthetic_stress"} == set(outputs.metrics["data_mode"])
    assert {
        "cash_flow_mode",
        "rank_score",
        "risk_flag",
        "max_recovery_days",
        "total_contributed",
        "simple_cash_return",
        "xirr",
        "robust_score",
        "worst_segment_drawdown",
        "worst_segment_return",
    }.issubset(outputs.metrics.columns)
    assert {CASH_FLOW_LUMP_SUM, CASH_FLOW_DCA} == set(outputs.metrics["cash_flow_mode"])


def metric_row(
    scenario_id: str,
    risk_flag: str,
    *,
    cash_flow_mode: str = CASH_FLOW_LUMP_SUM,
    cagr: float,
    calmar: float,
    sortino: float,
    sharpe: float,
) -> dict[str, object]:
    return {
        "data_mode": "synthetic_stress",
        "cash_flow_mode": cash_flow_mode,
        "scenario_id": scenario_id,
        "scenario_label": scenario_id,
        "short_label": scenario_id,
        "strategy_family": "test",
        "weights_summary": "{}",
        "start_date": "2024-01-02",
        "end_date": "2024-01-03",
        "ending_equity": 1.0,
        "total_contributed": 1.0,
        "total_return": 0.0,
        "simple_cash_return": 0.0,
        "xirr": cagr,
        "cagr": cagr,
        "volatility": 0.0,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": -0.1,
        "max_recovery_days": 0,
        "worst_segment_return": 0.0,
        "worst_segment_drawdown": -0.1,
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


def sample_products() -> list[ProductSpec]:
    return [
        ProductSpec("QQQ", 1.0, "QQQ 1x"),
        ProductSpec("QLD", 2.0, "QLD 2x"),
        ProductSpec("TQQQ", 3.0, "TQQQ 3x"),
    ]
