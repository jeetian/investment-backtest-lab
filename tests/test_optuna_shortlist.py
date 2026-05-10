from __future__ import annotations

import json

import pandas as pd

from investment_backtest_lab.optuna_shortlist import build_optuna_shortlist


def test_shortlist_excludes_rejected_and_keeps_v2_return_candidates():
    triage = pd.DataFrame(
        [
            triage_row(12641, "rejected", 0.136, 0.75, 0.099, -0.56, 0.012, 16.0),
            triage_row(12076, "candidate", 0.1334, 1.00, 0.106, -0.57, 0.0123, 16.0),
            triage_row(12931, "candidate", 0.1327, 1.00, 0.113, -0.55, 0.0123, 16.1),
            triage_row(12583, "candidate", 0.1330, 1.00, 0.108, -0.57, 0.0122, 16.2),
            triage_row(6544, "candidate", 0.0900, 0.75, 0.012, -0.43, 0.0280, 42.0),
            triage_row(2195, "candidate", 0.0470, 0.25, 0.026, -0.67, 0.0050, 6.0),
        ]
    )

    shortlist = build_optuna_shortlist(triage, max_candidates=5)

    selected = set(shortlist["trial_number"].astype(int))
    assert 12641 not in selected
    assert {12076, 12931, 12583}.issubset(selected)
    assert shortlist["shortlist_rank"].tolist() == [1, 2, 3, 4, 5]
    assert {"scenario_name", "scenario_label", "params_json"}.issubset(shortlist.columns)


def test_shortlist_deduplicates_params():
    duplicate_params = json.dumps(sample_params(seed=1), sort_keys=True)
    triage = pd.DataFrame(
        [
            triage_row(1, "candidate", 0.12, 1.0, 0.08, -0.50, 0.01, 8.0, duplicate_params),
            triage_row(2, "candidate", 0.11, 1.0, 0.07, -0.49, 0.01, 8.0, duplicate_params),
            triage_row(3, "candidate", 0.10, 1.0, 0.09, -0.48, 0.02, 9.0),
        ]
    )

    shortlist = build_optuna_shortlist(triage, max_candidates=3)

    assert len(shortlist) == 2
    assert set(shortlist["trial_number"].astype(int)) != {1, 2}


def triage_row(
    trial_number: int,
    status: str,
    expected: float,
    win: float,
    p05: float,
    drawdown: float,
    cost: float,
    turnover: float,
    params_json: str | None = None,
) -> dict[str, object]:
    return {
        "trial_number": trial_number,
        "triage_status": status,
        "triage_reason": "Passed return-first gates." if status == "candidate" else "rejected",
        "return_first_rank": trial_number,
        "expected_xirr": expected,
        "win_rate_vs_0050_dca": win,
        "p05_xirr": p05,
        "p05_max_drawdown": drawdown,
        "drawdown_breach_rate": 0.0,
        "cost_drag_on_contributed": cost,
        "turnover_sum": turnover,
        "holdout_expected_xirr": expected + 0.01,
        "params_json": params_json or json.dumps(sample_params(seed=trial_number), sort_keys=True),
    }


def sample_params(*, seed: int) -> dict[str, object]:
    return {
        "trend_window": 100,
        "slope_window": 21,
        "momentum_window": 63,
        "vol_window": 21,
        "vol_percentile_window": 252,
        "vol_target": 0.25 + seed * 0.000001,
        "risk_on_leverage": 1.5,
        "risk_off_leverage": 0.5,
        "cash_leverage": 0.0,
        "drawdown_guard": -0.2,
        "severe_drawdown_guard": -0.5,
        "vol_spike_quantile": 0.8,
        "rebalance_cadence": "monthly",
        "rebalance_threshold": 0.05,
        "min_holding_days": 21,
        "signal_hysteresis": 0.01,
        "max_annual_turnover": 20,
        "sentiment_mode": "off",
        "fear_threshold": 20,
        "greed_threshold": 80,
        "sentiment_deleverage": 0.0,
    }
