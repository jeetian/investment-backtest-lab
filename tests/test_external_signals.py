from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from investment_backtest_lab.external_signals import (
    CORE_EXTERNAL_SIGNAL_FEATURE_COLUMNS,
    build_external_signal_audit_checks,
    build_market_regime_features,
    external_signal_columns_for_feature_set,
    load_external_signal_features,
    validate_external_signal_features,
)


def test_external_features_are_validated_and_shifted(tmp_path: Path):
    path = tmp_path / "market_regime_features_tw50.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "fear_greed_score": [10.0, 20.0, 30.0],
            "vix_percentile_252": [80.0, 70.0, 60.0],
        }
    ).to_csv(path, index=False)

    aligned = load_external_signal_features(
        path,
        trading_index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
    )

    assert pd.isna(aligned["fear_greed_score"].iloc[0])
    assert aligned["fear_greed_score"].iloc[1] == 10.0
    assert aligned["vix_percentile_252"].iloc[2] == 70.0


def test_core_external_features_trim_to_common_complete_coverage(tmp_path: Path):
    path = tmp_path / "market_regime_features_tw50.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"],
            "fear_greed_score": [10.0, 20.0, 30.0, 40.0],
            "vix_percentile_252": [80.0, 70.0, 60.0, 50.0],
            "usdtwd_return_63d_percentile_252": [None, None, 45.0, 46.0],
            "tw_margin_balance_percentile_252": [55.0, 56.0, 57.0, 58.0],
            "tw_institutional_net_buy_21d_percentile_252": [65.0, 66.0, 67.0, 68.0],
        }
    ).to_csv(path, index=False)

    aligned = load_external_signal_features(
        path,
        trading_index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"]),
        feature_columns=CORE_EXTERNAL_SIGNAL_FEATURE_COLUMNS,
        require_complete=True,
    )

    assert tuple(aligned.columns) == CORE_EXTERNAL_SIGNAL_FEATURE_COLUMNS
    assert "fear_greed_score" not in aligned.columns
    assert aligned.index.min() == pd.Timestamp("2020-01-06")
    assert aligned.isna().sum().sum() == 0


def test_core_external_features_fail_when_required_feature_missing(tmp_path: Path):
    path = tmp_path / "market_regime_features_tw50.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "vix_percentile_252": [80.0, 70.0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        load_external_signal_features(
            path,
            trading_index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
            feature_columns=CORE_EXTERNAL_SIGNAL_FEATURE_COLUMNS,
            require_complete=True,
        )


def test_external_features_reject_bad_schema():
    with pytest.raises(ValueError, match="duplicate dates"):
        validate_external_signal_features(
            pd.DataFrame(
                {
                    "date": ["2020-01-01", "2020-01-01"],
                    "fear_greed_score": [10.0, 20.0],
                }
            )
        )
    with pytest.raises(ValueError, match="sorted"):
        validate_external_signal_features(
            pd.DataFrame(
                {
                    "date": ["2020-01-02", "2020-01-01"],
                    "fear_greed_score": [10.0, 20.0],
                }
            )
        )


def test_market_regime_features_builds_expected_columns():
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    features = build_market_regime_features(
        fear_greed=pd.DataFrame(
            {
                "date": dates,
                "score": range(300),
                "rating": ["neutral"] * 300,
            }
        ),
        vix=pd.DataFrame({"date": dates, "vixcls": range(300)}),
        usdtwd=pd.DataFrame({"date": dates, "usdtwd": [30 + i * 0.01 for i in range(300)]}),
        margin=pd.DataFrame(
            {
                "date": dates,
                "name": ["融資"] * 300,
                "TodayBalance": range(1_000, 1_300),
            }
        ),
        institutional=pd.DataFrame(
            {
                "date": dates,
                "name": ["total"] * 300,
                "buy": range(300),
                "sell": range(100, 400),
            }
        ),
    )

    assert "fear_greed_score" in features.columns
    assert "vix_percentile_252" in features.columns
    assert "tw_margin_balance_percentile_252" in features.columns
    assert features["vix_percentile_252"].notna().any()


def test_external_signal_audit_fails_missing_artifacts(tmp_path: Path):
    checks = build_external_signal_audit_checks(family="tw50", external_dir=tmp_path)

    assert "fail" in set(checks["status"])
    assert "fear_greed" in set(checks["check_id"])


def test_core_external_signal_audit_excludes_fear_greed(tmp_path: Path):
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    pd.DataFrame({"date": dates, "vixcls": range(30)}).to_csv(
        tmp_path / "fred_vixcls.csv",
        index=False,
    )
    pd.DataFrame({"date": dates, "usdtwd": range(30)}).to_csv(
        tmp_path / "yfinance_usdtwd.csv",
        index=False,
    )
    pd.DataFrame({"date": dates, "TodayBalance": range(30)}).to_csv(
        tmp_path / "finmind_tw_margin_total.csv",
        index=False,
    )
    pd.DataFrame({"date": dates, "buy": range(30), "sell": range(30)}).to_csv(
        tmp_path / "finmind_tw_institutional_total.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "date": dates,
            "vix_percentile_252": range(30),
            "usdtwd_return_63d_percentile_252": range(30),
            "tw_margin_balance_percentile_252": range(30),
            "tw_institutional_net_buy_21d_percentile_252": range(30),
        }
    ).to_csv(tmp_path / "market_regime_features_tw50.csv", index=False)
    (tmp_path / "market_regime_manifest_tw50.json").write_text("{}", encoding="utf-8")

    checks = build_external_signal_audit_checks(
        family="tw50",
        external_dir=tmp_path,
        feature_set="core",
    )

    assert "fear_greed" not in set(checks["check_id"])
    assert "fear_greed_excluded_from_core" in set(checks["check_id"])
    assert "core_common_coverage" in set(checks["check_id"])
    assert external_signal_columns_for_feature_set("core") == CORE_EXTERNAL_SIGNAL_FEATURE_COLUMNS
