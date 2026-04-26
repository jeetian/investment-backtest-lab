from investment_backtest_lab.cross_validation import (
    buy_hold_ledger_vs_vectorbt,
    dividend_reinvestment_ledger_vs_adjusted_price,
    rebalance_ledger_vs_bt,
    run_cross_validation,
)


def test_buy_hold_cross_validation_matches_vectorbt():
    check = buy_hold_ledger_vs_vectorbt()

    assert check.passed
    assert check.primary == "AccountLedger"
    assert check.reference == "vectorbt"
    assert check.abs_diff <= check.tolerance


def test_rebalance_cross_validation_matches_bt():
    check = rebalance_ledger_vs_bt()

    assert check.passed
    assert check.primary == "PortfolioLedger"
    assert check.reference == "bt"
    assert check.abs_diff <= check.tolerance


def test_dividend_reinvestment_matches_adjusted_total_return_price():
    check = dividend_reinvestment_ledger_vs_adjusted_price()

    assert check.passed
    assert check.primary == "AccountLedger"
    assert check.reference == "vectorbt_adjusted_total_return"
    assert check.abs_diff <= check.tolerance


def test_run_cross_validation_returns_passed_dataframe():
    checks = run_cross_validation()

    assert set(checks["case"]) == {
        "buy_hold_price_only",
        "raw_dividend_reinvest_vs_adjusted_price",
        "monthly_rebalance_price_only",
    }
    assert checks["passed"].all()
