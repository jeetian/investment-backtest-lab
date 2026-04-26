from __future__ import annotations

import argparse
from pathlib import Path

from investment_backtest_lab.cross_validation import run_cross_validation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run synthetic cross-tool validation for ledger correctness."
    )
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checks = run_cross_validation(tolerance=args.tolerance)
    csv_path = output_dir / "cross_validation.csv"
    markdown_path = output_dir / "cross_validation.md"
    checks.to_csv(csv_path, index=False, encoding="utf-8")
    markdown_path.write_text(render_markdown(checks, csv_path), encoding="utf-8")

    printable = checks.copy()
    printable["passed"] = printable["passed"].map(lambda value: "PASS" if value else "FAIL")
    print(printable.to_string(index=False))
    print(f"\nCross-validation CSV: {csv_path}")
    print(f"Cross-validation report: {markdown_path}")
    if not bool(checks["passed"].all()):
        raise SystemExit(1)


def render_markdown(checks, csv_path: Path) -> str:
    table = checks.to_markdown(index=False, disable_numparse=True)
    return f"""# Cross-Tool Validation

這份報表用合成資料做 L3 cross-tool validation。案例刻意保持簡單，讓每個檢查只回答一個問題。

第一組 price-only、zero-fee、no-dividend，確認 ledger 的核心交易與再平衡數學能和成熟框架對上。
第二組 raw price + dividend reinvestment，確認 ledger 沒有漏算股息，
也沒有把 adjusted price 和股息重複計入。

- CSV：`{csv_path}`
- 通過標準：`abs_diff <= tolerance`

## 結果

{table}

## 解讀

- `buy_hold_price_only`：AccountLedger 與 vectorbt 比對單資產 buy-and-hold 期末資產。
- `raw_dividend_reinvest_vs_adjusted_price`：AccountLedger 用 raw close 加股息再投入，
  對照人工 total-return adjusted close。
- `monthly_rebalance_price_only`：PortfolioLedger 與 bt 比對 SPY/QQQ 60/40 月再平衡總報酬。
- 這不是資料品質驗證，也不涵蓋稅、費用與匯率；那些由 ledger golden tests 和實際報表測試處理。
"""


if __name__ == "__main__":
    main()
