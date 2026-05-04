# 月度配置工作流

這份工具的目標是提供研究訊號，幫助人工決策 ETF / 基金配置。它不自動下單，
不串券商，也不保證未來績效。

## 每月流程

1. 更新環境與資料。

```powershell
uv sync --extra dev
uv run python scripts\analyze_dca_policy_optimizer.py --config configs\mvp_example.yaml --family qqq --scan-mode fast --cohort-validation
```

2. 產生月度決策包。

```powershell
uv run python scripts\analyze_monthly_decision_pack.py --config configs\mvp_example.yaml --family qqq
```

3. 打開主要入口。

```text
reports/monthly_decision_pack_qqq.html
```

先確認：

- 本月研究配置：QQQ / QLD / TQQQ / CASH 權重。
- 目標有效槓桿。
- regime 與 reason。
- 是否需要 manual review。
- 下一次月度調整日。
- 上期 vs 本期配置是否改變。

4. 若需要查細節，再打開研究後台。

```text
reports/dca_policy_optimizer_qqq.html
```

重點看：

- Eligible Candidates。
- Rejected / Watchlist。
- Actual ETF 與 Synthetic Stress 是否互相矛盾。
- Walk-Forward Validation。
- Cohort Robustness。
- signal explainability CSV。

## Synthetic-Primary 對照流程

如果你想讓壓測結果成為優先排序來源，請額外執行：

```powershell
uv run python scripts\analyze_monthly_decision_replay.py --config configs\mvp_example.yaml --family qqq --selector synthetic_primary
```

輸出：

```text
reports/monthly_decision_replay_qqq.html
```

這份報表用 synthetic stress ranking 和 rolling cohort replay 來評估每月決策流程。
它會比較策略是否在不同起點與終點下打敗 `QQQ DCA` benchmark，並檢查是否跌破
`-95%` 最大回撤硬線。

目前兩份報表的定位不同：

- `monthly_decision_pack_qqq.html`：actual-primary 主流程，適合每月例行閱讀。
- `monthly_decision_replay_qqq.html`：synthetic-primary 壓測對照，適合檢查極端市場穩健性。

在 synthetic-primary 邏輯正式成熟前，它不會自動覆蓋原本的 Monthly Decision Pack。

## 每週流程

每週只做風險監控，不做自動調倉。觀察：

- `review_now` 是否為 true。
- 是否跌破關鍵均線或風險線。
- synthetic stress 或 cohort 是否出現高風險警示。
- 是否需要人工檢查部位，而不是立即按工具自動交易。

## 人工決策檢查表

每次實際調整前，至少確認：

- 策略不是只靠單一全期間 XIRR 排名。
- 沒有跌破 `-95%` 最大回撤硬線。
- Synthetic stress 的最差 cohort 可以接受。
- Walk-forward 與 rolling cohort 結果沒有明顯不穩。
- 目前配置符合你自己的現金流、風險承受度與投資限制。

這些報表是研究工具，不是投資建議。
