# 月度配置工作流

這份文件描述如何使用 DCA Policy Optimizer 產生研究版配置訊號。這不是投資建議，也不是自動下單流程。

## 每月流程

1. 更新依賴與研究資料。

```powershell
uv sync --extra dev
uv run python scripts\analyze_dca_policy_optimizer.py --config configs\mvp_example.yaml --family qqq --scan-mode fast --cohort-validation
uv run python scripts\analyze_monthly_decision_pack.py --config configs\mvp_example.yaml --family qqq
```

2. 先打開月度決策入口。

```text
reports/monthly_decision_pack_qqq.html
```

這頁只回答四件事：

- 本月研究配置是什麼。
- 是否需要人工 review。
- 為什麼需要或不需要 review。
- 上期與本期配置是否改變。

3. 再打開研究後台。

```text
reports/dca_policy_optimizer_qqq.html
```

這頁用來查 optimizer 細節，不是第一入口。先看 `研究摘要`，再看 `Eligible Candidates` 與 `Rejected / Watchlist`。

4. 在月度決策入口先看 `本月結論`。

重點欄位：

- `as_of_date`
- `scenario_label`
- `regime`
- `reason`
- `target_effective_leverage`
- `QQQ_weight`
- `QLD_weight`
- `TQQQ_weight`
- `CASH_weight`
- `next_rebalance_date`
- `next_monitor_date`
- `manual_review_required`
- `review_reasons`

5. 在 optimizer 後台看 `Eligible Candidates`。

確認策略是否通過：

- walk-forward。
- rolling cohort。
- synthetic stress。
- cross-mode drawdown filter。

`Rejected / Watchlist` 不是刪除策略，而是把高 XIRR 但風險或穩健性不足的策略放在研究觀察區。

6. 最後看 CSV。

- `reports/dca_policy_optimizer_qqq_policy.csv`
- `reports/dca_policy_optimizer_qqq_allocation_signal.csv`
- `reports/dca_policy_optimizer_qqq_cohort_summary.csv`
- `reports/monthly_decision_pack_qqq.csv`
- `reports/monthly_decision_pack_qqq_signal_history.csv`

## 週度監控

週度監控只回答一件事：風險狀態是否惡化到需要人工檢查。

- 正式調整節奏仍是每月。
- `review_now = true` 時，只代表需要人工檢查，不代表自動交易。
- 若市場狀態突然進入 defensive regime，可以人工決定是否提前處理。

## 人工判斷清單

在採用任何研究訊號前，至少確認：

- 是否理解策略規則。
- 是否能接受最大回撤。
- 是否能接受長時間修復期。
- 是否願意承擔槓桿 ETF product 的追蹤誤差與產品風險。
- 是否已確認這不是 margin loan 模型。
- 是否知道 synthetic stress 不是實際 ETF 歷史。

## 不做的事

- 不自動下單。
- 不串券商。
- 不產生正式投資建議。
- 不保證未來績效。
