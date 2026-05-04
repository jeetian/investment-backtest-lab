# 接手說明

## 目前狀態

- 專案已可在這台電腦用 `python -m uv` 執行。
- 目前資料 cutoff 設為 `2026-05-01`，用來包含 `2026-04-30` 月底收盤。
- comparison CLI 會檢查 `recommended_as_of_date` 是否符合 config end-exclusive cutoff。
- 月度正式 replay selector 已改成 `hybrid_primary`。
- `hybrid_primary` 的資料邏輯是：ETF 上市後使用 actual ETF，上市前使用 scaled synthetic backfill。
- 正式策略排名由 Monte Carlo replay 主導，deterministic rolling cohort 保留為硬風控 gate 與審計軌跡。
- 目前最重要的結果入口是 `reports/monthly_decision_comparison_qqq.html`。

## 最新觀察結果

- replay authority：`hybrid_primary_monte_carlo`
- 正式推薦策略：`Vol Target 63D 25%`
- latest actual ETF 可交易權重：`QQQ 73% / QLD 27% / TQQQ 0% / CASH 0%`
- actual-primary reference：`Momentum+Trend 126D/200MA 3.0x to 1.0x`
- actual-primary reference 與 hybrid-primary authority 不同時，不視為錯誤；這代表參考訊號分歧，需要在一頁決策包中人工確認。

## 日常結果入口

平常先看：

```text
reports/monthly_decision_comparison_qqq.html
```

這份一頁決策包會顯示：

- 正式推薦策略。
- 可交易權重。
- manual review 狀態與原因。
- Hybrid-primary authority 與 actual-primary reference 的差異。
- MC risk 摘要：expected XIRR、p05 XIRR、win rate、drawdown breach rate、cohort gate。
- source coverage：QQQ/QLD/TQQQ 哪段使用 actual，哪段使用 synthetic backfill。

人工確認後再產生 review record：

```powershell
python -m uv run python scripts\record_monthly_decision_review.py --family qqq --status pending_review --reviewer "Ian"
```

comparison HTML 是決策入口；review record 是人工確認留痕。它只記錄本月 review 狀態、reviewer、notes 與審計證據路徑，不會改變 ranking 或權重。

日常可讀 CSV：

- `reports/monthly_decision_comparison_qqq.csv`
- `reports/monthly_decision_comparison_qqq_top_candidates.csv`
- `reports/monthly_decision_review_qqq.csv`
- `reports/monthly_decision_replay_qqq_mc_summary.csv`
- `reports/monthly_decision_replay_qqq_source_coverage.csv`

深度審計才看：

- `reports/monthly_decision_replay_qqq_mc_trials.csv.gz`
- `reports/monthly_decision_replay_qqq_cohorts.csv`
- `reports/monthly_decision_replay_qqq_decisions.csv`
- `reports/dca_policy_optimizer_qqq_policy.csv`

## 常用命令

```powershell
python -m uv run pytest
python -m uv run ruff check src tests scripts
python -m uv run python scripts\analyze_dca_policy_optimizer.py --config configs\mvp_example.yaml --family qqq --scan-mode fast --cohort-validation
python -m uv run python scripts\analyze_monthly_decision_pack.py --config configs\mvp_example.yaml --family qqq
python -m uv run python scripts\analyze_monthly_decision_replay.py --config configs\mvp_example.yaml --family qqq --selector hybrid_primary --fast
python -m uv run python scripts\analyze_monthly_decision_comparison.py --config configs\mvp_example.yaml --family qqq
```

正式 full MC replay：

```powershell
python -m uv run python scripts\analyze_monthly_decision_replay.py --config configs\mvp_example.yaml --family qqq --selector hybrid_primary --full
```

Windows pytest 若遇到 temp ACL 問題，使用 fresh basetemp：

```powershell
python -m uv run pytest --basetemp=C:\Users\Ian Lai\Desktop\Python\pytest-fresh-20260505
```

## Runtime

- `hybrid_primary --fast`：約 10 分鐘。
- `hybrid_primary --full`：目前觀察約 49 分鐘。
- 主要瓶頸在 deterministic rolling cohort gate。
- CLI 已有開始時間、selector、scan mode、horizons、coverage、階段進度、總耗時與輸出檔案，長時間執行時可用來確認不是卡死。

## 下一步建議

在決策頁與審計資料結構穩定後，再考慮 Optuna 或擴大策略搜尋空間。現在優先順序是：

1. 固定月度決策包的閱讀與審計流程。
2. 確認 source coverage 與 MC summary 足夠回答人工 review 問題。
3. 再把同一套流程擴到其他 ETF family 或 Optuna objective。
