# Investment Backtest Lab

Investment Backtest Lab 是一個投資回測研究專案，目前重點是 QQQ/QLD/TQQQ/CASH 的 DCA policy optimizer、monthly decision pack，以及 hybrid-primary Monte Carlo replay。

## 快速開始

```powershell
python -m uv sync --extra dev
python -m uv run pytest
python -m uv run python scripts\smoke_imports.py
```

Python 需求是 `>=3.12,<3.13`，依賴由 `uv.lock` 鎖定。

Windows pytest 若遇到 temp ACL 問題，可使用 fresh basetemp：

```powershell
python -m uv run pytest --basetemp=C:\Users\Ian Lai\Desktop\Python\pytest-fresh-20260505
```

## 主要月度流程

目前 `configs/mvp_example.yaml` 的資料截止設定是 `end_date: 2026-05-01`。這是
yfinance 的 end-exclusive cutoff，用來包含 `2026-04-30` 月底收盤；月度決策
comparison 會檢查 `recommended_as_of_date` 是否等於 `2026-04-30`。

先產生 actual ETF policy 與 cohort gate：

```powershell
python -m uv run python scripts\analyze_dca_policy_optimizer.py --config configs\mvp_example.yaml --family qqq --scan-mode fast --cohort-validation
```

產生 actual-primary 參考月度包：

```powershell
python -m uv run python scripts\analyze_monthly_decision_pack.py --config configs\mvp_example.yaml --family qqq
```

產生 hybrid-primary replay：

```powershell
python -m uv run python scripts\analyze_monthly_decision_replay.py --config configs\mvp_example.yaml --family qqq --selector hybrid_primary --fast
```

正式 full replay：

```powershell
python -m uv run python scripts\analyze_monthly_decision_replay.py --config configs\mvp_example.yaml --family qqq --selector hybrid_primary --full
```

產生一頁決策包：

```powershell
python -m uv run python scripts\analyze_monthly_decision_comparison.py --config configs\mvp_example.yaml --family qqq
```

產生人工 review 紀錄：

```powershell
python -m uv run python scripts\record_monthly_decision_review.py --family qqq --status pending_review --reviewer "Ian"
```

## 最重要的輸出

日常先看：

- `reports/monthly_decision_comparison_qqq.html`：一頁決策包，顯示正式推薦策略、可交易權重、review 狀態、MC 風控摘要與 source coverage。
- `reports/monthly_decision_review_qqq.md`：人工 review 紀錄，留下本月是否接受、延後或覆核的決策痕跡。
- `reports/monthly_decision_comparison_qqq_top_candidates.csv`：top candidates 摘要。
- `reports/monthly_decision_replay_qqq_mc_summary.csv`：Monte Carlo trials 分層摘要。

深度審計才看：

- `reports/monthly_decision_replay_qqq_mc_trials.csv.gz`：完整 Monte Carlo trials 壓縮檔。
- `reports/monthly_decision_replay_qqq_cohorts.csv`：deterministic rolling cohort gate 明細。
- `reports/dca_policy_optimizer_qqq_policy.csv`：每日 actual ETF policy state。

## 目前決策語意

- 正式月度 authority：`hybrid_primary_monte_carlo`
- `hybrid_primary` 上市後使用 actual ETF，上市前使用 scaled synthetic backfill。
- Monte Carlo ranking 是正式主排名，expected XIRR 是主要 objective。
- deterministic rolling cohort gate 是硬風控 gate。
- actual-primary monthly decision pack 保留為參考訊號；和 hybrid-primary authority 不同不代表錯誤，而是人工 review 要看的分歧。
- review record 只記錄人工判斷，不會改變 ranking 或權重計算。

## Runtime

- `hybrid_primary --fast` 約 10 分鐘。
- `hybrid_primary --full` 目前觀察約 49 分鐘。
- 主要瓶頸在 deterministic rolling cohort gate。
- CLI 會印出開始時間、selector、scan mode、horizons、coverage、階段進度、總耗時與輸出檔案。

## 其他研究入口

```powershell
python -m uv run python scripts\run_prototype.py --config configs\mvp_example.yaml --offline-demo
python -m uv run python scripts\analyze_ledger.py --config configs\mvp_example.yaml --tickers SPY QQQ
python -m uv run python scripts\analyze_leveraged_etf_lab.py --config configs\mvp_example.yaml --family qqq --cash-flow-mode both
```

更多接手說明：

- `RESULTS_ZH.md`
- `docs/HANDOFF_ZH.md`
- `docs/QUICKSTART_ZH.md`
- `docs/ALLOCATION_WORKFLOW_ZH.md`
- `docs/ROADMAP_ZH.md`

長期方向文件：

- [專案範圍](docs/PROJECT_SCOPE_ZH.md)
- [Roadmap](docs/ROADMAP_ZH.md)
- [驗證策略](docs/VALIDATION_STRATEGY_ZH.md)
- [框架維護規範](docs/FRAMEWORK_HYGIENE_ZH.md)
- [策略研究計畫](docs/STRATEGY_RESEARCH_PLAN_ZH.md)
- [月度配置工作流](docs/ALLOCATION_WORKFLOW_ZH.md)

## FinMind Token

不要把 token 寫入檔案。需要 live data 時，只在目前 shell 設定：

```powershell
$env:FINMIND_TOKEN = "你的 token"
python -m uv run python scripts\smoke_data.py --network
```
