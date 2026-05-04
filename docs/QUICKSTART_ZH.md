# 快速上手

這份專案目前最重要的主線是 QQQ family（QQQ / QLD / TQQQ / CASH）的 DCA
槓桿 ETF 策略研究。報表只提供研究訊號，不是投資建議，也不會自動下單。

## 1. 確認環境

```powershell
python --version
uv --version
git --version
uv sync --extra dev
```

基本驗證：

```powershell
uv run pytest
uv run python scripts\smoke_imports.py
uv run python scripts\run_prototype.py --config configs\mvp_example.yaml --offline-demo
```

## 2. 一般 SPY / QQQ 報表

```powershell
uv run python scripts\analyze_results.py --config configs\mvp_example.yaml --tickers SPY QQQ
uv run python scripts\analyze_ledger.py --config configs\mvp_example.yaml --tickers SPY QQQ
```

主要輸出：

- `reports/quickstart_spy_qqq.md`
- `reports/ledger_spy_qqq.html`

## 3. 槓桿 ETF Product Lab

```powershell
uv run python scripts\analyze_leveraged_etf_lab.py --config configs\mvp_example.yaml --family qqq --cash-flow-mode both
```

主要輸出：

- `reports/leveraged_etf_qqq.html`
- `reports/leveraged_etf_qqq_metrics.csv`

這頁用來理解 QQQ / QLD / TQQQ 的一次投入與 DCA 候選策略，並分開看
`actual_etf` 與 `synthetic_stress`。Synthetic stress 是壓測，不是真實 ETF 歷史。

## 4. DCA Policy Optimizer

```powershell
uv run python scripts\analyze_dca_policy_optimizer.py --config configs\mvp_example.yaml --family qqq --scan-mode fast --cohort-validation
```

主要輸出：

- `reports/dca_policy_optimizer_qqq.html`
- `reports/dca_policy_optimizer_qqq_metrics.csv`
- `reports/dca_policy_optimizer_qqq_policy.csv`
- `reports/dca_policy_optimizer_qqq_cohort_summary.csv`
- `reports/dca_policy_optimizer_qqq_allocation_signal.csv`

閱讀順序：

1. 先看「研究摘要」與 `Eligible Candidates`。
2. 再看 Actual ETF / Synthetic Stress 是否同時可接受。
3. 接著看 Walk-Forward 與 Cohort Robustness。
4. 最後看 Monthly Allocation Signal 與 signal explainability。

## 5. Monthly Decision Pack

```powershell
uv run python scripts\analyze_monthly_decision_pack.py --config configs\mvp_example.yaml --family qqq
```

主要輸出：

- `reports/monthly_decision_pack_qqq.html`
- `reports/monthly_decision_pack_qqq.csv`
- `reports/monthly_decision_pack_qqq_signal_history.csv`

這是每月使用入口。它目前仍使用 `actual_primary` 邏輯，也就是 actual ETF 排名優先，
synthetic stress 用來淘汰或警示。打開這頁先看：

- 本月研究配置。
- 目標有效槓桿。
- 是否需要 manual review。
- 上期與本期配置差異。
- 下一次月度調整日與週度監控日。

## 6. Synthetic-Primary Monthly Replay

```powershell
uv run python scripts\analyze_monthly_decision_replay.py --config configs\mvp_example.yaml --family qqq --selector synthetic_primary
```

主要輸出：

- `reports/monthly_decision_replay_qqq.html`
- `reports/monthly_decision_replay_qqq_ranking.csv`
- `reports/monthly_decision_replay_qqq_cohorts.csv`
- `reports/monthly_decision_replay_qqq_decisions.csv`

這份報表是壓測優先的對照流程，不會覆蓋 Monthly Decision Pack。它會用不同起點與
不同終點的 DCA cohort replay，檢查策略是否能穩定打敗 `QQQ DCA` benchmark。
排序重點是 win rate、drawdown breach rate、worst drawdown 與 replay score，
不是 ending equity 或單一全期間 XIRR。

## 7. FinMind Token

台股資料需要 FinMind token：

```powershell
$env:FINMIND_TOKEN = "你的 token"
uv run python scripts\smoke_data.py --network
```

沒有 token 時，美股研究流程仍可執行；台股 live data 測試會跳過。
