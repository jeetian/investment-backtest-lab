# 快速上手

## 1. 環境檢查

```powershell
python --version
uv --version
git --version
```

同步依賴：

```powershell
uv sync --extra dev
```

基本驗證：

```powershell
uv run pytest
uv run python scripts\smoke_imports.py
uv run python scripts\run_prototype.py --config configs\mvp_example.yaml --offline-demo
```

## 2. 一般 SPY/QQQ 報表

```powershell
uv run python scripts\analyze_results.py --config configs\mvp_example.yaml --tickers SPY QQQ
```

輸出：

- `reports/quickstart_spy_qqq.md`
- `reports/quickstart_spy_qqq_metrics.csv`

## 3. Ledger 報表

```powershell
uv run python scripts\analyze_ledger.py --config configs\mvp_example.yaml --tickers SPY QQQ
```

輸出：

- `reports/ledger_spy_qqq.html`
- `reports/ledger_spy_qqq_metrics.csv`
- `reports/ledger_spy_qqq_trades.csv`
- `reports/ledger_spy_qqq_dividends.csv`
- `reports/ledger_spy_qqq_cash_flows.csv`

Ledger 報表用來檢查 raw price、股息、稅、費用、DCA、再平衡與每日資產。

## 4. 槓桿 ETF Product Lab

```powershell
uv run python scripts\analyze_leveraged_etf_lab.py --config configs\mvp_example.yaml --family qqq --cash-flow-mode both
```

輸出：

- `reports/leveraged_etf_qqq.html`
- `reports/leveraged_etf_qqq_metrics.csv`
- `reports/leveraged_etf_qqq_dca_optimizer.csv`

這份報表比較 QQQ、QLD、TQQQ，以及 static mix、trend guard、drawdown guard。Synthetic stress 是壓力測試，不是實際 TQQQ 歷史。

## 5. DCA Policy Optimizer

```powershell
uv run python scripts\analyze_dca_policy_optimizer.py --config configs\mvp_example.yaml --family qqq --scan-mode fast --cohort-validation
```

輸出：

- `reports/dca_policy_optimizer_qqq.html`
- `reports/dca_policy_optimizer_qqq_metrics.csv`
- `reports/dca_policy_optimizer_qqq_policy.csv`
- `reports/dca_policy_optimizer_qqq_walk_forward.csv`
- `reports/dca_policy_optimizer_qqq_cohorts.csv`
- `reports/dca_policy_optimizer_qqq_cohort_summary.csv`
- `reports/dca_policy_optimizer_qqq_allocation_signal.csv`
- `reports/dca_policy_optimizer_qqq_signal_explainability.csv`

這份報表是研究後台，用來理解候選策略怎麼被挑出來。閱讀順序：

1. `研究摘要`：看目前採用候選、為什麼不是只選最高 XIRR。
2. `Monthly Allocation Signal`：看目前研究配置、regime、reason、下一次月度調整日。
3. `signal_explainability CSV`：查最新指標、觸發條件與 policy 理由。
4. `Eligible Candidates`：看通過驗證的候選策略。
5. `Rejected / Watchlist`：看報酬高但風險或穩健性不足的策略。
6. `Actual ETF Ranking`：看真實 QQQ/QLD/TQQQ 歷史。
7. `Synthetic Stress Ranking`：看 2000/2008 類壓力測試。
8. `Cohort Robustness`：看不同起點與持有期間下是否穩定。
9. `Compare Lab`：自由勾選策略疊圖。

## 6. Monthly Decision Pack

```powershell
uv run python scripts\analyze_monthly_decision_pack.py --config configs\mvp_example.yaml --family qqq
```

輸出：

- `reports/monthly_decision_pack_qqq.html`
- `reports/monthly_decision_pack_qqq.csv`
- `reports/monthly_decision_pack_qqq_signal_history.csv`

這份報表是每月第一入口，比 optimizer 後台更適合一般閱讀。閱讀順序：

1. `本月結論`：看目前配置、目標槓桿、下次調整日、是否需要人工 review。
2. `本月訊號解釋`：看 QQQ 價格、均線、動能、波動、回撤與下一個加減槓桿觸發條件。
3. `配置比例`：用長條圖看 QQQ/QLD/TQQQ/CASH 權重。
4. `為什麼需要 review`：看 synthetic stress、cohort、walk-forward、drawdown hard line 的實際數字。
5. `上期 vs 本期`：看權重、槓桿、regime 是否改變。
6. `Decision Checklist`：確認哪個風險檢查沒有通過。

建議日常順序是先看 `monthly_decision_pack_qqq.html`，需要查原因時再回到 `dca_policy_optimizer_qqq.html`。

## 7. FinMind Token

台股與台灣 ETF live data 需要：

```powershell
$env:FINMIND_TOKEN = "你的 token"
uv run python scripts\smoke_data.py --network
```

沒有 token 時，美股與離線測試仍可執行。

## 8. 常見提醒

- DCA 策略不要只看 ending equity，應優先看 XIRR、max drawdown 與 cohort robustness。
- 槓桿 ETF product 不是 margin loan。
- Synthetic stress 不是實際 ETF 歷史。
- 月度配置訊號是研究訊號，不是投資建議。
