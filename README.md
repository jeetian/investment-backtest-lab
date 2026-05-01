# Investment Backtest Lab

這是一個個人投資回測研究實驗室，目標是用可審計的資料、策略與報表，研究美股 ETF、台股、台灣 ETF 與台灣基金的長期投資配置。

目前專案重點已進入美股 ETF 研究：一般 ledger、margin loan 槓桿、槓桿 ETF product，以及 QQQ/QLD/TQQQ 的 DCA policy optimizer。工具輸出的是研究訊號，不是投資建議，也不會自動下單。

## 最快路線

```powershell
uv sync --extra dev
uv run pytest
uv run python scripts\smoke_imports.py
uv run python scripts\run_prototype.py --config configs\mvp_example.yaml --offline-demo
```

看 SPY/QQQ ledger 報表：

```powershell
uv run python scripts\analyze_ledger.py --config configs\mvp_example.yaml --tickers SPY QQQ
```

看 QQQ/QLD/TQQQ 槓桿 ETF product lab：

```powershell
uv run python scripts\analyze_leveraged_etf_lab.py --config configs\mvp_example.yaml --family qqq --cash-flow-mode both
```

看 DCA policy optimizer 與月度配置研究訊號：

```powershell
uv run python scripts\analyze_dca_policy_optimizer.py --config configs\mvp_example.yaml --family qqq --scan-mode fast --cohort-validation
```

主要輸出：

- `reports/ledger_spy_qqq.html`
- `reports/leverage_spy_qqq.html`
- `reports/leveraged_etf_qqq.html`
- `reports/dca_policy_optimizer_qqq.html`

## 專案方向

長期文件請先讀：

- [專案範圍](docs/PROJECT_SCOPE_ZH.md)
- [Roadmap](docs/ROADMAP_ZH.md)
- [策略研究計畫](docs/STRATEGY_RESEARCH_PLAN_ZH.md)
- [月度配置工作流](docs/ALLOCATION_WORKFLOW_ZH.md)
- [驗證策略](docs/VALIDATION_STRATEGY_ZH.md)
- [框架維護規範](docs/FRAMEWORK_HYGIENE_ZH.md)

目前優先順序：

1. 美股 ETF 的資料、股息、成本、稅與 ledger 正確性。
2. QQQ/QLD/TQQQ 的 DCA 槓桿策略研究。
3. Walk-forward、rolling cohort、synthetic stress 的防過擬合驗證。
4. 每月配置研究訊號與每週風險監控。
5. 之後再擴充 Optuna 搜尋、SPY/SSO/UPRO、台股、台灣 ETF 與台灣基金。

## 重要原則

- 研究級正確性優先於漂亮 UI。
- `AccountLedger`、`PortfolioLedger`、`MarginLoanLedger` 是可審計主線。
- `vectorbt`、`bt` 用於研究、掃描與 cross-tool validation，不直接取代 ledger audit trail。
- margin loan 和 leveraged ETF product 必須分開建模。
- Optuna 未來只作搜尋加速，不得取代 walk-forward、cohort validation 與 synthetic stress。
- 本專案不做自動交易、不串券商、不提供投資建議。

## 常用驗證

```powershell
uv run pytest
uv run ruff check src tests scripts
uv run python scripts\cross_validate.py
uv run python scripts\analyze_dca_policy_optimizer.py --config configs\mvp_example.yaml --family qqq --scan-mode fast --cohort-validation
```

## FinMind Token

台股與台灣 ETF live data 需要 FinMind token，請只用環境變數，不要寫入設定檔：

```powershell
$env:FINMIND_TOKEN = "你的 token"
uv run python scripts\smoke_data.py --network
```

沒有 token 時，美股與離線測試仍可正常執行。
