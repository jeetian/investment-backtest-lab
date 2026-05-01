# AGENTS.md

## 專案定位

這是一個個人投資回測研究實驗室，用來研究 ETF、基金與長期配置策略。近期主線是美股 ETF 與 QQQ/QLD/TQQQ 的 DCA 槓桿策略研究；長期會擴充到台股、台灣 ETF 與台灣基金。

本專案只做研究，不做自動交易、不串券商、不提供投資建議。

優先參考：

- `docs/PROJECT_SCOPE_ZH.md`
- `docs/ROADMAP_ZH.md`
- `docs/STRATEGY_RESEARCH_PLAN_ZH.md`
- `docs/ALLOCATION_WORKFLOW_ZH.md`
- `docs/VALIDATION_STRATEGY_ZH.md`
- `docs/FRAMEWORK_HYGIENE_ZH.md`

## 核心原則

- 研究級正確性優先於 UI。
- 策略結果必須能回溯資料、訊號、交易、現金流、股息、費用、稅與每日資產。
- `AccountLedger`、`PortfolioLedger`、`MarginLoanLedger` 是可審計主線。
- `vectorbt` 與 `bt` 可用於訊號研究、參數掃描、再平衡研究與 cross-tool validation。
- 不要把研究框架輸出直接當最終 audit trail。
- margin loan 與 leveraged ETF product 必須分開建模。
- Optuna 未來只能作搜尋加速器，不得跳過 walk-forward、rolling cohort 與 synthetic stress。

## 目前研究主線

QQQ family DCA policy optimizer：

- 使用 `QQQ/QLD/TQQQ/CASH` 表達 0x 到 3x 的產品型槓桿曝險。
- 以 DCA XIRR 作主要排名口徑。
- `max_drawdown < -95%` 為硬淘汰線。
- `-85%` 到 `-95%` 為高風險帶，但不直接淘汰。
- 必須分開看 `actual_etf` 與 `synthetic_stress`。
- 必須檢查 walk-forward 與 rolling cohort，不得只看全期間結果。
- 月度配置研究訊號可輸出 QQQ/QLD/TQQQ/CASH 權重、regime、reason 與 effective leverage。
- 每週只做風險監控，不作自動調倉。

## 資料策略

MVP 資料來源：

- `yfinance`：美股、美股 ETF、槓桿 ETF、USD/TWD FX。
- `FinMind`：台股與台灣 ETF。
- CSV/parquet：未來台灣基金 NAV 與備援資料。

規則：

- token 只從環境變數讀取，例如 `FINMIND_TOKEN`。
- 不提交秘密資訊。
- 清理後資料優先存 parquet cache。
- 策略不要直接依賴資料來源 adapter。
- 資料 adapter 不依賴 `vectorbt` 或 `bt`。

## 驗證要求

常用命令：

```powershell
uv run pytest
uv run ruff check src tests scripts
uv run python scripts\smoke_imports.py
uv run python scripts\run_prototype.py --config configs\mvp_example.yaml --offline-demo
uv run python scripts\cross_validate.py
uv run python scripts\analyze_dca_policy_optimizer.py --config configs\mvp_example.yaml --family qqq --scan-mode fast --cohort-validation
```

新增策略或 optimizer 功能時至少要有：

- golden tests。
- no-lookahead tests。
- max drawdown hard-line tests。
- walk-forward 或 cohort validation tests。
- HTML/CSV 報表存在性測試。
- 中文文件更新。

## 非目標

MVP 不做：

- 實盤交易。
- 券商串接。
- 自動下單。
- 高頻或日內交易。
- 期貨與選擇權。
- 完整個人稅務模擬。
- 用黑箱 optimizer 直接給最終答案。
