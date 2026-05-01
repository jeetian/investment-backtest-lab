# Roadmap

## Phase 1：美股可審計回測核心

已完成或已建立雛形：

- yfinance 資料 adapter 與 parquet cache。
- SPY/QQQ quickstart 報表。
- AccountLedger、PortfolioLedger。
- Buy-and-hold、DCA、SPY/QQQ 60/40 再平衡。
- raw price + dividend + tax + fee 的 ledger 報表。
- cross-tool validation：ledger vs vectorbt / bt。

## Phase 2：槓桿與策略研究

已完成或進行中：

- margin loan v1：debt、interest、safety buffer、margin call audit trail。
- leveraged ETF product lab：QQQ/QLD/TQQQ actual ETF 與 synthetic stress。
- DCA policy optimizer：trend、drawdown、volatility、momentum ladder。
- current allocation signal：QQQ/QLD/TQQQ/CASH 權重、regime、reason。

下一步：

- Phase 2.1：Rolling Cohort Robustness。
- Phase 2.2：月度配置研究訊號與週度風險監控。
- Phase 2.3：Optuna 搜尋加速器。
- Phase 2.4：SPY/SSO/UPRO 與更多 ETF family。

## Phase 3：台股、台灣 ETF、台灣基金

預計內容：

- FinMind 台股與台灣 ETF 資料主線。
- 台股交易成本、證交稅、最低手續費。
- 還原價與除權息限制揭露。
- 台灣基金 NAV CSV/parquet adapter。
- 基金配息與成本假設。

## Phase 4：互動 UI

預計內容：

- Streamlit UI 或等價互動工具。
- 選擇資產、策略、參數、回測期間。
- 查看圖表、配置訊號與 audit CSV。
- 下載報表。

## Optuna 導入原則

Optuna 只作搜尋加速器，不作最終判斷。任何 Optuna 找到的策略都必須通過：

- no-lookahead tests。
- walk-forward validation。
- rolling cohort validation。
- synthetic stress。
- 最大回撤硬線。
- 人可理解的 policy 規則。
