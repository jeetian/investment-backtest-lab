# 框架維護清單

本專案會一路從美股 prototype 擴充到台股、台灣基金、槓桿策略與 UI。為了避免功能長大後互相纏住，每次新增功能都要先通過這份清單。

## 分層原則

- data adapter 只負責取資料與標準化，不直接碰策略或 ledger。
- strategy 只產生訊號、權重、投入排程或目標曝險，不直接改現金帳。
- ledger 只負責可審計會計：交易、股息、費用、稅、借款、利息、現金流與每日 snapshot。
- report 只讀取 ledger 或策略結果，不反向修改回測邏輯。
- UI 只呼叫公開 CLI/API，不把資料來源、最佳化或會計邏輯寫在畫面層。

## 新功能合併前檢查

- 是否有 public config，且預設值保守、可覆寫。
- 是否有 audit trail：至少能輸出 trades、cash flows、fees/taxes、equity curve；槓桿功能還要輸出 debt、interest、safety buffer 與 margin events。
- 是否有 golden tests，可以用手算案例驗證核心會計。
- 是否需要 cross-tool validation：若有成熟工具可對照，應新增 synthetic comparison。
- 是否有中文文件，讓未來的自己能理解限制與使用方式。
- 是否不破壞既有 CLI、CSV、Markdown、HTML 報表。
- 是否明確標註研究假設，不把 prototype 結果包裝成投資建議。

## 槓桿功能特別規範

- margin loan 與 leveraged ETF product 必須分開建模。
- margin loan 主線要顯示 debt、borrow rate、interest paid、equity ratio、maintenance requirement、safety buffer、margin call 與 forced deleverage。
- leveraged ETF 只先當一般價格序列回測；報表要標註沒有模擬產品內部 swap、期貨、融資成本、每日重設與追蹤誤差細節。
- 最佳化排序不能只看 CAGR，必須同時看 max drawdown、Calmar、Sortino、Sharpe、worst safety buffer 與 margin call 次數。

## 常用驗證命令

```powershell
uv run ruff check src tests scripts\analyze_ledger.py scripts\analyze_leverage.py scripts\cross_validate.py
uv run pytest
uv run python scripts\cross_validate.py
uv run python scripts\run_prototype.py --config configs\mvp_example.yaml --offline-demo
uv run python scripts\analyze_ledger.py --config configs\mvp_example.yaml --tickers SPY QQQ
uv run python scripts\analyze_leverage.py --config configs\mvp_example.yaml --tickers SPY QQQ
```
