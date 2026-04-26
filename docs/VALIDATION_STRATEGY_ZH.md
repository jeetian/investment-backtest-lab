# 回測正確性驗證策略

本專案的驗證分成 L1、L2、L3。原則是先驗證資料與現金流，再驗證策略與框架結果。Ledger 是可審計主線，`vectorbt`、`bt`、`quantstats` 主要作為研究與交叉驗證工具。

## L1：資料品質驗證

目的：確認輸入資料本身可以被信任。

檢查項目：

- 日期是否可解析並排序。
- 價格欄位是否齊全：open、high、low、close、volume。
- close 是否缺值、重複日期、非正數。
- 是否有異常大幅跳動。
- cache 是否可重讀。
- FX 日期是否能與資產價格對齊。

L1 主要避免「資料壞掉卻跑出漂亮報表」。

## L2：Golden Case 手算驗證

目的：用非常小、可以手算的案例確認 ledger 現金流。

目前重點：

- 買進、賣出、費用、稅、現金與持股正確。
- 股息現金入帳：gross dividend、withholding tax、net cash 正確。
- 股息再投入：扣稅後新增股數正確。
- DCA 每期投入、買入、費用與期末資產正確。
- 再平衡會先賣超配，再買低配，且不超買。
- 每日 snapshot 滿足 `cash + market value = total equity`。
- 槓桿 snapshot 滿足 `cash + market value - debt = total equity`。
- margin loan 每日利息會降低 equity，並增加 debt。
- `safety_buffer < min_safety_buffer` 時會自動降槓桿到 `deleverage_to`。
- `equity_ratio <= maintenance_requirement` 時會記錄 margin call 與 forced deleverage。
- TWD 換算使用對齊後的 USD/TWD 匯率。

Golden case 是本專案最重要的防線。只要 ledger 行為有變，應優先補 golden tests。

## L3：Cross-Tool Validation

目的：用不同工具或不同實作互相校驗，降低「自己寫錯但測試也跟著錯」的風險。

目前已落地：

```powershell
uv run python scripts\cross_validate.py
```

第一批案例刻意使用 synthetic price-only、zero-fee、no-dividend：

- `buy_hold_price_only`：AccountLedger vs `vectorbt`，比對單資產 buy-and-hold 期末資產。
- `monthly_rebalance_price_only`：PortfolioLedger vs `bt`，比對 SPY/QQQ 60/40 月再平衡總報酬。
- `raw_dividend_reinvest_vs_adjusted_price`：AccountLedger 使用 raw close 加股息再投入，對照人工建立的 total-return adjusted close。

這批案例不驗證資料品質、稅、費用或匯率；那些由 L1/L2 與 ledger 報表測試處理。L3 的目的，是把核心交易、股息再投入與再平衡數學拿去和成熟框架或獨立公式對答案。

後續擴充方向：

- 報表輸出的 drawdown 指標 vs `quantstats`。
- raw price + dividend reinvestment vs adjusted price 的 live data 近似檢查。
- 槓桿策略與風險控制的簡化交叉案例。
- `target_leverage=1.0`、zero-interest、price-only 時，槓桿 ledger 應貼近無槓桿 ledger。
- 台股除權息與台灣基金配息接入後的代表案例。

## 標準驗證命令

常規開發至少跑：

```powershell
uv run pytest
uv run ruff check src tests scripts\analyze_ledger.py scripts\analyze_leverage.py scripts\cross_validate.py
```

修改資料、報表或 CLI 時，加跑：

```powershell
uv run python scripts\smoke_imports.py
uv run python scripts\run_prototype.py --config configs\mvp_example.yaml --offline-demo
uv run python scripts\analyze_results.py --config configs\mvp_example.yaml --tickers SPY QQQ
uv run python scripts\analyze_ledger.py --config configs\mvp_example.yaml --tickers SPY QQQ
uv run python scripts\analyze_leverage.py --config configs\mvp_example.yaml --tickers SPY QQQ
uv run python scripts\cross_validate.py
```

## 報表驗證原則

報表不是只看漂亮圖表，還要能追溯：

- 資料來源與日期範圍。
- 策略、股息模式、成本、稅率與基準幣別。
- trades、dividends、cash flows、equity、positions、rebalance CSV。
- 槓桿報表必須能看到 debt、interest、actual leverage、equity ratio、safety buffer、margin events。
- 重要限制，例如 yfinance dividend date、fractional shares、raw/adjusted price 假設。

每個新增功能都應該能回答三個問題：

- 數字從哪裡來？
- 是否能用小案例手算？
- 是否能用另一個工具或簡化模型交叉檢查？
