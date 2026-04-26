# AGENTS.md

## 專案定位

這是一個個人投資回測研究實驗室，用來研究台灣與美國市場的長期投資策略。

長期 scope、roadmap 與驗證標準請優先參考：

- `docs/PROJECT_SCOPE_ZH.md`
- `docs/ROADMAP_ZH.md`
- `docs/VALIDATION_STRATEGY_ZH.md`
- `docs/FRAMEWORK_HYGIENE_ZH.md`

MVP 評估範圍：

- 台股
- 台灣 ETF
- 美股與美股 ETF
- 台灣基金先預留 NAV/CSV 介面，第二階段再接入

本專案只做投資研究，不是實盤交易系統，不得下單或串接券商。

研究級正確性優先於產品化 UI。先把美股與美股 ETF 做到價格、股息、匯率、成本、稅與 ledger 可審計，再擴充台股、台灣 ETF 與台灣基金。UI 第一版預計使用 Streamlit。

---

## 核心方向

不要從零開始寫完整回測引擎。優先整合成熟 Python 框架，再用薄薄的 adapter 包起來。

框架優先順序：

1. `vectorbt`
   - 主要用於訊號型策略、技術指標、參數掃描與快速研究。
   - 目前固定使用 `vectorbt==1.0.0`。

2. `bt`
   - 用於目標權重、月/季再平衡、配置型策略。
   - Windows + Python 3.12 需要 C++ Build Tools 才能 build。
   - 目前環境已成功安裝並驗證 `bt==1.1.5`。

3. `quantstats`
   - 用於績效報表與 HTML report。

4. Backtrader
   - 只在之後需要 event-driven 驗證時再考慮。

5. QuantConnect Lean
   - 只作為長期專業架構選項，不放進 MVP。

嚴謹現金流主線：

- `AccountLedger` 是後續嚴謹回測的核心。
- `MarginLoanLedger` 是美股 ETF 輕槓桿研究的核心，必須獨立保留借款、利息、維持率、安全緩衝與 margin call audit trail。
- `vectorbt` 與 `bt` 保留作策略訊號、參數掃描、配置研究與 cross-tool validation。
- 不要把研究框架的輸出直接當成最終 audit trail；需要能回溯交易、股息、費用、稅與每日資產。
- margin loan 和 leveraged ETF product 必須分開建模；槓桿 ETF 只當一般價格序列資產，不宣稱已模擬產品內部每日重設、swap 或期貨細節。

---

## 環境原則

在策略研究前，先確保環境可重現：

- Python 3.12
- uv
- Git
- JupyterLab
- Microsoft Visual Studio C++ Build Tools
- `FINMIND_TOKEN` 只從環境變數讀取

不要硬編碼 API token。不要提交秘密資訊。

---

## 資料策略

用 pandas DataFrame 作為共同資料格式，並分離 raw、cache、processed 資料。

MVP 資料來源：

- `FinMind`：台股與台灣 ETF
- `yfinance`：美股、美股 ETF、USD/TWD FX
- 本地 CSV/parquet：備援資料與未來台灣基金 NAV

標準 OHLCV 欄位：

- `open`
- `high`
- `low`
- `close`
- `volume`

重要規則：

- 優先把清理後資料存成 parquet。
- raw data 和 cleaned/cache data 不混在一起。
- 不做不穩定或授權不清楚的網站爬蟲。
- 若台股還原價不可用，報表要明確標註限制。

---

## 策略範圍

MVP 策略：

- 定期定額 DCA
- Buy and hold
- 均線進出
- 月/季投組再平衡
- Rolling 3Y / 5Y / 10Y 分析

第二階段：

- 動能輪動
- 股息再投入
- 研究版槓桿與風險指標調整曝險
- 台灣基金 NAV 回測
- 更細的配息與稅務處理

---

## 成本與幣別

預設報表基準幣別為 TWD。

美股資產：

- 保留 USD 原幣績效。
- 最終用 USD/TWD FX 換算 TWD 績效。
- 可行時輸出 FX contribution。

MVP 成本模型處理交易層成本：

- 台股手續費
- 台股手續費折扣
- 台股最低手續費
- 台股證交稅
- 美股佣金
- 美股 SEC fee
- 美股 FINRA TAF
- 換匯 spread
- 滑價

MVP 不做完整個人所得稅模擬。

稅務預設觀點：

- 預設為台灣個人投資者視角。
- Phase 1 只做交易層成本與美股股息預扣稅。
- 台灣個人所得稅、海外所得與最低稅負情境留到後續階段。

---

## 架構原則

清楚分離：

1. 資料 adapter
2. 資料標準化
3. 策略訊號產生
4. 框架 wrapper
5. 成本與匯率換算
6. 績效報表
7. Notebook 或 UI

策略程式不要直接依賴資料來源 adapter。
資料 adapter 不要依賴 `vectorbt` 或 `bt`。

---

## 非目標

MVP 不做：

- 實盤交易
- 券商串接
- 自製撮合引擎
- 日內交易
- 期貨或選擇權
- 高頻回測
- 完整個人稅務模擬
- 複雜交割模擬

---

## 常用驗證

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\smoke_imports.py
.\.venv\Scripts\python.exe scripts\run_prototype.py --config configs\mvp_example.yaml --offline-demo
.\.venv\Scripts\python.exe scripts\analyze_leverage.py --config configs\mvp_example.yaml --tickers SPY QQQ
```
