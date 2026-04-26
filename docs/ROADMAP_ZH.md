# Roadmap

這份 roadmap 用來切分工作順序。原則是先把美股研究核心做正確，再擴充市場、策略與 UI。

## Phase 1：美股嚴謹核心

目標：建立可審計的美股回測地基。

完成條件：

- 價格、股息、匯率資料可讀取、快取與品質檢查。
- Account/portfolio ledger 能記錄交易、股息、費用、稅、現金、持股、權重與每日總資產。
- 支援 buy and hold、DCA、SPY/QQQ 再平衡、股息現金入帳、股息再投入。
- 支援基本美股交易成本與美股股息預扣稅。
- 報表同時提供投資者摘要與 audit appendix。
- L1 資料品質測試與 L2 golden tests 通過。

目前狀態：

- 美股價格與 USD/TWD 資料 smoke test 已可跑。
- quickstart SPY/QQQ 報表已可輸出。
- account ledger 與 portfolio ledger 已開始建立。
- 美股 ledger 報表 v1 已接入 raw price、股息、股息再投入、DCA 外部投入、SPY/QQQ 再平衡、USD/TWD 與 HTML 圖表。

## Phase 2：策略與驗證擴充

目標：把策略研究做成可比較、可掃描、可交叉驗證的流程。

完成條件：

- 策略 registry 可用 config 選擇策略與參數。
- 再平衡從 SPY/QQQ 擴充到更多美股 ETF，並加入 cross-tool validation。
- 動能輪動、波動度目標、研究版槓桿策略可以跑。
- 美股 ETF margin loan 風險模型可輸出借款、每日利息、維持率、安全緩衝、margin call 與 forced deleverage。
- 槓桿策略比較以 Calmar、Sortino、Sharpe、max drawdown、worst safety buffer 與 margin call 次數為主，不用 CAGR 單獨排序。
- `vectorbt` 用於訊號、參數掃描與快速比較。
- `bt` 用於配置型策略交叉檢查。
- L3 cross-tool validation 有至少一批代表案例。

目前狀態：

- 已新增美股 ETF margin loan v1，支援固定槓桿與動態槓桿：buy-hold、DCA、SPY/QQQ 60/40 rebalance。
- 動態槓桿 v1 會用趨勢均線、回撤、波動度與安全緩衝決定 1.0x / 1.1x / 1.3x，並輸出每日 policy audit trail。
- 下一步應加入槓桿策略 cross-tool / 手算情境驗證，以及參數掃描。
- leveraged ETF product 先只作為一般價格序列資產，不與 margin loan 會計混用。

## Phase 3：台股、台灣 ETF、台灣基金

目標：在美股核心穩定後擴充到台灣市場。

完成條件：

- FinMind adapter 支援台股與台灣 ETF 的正式研究流程。
- 台股交易成本模型完整接入 ledger。
- 台股除權息與還原價限制在報表中清楚揭露。
- 台灣基金 NAV CSV/parquet adapter 可用。
- 基金申購、贖回與配息假設可設定。

## Phase 4：Streamlit UI

目標：讓不想改 Python 的使用者也能操作回測。

完成條件：

- Streamlit 可選 config、資產池、策略與參數。
- 可執行回測並看圖表。
- 可下載 Markdown/CSV/HTML 報表。
- 常見錯誤能在 UI 中顯示清楚訊息。

## 延後決策

- 是否接 paid data：等免費資料流程與 adapter 穩定後再決定。
- 是否做正式 Web app：等 Streamlit 版卡到明確限制再決定。
- 是否引入 Backtrader：等 event-driven 需求明確後再決定。
- 是否引入 QuantConnect Lean：只作為長期專業架構參考。
