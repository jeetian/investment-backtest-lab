# 專案範圍

Investment Backtest Lab 是個人投資研究工具，目標是用可重現、可審計的方式評估 ETF、基金與長期配置策略。工具可以協助產生研究訊號，但不提供投資建議，也不自動下單。

## 最終目標

- 回測美股 ETF、台股、台灣 ETF 與台灣基金。
- 支援 buy-and-hold、DCA、再平衡、股息再投入、槓桿與防守策略。
- 建立可讀的 Markdown、CSV、HTML 報表與未來 UI。
- 嚴謹驗證回測正確性，避免只看全期間績效。
- 產生月度配置研究訊號，協助人工判斷目前 QQQ/QLD/TQQQ/CASH 或其他資產應如何配置。

## 近期主線

近期主線是美股 ETF，尤其是 QQQ family：

- `QQQ`：1x。
- `QLD`：2x。
- `TQQQ`：3x。
- `CASH`：防守或降曝險。

DCA Policy Optimizer 會在 `max_drawdown >= -95%` 的硬限制下，以 DCA XIRR、walk-forward、rolling cohort 與 synthetic stress 找候選策略。

## 長期市場範圍

- Phase 1：美股與美股 ETF ledger。
- Phase 2：槓桿策略、DCA policy optimizer、cohort validation、Optuna 搜尋加速。
- Phase 3：台股、台灣 ETF、台灣基金 NAV。
- Phase 4：Streamlit 或其他互動 UI。

## 非目標

- 不做自動交易。
- 不串券商。
- 不做高頻或日內交易。
- 不做期貨與選擇權。
- 不做完整個人稅務申報模擬。
- 不把 Optuna 或任何黑箱最佳化結果直接當作最終策略。
