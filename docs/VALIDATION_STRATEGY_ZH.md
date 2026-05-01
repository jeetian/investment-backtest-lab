# 驗證策略

本專案的回測結果必須可審計、可重現、可被壓力測試。漂亮報酬不是足夠理由，尤其是槓桿策略。

## L1：資料驗證

- OHLCV 欄位完整。
- 日期排序正確，沒有重複日期。
- 價格與成交量沒有明顯不合理值。
- yfinance / FinMind / CSV cache 可重跑。
- raw price 與 adjusted price 使用情境必須明確標示。

## L2：Golden Tests

用小型合成資料手算：

- buy-and-hold 的 shares、cash、market value、total equity。
- DCA 投入次數、累計投入、期末資產。
- 股息、預扣稅、再投入股數。
- 再平衡後權重與交易成本。
- margin loan 的 debt、interest、equity ratio、safety buffer。
- leveraged ETF product 的 daily reset synthetic price。

## L3：Cross-Tool Validation

使用 `scripts/cross_validate.py`：

- AccountLedger vs vectorbt。
- PortfolioLedger vs bt。
- raw price + dividend reinvestment vs adjusted total-return price。

## L4：防過擬合驗證

槓桿與 optimizer 策略必須加做：

- no-lookahead：所有技術指標使用前一日已知資料。
- walk-forward：train period 選策略，test period 重新跑 DCA。
- rolling cohort：不同起點與不同持有期間都要檢查。
- synthetic stress：用 QQQ 合成 2x/3x daily reset，補 2000/2008 類壓力情境。
- cross-mode filter：actual ETF 很漂亮但 synthetic stress 穿越硬線，不得成為正式候選。

## DCA Policy Optimizer 驗證口徑

- 主要排名指標：DCA XIRR。
- 輔助指標：ending equity、simple cash return、max drawdown、recovery days。
- 硬淘汰：`max_drawdown < -95%`。
- 高風險帶：`-85%` 到 `-95%`，保留排名但明確標記。
- 月度配置訊號只能從通過 walk-forward、cohort 與 cross-mode filter 的候選策略產生。

## 常用命令

```powershell
uv run pytest
uv run ruff check src tests scripts
uv run python scripts\cross_validate.py
uv run python scripts\analyze_dca_policy_optimizer.py --config configs\mvp_example.yaml --family qqq --scan-mode fast --cohort-validation
```
