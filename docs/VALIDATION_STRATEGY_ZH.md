# 正確性驗證策略

本專案的驗證分成 L1、L2、L3。原則是先驗證資料與現金流，再驗證框架結果。

## L1：資料品質驗證

目的：確認輸入資料沒有明顯錯誤。

檢查項目：

- 日期範圍是否符合 config。
- 是否有缺值。
- 是否有重複日期。
- 價格是否小於等於零。
- 是否有異常大漲跌。
- 是否有足夠資料計算策略需要的 window。
- cache 檔案是否可重複讀取。
- 匯率資料是否可對齊交易日。

L1 失敗時，報表要明確顯示警告，不可以默默輸出誤導結果。

## L2：Golden Case 驗證

目的：用人工可手算的小案例驗證核心邏輯。

必備案例：

- 買進 10 股，價格上漲後 `cash + market value = total equity`。
- 賣出時計算現金、持股、費用與稅。
- 現金股息入帳時計算 gross dividend、withholding tax、net cash。
- 股息再投入時計算新買入股數與剩餘現金。
- DCA 在每月第一個可交易日投入。
- FX 換算能對齊日期並保留原幣結果。

Golden case 應該小到可以用人眼檢查，不依賴外部資料來源。

## L3：Cross-tool Validation

目的：用不同工具或不同實作互相校驗。

候選方式：

- Account ledger 手算結果 vs `vectorbt`。
- 配置再平衡結果 vs `bt`。
- 報酬與 drawdown 指標 vs `quantstats`。
- yfinance adjusted price vs 明確股息再投入流程。

L3 不要求一開始完整完成，但每當核心邏輯變複雜，例如槓桿、台股除權息、基金配息，就應該補上代表案例。

## 驗證命令

常用命令：

```powershell
uv run pytest
uv run python scripts\smoke_imports.py
uv run python scripts\run_prototype.py --config configs\mvp_example.yaml --offline-demo
uv run python scripts\analyze_results.py --config configs\mvp_example.yaml --tickers SPY QQQ
```

新增功能時，至少要跑 `uv run pytest`。若修改資料或報表流程，還要跑 quickstart report。

## 報表驗證原則

每份報表都要能回答：

- 用了哪些資料來源。
- 日期範圍是什麼。
- 策略如何解讀。
- 結果是原幣還是 TWD。
- 成本、稅與匯率如何處理。
- 哪些地方目前只是 prototype 限制。
