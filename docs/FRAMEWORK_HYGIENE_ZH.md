# 框架維護規範

## 分層原則

新增功能時要維持分層：

1. data adapter。
2. data normalization。
3. strategy / policy generation。
4. ledger or simulator。
5. metrics / validation。
6. report / UI。

策略不要直接依賴 yfinance、FinMind 或任何資料來源 adapter。報表不要反向改動策略邏輯。

## Optimizer Hygiene

任何 optimizer 必須遵守：

- 策略規則要能解釋。
- 不得只用全期間 XIRR、CAGR 或 ending equity 排名。
- 必須輸出 policy CSV，包含每日 regime、reason、target leverage、權重與指標值。
- 必須做 no-lookahead 測試。
- 必須做 walk-forward 或 rolling cohort。
- 必須分開 actual ETF 與 synthetic stress 結論。
- Optuna 只能作搜尋加速，不得跳過驗證。

## 槓桿建模

- margin loan：用 debt、borrow rate、interest、maintenance requirement、safety buffer、margin call 建模。
- leveraged ETF product：只當價格序列資產，不宣稱已模擬 swap、futures、費用與產品內部細節。
- 兩者不得混在同一套會計邏輯。

## 報表要求

每個重要研究報表至少要有：

- HTML dashboard。
- metrics CSV。
- policy 或 audit CSV。
- 重要限制與假設。
- 人可以讀懂的情境名稱。
- 下載連結。

## 每次新增功能的檢查表

- 是否新增 public config。
- 是否有 golden tests。
- 是否有 regression tests。
- 是否保留 audit trail。
- 是否更新中文文件。
- 是否避免提交 secrets、cache、報表產物。
