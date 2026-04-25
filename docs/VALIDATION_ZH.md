# 驗證與排錯指南

這份文件記錄本專案的標準驗證命令與預期結果。

## 1. 系統工具驗證

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts\check_environment.ps1
python --version
py --list
git --version
uv --version
```

目前已驗證：

- Python 3.12.10
- Git 2.54.0
- uv 0.11.7
- Python launcher 可看到 Python 3.12
- Visual Studio C++ Build Tools 已安裝，且 `bt==1.1.5` 已成功 build

## 2. 依賴安裝驗證

```powershell
uv sync --extra dev
```

預期會安裝並可匯入：

- `pandas`
- `numpy`
- `vectorbt`
- `bt`
- `quantstats`
- `yfinance`
- `FinMind`

`bt` 在 Windows + Python 3.12 需要 C++ Build Tools。本機已成功 build `bt==1.1.5`。

## 3. 單元測試

```powershell
.\.venv\Scripts\python.exe -m pytest
```

目前結果：

```text
全部通過
```

測試涵蓋：

- YAML config 載入
- 台股/美股成本模型
- 績效指標
- 均線訊號
- DCA 現金流計算
- 美股 account ledger golden cases
- 美股 dividend data/cache golden cases
- 美股 ledger report golden cases
- 美股 DCA ledger golden cases
- 中文文件 UTF-8 與 scope 連結

## 4. Import Smoke Test

```powershell
.\.venv\Scripts\python.exe scripts\smoke_imports.py
```

目前結果：

```text
[OK] pandas
[OK] numpy
[OK] vectorbt
[OK] bt
[OK] quantstats
[OK] yfinance
[OK] FinMind
[OK] investment_backtest_lab
```

## 5. 離線 Prototype Demo

```powershell
.\.venv\Scripts\python.exe scripts\run_prototype.py --config configs\mvp_example.yaml --offline-demo
```

預期輸出：

- 均線策略最近訊號
- Signal performance summary
- DCA summary
- Rolling 3Y CAGR tail

這個 demo 使用 synthetic price，不需要網路與 API token。

## 6. Live Data Smoke Test

```powershell
.\.venv\Scripts\python.exe scripts\smoke_data.py --network
```

行為：

- 一定會測 `SPY`、`QQQ`、`USDTWD=X`
- 若有 `FINMIND_TOKEN`，會加測 `0050`、`2330`
- 若沒有 `FINMIND_TOKEN`，台股資料測試會跳過

目前已驗證：

```text
[OK] SPY: 61 rows
[OK] QQQ: 61 rows
[OK] USDTWD=X: 65 rows
```

目前未設定 `FINMIND_TOKEN`，所以 `0050`、`2330` 尚未測 live data。

## 7. 排錯順序

## 7. Ledger 報表驗證

```powershell
.\.venv\Scripts\python.exe scripts\analyze_ledger.py --config configs\mvp_example.yaml --tickers SPY QQQ
```

預期輸出：

- `reports/ledger_spy_qqq.md`
- `reports/ledger_spy_qqq_metrics.csv`
- `reports/ledger_spy_qqq_trades.csv`
- `reports/ledger_spy_qqq_dividends.csv`
- `reports/ledger_spy_qqq_cash_flows.csv`
- `reports/ledger_spy_qqq_equity.csv`
- `reports/ledger_spy_qqq_positions.csv`
- `reports/ledger_spy_qqq.html`

驗證重點：

- price source 應該是 raw price，不是 adjusted price。
- `strategy=ledger_buy_and_hold`、`strategy=ledger_dca`、`strategy=ledger_rebalance` 都應出現在 metrics。
- `dividend_mode=cash` 與 `dividend_mode=reinvest` 都應出現在 metrics。
- `basis=USD` 與 `basis=TWD` 都應出現在 metrics。
- dividends CSV 要能看到 gross dividend、withholding tax、net amount。
- cash flows CSV 要能看到 DCA 每月投入。
- positions CSV 要能看到 SPY/QQQ 的 quantity、market value、weight。
- HTML 要能打開並看到中文 dashboard shell、設定總覽、KPI、情境表、TWD/USD equity curve、drawdown、cash/market value、股息/稅/費用圖、權重漂移圖與 CSV 下載連結。

## 8. 排錯順序

若驗證失敗，建議順序：

1. 重開 PowerShell。
2. 跑 `scripts\check_environment.ps1`。
3. 確認 `python --version` 是 Python 3.12。
4. 確認 `uv --version` 可執行。
5. 重跑 `uv sync --extra dev`。
6. 若 `bt` build 失敗，重開機後再試。
7. 若 live data 失敗，先確認網路、Yahoo/FinMind 狀態與 `FINMIND_TOKEN`。
