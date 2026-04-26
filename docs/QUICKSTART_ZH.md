# 快速上手教學

這份文件是本專案的第一條路線：從確認環境開始，跑到第一個離線回測結果，再進一步測 yfinance live data。照著做一輪，你就會知道目前 prototype 能做什麼、哪裡可以開始改。

## 0. 你會跑出什麼

完成後你會得到：

- 一個可用的 Python 3.12 + uv 虛擬環境
- 一次完整測試結果：全部通過
- 一次 import smoke test：`vectorbt`、`bt`、`quantstats`、`yfinance`、`FinMind` 都可匯入
- 一次離線 prototype 結果：均線策略摘要、DCA 摘要、Rolling 3Y CAGR
- 一次 yfinance live data 檢查：`SPY`、`QQQ`、`USDTWD=X`
- 一份可審計的美股 ledger 報表：raw price、股息、稅、成本、DCA 現金流、USD/TWD
- 一份美股 ETF 輕槓桿風險報表：margin loan、每日利息、安全緩衝、自動降槓桿

## 1. 1 分鐘確認環境

在專案根目錄執行：

```powershell
python --version
uv --version
git --version
PowerShell -ExecutionPolicy Bypass -File scripts\check_environment.ps1
```

預期看到：

- Python 3.12.x
- uv 0.11.x
- Git 2.54.x
- Visual Studio Build Tools 已找到

如果 `uv` 找不到，先重開 PowerShell。若還是不行，暫時用完整路徑：

```powershell
& "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe" --version
```

## 2. 3 分鐘同步依賴

```powershell
uv sync --extra dev
```

這會依照 `pyproject.toml` 和 `uv.lock` 建立或更新 `.venv`。

如果你想確認 lock file 沒有被意外改動，可以用：

```powershell
uv sync --extra dev --locked
```

## 3. 5 分鐘跑完整離線驗證

```powershell
uv run pytest
uv run python scripts\smoke_imports.py
uv run python scripts\run_prototype.py --config configs\mvp_example.yaml --offline-demo
```

預期結果：

- `pytest` 顯示全部通過
- `smoke_imports.py` 每個套件都顯示 `[OK]`
- `run_prototype.py` 輸出：
  - `Moving-average signal sample`
  - `Signal performance summary`
  - `DCA summary`
  - `Rolling 3Y CAGR tail`

這個離線 demo 使用 synthetic price，不需要網路，也不需要 FinMind token。

## 4. 5 分鐘理解設定檔

第一個要看的設定檔是：

```text
configs/mvp_example.yaml
```

幾個最重要的區塊：

```yaml
base_currency: TWD
start_date: "2020-01-01"
end_date: "2025-12-31"
```

這決定回測期間和主要報表幣別。

```yaml
universe:
  - ticker: SPY
    market: US
    data_source: yfinance
  - ticker: "0050"
    market: TW
    data_source: finmind
```

這決定資產池。美股先用 `yfinance`，台股先用 `FinMind`。

```yaml
strategy:
  name: moving_average_timing
  params:
    fast_window: 50
    slow_window: 200
```

這是第一個可以改的策略參數。想讓均線更敏感，可以降低 `fast_window` 或 `slow_window`；想讓策略更慢、更長期，就提高它們。

```yaml
rebalance:
  frequency: monthly
  target_weights:
    SPY: 0.60
    QQQ: 0.40
```

這是多資產 ledger 會用到的配置型策略設定。v1 先只做 SPY/QQQ 美股再平衡，台股與基金等美股核心穩定後再接。

## 5. 5 分鐘跑 yfinance live data

```powershell
uv run python scripts\smoke_data.py --network
```

目前沒有 `FINMIND_TOKEN` 時，台股會跳過，但仍會測：

- `SPY`
- `QQQ`
- `USDTWD=X`

成功時會看到類似：

```text
[OK] SPY: 61 rows
[OK] QQQ: 61 rows
[OK] USDTWD=X: 65 rows
```

資料會快取到：

```text
data/cache/
```

下次再跑時，會優先從 parquet cache 讀取。

## 6. 抓完資料後怎麼看結果

`smoke_data.py` 只回答一件事：資料管線能不能抓資料、能不能寫入 cache。看到 `[OK] SPY: 61 rows` 不代表已經完成回測，只代表資料已經能讀。

要看 SPY/QQQ 的資料品質與策略結果，請跑：

```powershell
uv run python scripts\analyze_results.py --config configs\mvp_example.yaml --tickers SPY QQQ
```

這個命令會做幾件事：

- 檢查 SPY/QQQ 的資料筆數、日期範圍、缺值、重複日期、異常價格。
- 用 `2020-01-01` 到 `2025-12-31` 的資料跑結果，而不是只看 quickstart smoke test 的 2024 Q1。
- 比較 SPY/QQQ 的 buy-and-hold。
- 比較 SPY/QQQ 的均線策略。
- 比較 SPY/QQQ 的 DCA 結果。
- 同時輸出 USD 原幣績效和 TWD 換匯後績效。

輸出檔案會在：

```text
reports/quickstart_spy_qqq.md
reports/quickstart_spy_qqq_metrics.csv
```

先打開 `reports/quickstart_spy_qqq.md`。它會告訴你：

- `buy_and_hold`：買進並持有結果，拿來當 benchmark。
- `moving_average_50_200`：50/200 均線進出結果。
- `dca`：定期投入結果，重點看 `ending_equity`、`total_contributed`、`simple_cash_return`。
- `basis=USD`：原幣績效。
- `basis=TWD`：用 USD/TWD 匯率換算後的台幣績效。

## 7. 跑可審計 Ledger 報表

`analyze_results.py` 適合快速比較策略；`analyze_ledger.py` 則用 raw price 加上明確股息現金流，避免 adjusted price 和股息重複計算。這份報表是後續嚴謹回測的主線。

```powershell
uv run python scripts\analyze_ledger.py --config configs\mvp_example.yaml --tickers SPY QQQ
```

輸出檔案會在：

```text
reports/ledger_spy_qqq.md
reports/ledger_spy_qqq_metrics.csv
reports/ledger_spy_qqq_trades.csv
reports/ledger_spy_qqq_dividends.csv
reports/ledger_spy_qqq_cash_flows.csv
reports/ledger_spy_qqq_equity.csv
reports/ledger_spy_qqq_positions.csv
reports/ledger_spy_qqq_rebalance.csv
reports/ledger_spy_qqq.html
```

先打開 `reports/ledger_spy_qqq.md`。它會告訴你：

- `dividend_mode=cash`：股息扣除美股預扣稅後留現金。
- `dividend_mode=reinvest`：股息扣稅後用對齊後交易日收盤價再投入。
- `strategy=ledger_buy_and_hold`：期初一次投入後長期持有。
- `strategy=ledger_dca`：每月第一個可交易日投入 `dca.contribution`，目前範例是 1,000 USD，並把每次外部投入記錄到 cash flows。
- `strategy=ledger_rebalance`：期初投入 10,000 USD 到 SPY 60% / QQQ 40%，每月第一個可交易日拉回目標權重。
- `total_contributed`：投入本金；DCA 的 TWD 版本會用投入日 USD/TWD 匯率換算。
- `simple_cash_return`：期末資產除以投入本金後的簡單現金報酬，適合 DCA 第一版閱讀。
- `gross_dividends`：收到的稅前股息。
- `withholding_tax`：美股股息預扣稅。
- `fees_paid`：交易成本。
- `final_shares`：最後持股數，會反映再投入。
- `final_weights`：再平衡投組最後的 SPY/QQQ 權重。
- `cash`：最後現金餘額。

想看圖表，打開：

```text
reports/ledger_spy_qqq.html
```

新版 HTML 是第一版投資 dashboard。建議閱讀順序：

- 第一屏先看「設定總覽」：期間、標的、策略、DCA 金額、初始資金、再平衡權重、股息模式、基準幣別、資料來源、成本與稅率。
- 再看「關鍵績效」：期末資產、投入本金、simple cash return、最大回撤、股息、預扣稅、費用與最後持股。
- 「策略與股息模式比較」會保留 `ledger_buy_and_hold`、`ledger_dca`、`ledger_rebalance`、`cash`、`reinvest`，方便回到 CSV 追查。
- 「再平衡讀法」會列出每次再平衡的交易數、買賣方向、交易金額、調整後權重與買賣原因。
- 圖表圖例會使用短名稱，例如 `SPY B&H 現金`、`QQQ DCA 再投`、`SPY_QQQ Rebal 現金`；完整描述可看 hover 或上方情境表。
- 最下方「審計明細與 CSV 下載」可以打開 trades、dividends、cash flows、equity、positions、rebalance 等明細檔。

目前 v1 規則：yfinance dividend date 先視為可入帳日期；若遇到非交易日，會對齊到下一個可交易日。精確 ex-date/payment-date 差異會在後續版本強化。

如果只想跑其中一種 ledger 策略，可以加上 `--strategies`：

```powershell
uv run python scripts\analyze_ledger.py --config configs\mvp_example.yaml --tickers SPY QQQ --strategies dca
```

## 8. 跑輕槓桿風險報表

槓桿報表目前是美股 ETF margin loan 研究模型，預設 1.3x 目標槓桿、6.5% 年化借款利率、35% 維持率、安全緩衝低於 25 percentage points 時自動降到 1.1x。

```powershell
uv run python scripts\analyze_leverage.py --config configs\mvp_example.yaml --tickers SPY QQQ
```

輸出檔案會在：

```text
reports/leverage_spy_qqq.md
reports/leverage_spy_qqq_metrics.csv
reports/leverage_spy_qqq_trades.csv
reports/leverage_spy_qqq_interest.csv
reports/leverage_spy_qqq_events.csv
reports/leverage_spy_qqq_curve.csv
reports/leverage_spy_qqq.html
```

先看 `metrics.csv` 或 HTML 裡的 `worst_safety_buffer`、`margin_call_count`、`forced_deleverage_count`、`interest_paid`。這份報表第一版用來確認「會不會太接近爆倉」，不是用來直接挑最高 CAGR。

目前 v1 支援 `buy_hold_leveraged`、`dca_leveraged` 與 SPY/QQQ 60/40 的 `rebalance_leveraged`。槓桿 ETF 產品先當一般價格序列，不混入 margin loan 借款模型。

## 9. 設定 FinMind Token

台股與台灣 ETF live data 需要 FinMind token。先用暫時環境變數：

```powershell
$env:FINMIND_TOKEN = "你的 token"
uv run python scripts\smoke_data.py --network
```

有 token 後會額外測：

- `0050`
- `2330`

不要把 token 寫進任何會提交的檔案。

## 9. 第一個可以改的策略

從均線參數開始最安全：

```yaml
strategy:
  name: moving_average_timing
  params:
    fast_window: 20
    slow_window: 120
    init_cash: 10000
```

改完後先跑離線 demo：

```powershell
uv run python scripts\run_prototype.py --config configs\mvp_example.yaml --offline-demo
```

這個 demo 目前是 synthetic price，所以目的是確認流程和輸出，不是做真實投資結論。

## 10. 第一個可以改的資產池

美股可以先改成其他 yfinance ticker，例如：

```yaml
universe:
  - ticker: VTI
    name: Vanguard Total Stock Market ETF
    market: US
    asset_type: etf
    currency: USD
    data_source: yfinance
  - ticker: VXUS
    name: Vanguard Total International Stock ETF
    market: US
    asset_type: etf
    currency: USD
    data_source: yfinance
```

台股可保留 `0050`、`2330`，等 `FINMIND_TOKEN` 設好後再測 live data。

## 11. 常見錯誤

### 找不到 `uv`

先重開 PowerShell，再跑：

```powershell
uv --version
```

若仍找不到，用完整路徑：

```powershell
& "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe" sync --extra dev
```

### 沒有 `FINMIND_TOKEN`

這不是錯誤。沒有 token 時台股 live data 會跳過，美股與離線測試仍可跑。

### `bt` build 失敗

`bt` 在 Windows + Python 3.12 需要 Microsoft Visual Studio C++ Build Tools。若剛安裝完 Build Tools，請先重開機，再跑：

```powershell
uv sync --extra dev
```

### Yahoo 連線失敗

`yfinance` 可能因網路、Yahoo 暫時限制或防火牆失敗。先重跑一次：

```powershell
uv run python scripts\smoke_data.py --network
```

若仍失敗，先確認瀏覽器可連 Yahoo Finance，再檢查公司/學校網路是否擋外部 API。

## 12. 下一步建議

照這份 quickstart 跑通後，下一個自然步驟是：

- 擴充再平衡報表，檢查權重漂移、買賣原因與交易摩擦成本。
- 加入 FinMind token 後驗證 `0050`、`2330`。
- Phase 1 收尾時另開一個 chat 做冷讀驗證。
