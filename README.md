# Investment Backtest Lab

這是一個個人投資回測研究專案，用來驗證台股、台灣 ETF、美股與美股 ETF 的長期投資策略。第一版重點是建立可重現的研究環境和可測試的 prototype，不做實盤交易。

## 最快路線

第一次打開專案，請先看：

- [快速上手教學](docs/QUICKSTART_ZH.md)
- [專案 Scope](docs/PROJECT_SCOPE_ZH.md)
- [Roadmap](docs/ROADMAP_ZH.md)

最短驗證路線：

```powershell
uv sync --extra dev
uv run pytest
uv run python scripts\smoke_imports.py
uv run python scripts\run_prototype.py --config configs\mvp_example.yaml --offline-demo
uv run python scripts\analyze_ledger.py --config configs\mvp_example.yaml --tickers SPY QQQ
```

## 目前狀態

已完成：

- Python 3.12.10
- Git 2.54.0
- uv 0.11.7
- Microsoft Visual Studio C++ Build Tools
- `.venv` 虛擬環境
- 專案依賴安裝
- `bt==1.1.5` 編譯與匯入驗證
- 離線測試與 prototype demo
- yfinance live data smoke test：`SPY`、`QQQ`、`USDTWD=X` 通過
- FinMind live data smoke test：目前因未設定 `FINMIND_TOKEN` 而跳過
- quickstart SPY/QQQ 結果檢視報表
- 美股 account ledger 核心與 ledger 報表 v1
- git baseline commit 已建立，後續里程碑可回溯

驗證結果：

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\smoke_imports.py
.\.venv\Scripts\python.exe scripts\run_prototype.py --config configs\mvp_example.yaml --offline-demo
.\.venv\Scripts\python.exe scripts\analyze_ledger.py --config configs\mvp_example.yaml --tickers SPY QQQ
```

預期結果：

- `pytest`: 全部通過
- `smoke_imports.py`: `pandas`、`vectorbt`、`bt`、`quantstats`、`yfinance`、`FinMind` 都顯示 `[OK]`
- `run_prototype.py`: 會輸出均線策略摘要、DCA 摘要、Rolling 3Y CAGR
- `analyze_ledger.py`: 會輸出 raw price + 股息 + 稅 + 成本的可審計 ledger 報表

## 專案目標

本專案先做研究級回測實驗室，不先做產品級 UI。第一版先把美股與美股 ETF 做完整，包含價格、股息、匯率、成本、稅與可審計 ledger；美股核心穩定後，再擴充台股、台灣 ETF 與台灣基金。UI 方向先採 Streamlit。

第一版聚焦：

- 美股與美股 ETF：透過 `yfinance`
- 台股與台灣 ETF：透過 `FinMind`
- 台灣基金：第二階段用 NAV CSV/parquet 接入
- 績效預設以 TWD 呈現，並保留 USD 原幣參考
- 成本模型處理交易層費用，不做完整個人稅務模擬
- ledger 作為嚴謹現金流與 audit trail 主線；`vectorbt` / `bt` 保留作研究與交叉驗證

## 專案結構

```text
configs/                  回測設定檔
data/                     raw/cache/processed 資料目錄說明
docs/                     中文安裝與驗證指南
notebooks/                Jupyter 研究入口
scripts/                  環境檢查、smoke test、prototype runner
src/investment_backtest_lab/  核心 Python package
tests/                    離線測試
```

## 常用命令

如果你剛重開 PowerShell，可以先確認工具：

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts\check_environment.ps1
python --version
git --version
uv --version
```

建立或同步環境：

```powershell
uv venv --python 3.12
uv sync --extra dev
```

如果目前 terminal 找不到 `uv`，先重開 PowerShell。若仍找不到，可以暫時用完整路徑：

```powershell
& "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe" sync --extra dev
```

跑完整離線驗證：

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\smoke_imports.py
.\.venv\Scripts\python.exe scripts\run_prototype.py --config configs\mvp_example.yaml --offline-demo
.\.venv\Scripts\python.exe scripts\analyze_ledger.py --config configs\mvp_example.yaml --tickers SPY QQQ
```

## FinMind Token

台股與台灣 ETF 會使用 `FinMind`。Token 不要寫進程式碼，請用環境變數：

```powershell
$env:FINMIND_TOKEN = "你的 token"
```

若沒有 `FINMIND_TOKEN`，台股 live data smoke test 會跳過，但離線測試與 CSV adapter 仍可使用。

測試 live data：

```powershell
.\.venv\Scripts\python.exe scripts\smoke_data.py --network
```

若在 Codex 沙盒內執行 live data，可能需要 elevated network permission；一般 PowerShell 通常不需要。

## bt 為什麼需要 C++ Build Tools

`bt` 是投組目標權重、月/季再平衡與配置型策略的重要框架。它不是第一個訊號策略 prototype 的必要條件，但對「多資產配置與再平衡」很有用。

在 Windows + Python 3.12 上，`bt==1.1.5` 目前需要本機編譯 C extension，因此要安裝 Microsoft Visual Studio C++ Build Tools。這台環境已經安裝並成功 build `bt`。

## 文件

- [快速上手教學](docs/QUICKSTART_ZH.md)
- [專案 Scope](docs/PROJECT_SCOPE_ZH.md)
- [Roadmap](docs/ROADMAP_ZH.md)
- [正確性驗證策略](docs/VALIDATION_STRATEGY_ZH.md)
- [環境設定指南](docs/SETUP_ZH.md)
- [驗證與排錯指南](docs/VALIDATION_ZH.md)

## 參考來源

- Python Windows 文件：https://docs.python.org/3.12/using/windows.html
- uv 安裝文件：https://docs.astral.sh/uv/getting-started/installation/
- Git for Windows：https://git-scm.com/install/windows
- Visual Studio command-line install：https://learn.microsoft.com/en-us/visualstudio/install/use-command-line-parameters-to-install-visual-studio
- FinMind：https://finmind.github.io/
- yfinance：https://ranaroussi.github.io/yfinance/
