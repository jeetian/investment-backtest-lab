# 環境設定指南

這份文件記錄 Windows 上建立本專案研究環境的標準流程。

## 1. 系統工具

本專案使用：

- Python 3.12
- Git
- uv
- Microsoft Visual Studio C++ Build Tools

已使用 `winget` 安裝：

```powershell
winget install --id Python.Python.3.12 -e --source winget --scope machine --accept-source-agreements --accept-package-agreements
winget install --id Git.Git -e --source winget --scope machine --accept-source-agreements --accept-package-agreements
winget install --id astral-sh.uv -e --source winget --accept-source-agreements --accept-package-agreements
winget install -e --id Microsoft.VisualStudio.2022.BuildTools --source winget --accept-source-agreements --accept-package-agreements --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

Build Tools installer 可能會提示重新啟動。即使目前已能 build `bt`，若之後 C++ 編譯出現奇怪錯誤，先重開機再重跑 `uv sync --extra dev`。

## 2. 驗證系統工具

重開 PowerShell 後執行：

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts\check_environment.ps1
python --version
py --list
git --version
uv --version
```

預期：

- `python --version` 顯示 Python 3.12.x
- `py --list` 看得到 Python 3.12
- `git --version` 顯示 Git 版本
- `uv --version` 顯示 uv 版本

如果 `uv` 找不到，先重開 PowerShell。若仍找不到，可暫時用完整路徑：

```powershell
& "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe" --version
```

## 3. 建立虛擬環境

```powershell
uv venv --python 3.12
uv sync --extra dev
```

若目前 terminal 找不到 `uv`，可以改用完整路徑：

```powershell
& "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe" venv --python 3.12
& "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe" sync --extra dev
```

## 4. FinMind Token

台股 live data 需要 `FINMIND_TOKEN`：

```powershell
$env:FINMIND_TOKEN = "你的 token"
```

若要永久設定，可以使用 Windows 使用者環境變數。不要把 token 寫進 `.env.example` 或任何會提交到 git 的檔案。

## 5. 常見狀況

### `python` 指到 WindowsApps alias

這代表系統尚未正確找到 Python 安裝。重開 PowerShell 後再試：

```powershell
python --version
py -3.12 --version
```

### `bt` build 失敗

確認 Build Tools 已安裝並重開機。然後重跑：

```powershell
uv sync --extra dev
```

### Matplotlib cache 權限問題

`scripts/smoke_imports.py` 已把 `MPLCONFIGDIR` 指到專案內 `.cache/matplotlib`，正常情況下不需要手動處理。

### Windows pytest 暫存資料夾權限問題

如果曾經用不同權限層級執行 pytest，Windows 可能留下無法讀取的
`%TEMP%\pytest-of-<user>` 資料夾。症狀是測試在 `tmp_path` setup 階段出現
`PermissionError [WinError 5]`，但多數測試本身沒有 assertion failure。

可用 fresh basetemp 避開舊 ACL：

```powershell
uv run pytest --basetemp=C:\Users\Ian Lai\Desktop\Python\pytest-fresh-20260504
```

若要清理舊資料夾，請用 Windows 檔案總管或系統管理員 PowerShell 檢查 ACL 後再刪除；
不要把 pytest 暫存資料夾加入 git。
