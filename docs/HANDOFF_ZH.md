# Codex 交接紀錄

更新日期：2026-05-04

## 目前 checkpoint

本專案目前停在一個可交接的穩定點：

- Git commit：請用 `git log -1 --oneline` 確認最新 checkpoint。
- 已新增 `Synthetic-Primary Monthly Replay Ranking`。
- 新增 CLI：`scripts/analyze_monthly_decision_replay.py`。
- 新增核心模組：`src/investment_backtest_lab/monthly_decision_replay.py`。
- 新增測試：`tests/test_monthly_decision_replay.py`。
- `configs/mvp_example.yaml` 已新增 `monthly_decision_replay` 設定。
- `docs/QUICKSTART_ZH.md` 與 `docs/ALLOCATION_WORKFLOW_ZH.md` 已重寫成可讀中文。

這個 replay 流程不覆蓋既有 Monthly Decision Pack，而是新增一份壓測優先的對照報表。

## 最新驗證結果

已執行並通過：

```powershell
uv run ruff check src tests scripts
uv run pytest
uv run python scripts\analyze_monthly_decision_replay.py --config configs\mvp_example.yaml --family qqq --selector synthetic_primary
```

結果摘要：

- Ruff：通過。
- Pytest：`107 passed`。
- Monthly replay CLI：成功產生報表。

Replay 報表輸出：

- `reports/monthly_decision_replay_qqq.html`
- `reports/monthly_decision_replay_qqq_decisions.csv`
- `reports/monthly_decision_replay_qqq_cohorts.csv`
- `reports/monthly_decision_replay_qqq_ranking.csv`
- `reports/monthly_decision_replay_qqq_equity.csv`

本次 synthetic-primary replay 的前幾名候選：

1. `Vol Target 63D 35%`
2. `Momentum+Trend 126D/200MA 2.0x to 1.0x`
3. `Trend Ladder 200MA 2.0x to 1.0x`
4. `Vol Target 63D 25%`
5. `Momentum+Trend 126D/200MA 3.0x to 1.0x`

這和目前 actual-primary 月度訊號可能不同，代表「壓測優先」確實會影響決策排序。

## 重要設計決定

目前有兩條月度決策邏輯：

- `monthly_decision_pack_qqq.html`：actual-primary 主流程。actual ETF ranking 優先，synthetic stress 主要作為淘汰或警示。
- `monthly_decision_replay_qqq.html`：synthetic-primary 對照流程。以 synthetic stress ranking 與 cohort replay 穩定性優先。

Synthetic-primary replay 的 ranking 不看單一 ending equity，也不只看全期間 XIRR。主要依據是：

- `win_rate_vs_qqq_dca`
- `drawdown_breach_rate`
- `worst_max_drawdown`
- `median_xirr`
- `worst_xirr`
- `replay_score`

Benchmark 是同起點、同終點、同投入金額的 `QQQ DCA`。

## 轉移到另一台電腦

建議步驟：

1. 安裝 Python 3.12、Git、uv。
2. 複製 repo 或解壓備份包。
3. 在專案根目錄執行：

```powershell
uv sync --extra dev
uv run pytest
uv run python scripts\analyze_dca_policy_optimizer.py --config configs\mvp_example.yaml --family qqq --scan-mode fast --cohort-validation
uv run python scripts\analyze_monthly_decision_replay.py --config configs\mvp_example.yaml --family qqq --selector synthetic_primary
uv run python scripts\analyze_monthly_decision_pack.py --config configs\mvp_example.yaml --family qqq
```

若只想快速確認 replay 功能，可先跑：

```powershell
uv run python scripts\analyze_monthly_decision_replay.py --config configs\mvp_example.yaml --family qqq --selector synthetic_primary
```

注意：`reports/`、`data/cache/`、`.venv/` 都不進 git。換電腦後需要重新產生報表與資料快取。

## 本機備份檔案

已在專案上一層建立兩個備份：

- Git bundle：`C:\Users\Hsuan\Desktop\Market_Investment\investment-backtest-lab_latest_handoff_20260504.bundle`
- Checkpoint zip：`C:\Users\Hsuan\Desktop\Market_Investment\investment-backtest-lab_latest_handoff_20260504.zip`

用途：

- `.bundle` 保留 git 歷史，適合在另一台電腦還原成 git repo。
- `.zip` 包含目前 source、docs、tests、scripts、configs 與 `reports/`，適合直接解壓查看報表。

在另一台電腦還原 git bundle：

```powershell
git clone investment-backtest-lab_latest_handoff_20260504.bundle investment-backtest-lab
cd investment-backtest-lab
uv sync --extra dev
```

## 下一步建議

1. 檢查 `monthly_decision_replay_qqq.html` 的 UI 是否足夠易讀。
2. 把 synthetic-primary replay 的 top candidate 和 actual-primary monthly pack 做並排比較。
3. 決定是否把 monthly decision pack 的正式 selector 改成 synthetic-primary，或先維持雙軌。
4. 若 replay runtime 太久，做 cohort 計算快取或 `--quick-horizons` 模式。
5. 等 synthetic-primary 流程穩定後，再導入 Optuna 作搜尋加速，但 Optuna 不能跳過 walk-forward、cohort replay 與 synthetic stress。
