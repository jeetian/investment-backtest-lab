# 結果入口

專案位置：

```text
C:\Users\Ian Lai\Desktop\Python\investment-backtest-lab
```

報表位置：

```text
C:\Users\Ian Lai\Desktop\Python\investment-backtest-lab\reports
```

## 平常先看這個

1. `reports\monthly_decision_comparison_qqq.html`
   - 這是目前最重要的「一頁決策包」。
   - 第一屏會直接回答：research authority、actionable default、可交易權重、是否需要人工 review、review 原因。
   - research authority 來自 `hybrid_primary` Monte Carlo expected-XIRR ranking。
   - actionable default 是未人工 override 前的 zero-breach 可行動預設。

2. `reports\monthly_decision_comparison_qqq_top_candidates.csv`
   - 日常檢查 top candidates 用。
   - 比大型 MC trials 好讀很多。

3. `reports\monthly_decision_replay_qqq_mc_summary.csv`
   - Monte Carlo trials 的分層摘要。
   - 日常查 expected XIRR、p05 XIRR、win rate、drawdown breach rate 時看這個。

4. `reports\monthly_decision_review_qqq.md`
   - 月度人工 review 紀錄。
   - 用來留下本月是 `pending_review`、`accepted`、`deferred`、`rejected` 或 `override`。
   - 這份檔案只記錄人工判斷，不會改變 ranking 或權重計算。

## 目前最新結果

- 本次資料更新目標：`2026-04-30` 月底收盤。
- 設定檔使用 `end_date: 2026-05-01`，因為 yfinance 的 `end` 是排他式，用來包含 `2026-04-30`。
- 重新產生 comparison 時會自動檢查 `recommended_as_of_date` 是否為 `2026-04-30`。
- 決策 authority：`hybrid_primary_monte_carlo`
- Research authority：`Momentum+Trend 126D/200MA 3.0x to 1.0x`
- Actionable default：`Vol Target 63D 25%`
- 未人工 override 前的 actual ETF 可交易權重：`QQQ 73% / QLD 27% / TQQQ 0% / CASH 0%`
- actual-primary 參考策略：`Momentum+Trend 126D/200MA 3.0x to 1.0x`
- Research authority 的 MC breach rate 約 `3.83%`，expected XIRR 約 `21.78%`；若要採用它，必須用 review `override`。
- Actionable default 的 MC breach rate 是 `0%`，expected XIRR 約 `16.30%`。
- actual-primary 與 hybrid-primary 不同不是程式錯誤，而是參考訊號分歧。未 override 前以 actionable default 為可行動預設。

## 大型審計檔

平常不需要直接打開這些檔：

- `reports\monthly_decision_replay_qqq_mc_trials.csv.gz`
  - 完整 Monte Carlo trials 壓縮審計檔。
  - full replay 原始 CSV 約 302 MB，已改成 gzip 輸出，避免日常誤開。

- `reports\monthly_decision_replay_qqq_cohorts.csv`
  - deterministic rolling cohort gate 的明細。
  - 用來審計 cohort gate，不是日常決策入口。

- `reports\dca_policy_optimizer_qqq_policy.csv`
  - 每日 policy state 明細，檔案可能很大。
  - 只有追查某一天權重來源時才需要看。

## 常用命令

先進專案資料夾：

```powershell
cd "C:\Users\Ian Lai\Desktop\Python\investment-backtest-lab"
```

重新產生一頁決策包：

```powershell
python -m uv run python scripts\analyze_monthly_decision_comparison.py --config configs\mvp_example.yaml --family qqq
```

月度流程最後一步：產生人工 review record：

```powershell
python -m uv run python scripts\record_monthly_decision_review.py --family qqq --status pending_review --reviewer "Ian"
```

若你已人工確認仍接受 recommendation，可以改成：

```powershell
python -m uv run python scripts\record_monthly_decision_review.py --family qqq --status accepted --reviewer "Ian" --notes "Reviewed actual-primary divergence."
```

若你要刻意採用 research authority 而非 zero-breach actionable default，必須明確 override：

```powershell
python -m uv run python scripts\record_monthly_decision_review.py --family qqq --status override --selected-layer research_authority --reviewer "Ian" --notes "Explicitly accepting research-authority tail risk."
```

重新跑 actual-primary 月度參考包：

```powershell
python -m uv run python scripts\analyze_monthly_decision_pack.py --config configs\mvp_example.yaml --family qqq
```

重新跑 hybrid-primary replay fast 版：

```powershell
python -m uv run python scripts\analyze_monthly_decision_replay.py --config configs\mvp_example.yaml --family qqq --selector hybrid_primary --fast
```

重新跑 hybrid-primary replay full 版：

```powershell
python -m uv run python scripts\analyze_monthly_decision_replay.py --config configs\mvp_example.yaml --family qqq --selector hybrid_primary --full
```

驗證：

```powershell
python -m uv run ruff check src tests scripts
python -m uv run pytest --basetemp=C:\Users\Ian Lai\Desktop\Python\pytest-fresh-20260505
```

## Runtime

- `hybrid_primary --fast`：約 10 分鐘。
- `hybrid_primary --full`：目前觀察約 49 分鐘。
- 主要瓶頸是 deterministic rolling cohort gate，不是 Monte Carlo 抽樣本身。
- CLI 會印開始時間、selector、scan mode、horizons、coverage、重要階段進度、總耗時與輸出檔案。完整 replay 完成後，結果才適合拿來做人工決策檢查。
