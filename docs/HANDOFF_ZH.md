# 接手說明

## 目前狀態

- 專案已可在這台電腦用 `python -m uv` 執行。
- TW50 目前資料 cutoff 設為 `2026-05-09`，用來包含 `2026-05-08` 收盤。
- TW50 V5 正式執行口徑已改為月度主 rebalance + 週度差距觸發；DCA 現金流仍維持 monthly contribution。
- comparison CLI 會檢查 `recommended_as_of_date` 是否符合 config end-exclusive cutoff。
- 月度正式 replay selector 已改成 `hybrid_primary`。
- `hybrid_primary` 的資料邏輯是：ETF 上市後使用 actual ETF，上市前使用 scaled synthetic backfill。
- 正式策略排名由 Monte Carlo replay 主導，deterministic rolling cohort 保留為硬風控 gate 與審計軌跡。
- 月度報表採雙層語意：research authority 追求 expected XIRR，actionable default 要求 zero MC breach。
- optimizer、monthly replay、comparison 預設改為 `net_of_cost`。成本摘要會包含 `total_trade_cost`、`cost_drag_on_contributed`、`turnover_sum`。
- 目前最重要的結果入口是 `reports/monthly_decision_comparison_qqq.html`。
- TW50 含息 MVP 已加入：`configs/tw50_example.yaml`，family 是 `0050 / 00631L / CASH`。
- TW50 使用稅前總報酬與配息再投入；`tw50_total_return_sources.csv` 與 `tw50_dividend_audit.csv` 是資料來源審計入口。
- TW50 現在是含息、含台股 ETF 手續費/證交稅/slippage、未含整股/零股限制的研究口徑。
- TW50 進入策略搜尋前，先跑 pre-optimization audit；若 audit 有 `fail`，Optuna 搜尋預設會拒絕執行。
- TW50 Optuna V5 是 monthly-core + weekly-delta / core external / return-first research search；正式 baseline 是固定 `1.5x DCA`，候選策略仍必須回到既有 replay、comparison、review record 流程人工確認。

## 最新觀察結果

### QQQ

- replay authority：`hybrid_primary_monte_carlo`
- research authority：`Momentum+Trend 126D/200MA 3.0x to 1.0x`
- actionable default：`Vol Target 63D 25%`
- latest actual ETF actionable 權重：`QQQ 73% / QLD 27% / TQQQ 0% / CASH 0%`
- research authority 有 MC breach，需要人工 override 才能採用；actionable default 是 zero-breach 預設。
- actual-primary reference：`Momentum+Trend 126D/200MA 3.0x to 1.0x`
- actual-primary reference 與 hybrid-primary authority 不同時，不視為錯誤；這代表參考訊號分歧，需要在一頁決策包中人工確認。

### TW50

- comparison 入口：`reports/monthly_decision_comparison_tw50.html`
- review record：`reports/monthly_decision_review_tw50.md`
- source coverage：`reports/tw50_total_return_sources.csv`
- dividend audit：`reports/tw50_dividend_audit.csv`
- fast replay as-of：`2026-05-08`
- research authority：`Vol Target 63D 25%`
- actual weekly policy state：`0050 78% / 00631L 0% / CASH 22%`，等效槓桿約 `0.78x`。
- 含成本後 cost mode 是 `net_of_cost`；最低手續費已改為 `1` 元。
- research authority 總交易成本約 `761,598`，成本拖累約 `22.60%`。
- MC breach rate 約 `0.50%`；目前 fast replay 沒有 zero-breach actionable default，所以不輸出可交易權重，必須人工 review。
- `1999-2002` 為 `^TWII` price proxy，不是完整含息；從 `2003-01-02` 起使用 TAIEX total return。

## 日常結果入口

平常先看：

```text
reports/monthly_decision_comparison_qqq.html
```

這份一頁決策包會顯示：

- 正式推薦策略。
- 可交易權重。
- manual review 狀態與原因。
- Research authority、actionable default 與 actual-primary reference 的差異。
- MC risk 摘要：expected XIRR、p05 XIRR、win rate、drawdown breach rate、cohort gate。
- 成本摘要：cost mode、總交易成本、成本拖累與 turnover。
- source coverage：QQQ/QLD/TQQQ 哪段使用 actual，哪段使用 synthetic backfill。

TW50 comparison 的第一屏額外會顯示：

- 投入設定：初始 `100,000`，每月 `10,000`。
- actual ETF optimizer 與 hybrid replay 的期間、總投入。
- 台股成本模型：手續費、最低手續費、ETF 賣出稅、slippage。
- actual reference 權重；若沒有 zero-breach actionable default，會明確標示它只是人工 review 參考。
- 完整候選互動趨勢圖，可切換累積交易成本與 turnover。

TW50 搜尋前 audit 入口：

```powershell
python -m uv run python scripts\audit_pre_optimization.py --config configs\tw50_example.yaml --family tw50
```

主要輸出：

- `reports/pre_optimization_audit_tw50.html`
- `reports/pre_optimization_audit_tw50.md`
- `reports/pre_optimization_audit_tw50.csv`

TW50 Optuna 搜尋入口：

```powershell
python -m uv run python scripts\search_strategy_optuna.py --config configs\tw50_example.yaml --family tw50 --study-name tw50_v1 --trials 1000
```

若目標改成 XIRR / 勝率優先，正式主線請使用 V5 return-first study，不要覆蓋 V1/V4：

```powershell
python -m uv run python scripts\search_strategy_optuna.py --config configs\tw50_example.yaml --family tw50 --study-name tw50_v5_return_first_core_external_monthly_weekly_delta_benchmark15x_minfee1_20260508 --storage reports\optuna_tw50_v5_return_first_core_external_monthly_weekly_delta_benchmark15x_minfee1_20260508.db --trials 50000 --include-external-signals --objective-profile return_first --timeout-hours 10.5
```

return-first 的正式勝率欄位是 `win_rate_vs_fixed_1p5x_dca`，代表打敗固定 1.5x DCA。
報表預設依 `return_first_rank` 看 `Return Leaders`，同時保留 `High Win Rate Leaders`
與 `Balanced Backup`。候選策略仍只是 research candidates。

長跑中斷後，先從 SQLite 重建報表，不要重跑已完成 trials：

```powershell
python -m uv run python scripts\search_strategy_optuna.py --config configs\tw50_example.yaml --family tw50 --study-name tw50_v1 --storage reports\optuna_tw50_v1.db --export-only
```

目前 `tw50_v1` 已從 SQLite 匯出 `9,379` 個 COMPLETE trials、`1` 個 FAIL、
`707` 個 Pareto candidates。下一步不是繼續堆 trials，而是從
`optuna_tw50_v1_candidate_triage.csv` 挑前 `3~5` 個去重候選，轉成可 replay 的 scenario。

搜尋會使用 SQLite storage，可中斷續跑：

- `reports/optuna_tw50_v1.db`
- `reports/optuna_tw50_v1_trials.csv`
- `reports/optuna_tw50_v1_pareto.csv`
- `reports/optuna_tw50_v1_best_candidates.csv`
- `reports/optuna_tw50_v1_candidate_triage.csv`
- `reports/optuna_tw50_v1.html`

若要啟用 Fear & Greed，先放 frozen CSV：

```text
data/external/fear_greed.csv
```

欄位必須是 `date,score,rating`。這份 CSV 是固定資料集，不從 live API 直接最佳化；訊號在程式內會 shift 到 `t-1`，避免 lookahead。若只是 smoke test，可以先用：

```powershell
python -m uv run python scripts\search_strategy_optuna.py --config configs\tw50_example.yaml --family tw50 --study-name tw50_smoke --trials 5 --storage reports\optuna_tw50_smoke.db --no-external-signals --no-sentiment
```

目前更建議跑完整外部 regime 特徵流程：

```powershell
python -m uv run python scripts\fetch_external_signals.py --config configs\tw50_example.yaml --family tw50 --end-date 2026-05-09
python -m uv run python scripts\audit_external_signals.py --config configs\tw50_example.yaml --family tw50
python -m uv run python scripts\search_strategy_optuna.py --config configs\tw50_example.yaml --family tw50 --study-name tw50_v5_smoke --trials 5 --storage reports\optuna_tw50_v5_smoke.db --include-external-signals --objective-profile return_first
```

正式 core external 特徵只包括 VIX、USD/TWD、台股融資融券與三大法人買賣超。Fear & Greed 目前約一年歷史，已從正式搜尋移除；若未來使用，只能作為實驗附錄。資料本體放在 `data/external/`，不提交 GitHub。

V5 core external 的正式 replay 現在只使用核心外部指標共同齊全區間；目前 smoke
驗證會裁到 `2004-07-23 ~ 2026-05-08`。`1999-03-10` 起的資料保留給 long stress
附錄，不再混入 V5 external 正式成績。V5 固定 official benchmark 為
`0050 50% / 00631L 50%` 的 1.5x DCA；策略執行是月度主 rebalance + 週度差距觸發，
monthly contribution、台股最低手續費 `1` 元。若候選策略需要 replay，使用：

```powershell
python -m uv run python scripts\analyze_monthly_decision_replay.py --config configs\tw50_example.yaml --family tw50 --selector hybrid_primary --fast --optuna-scenarios reports\optuna_tw50_v5_return_first_core_external_monthly_weekly_delta_benchmark15x_minfee1_20260508_shortlist.csv
```

一條龍 loop runner 會在每輪完成後產生 summary 與 snapshot：

```powershell
python -m uv run python scripts\run_optuna_replay_loop.py --config configs\tw50_example.yaml --family tw50 --study-name tw50_v5_return_first_core_external_monthly_weekly_delta_benchmark15x_minfee1_20260508 --storage reports\optuna_tw50_v5_return_first_core_external_monthly_weekly_delta_benchmark15x_minfee1_20260508.db --loops 8 --search-hours 1 --max-candidates 5 --final-full-replay --final-candidates 5
```

每輪預設會先跑正式成本 replay，再對同一份 shortlist 跑 `2x` 成本壓測 replay。壓測只把
估算出的交易成本乘以 2，不改 `configs/tw50_example.yaml` 的正式費率。若臨時不想跑壓測，
可加 `--skip-cost-stress`；若想改倍率，可加 `--cost-stress-multiplier 3.0`。

優先看：

- `reports/optuna_tw50_v5_return_first_core_external_monthly_weekly_delta_benchmark15x_minfee1_20260508_loop_summary.html`
- `reports/loop_snapshots/tw50_v5_return_first_core_external_monthly_weekly_delta_benchmark15x_minfee1_20260508/cycle_0001/index.html`
- `reports/loop_snapshots/tw50_v5_return_first_core_external_monthly_weekly_delta_benchmark15x_minfee1_20260508/final/index.html`

snapshot 的 `index.html` 是入口；真正的圖表比較在同資料夾的
`monthly_decision_comparison_tw50.html`。新版 comparison 會優先讀 replay compare
payload，必含 research top、固定 1.5x DCA official benchmark、0050 1x DCA、00631L 2x DCA、actual reference、等效槓桿、
累積投入、累積交易成本與 turnover。

成本診斷入口：

- `monthly_decision_replay_tw50_trade_audit.html`
- `monthly_decision_replay_tw50_stress_2x.html`
- `monthly_decision_replay_tw50_stress_2x_trade_audit.html`

人工確認後再產生 review record：

```powershell
python -m uv run python scripts\record_monthly_decision_review.py --family qqq --status pending_review --reviewer "Ian"
```

comparison HTML 是決策入口；review record 是人工確認留痕。它只記錄本月 review 狀態、reviewer、notes、成本摘要與審計證據路徑，不會改變 ranking 或權重。

若要採用有 MC breach 的 research authority，必須使用：

```powershell
python -m uv run python scripts\record_monthly_decision_review.py --family qqq --status override --selected-layer research_authority --reviewer "Ian"
```

日常可讀 CSV：

- `reports/monthly_decision_comparison_qqq.csv`
- `reports/monthly_decision_comparison_qqq_top_candidates.csv`
- `reports/monthly_decision_review_qqq.csv`
- `reports/monthly_decision_replay_qqq_mc_summary.csv`
- `reports/monthly_decision_replay_qqq_source_coverage.csv`

深度審計才看：

- `reports/monthly_decision_replay_qqq_mc_trials.csv.gz`
- `reports/monthly_decision_replay_qqq_cohorts.csv`
- `reports/monthly_decision_replay_qqq_decisions.csv`
- `reports/dca_policy_optimizer_qqq_policy.csv`

## 常用命令

```powershell
python -m uv run pytest
python -m uv run ruff check src tests scripts
python -m uv run python scripts\analyze_dca_policy_optimizer.py --config configs\mvp_example.yaml --family qqq --scan-mode fast --cohort-validation
python -m uv run python scripts\analyze_monthly_decision_pack.py --config configs\mvp_example.yaml --family qqq
python -m uv run python scripts\analyze_monthly_decision_replay.py --config configs\mvp_example.yaml --family qqq --selector hybrid_primary --fast
python -m uv run python scripts\analyze_monthly_decision_comparison.py --config configs\mvp_example.yaml --family qqq
```

TW50 含息 fast 流程：

```powershell
python -m uv run python scripts\analyze_dca_policy_optimizer.py --config configs\tw50_example.yaml --family tw50 --scan-mode fast --cohort-validation
python -m uv run python scripts\analyze_monthly_decision_pack.py --config configs\tw50_example.yaml --family tw50
python -m uv run python scripts\analyze_monthly_decision_replay.py --config configs\tw50_example.yaml --family tw50 --selector hybrid_primary --fast
python -m uv run python scripts\analyze_monthly_decision_comparison.py --config configs\tw50_example.yaml --family tw50
python -m uv run python scripts\record_monthly_decision_review.py --family tw50 --status pending_review --reviewer "Ian"
```

正式 full MC replay：

```powershell
python -m uv run python scripts\analyze_monthly_decision_replay.py --config configs\mvp_example.yaml --family qqq --selector hybrid_primary --full
```

Windows pytest 若遇到 temp ACL 問題，使用 fresh basetemp：

```powershell
python -m uv run pytest --basetemp=C:\Users\Ian Lai\Desktop\Python\pytest-fresh-20260505
```

## Runtime

- `hybrid_primary --fast`：約 10 分鐘。
- `hybrid_primary --full`：目前觀察約 49 分鐘。
- TW50 `hybrid_primary --fast`：含成本後本次約 6.5 分鐘。
- 主要瓶頸在 deterministic rolling cohort gate。
- CLI 已有開始時間、selector、scan mode、horizons、coverage、階段進度、總耗時與輸出檔案，長時間執行時可用來確認不是卡死。

## 下一步建議

在策略搜尋開始後，優先順序是：

1. 先看 `pre_optimization_audit_tw50.html`，確認資料、含息、成本、no-lookahead 與 artifacts 沒有 fail。
2. 用小 trials smoke test 確認 Optuna 可續跑、輸出 schema 正常。
3. 再長跑 `tw50_v1`，看 Pareto 與 best candidates。
4. 對 top candidates 重跑既有 TW50 replay、comparison、review record；搜尋結果本身不能直接當交易建議。
