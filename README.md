# Investment Backtest Lab

Investment Backtest Lab 是一個投資回測研究專案，目前重點是 QQQ/QLD/TQQQ/CASH 與 TW50
`0050/00631L/CASH` 的 DCA policy optimizer、monthly decision pack，以及
hybrid-primary Monte Carlo replay。

正式 optimizer、monthly replay 與 comparison 預設都是 `net_of_cost`。QQQ 會套用 US
SEC fee / FINRA TAF / slippage 設定；TW50 會套用台股 ETF 手續費、賣出證交稅與
slippage。研究模型仍使用權重，不做整股、零股、成交量或市場衝擊限制。

## 快速開始

```powershell
python -m uv sync --extra dev
python -m uv run pytest
python -m uv run python scripts\smoke_imports.py
```

Python 需求是 `>=3.12,<3.13`，依賴由 `uv.lock` 鎖定。

Windows pytest 若遇到 temp ACL 問題，可使用 fresh basetemp：

```powershell
python -m uv run pytest --basetemp=C:\Users\Ian Lai\Desktop\Python\pytest-fresh-20260505
```

## 主要月度流程

目前 `configs/mvp_example.yaml` 的資料截止設定是 `end_date: 2026-05-01`。這是
yfinance 的 end-exclusive cutoff，用來包含 `2026-04-30` 月底收盤；月度決策
comparison 會檢查 `recommended_as_of_date` 是否等於 `2026-04-30`。

先產生 actual ETF policy 與 cohort gate：

```powershell
python -m uv run python scripts\analyze_dca_policy_optimizer.py --config configs\mvp_example.yaml --family qqq --scan-mode fast --cohort-validation
```

產生 actual-primary 參考月度包：

```powershell
python -m uv run python scripts\analyze_monthly_decision_pack.py --config configs\mvp_example.yaml --family qqq
```

產生 hybrid-primary replay：

```powershell
python -m uv run python scripts\analyze_monthly_decision_replay.py --config configs\mvp_example.yaml --family qqq --selector hybrid_primary --fast
```

正式 full replay：

```powershell
python -m uv run python scripts\analyze_monthly_decision_replay.py --config configs\mvp_example.yaml --family qqq --selector hybrid_primary --full
```

產生一頁決策包：

```powershell
python -m uv run python scripts\analyze_monthly_decision_comparison.py --config configs\mvp_example.yaml --family qqq
```

產生人工 review 紀錄：

```powershell
python -m uv run python scripts\record_monthly_decision_review.py --family qqq --status pending_review --reviewer "Ian"
```

## TW50 含息流程

`configs/tw50_example.yaml` 使用 `0050 + 00631L + CASH`，口徑是稅前總報酬、配息再投入。
TW50 builder 會用 FinMind raw price + dividend audit 建構 0050/00631L actual total return，
0050 的 `2025-06-18` 1 拆 4 會同步調整歷史價格與配息。1999-2002 因官方 TAIEX
total return 尚不可用，會明確標示為 `^TWII` price proxy，不視為完整含息資料。

```powershell
python -m uv run python scripts\analyze_dca_policy_optimizer.py --config configs\tw50_example.yaml --family tw50 --scan-mode fast --cohort-validation
python -m uv run python scripts\analyze_monthly_decision_pack.py --config configs\tw50_example.yaml --family tw50
python -m uv run python scripts\analyze_monthly_decision_replay.py --config configs\tw50_example.yaml --family tw50 --selector hybrid_primary --fast
python -m uv run python scripts\analyze_monthly_decision_comparison.py --config configs\tw50_example.yaml --family tw50
python -m uv run python scripts\record_monthly_decision_review.py --family tw50 --status pending_review --reviewer "Ian"
```

TW50 目前口徑是稅前含息、含交易成本、未含股數限制。

## TW50 搜尋前 Audit 與 Optuna

進入長時間策略搜尋前，先跑 pre-optimization audit：

```powershell
python -m uv run python scripts\audit_pre_optimization.py --config configs\tw50_example.yaml --family tw50
```

Audit 會輸出 `reports/pre_optimization_audit_tw50.html`、`.csv`、`.md`，
檢查 TW50 source coverage、配息/split audit、成本模型、no-lookahead、optimizer/replay/comparison
artifact 與動態回撤硬線。TW50 Optuna v1 的 hard drawdown limit 會用
`1x TW50 synthetic stress max drawdown * 1.20` 推導，而不是寫死。

外部 regime 指標先凍結到本機：

```powershell
python -m uv run python scripts\fetch_external_signals.py --config configs\tw50_example.yaml --family tw50 --end-date 2026-05-09
python -m uv run python scripts\audit_external_signals.py --config configs\tw50_example.yaml --family tw50
```

會產生 `data/external/market_regime_features_tw50.csv` 與
`reports/external_signal_audit_tw50.html`。資料本體不提交 GitHub。正式 TW50
搜尋使用 `core` 外部指標集，只包含 VIX、USD/TWD、台股融資融券與法人買賣超；
CNN Fear & Greed 因歷史長度不足，已排除於正式搜尋，只能作為實驗附錄。所有外部特徵在搜尋時都會 shift 到 `t-1`。

Optuna 搜尋入口：

```powershell
python -m uv run python scripts\search_strategy_optuna.py --config configs\tw50_example.yaml --family tw50 --study-name tw50_v1 --trials 1000 --include-external-signals
```

TW50 V5 正式比較基準是固定 `1.5x DCA`：每月投入後維持
`0050 50% / 00631L 50% / CASH 0%`。正式勝率欄位改看
`win_rate_vs_fixed_1p5x_dca`；`0050 1x DCA` 與 `00631L 2x DCA`
仍保留在報表與 compare lab 作參考。V5 策略執行口徑是
`monthly_core_weekly_delta`：月度主 rebalance，每週只在新目標等效槓桿與目前執行目標
差距超過 `weekly_delta_threshold` 時才交易。DCA 現金流仍是 monthly contribution，最低手續費 1 元。

```powershell
python -m uv run python scripts\search_strategy_optuna.py --config configs\tw50_example.yaml --family tw50 --study-name tw50_v5_return_first_core_external_monthly_weekly_delta_benchmark15x_minfee1_20260508 --storage reports\optuna_tw50_v5_return_first_core_external_monthly_weekly_delta_benchmark15x_minfee1_20260508.db --trials 50000 --include-external-signals --objective-profile return_first --timeout-hours 10.5
```

return-first 報表會新增 `return_first_rank`、`return_first_score`、`win_rate_vs_fixed_1p5x_dca`，
並分成 `Return Leaders`、`High Win Rate Leaders`、`Balanced Backup`。回撤、成本與
turnover 仍是 gate / watchlist 訊號；高成本策略需再通過 2x 成本壓測，候選仍不得直接變成 actionable default。

V5 core external 策略的正式 replay 只使用核心外部指標共同齊全區間；目前會自動裁到
`2004-07-23 ~ 2026-05-08`。`1999-03-10` 起的 TW50 hybrid 資料仍保留，但只作未來
long-stress 附錄，不混入 V5 external 正式成績。

若要讓電腦自動一輪一輪搜尋並驗證候選，可用 loop runner：

```powershell
python -m uv run python scripts\run_optuna_replay_loop.py --config configs\tw50_example.yaml --family tw50 --study-name tw50_v5_return_first_core_external_monthly_weekly_delta_benchmark15x_minfee1_20260508 --storage reports\optuna_tw50_v5_return_first_core_external_monthly_weekly_delta_benchmark15x_minfee1_20260508.db --loops 8 --search-hours 1 --max-candidates 5 --final-full-replay --final-candidates 5
```

每輪會產生 loop summary 與 snapshot；預設會多跑一次 `2x` 交易成本壓測，不覆蓋正式
`1x` replay。優先開：

- `reports/optuna_tw50_v5_return_first_core_external_monthly_weekly_delta_benchmark15x_minfee1_20260508_loop_summary.html`
- `reports/loop_snapshots/tw50_v5_return_first_core_external_monthly_weekly_delta_benchmark15x_minfee1_20260508/cycle_0001/index.html`
- `reports/loop_snapshots/tw50_v5_return_first_core_external_monthly_weekly_delta_benchmark15x_minfee1_20260508/final/index.html`

snapshot 的 `index.html` 只是入口；完整圖表請開同資料夾內的
`monthly_decision_comparison_tw50.html`，其中包含 replay compare lab、固定 1.5x DCA
official benchmark、0050 1x DCA、00631L 2x DCA、Optuna candidates、actual reference 與等效槓桿。

成本診斷請看同資料夾內：

- `monthly_decision_replay_tw50_trade_audit.html`：正式成本下的交易行為審計。
- `monthly_decision_replay_tw50_stress_2x.html`：交易成本乘以 2 的壓測 replay。
- `monthly_decision_replay_tw50_stress_2x_trade_audit.html`：2x 壓測下的交易成本拆解。

若只想跑搜尋/replay、不跑 2x 壓測，可加 `--skip-cost-stress`；若想改壓測倍率，可加
`--cost-stress-multiplier 3.0`。

若長跑中斷，不要重跑已完成 trials；先用 export-only 重建報表：

```powershell
python -m uv run python scripts\search_strategy_optuna.py --config configs\tw50_example.yaml --family tw50 --study-name tw50_v1 --storage reports\optuna_tw50_v1.db --export-only
```

export-only 會重建 trials、Pareto、best candidates、candidate triage 與 HTML，
並顯示 completed/failed trials、last trial number、external signal set。它不新增 trial。

若只是測試搜尋管線，可先用：

```powershell
python -m uv run python scripts\search_strategy_optuna.py --config configs\tw50_example.yaml --family tw50 --study-name tw50_smoke --trials 5 --storage reports\optuna_tw50_smoke.db --no-external-signals --no-sentiment
```

搜尋結果會輸出 SQLite study、trials CSV、Pareto CSV、best candidates CSV 與 HTML。
結果不得直接變成 actionable default；候選策略仍要回到 monthly replay / comparison / review 流程。

## 最重要的輸出

日常先看：

- `reports/monthly_decision_comparison_qqq.html`：一頁決策包，顯示正式推薦策略、可交易權重、review 狀態、MC 風控摘要與 source coverage。
- `reports/monthly_decision_review_qqq.md`：人工 review 紀錄，留下本月是否接受、延後或覆核的決策痕跡。
- `reports/monthly_decision_comparison_qqq_top_candidates.csv`：top candidates 摘要。
- `reports/monthly_decision_replay_qqq_mc_summary.csv`：Monte Carlo trials 分層摘要。
- `reports/monthly_decision_comparison_tw50.html`：TW50 一頁決策包。
- `reports/tw50_total_return_sources.csv`：TW50 source coverage，揭露 `^TWII` price proxy、TAIEX total return、0050 actual、00631L synthetic/actual 區間。
- `reports/tw50_dividend_audit.csv`：0050/00631L 配息與 split-adjusted dividend audit。

comparison 與 top candidates 會顯示成本摘要：`cost_mode`、`total_trade_cost`、
`cost_drag_on_contributed`、`cost_to_final_equity`、`turnover_sum`。review record 也會留下
actionable default 的成本摘要，方便月度審計。`cost_drag_on_contributed` 是總交易成本
除以總投入，不是年化成本；高成長策略也要同時看 `cost_to_final_equity`。

TW50 comparison 第一屏會把 actual ETF optimizer 與 hybrid replay 的期間、總投入、
成本模型、本月 reference 權重、p05 XIRR、p05 drawdown、cost drag、turnover 與 MC
breach 翻成白話。頁面也內建完整候選互動趨勢圖，可切換淨資產、回撤、有效槓桿、
累積投入、累積交易成本與 turnover。

深度審計才看：

- `reports/monthly_decision_replay_qqq_mc_trials.csv.gz`：完整 Monte Carlo trials 壓縮檔。
- `reports/monthly_decision_replay_qqq_cohorts.csv`：deterministic rolling cohort gate 明細。
- `reports/dca_policy_optimizer_qqq_policy.csv`：每日 actual ETF policy state。

## 目前決策語意

- 正式月度 authority：`hybrid_primary_monte_carlo`
- `hybrid_primary` 上市後使用 actual ETF，上市前使用 scaled synthetic backfill。
- Monte Carlo ranking 是 research ranking，expected XIRR 是主要 objective。
- Actionable default 是未人工 override 前的可行動層，只能使用 zero MC breach 且 cohort gate passed 的最高候選。
- 目前 research authority 是 `Momentum+Trend 126D/200MA 3.0x to 1.0x`；actionable default 是 `Vol Target 63D 25%`。
- deterministic rolling cohort gate 是硬風控 gate。
- actual-primary monthly decision pack 保留為參考訊號；和 hybrid-primary authority 不同不代表錯誤，而是人工 review 要看的分歧。
- review record 只記錄人工判斷，不會改變 ranking 或權重計算。
- TW50 V5 之後請以固定 1.5x DCA 作 official benchmark。舊 V4 weekly execution 結果
  只保留作歷史參考，不再作正式候選 gate。

## Runtime

- `hybrid_primary --fast` 約 10 分鐘。
- `hybrid_primary --full` 目前觀察約 49 分鐘。
- TW50 `hybrid_primary --fast` 含成本後本次約 6.5 分鐘；主要瓶頸同樣在 deterministic rolling cohort gate。
- 主要瓶頸在 deterministic rolling cohort gate。
- CLI 會印出開始時間、selector、scan mode、horizons、coverage、階段進度、總耗時與輸出檔案。

## 其他研究入口

```powershell
python -m uv run python scripts\run_prototype.py --config configs\mvp_example.yaml --offline-demo
python -m uv run python scripts\analyze_ledger.py --config configs\mvp_example.yaml --tickers SPY QQQ
python -m uv run python scripts\analyze_leveraged_etf_lab.py --config configs\mvp_example.yaml --family qqq --cash-flow-mode both
```

更多接手說明：

- `RESULTS_ZH.md`
- `docs/HANDOFF_ZH.md`
- `docs/QUICKSTART_ZH.md`
- `docs/ALLOCATION_WORKFLOW_ZH.md`
- `docs/ROADMAP_ZH.md`

長期方向文件：

- [專案範圍](docs/PROJECT_SCOPE_ZH.md)
- [Roadmap](docs/ROADMAP_ZH.md)
- [驗證策略](docs/VALIDATION_STRATEGY_ZH.md)
- [框架維護規範](docs/FRAMEWORK_HYGIENE_ZH.md)
- [策略研究計畫](docs/STRATEGY_RESEARCH_PLAN_ZH.md)
- [月度配置工作流](docs/ALLOCATION_WORKFLOW_ZH.md)

## FinMind Token

不要把 token 寫入檔案。需要 live data 時，只在目前 shell 設定：

```powershell
$env:FINMIND_TOKEN = "你的 token"
python -m uv run python scripts\smoke_data.py --network
```
