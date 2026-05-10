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
   - 成本口徑會顯示為 `net_of_cost`：交易成本、成本拖累與 turnover 都會列在 comparison 與 top candidates。

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

## 台股 TW50 含息入口

1. `reports\monthly_decision_comparison_tw50.html`
   - TW50 的一頁決策包，family 是 `0050 / 00631L / CASH`。
   - 口徑是稅前含息、含交易成本、未做整股/零股限制。
   - 第一屏會直接列出投入設定、actual/replay 期間、總投入、成本模型與本月結論。
   - 沒有 zero-breach actionable default 時，actual reference 權重只作人工 review 參考，不是自動下單建議。
   - 內建完整候選互動趨勢圖，可切換淨資產、標準化淨值、回撤、有效槓桿、累積投入、累積交易成本與 turnover。
   - 權重採動態 schema：`recommended_weights_json`，不再硬塞進 QQQ 欄位。
   - 若沒有 zero-breach actionable default，會明確標示人工 review，且不輸出可交易權重。

2. `reports\tw50_total_return_sources.csv`
   - 台股資料來源審計入口。
   - `0050`：`1999-2002` 是 `^TWII` price proxy，不是完整含息；`2003-01-02` 起用 TAIEX total return；0050 上市後用 raw price + dividend + split-adjusted total return。
   - `00631L`：上市前用 0050 hybrid total-return daily return 合成 2x daily-reset，上市後用 00631L actual total return。

3. `reports\tw50_dividend_audit.csv`
   - 0050/00631L 配息審計。
   - 0050 的 `2025-06-18` 1 拆 4 已反映在 split-adjusted dividend。

4. `reports\monthly_decision_review_tw50.md`
   - TW50 月度人工 review 留痕。

5. `reports\pre_optimization_audit_tw50.html`
   - 進入 Optuna 長時間搜尋前先看。
   - 檢查 TW50 主鏈路：source coverage、配息/split、成本模型、no-lookahead、optimizer/replay/comparison artifact、schema 與動態回撤硬線。
   - 若 overall status 是 `fail`，不應進入正式搜尋。

6. `reports\optuna_tw50_v1.html`
   - TW50 Optuna-first 搜尋結果入口。
   - 顯示 Pareto candidates、成本拖累、turnover、p05 XIRR、p05 drawdown 與候選狀態。
   - 搜尋結果只是 research candidates，不能直接視為 actionable default。

7. `reports\optuna_tw50_v1_candidate_triage.csv`
   - 中斷後 export-only 產生的候選篩選表。
   - 會標示 `candidate / watchlist / rejected`，並保留 holdout、成本、turnover 與 external signal 欄位。
   - 這不是 replay 驗證結果；下一步要挑前 3~5 個去重候選重跑 TW50 replay。

## 目前最新結果

### QQQ

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
- 成本模型導入後，重新產生的 QQQ optimizer / replay / comparison 會改讀為 `net_of_cost`；舊報表若未重跑，數字仍可能是導入成本前的版本。

### TW50 fast replay

- 設定檔：`configs\tw50_example.yaml`
- 資料 cutoff：`2026-05-09` end-exclusive，報表 as-of 是 `2026-05-08`。
- 正式執行口徑：每週檢查 / 可換倉，月投入仍是每月 `10,000`。
- actual-primary 週度參考策略：`Vol Target 63D 25%`
- latest actual TW50 policy state：`0050 78% / 00631L 0% / CASH 22%`，等效槓桿約 `0.78x`。
- hybrid-primary research authority：`Vol Target 63D 25%`
- 含成本後 replay ranking 的 cost mode：`net_of_cost`
- 台股最低手續費已改為 `1` 元。
- research authority 的總交易成本約 `761,598`，成本拖累約 `22.60%`。
- MC breach rate 約 `0.50%`，因此需要人工 review。
- fast replay 目前沒有 zero-breach actionable default，所以 comparison 不輸出可交易權重；這是風控語意，不是資料失敗。
- `1999-2002` 只是 `^TWII` price proxy，不是完整含息；正式含息 proxy 從 `2003-01-02` 的 TAIEX total return 開始。
- 成本模型導入後，TW50 會扣台股 ETF 買賣手續費、賣出證交稅與 slippage；仍不做整股、零股、成交量或市場衝擊限制。

## 常見指標白話

- `p05 XIRR`：偏壞 5% Monte Carlo 情境的年化報酬。
- `p05 drawdown`：偏壞 5% Monte Carlo 情境的最大回撤。
- `cost drag`：總交易成本 / 總投入，不是年化值。
- `cost_to_final_equity`：總交易成本 / 最終資產，用來避免高成長策略被 `cost drag`
  單一比例誤判。
- `turnover`：累積換倉強度，數字越高代表策略越常大幅調整。
- `MC breach`：Monte Carlo 路徑中跌破硬性回撤門檻的比例；大於 0 就需要人工 review。

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

重新跑 TW50 含息 fast 流程：

```powershell
python -m uv run python scripts\analyze_dca_policy_optimizer.py --config configs\tw50_example.yaml --family tw50 --scan-mode fast --cohort-validation
python -m uv run python scripts\analyze_monthly_decision_pack.py --config configs\tw50_example.yaml --family tw50
python -m uv run python scripts\analyze_monthly_decision_replay.py --config configs\tw50_example.yaml --family tw50 --selector hybrid_primary --fast
python -m uv run python scripts\analyze_monthly_decision_comparison.py --config configs\tw50_example.yaml --family tw50
python -m uv run python scripts\record_monthly_decision_review.py --family tw50 --status pending_review --reviewer "Ian"
```

上述 optimizer、replay、comparison 皆為 net-of-cost；comparison HTML 與 review record 會揭露 `cost_mode`、`total_trade_cost`、`cost_drag_on_contributed` 與 `turnover_sum`。

TW50 搜尋前 audit：

```powershell
python -m uv run python scripts\audit_pre_optimization.py --config configs\tw50_example.yaml --family tw50
```

TW50 外部 regime 指標：

```powershell
python -m uv run python scripts\fetch_external_signals.py --config configs\tw50_example.yaml --family tw50 --end-date 2026-05-09
python -m uv run python scripts\audit_external_signals.py --config configs\tw50_example.yaml --family tw50
```

- frozen data 會放在 `data\external\`，不提交 GitHub。
- `reports\external_signal_audit_tw50.html` 是外部指標審計入口。
- 正式搜尋使用 `core` 外部指標集：VIX、USD/TWD、台股融資融券、三大法人買賣超。
- Fear & Greed 因目前可審計歷史太短，已從正式搜尋移除，只能作為實驗附錄。
- Optuna 讀取 `market_regime_features_tw50.csv` 後會對齊交易日、shift 到 `t-1`，並裁切到 core 指標共同完整區間。

TW50 Optuna 搜尋：

```powershell
python -m uv run python scripts\search_strategy_optuna.py --config configs\tw50_example.yaml --family tw50 --study-name tw50_v1 --trials 1000 --include-external-signals
```

TW50 V5 core external return-first 搜尋會把 `expected_xirr` 與
`win_rate_vs_fixed_1p5x_dca` 放在最前面，並排除 Fear & Greed。正式 baseline 改為
固定 `1.5x DCA`：每月投入後維持 `0050 50% / 00631L 50% / CASH 0%`。
V5 執行口徑是月度主 rebalance + 週度差距觸發；DCA 現金流仍是 monthly contribution、
台股最低手續費 `1` 元：

```powershell
python -m uv run python scripts\search_strategy_optuna.py --config configs\tw50_example.yaml --family tw50 --study-name tw50_v5_return_first_core_external_monthly_weekly_delta_benchmark15x_minfee1_20260508 --storage reports\optuna_tw50_v5_return_first_core_external_monthly_weekly_delta_benchmark15x_minfee1_20260508.db --trials 50000 --include-external-signals --objective-profile return_first --timeout-hours 10.5
```

輸出入口：

- `reports\optuna_tw50_v5_return_first_core_external_monthly_weekly_delta_benchmark15x_minfee1_20260508.html`
- `reports\optuna_tw50_v5_return_first_core_external_monthly_weekly_delta_benchmark15x_minfee1_20260508_candidate_triage.csv`

報表會分成 `Return Leaders`、`High Win Rate Leaders`、`Balanced Backup`。
這些仍是 research candidates，需要再回到 TW50 replay / comparison / review record。

TW50 V5 core external 的正式 replay 現在只使用核心外部指標共同齊全區間。
目前 smoke 驗證出的 official window 是 `2004-07-23 ~ 2026-05-08`；
`1999-03-10` 起的 hybrid 資料仍保留，但只作未來 long stress 附錄，不混入 V5
external 正式成績。

一條龍 loop runner 入口：

```powershell
python -m uv run python scripts\run_optuna_replay_loop.py --config configs\tw50_example.yaml --family tw50 --study-name tw50_v5_return_first_core_external_monthly_weekly_delta_benchmark15x_minfee1_20260508 --storage reports\optuna_tw50_v5_return_first_core_external_monthly_weekly_delta_benchmark15x_minfee1_20260508.db --loops 8 --search-hours 1 --max-candidates 5 --final-full-replay --final-candidates 5
```

每輪會搜尋、export、shortlist、fast replay、2x 成本壓測、comparison、review，並寫出：

- `reports\optuna_tw50_v5_return_first_core_external_monthly_weekly_delta_benchmark15x_minfee1_20260508_loop_summary.html`
- `reports\loop_snapshots\tw50_v5_return_first_core_external_monthly_weekly_delta_benchmark15x_minfee1_20260508\cycle_0001\index.html`
- 每輪 snapshot 內的 `monthly_decision_comparison_tw50.html`
- 每輪 snapshot 內的 `monthly_decision_replay_tw50_compare_payload.json`
- 每輪 snapshot 內的 `monthly_decision_replay_tw50_trade_audit.html`
- 每輪 snapshot 內的 `monthly_decision_replay_tw50_stress_2x.html`
- 每輪 snapshot 內的 `monthly_decision_replay_tw50_stress_2x_trade_audit.html`

`2x` 成本壓測不改正式成本模型；它只把估算出的手續費、證交稅、slippage 等交易成本乘以
`2`，用來檢查高換手策略在交易摩擦加倍後是否仍能勝過 baseline。

長跑結束後若有 `--final-full-replay`，會再產生 final snapshot：

- `reports\loop_snapshots\tw50_v5_return_first_core_external_monthly_weekly_delta_benchmark15x_minfee1_20260508\final\index.html`

每個 snapshot 的 `index.html` 是入口；`monthly_decision_comparison_tw50.html`
是完整投資決策頁與 replay compare lab，包含 research top、固定 1.5x DCA official benchmark、
0050 1x DCA、00631L 2x DCA、actual reference、等效槓桿、成本、turnover 與趨勢圖。

長跑中斷後重建報表：

```powershell
python -m uv run python scripts\search_strategy_optuna.py --config configs\tw50_example.yaml --family tw50 --study-name tw50_v1 --storage reports\optuna_tw50_v1.db --export-only
```

目前 `reports\optuna_tw50_v1.db` 已匯出：

- COMPLETE trials：`9,379`
- failed trials：`1`
- Pareto candidates：`707`
- candidate triage rows：`9,379`

如果只是測試搜尋管線，可先用：

```powershell
python -m uv run python scripts\search_strategy_optuna.py --config configs\tw50_example.yaml --family tw50 --study-name tw50_smoke --trials 5 --storage reports\optuna_tw50_smoke.db --no-external-signals --no-sentiment
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
- TW50 `hybrid_primary --fast`：含成本後本次約 6.5 分鐘。
- 主要瓶頸是 deterministic rolling cohort gate，不是 Monte Carlo 抽樣本身。
- CLI 會印開始時間、selector、scan mode、horizons、coverage、重要階段進度、總耗時與輸出檔案。完整 replay 完成後，結果才適合拿來做人工決策檢查。
