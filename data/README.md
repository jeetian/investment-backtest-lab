# data 資料夾

這個資料夾只放本機資料與可重建的資料快取，預設不提交到 git。

建議用途：

- `raw/`：原始 API 下載資料。
- `cache/`：adapter 或 pipeline 產生的 parquet/cache。
- `processed/`：可重建的中間處理資料。
- `external/`：人工放入的 frozen external CSV，例如 `fear_greed.csv`。

注意事項：

- 不要提交 API token、券商帳密或任何私人資料。
- 不要把 live API 回傳結果直接當成策略搜尋的最佳化依據；需要先 frozen 成 CSV。
- `data/external/fear_greed.csv` 若要用於 Optuna V1，schema 必須是 `date,score,rating`。
- Fear & Greed 分數必須固定、可追溯，程式會在交易日對齊後使用 `t-1` 或更早資料，避免 lookahead。
- `data/external/market_regime_features_tw50.csv` 是 Optuna 多特徵搜尋使用的 frozen feature table。
- 外部資料目前包含 Fear & Greed、VIX、USD/TWD、台股融資融券與三大法人買賣超。
