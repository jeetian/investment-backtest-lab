# 策略研究計畫

## 研究階段

1. Grid / ladder search  
   先用可解釋的規則掃描策略，例如均線、動能、回撤、波動率與槓桿階梯。

2. Walk-forward validation  
   用 train period 選策略，再用下一段 test period 驗證。

3. Rolling cohort validation  
   用不同 DCA 起點與不同持有期間檢查排名穩定度。

4. Optuna 搜尋加速  
   等策略 family 與驗證框架穩定後，再用 Optuna 加速參數搜尋。

5. 多標的擴充  
   從 QQQ family 擴到 SPY family、其他 ETF、台股與基金。

## 目前 QQQ Family 策略族群

- `constant_leverage`：固定 0x、1x、2x、3x。
- `trend_ladder`：依 QQQ 是否高於 MA 決定槓桿。
- `drawdown_ladder`：依 QQQ 回撤深度分段降槓桿。
- `vol_target_ladder`：依 realized volatility 調整目標槓桿。
- `momentum_trend_ladder`：結合 momentum 與 MA trend。

## 排名原則

DCA 策略主要看：

- XIRR。
- max drawdown。
- recovery days。
- rolling cohort top-3 hit rate。
- drawdown breach rate。
- synthetic stress 是否穿越硬線。

不得只用 ending equity 或全期間 CAGR 決定最佳策略。

## Optuna 導入原則

Optuna 將來可用來搜尋：

- MA window。
- momentum window。
- volatility target。
- drawdown guard。
- leverage ladder。

但 Optuna 找到的策略仍必須通過：

- no-lookahead。
- walk-forward。
- rolling cohort。
- synthetic stress。
- 最大回撤硬線。
