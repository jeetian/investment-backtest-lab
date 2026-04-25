from investment_backtest_lab.costs import CostModel, TradeSide
from investment_backtest_lab.models import AssetSpec, AssetType, DataSource, Market


def test_taiwan_sell_cost_includes_tax_and_commission():
    asset = AssetSpec("2330", Market.TW, AssetType.STOCK, "TWD", DataSource.FINMIND)
    cost = CostModel().estimate_trade(asset, side=TradeSide.SELL, quantity=1000, price=800)

    assert cost.commission > 0
    assert cost.transaction_tax == 800 * 1000 * 0.003
    assert cost.total > cost.transaction_tax


def test_us_sell_cost_includes_regulatory_fees():
    asset = AssetSpec("SPY", Market.US, AssetType.ETF, "USD", DataSource.YFINANCE)
    cost = CostModel().estimate_trade(asset, side="sell", quantity=10, price=500)

    assert cost.sec_fee > 0
    assert cost.finra_taf > 0
    assert cost.total > 0


def test_fx_spread_cost():
    cost = CostModel.from_dict({"fx": {"spread_bps": 15}}).estimate_fx_conversion(100_000)

    assert cost.fx_spread == 150
