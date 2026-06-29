from src.core.data_loader import DataLoader
from src.models.market_data import MarketData

from src.indicators.ema import EMA


loader = DataLoader()

df = loader.load_csv(
    "data/sample/nifty_sample.csv"
)

market = MarketData(df)

ema20 = EMA(period=20)

market = ema20.calculate(market)

print()

print(market.dataframe.head())