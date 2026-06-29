from src.core.data_loader import DataLoader
from src.models.market_data import MarketData

loader = DataLoader()

df = loader.load_csv(
    "data/sample/nifty_sample.csv"
)

market = MarketData(df)

print()

print("Rows :", market.rows())

print("Columns :", market.columns())

print()

print(market.dataframe.head())