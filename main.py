from src.core.data_loader import DataLoader
from src.models.market_data import MarketData
from src.indicators.ema import EMA


def main():

    loader = DataLoader()

    df = loader.load_csv(
        "data/sample/nifty_sample.csv"
    )

    market = MarketData(df)

    EMA(20).calculate(market)
    EMA(50).calculate(market)
    EMA(100).calculate(market)

    print()

    print("Rows :", market.rows)
    print("Columns :", market.columns)

    print()

    print(market.head())


if __name__ == "__main__":
    main()