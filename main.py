from src.core.data_loader import DataLoader
from src.models.market_data import MarketData

from src.indicators.ema import EMA

from src.smc.swing_detector import SwingDetector
from src.smc.market_structure import MarketStructure
from src.smc.bos import BOS
from src.smc.choch import CHOCH
from src.smc.fvg import FVG


def main() -> None:
    """
    SmartMoneyEngine Entry Point
    """

    # -----------------------------------
    # Load Sample Data
    # -----------------------------------

    loader = DataLoader()

    df = loader.load_csv(
        "tests/sample_data/swing_test.csv"
    )

    market = MarketData(df)

    # -----------------------------------
    # Indicators
    # -----------------------------------

    EMA(20).calculate(market)
    EMA(50).calculate(market)
    EMA(100).calculate(market)

    # -----------------------------------
    # Smart Money Concepts
    # -----------------------------------

    SwingDetector().detect(market)

    MarketStructure().detect(market)

    BOS().detect(market)

    CHOCH().detect(market)

    FVG().detect(market)

    # -----------------------------------
    # Output
    # -----------------------------------

    print("\n")
    print("=" * 90)
    print("SMART MONEY ENGINE")
    print("=" * 90)

    print("\nRows :", market.rows)
    print("\nColumns :")

    for column in market.columns:
        print("   ", column)

    print("\n")
    print("=" * 90)
    print("SMC VALIDATION")
    print("=" * 90)

    print(
        market.data[
            [
                "Date",
                "High",
                "Low",
                "Close",

                "Swing_High",
                "Swing_Low",

                "HH",
                "HL",
                "LH",
                "LL",

                "Bullish_BOS",
                "Bearish_BOS",

                "Bullish_CHOCH",
                "Bearish_CHOCH",

                "Bullish_FVG_Top",
                "Bullish_FVG_Bottom",

                "Bearish_FVG_Top",
                "Bearish_FVG_Bottom",
            ]
        ]
    )


if __name__ == "__main__":
    main()