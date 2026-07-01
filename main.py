from src.core.data_loader import DataLoader
from src.models.market_data import MarketData

from src.indicators.ema import EMA

from src.smc.swing_detector import SwingDetector
from src.smc.market_structure import MarketStructure


def main() -> None:
    """
    SmartMoneyEngine Entry Point
    """

    loader = DataLoader()

    # ---------------------------------------
    # Load Test Dataset
    # ---------------------------------------
    df = loader.load_csv(
        "tests/sample_data/swing_test.csv"
    )

    market = MarketData(df)

    # ---------------------------------------
    # Indicators
    # ---------------------------------------

    EMA(20).calculate(market)
    EMA(50).calculate(market)
    EMA(100).calculate(market)

    # ---------------------------------------
    # Smart Money Concepts
    # ---------------------------------------

    SwingDetector().detect(market)
    MarketStructure().detect(market)

    # ---------------------------------------
    # Summary
    # ---------------------------------------

    print()
    print("=" * 80)
    print("SMART MONEY ENGINE TEST")
    print("=" * 80)

    print("\nRows :", market.rows)
    print("Columns :", market.columns)

    # ---------------------------------------
    # Complete Validation Table
    # ---------------------------------------

    print("\nSwing Validation\n")

    print(
        market.data[
            [
                "High",
                "Low",
                "Swing_High",
                "Swing_Low",
                "HH",
                "HL",
                "LH",
                "LL",
            ]
        ]
    )


if __name__ == "__main__":
    main()