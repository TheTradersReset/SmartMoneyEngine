from __future__ import annotations

import pandas as pd

from src.models.market_data import MarketData
from src.smc.base_smc import BaseSMC


class FairValueGap(BaseSMC):
    """
    Detect Fair Value Gaps (FVG) from raw OHLC data.

    Evaluates three-candle patterns where the wicks of the outer
    candles do not overlap:

    - **Bullish FVG** — low of candle 3 is above high of candle 1
    - **Bearish FVG** — high of candle 3 is below low of candle 1

    Gap boundaries are stored on the third candle of each pattern;
    all other rows remain ``NaN``.
    """

    HIGH_COLUMN = "High"
    LOW_COLUMN = "Low"
    BULLISH_FVG_TOP_COLUMN = "Bullish_FVG_Top"
    BULLISH_FVG_BOTTOM_COLUMN = "Bullish_FVG_Bottom"
    BEARISH_FVG_TOP_COLUMN = "Bearish_FVG_Top"
    BEARISH_FVG_BOTTOM_COLUMN = "Bearish_FVG_Bottom"
    REQUIRED_COLUMNS = (HIGH_COLUMN, LOW_COLUMN)
    MIN_ROWS = 3

    def __init__(self) -> None:
        super().__init__("Fair Value Gap")

    def detect(self, market: MarketData) -> MarketData:
        """
        Detect bullish and bearish fair value gaps.

        Parameters
        ----------
        market : MarketData
            Market data containing ``High`` and ``Low`` columns.

        Returns
        -------
        MarketData
            Same instance with FVG boundary columns added. Each column
            stores the gap price where applicable.

        Raises
        ------
        ValueError
            If required columns are missing or row count is insufficient.
        """
        self.log_start()
        self._validate_market(market)

        (
            bullish_top,
            bullish_bottom,
            bearish_top,
            bearish_bottom,
        ) = self.detect_gaps(market)

        market.add_column(self.BULLISH_FVG_TOP_COLUMN, bullish_top)
        market.add_column(self.BULLISH_FVG_BOTTOM_COLUMN, bullish_bottom)
        market.add_column(self.BEARISH_FVG_TOP_COLUMN, bearish_top)
        market.add_column(self.BEARISH_FVG_BOTTOM_COLUMN, bearish_bottom)

        self.log_finish()

        return market

    def detect_gaps(
        self,
        market: MarketData,
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Scan three-candle sequences for fair value gaps.

        Parameters
        ----------
        market : MarketData
            Market data containing ``High`` and ``Low`` columns.

        Returns
        -------
        tuple[pd.Series, pd.Series, pd.Series, pd.Series]
            Bullish top, bullish bottom, bearish top, and bearish bottom
            gap boundary series respectively.
        """
        highs = market.get_column(self.HIGH_COLUMN)
        lows = market.get_column(self.LOW_COLUMN)
        index = highs.index

        bullish_top = self._empty_price_series(index)
        bullish_bottom = self._empty_price_series(index)
        bearish_top = self._empty_price_series(index)
        bearish_bottom = self._empty_price_series(index)

        for position in range(2, len(index)):
            row_index = index[position]
            first_high = float(highs.iloc[position - 2])
            first_low = float(lows.iloc[position - 2])
            third_high = float(highs.iloc[position])
            third_low = float(lows.iloc[position])

            if third_low > first_high:
                bullish_bottom.loc[row_index] = first_high
                bullish_top.loc[row_index] = third_low

            if third_high < first_low:
                bearish_top.loc[row_index] = first_low
                bearish_bottom.loc[row_index] = third_high

        return bullish_top, bullish_bottom, bearish_top, bearish_bottom

    def _validate_market(self, market: MarketData) -> None:
        """
        Validate required columns and minimum row count.

        Parameters
        ----------
        market : MarketData
            Market data to validate.

        Raises
        ------
        ValueError
            If validation fails.
        """
        for column in self.REQUIRED_COLUMNS:
            if not market.has_column(column):
                raise ValueError(f"{column} column not found.")

        if market.rows < self.MIN_ROWS:
            raise ValueError(
                f"Insufficient rows for FVG detection: "
                f"need at least {self.MIN_ROWS}, got {market.rows}."
            )

    @staticmethod
    def _empty_price_series(index: pd.Index) -> pd.Series:
        """
        Create an empty price series initialized with ``NaN``.

        Parameters
        ----------
        index : pd.Index
            Index aligned with the market data frame.

        Returns
        -------
        pd.Series
            Empty float series.
        """
        return pd.Series(pd.NA, index=index, dtype="Float64")
