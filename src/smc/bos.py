from __future__ import annotations

import pandas as pd

from src.models.market_data import MarketData
from src.smc.base_smc import BaseSMC


class BreakOfStructure(BaseSMC):
    """
    Detect Break of Structure (BOS) events from market structure labels.

    Reads ``HH``, ``HL``, ``LH``, and ``LL`` columns produced by
    ``MarketStructure`` and compares candle closes against the most
    recent structural levels:

    - **Bullish BOS** — close breaks above the previous Higher High
    - **Bearish BOS** — close breaks below the previous Lower Low

    Break prices are stored at the confirming candle; all other
    rows remain ``NaN``.
    """

    HH_COLUMN = "HH"
    HL_COLUMN = "HL"
    LH_COLUMN = "LH"
    LL_COLUMN = "LL"
    CLOSE_COLUMN = "Close"
    BULLISH_BOS_COLUMN = "Bullish_BOS"
    BEARISH_BOS_COLUMN = "Bearish_BOS"
    REQUIRED_COLUMNS = (HH_COLUMN, HL_COLUMN, LH_COLUMN, LL_COLUMN, CLOSE_COLUMN)

    def __init__(self) -> None:
        super().__init__("Break of Structure")

    def detect(self, market: MarketData) -> MarketData:
        """
        Detect bullish and bearish breaks of structure.

        Parameters
        ----------
        market : MarketData
            Market data containing structure label and ``Close`` columns.

        Returns
        -------
        MarketData
            Same instance with ``Bullish_BOS`` and ``Bearish_BOS`` columns
            added. Each column stores the break price where applicable.

        Raises
        ------
        ValueError
            If required columns are missing.
        """
        self.log_start()
        self._validate_market(market)

        bullish_bos, bearish_bos = self.detect_breaks(market)

        market.add_column(self.BULLISH_BOS_COLUMN, bullish_bos)
        market.add_column(self.BEARISH_BOS_COLUMN, bearish_bos)

        self.log_finish()

        return market

    def detect_breaks(
        self,
        market: MarketData,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Scan closes for bullish and bearish structure breaks.

        Parameters
        ----------
        market : MarketData
            Market data containing structure and close columns.

        Returns
        -------
        tuple[pd.Series, pd.Series]
            Bullish and bearish BOS price series respectively.
        """
        index = market.get_column(self.CLOSE_COLUMN).index
        closes = market.get_column(self.CLOSE_COLUMN)
        higher_highs = market.get_column(self.HH_COLUMN)
        lower_lows = market.get_column(self.LL_COLUMN)

        bullish_bos = self._empty_price_series(index)
        bearish_bos = self._empty_price_series(index)

        last_hh_price: float | None = None
        last_ll_price: float | None = None

        for row_index in index:
            close_price = float(closes.loc[row_index])
            previous_close = (
                float(closes.shift(1).loc[row_index])
                if row_index != index[0]
                else None
            )

            if last_hh_price is not None and close_price > last_hh_price:
                if previous_close is None or previous_close <= last_hh_price:
                    bullish_bos.loc[row_index] = close_price

            if last_ll_price is not None and close_price < last_ll_price:
                if previous_close is None or previous_close >= last_ll_price:
                    bearish_bos.loc[row_index] = close_price

            if pd.notna(higher_highs.loc[row_index]):
                last_hh_price = float(higher_highs.loc[row_index])

            if pd.notna(lower_lows.loc[row_index]):
                last_ll_price = float(lower_lows.loc[row_index])

        return bullish_bos, bearish_bos

    def _validate_market(self, market: MarketData) -> None:
        """
        Validate that required columns are present.

        Parameters
        ----------
        market : MarketData
            Market data to validate.

        Raises
        ------
        ValueError
            If a required column is missing.
        """
        for column in self.REQUIRED_COLUMNS:
            if not market.has_column(column):
                raise ValueError(
                    f"{column} column not found. "
                    "Run MarketStructure before BreakOfStructure."
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
