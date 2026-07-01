from __future__ import annotations

from enum import Enum

import pandas as pd

from src.models.market_data import MarketData
from src.smc.base_smc import BaseSMC


class _StructureBias(str, Enum):
    """Internal market structure bias states."""

    BULLISH = "bullish"
    BEARISH = "bearish"


class ChangeOfCharacter(BaseSMC):
    """
    Detect Change of Character (CHOCH) events from market structure.

    Reads ``HH``, ``HL``, ``LH``, and ``LL`` columns produced by
    ``MarketStructure`` and identifies early reversals:

    - **Bullish CHOCH** — after bearish structure, close breaks above
      the previous Lower High
    - **Bearish CHOCH** — after bullish structure, close breaks below
      the previous Higher Low

    Break prices are stored at the confirming candle; all other
    rows remain ``NaN``.
    """

    HH_COLUMN = "HH"
    HL_COLUMN = "HL"
    LH_COLUMN = "LH"
    LL_COLUMN = "LL"
    CLOSE_COLUMN = "Close"
    BULLISH_CHOCH_COLUMN = "Bullish_CHOCH"
    BEARISH_CHOCH_COLUMN = "Bearish_CHOCH"
    REQUIRED_COLUMNS = (HH_COLUMN, HL_COLUMN, LH_COLUMN, LL_COLUMN, CLOSE_COLUMN)

    def __init__(self) -> None:
        super().__init__("Change of Character")

    def detect(self, market: MarketData) -> MarketData:
        """
        Detect bullish and bearish changes of character.

        Parameters
        ----------
        market : MarketData
            Market data containing structure label and ``Close`` columns.

        Returns
        -------
        MarketData
            Same instance with ``Bullish_CHOCH`` and ``Bearish_CHOCH``
            columns added. Each column stores the break price where
            applicable.

        Raises
        ------
        ValueError
            If required columns are missing.
        """
        self.log_start()
        self._validate_market(market)

        bullish_choch, bearish_choch = self.detect_changes(market)

        market.add_column(self.BULLISH_CHOCH_COLUMN, bullish_choch)
        market.add_column(self.BEARISH_CHOCH_COLUMN, bearish_choch)

        self.log_finish()

        return market

    def detect_changes(
        self,
        market: MarketData,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Scan closes for bullish and bearish character changes.

        Parameters
        ----------
        market : MarketData
            Market data containing structure and close columns.

        Returns
        -------
        tuple[pd.Series, pd.Series]
            Bullish and bearish CHOCH price series respectively.
        """
        index = market.get_column(self.CLOSE_COLUMN).index
        closes = market.get_column(self.CLOSE_COLUMN)
        higher_highs = market.get_column(self.HH_COLUMN)
        higher_lows = market.get_column(self.HL_COLUMN)
        lower_highs = market.get_column(self.LH_COLUMN)
        lower_lows = market.get_column(self.LL_COLUMN)

        bullish_choch = self._empty_price_series(index)
        bearish_choch = self._empty_price_series(index)

        structure_bias: _StructureBias | None = None
        last_lh_price: float | None = None
        last_hl_price: float | None = None

        for row_index in index:
            close_price = float(closes.loc[row_index])
            previous_close = (
                float(closes.shift(1).loc[row_index])
                if row_index != index[0]
                else None
            )

            if (
                structure_bias == _StructureBias.BEARISH
                and last_lh_price is not None
                and close_price > last_lh_price
                and (
                    previous_close is None
                    or previous_close <= last_lh_price
                )
            ):
                bullish_choch.loc[row_index] = close_price

            if (
                structure_bias == _StructureBias.BULLISH
                and last_hl_price is not None
                and close_price < last_hl_price
                and (
                    previous_close is None
                    or previous_close >= last_hl_price
                )
            ):
                bearish_choch.loc[row_index] = close_price

            structure_bias = self._update_structure_bias(
                structure_bias=structure_bias,
                higher_high=higher_highs.loc[row_index],
                higher_low=higher_lows.loc[row_index],
                lower_high=lower_highs.loc[row_index],
                lower_low=lower_lows.loc[row_index],
            )

            if pd.notna(lower_highs.loc[row_index]):
                last_lh_price = float(lower_highs.loc[row_index])

            if pd.notna(higher_lows.loc[row_index]):
                last_hl_price = float(higher_lows.loc[row_index])

        return bullish_choch, bearish_choch

    @staticmethod
    def _update_structure_bias(
        structure_bias: _StructureBias | None,
        higher_high: float,
        higher_low: float,
        lower_high: float,
        lower_low: float,
    ) -> _StructureBias | None:
        """
        Update structure bias from labels present on the current candle.

        Parameters
        ----------
        structure_bias : _StructureBias | None
            Current structure bias.
        higher_high : float
            HH label value at the current candle.
        higher_low : float
            HL label value at the current candle.
        lower_high : float
            LH label value at the current candle.
        lower_low : float
            LL label value at the current candle.

        Returns
        -------
        _StructureBias | None
            Updated structure bias.
        """
        if pd.notna(higher_high) or pd.notna(higher_low):
            return _StructureBias.BULLISH

        if pd.notna(lower_high) or pd.notna(lower_low):
            return _StructureBias.BEARISH

        return structure_bias

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
                    "Run MarketStructure before ChangeOfCharacter."
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
