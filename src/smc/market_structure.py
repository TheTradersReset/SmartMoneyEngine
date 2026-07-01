from __future__ import annotations

import pandas as pd

from src.models.market_data import MarketData
from src.smc.base_smc import BaseSMC


class MarketStructure(BaseSMC):
    """
    Classify detected swing points into market structure labels.

    Reads ``Swing_High`` and ``Swing_Low`` columns produced by
    ``SwingDetector`` and labels each subsequent swing relative to
    its predecessor:

    - **HH** — higher high (current swing high > previous swing high)
    - **LH** — lower high (current swing high < previous swing high)
    - **HL** — higher low (current swing low > previous swing low)
    - **LL** — lower low (current swing low < previous swing low)

    Each label column stores the swing price at matching candles;
    all other rows remain ``NaN``. The first swing of each type
    cannot be classified and remains ``NaN``.
    """

    SWING_HIGH_COLUMN = "Swing_High"
    SWING_LOW_COLUMN = "Swing_Low"
    HH_COLUMN = "HH"
    HL_COLUMN = "HL"
    LH_COLUMN = "LH"
    LL_COLUMN = "LL"
    REQUIRED_COLUMNS = (SWING_HIGH_COLUMN, SWING_LOW_COLUMN)
    STRUCTURE_COLUMNS = (HH_COLUMN, HL_COLUMN, LH_COLUMN, LL_COLUMN)

    def __init__(self) -> None:
        super().__init__("Market Structure")

    def detect(self, market: MarketData) -> MarketData:
        """
        Label swing highs and lows with market structure columns.

        Parameters
        ----------
        market : MarketData
            Market data containing ``Swing_High`` and ``Swing_Low`` columns.

        Returns
        -------
        MarketData
            Same instance with ``HH``, ``HL``, ``LH``, and ``LL`` columns
            added. Each column holds the swing price where the label
            applies and ``NaN`` elsewhere.

        Raises
        ------
        ValueError
            If required swing columns are missing.
        """
        self.log_start()
        self._validate_market(market)

        higher_highs, lower_highs = self.classify_swing_highs(market)
        self.name = "Swing High Structure"
        self.log_finish()

        higher_lows, lower_lows = self.classify_swing_lows(market)
        self.name = "Swing Low Structure"
        self.log_finish()

        market.add_column(self.HH_COLUMN, higher_highs)
        market.add_column(self.HL_COLUMN, higher_lows)
        market.add_column(self.LH_COLUMN, lower_highs)
        market.add_column(self.LL_COLUMN, lower_lows)

        self.name = "Market Structure"
        self.log_finish()

        return market

    def classify_swing_highs(
        self,
        market: MarketData,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Classify consecutive swing highs as HH or LH.

        Parameters
        ----------
        market : MarketData
            Market data containing a ``Swing_High`` column.

        Returns
        -------
        tuple[pd.Series, pd.Series]
            Two series for ``HH`` and ``LH`` labels respectively.
            Each stores the swing price where the label applies.
        """
        swing_highs = market.get_column(self.SWING_HIGH_COLUMN)
        hh = self._empty_structure_series(swing_highs.index)
        lh = self._empty_structure_series(swing_highs.index)

        previous_swing_price: float | None = None

        for index, current_price in swing_highs.dropna().items():
            price = float(current_price)

            if previous_swing_price is not None:
                if price > previous_swing_price:
                    hh.loc[index] = price
                elif price < previous_swing_price:
                    lh.loc[index] = price

            previous_swing_price = price

        return hh, lh

    def classify_swing_lows(
        self,
        market: MarketData,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Classify consecutive swing lows as HL or LL.

        Parameters
        ----------
        market : MarketData
            Market data containing a ``Swing_Low`` column.

        Returns
        -------
        tuple[pd.Series, pd.Series]
            Two series for ``HL`` and ``LL`` labels respectively.
            Each stores the swing price where the label applies.
        """
        swing_lows = market.get_column(self.SWING_LOW_COLUMN)
        hl = self._empty_structure_series(swing_lows.index)
        ll = self._empty_structure_series(swing_lows.index)

        previous_swing_price: float | None = None

        for index, current_price in swing_lows.dropna().items():
            price = float(current_price)

            if previous_swing_price is not None:
                if price > previous_swing_price:
                    hl.loc[index] = price
                elif price < previous_swing_price:
                    ll.loc[index] = price

            previous_swing_price = price

        return hl, ll

    def _validate_market(self, market: MarketData) -> None:
        """
        Validate that required swing columns are present.

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
                    "Run SwingDetector before MarketStructure."
                )

    @staticmethod
    def _empty_structure_series(index: pd.Index) -> pd.Series:
        """
        Create an empty market-structure price series.

        Parameters
        ----------
        index : pd.Index
            Index aligned with the market data frame.

        Returns
        -------
        pd.Series
            Series initialized with ``NaN`` for every row.
        """
        return pd.Series(pd.NA, index=index, dtype="Float64")
