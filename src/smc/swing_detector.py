from __future__ import annotations

from typing import Callable

import pandas as pd

from src.models.market_data import MarketData
from src.smc.base_smc import BaseSMC


class SwingDetector(BaseSMC):
    """
    Detect swing highs and swing lows in OHLC market data.

    A swing high occurs when the high at index ``i`` is strictly
    greater than the highs of ``lookback`` candles on each side.
    A swing low occurs when the low at index ``i`` is strictly
    lower than the lows of ``lookback`` candles on each side.

    Detected swings are stored as actual price values; non-swing
    and edge candles are stored as ``NaN``.

    Parameters
    ----------
    lookback : int, default=2
        Number of candles on each side used for comparison.
    """

    SWING_HIGH_COLUMN = "Swing_High"
    SWING_LOW_COLUMN = "Swing_Low"
    REQUIRED_COLUMNS = ("High", "Low")

    def __init__(self, lookback: int = 2) -> None:
        if lookback <= 0:
            raise ValueError("lookback must be greater than zero.")

        super().__init__("Swing Detection")
        self.lookback = lookback

    def detect(self, market: MarketData) -> MarketData:
        """
        Detect swing highs and lows and append price columns.

        Parameters
        ----------
        market : MarketData
            Market data wrapper containing OHLC columns.

        Returns
        -------
        MarketData
            Same instance with ``Swing_High`` and ``Swing_Low`` columns added.
            Each column holds the swing price at detected candles and
            ``NaN`` elsewhere (including edge candles).

        Raises
        ------
        ValueError
            If required columns are missing or row count is insufficient.
        """
        self.log_start()
        self._validate_market(market)

        swing_highs = self.detect_highs(market)
        self.name = "Swing High Detection"
        self.log_finish()

        swing_lows = self.detect_lows(market)
        self.name = "Swing Low Detection"
        self.log_finish()

        market.add_column(self.SWING_HIGH_COLUMN, swing_highs)
        market.add_column(self.SWING_LOW_COLUMN, swing_lows)

        self.name = "Swing Detection"
        self.log_finish()

        return market

    def detect_highs(self, market: MarketData) -> pd.Series:
        """
        Identify swing high prices.

        Parameters
        ----------
        market : MarketData
            Market data wrapper containing a ``High`` column.

        Returns
        -------
        pd.Series
            Series containing the high price at swing highs and ``NaN``
            at all other candles (including edges).
        """
        high = market.get_column("High")
        return self._detect_swing_prices(
            values=high,
            comparator=lambda current, neighbor: current > neighbor,
        )

    def detect_lows(self, market: MarketData) -> pd.Series:
        """
        Identify swing low prices.

        Parameters
        ----------
        market : MarketData
            Market data wrapper containing a ``Low`` column.

        Returns
        -------
        pd.Series
            Series containing the low price at swing lows and ``NaN``
            at all other candles (including edges).
        """
        low = market.get_column("Low")
        return self._detect_swing_prices(
            values=low,
            comparator=lambda current, neighbor: current < neighbor,
        )

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

        min_rows = self.lookback * 2 + 1
        if market.rows < min_rows:
            raise ValueError(
                f"Insufficient rows for swing detection: "
                f"need at least {min_rows}, got {market.rows}."
            )

    def _detect_swing_prices(
        self,
        values: pd.Series,
        comparator: Callable[[float, float], bool],
    ) -> pd.Series:
        """
        Detect swing points and return actual price values.

        Parameters
        ----------
        values : pd.Series
            Price series to evaluate (``High`` or ``Low``).
        comparator : Callable[[float, float], bool]
            Comparison function returning ``True`` when ``current`` qualifies
            against ``neighbor`` (strict greater-than for highs,
            strict less-than for lows).

        Returns
        -------
        pd.Series
            Series with swing prices at detected candles and ``NaN`` elsewhere.
        """
        swing_prices = pd.Series(
            pd.NA,
            index=values.index,
            dtype="Float64",
        )
        total_rows = len(values)

        for index in range(self.lookback, total_rows - self.lookback):
            current_value = float(values.iloc[index])
            is_swing = True

            for offset in range(1, self.lookback + 1):
                if not comparator(
                    current_value,
                    float(values.iloc[index - offset]),
                ):
                    is_swing = False
                    break
                if not comparator(
                    current_value,
                    float(values.iloc[index + offset]),
                ):
                    is_swing = False
                    break

            if is_swing:
                swing_prices.iloc[index] = current_value

        return swing_prices
