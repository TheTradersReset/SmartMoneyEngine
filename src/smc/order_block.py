from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import pandas as pd

from src.models.market_data import MarketData
from src.smc.base_smc import BaseSMC


class OrderBlockDirection(str, Enum):
    """Order block side."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


@dataclass(frozen=True)
class OrderBlockRecord:
    """
    Immutable order block record for downstream module consumption.

    Attributes
    ----------
    direction : OrderBlockDirection
        Order block side.
    index : object
        Index label of the originating order block candle.
    position : int
        Integer position of the originating order block candle.
    open : float
        Open price of the order block candle.
    high : float
        High price of the order block candle.
    low : float
        Low price of the order block candle.
    close : float
        Close price of the order block candle.
    bos_index : object
        Index label of the confirming BOS candle.
    bos_position : int
        Integer position of the confirming BOS candle.
    mitigated : bool
        Whether the order block has been mitigated.
    mitigation_index : object | None
        Index label where mitigation occurred, if mitigated.
    """

    direction: OrderBlockDirection
    index: object
    position: int
    open: float
    high: float
    low: float
    close: float
    bos_index: object
    bos_position: int
    mitigated: bool = False
    mitigation_index: object | None = None


@dataclass
class _ActiveOrderBlock:
    """Mutable order block used during detection and mitigation scans."""

    record: OrderBlockRecord
    mitigated: bool = False
    mitigation_index: object | None = None


class OrderBlockDetector(BaseSMC):
    """
    Detect institutional order blocks from confirmed BOS displacement.

    A bullish order block is the final bearish candle immediately before
    an impulsive bullish displacement that produces a valid ``Bullish_BOS``.
    A bearish order block is the final bullish candle immediately before
    an impulsive bearish displacement that produces a valid ``Bearish_BOS``.

    Detected blocks are exposed through ``order_blocks`` for downstream
    modules such as liquidity mapping and signal generation.
    """

    OPEN_COLUMN = "Open"
    HIGH_COLUMN = "High"
    LOW_COLUMN = "Low"
    CLOSE_COLUMN = "Close"
    BULLISH_BOS_COLUMN = "Bullish_BOS"
    BEARISH_BOS_COLUMN = "Bearish_BOS"

    BULLISH_OB_HIGH_COLUMN = "Bullish_OB_High"
    BULLISH_OB_LOW_COLUMN = "Bullish_OB_Low"
    BEARISH_OB_HIGH_COLUMN = "Bearish_OB_High"
    BEARISH_OB_LOW_COLUMN = "Bearish_OB_Low"
    BULLISH_OB_MITIGATED_COLUMN = "Bullish_OB_Mitigated"
    BEARISH_OB_MITIGATED_COLUMN = "Bearish_OB_Mitigated"

    REQUIRED_COLUMNS = (
        OPEN_COLUMN,
        HIGH_COLUMN,
        LOW_COLUMN,
        CLOSE_COLUMN,
        BULLISH_BOS_COLUMN,
        BEARISH_BOS_COLUMN,
    )

    DEFAULT_ROLLING_WINDOW = 14
    DEFAULT_MIN_BODY_RATIO = 0.35
    DEFAULT_MIN_DISPLACEMENT_BODY_RATIO = 0.50
    DEFAULT_MIN_DISPLACEMENT_MULTIPLIER = 1.25
    DEFAULT_EQUAL_LEVEL_TOLERANCE_RATIO = 0.05
    DEFAULT_OVERLAP_THRESHOLD = 0.50

    def __init__(
        self,
        rolling_window: int = DEFAULT_ROLLING_WINDOW,
        min_body_ratio: float = DEFAULT_MIN_BODY_RATIO,
        min_displacement_body_ratio: float = DEFAULT_MIN_DISPLACEMENT_BODY_RATIO,
        min_displacement_multiplier: float = DEFAULT_MIN_DISPLACEMENT_MULTIPLIER,
        equal_level_tolerance_ratio: float = DEFAULT_EQUAL_LEVEL_TOLERANCE_RATIO,
        overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
    ) -> None:
        if rolling_window <= 1:
            raise ValueError("rolling_window must be greater than one.")
        if not 0.0 < min_body_ratio <= 1.0:
            raise ValueError("min_body_ratio must be between 0 and 1.")
        if not 0.0 < min_displacement_body_ratio <= 1.0:
            raise ValueError("min_displacement_body_ratio must be between 0 and 1.")
        if min_displacement_multiplier <= 0.0:
            raise ValueError("min_displacement_multiplier must be positive.")
        if not 0.0 < equal_level_tolerance_ratio <= 1.0:
            raise ValueError("equal_level_tolerance_ratio must be between 0 and 1.")
        if not 0.0 < overlap_threshold <= 1.0:
            raise ValueError("overlap_threshold must be between 0 and 1.")

        super().__init__("Order Block")
        self.rolling_window = rolling_window
        self.min_body_ratio = min_body_ratio
        self.min_displacement_body_ratio = min_displacement_body_ratio
        self.min_displacement_multiplier = min_displacement_multiplier
        self.equal_level_tolerance_ratio = equal_level_tolerance_ratio
        self.overlap_threshold = overlap_threshold
        self._order_blocks: list[OrderBlockRecord] = ()

    @property
    def order_blocks(self) -> tuple[OrderBlockRecord, ...]:
        """
        Return detected order blocks for downstream module consumption.

        Returns
        -------
        tuple[OrderBlockRecord, ...]
            Immutable order block records detected in the last run.
        """
        return self._order_blocks

    def detect(self, market: MarketData) -> MarketData:
        """
        Detect bullish and bearish order blocks.

        Parameters
        ----------
        market : MarketData
            Market data containing OHLC and BOS columns.

        Returns
        -------
        MarketData
            Same instance with order block columns added.

        Raises
        ------
        ValueError
            If required columns are missing.
        """
        self.log_start()
        self._validate_market(market)

        active_blocks = self._detect_order_blocks(market)
        self._apply_mitigation(active_blocks, market)
        self._order_blocks = tuple(
            self._finalize_record(block) for block in active_blocks
        )

        self._write_output_columns(market, active_blocks)

        self.log_finish()
        return market

    def _detect_order_blocks(self, market: MarketData) -> list[_ActiveOrderBlock]:
        """
        Detect raw order blocks from confirmed BOS events.

        Parameters
        ----------
        market : MarketData
            Market data containing OHLC and BOS columns.

        Returns
        -------
        list[_ActiveOrderBlock]
            Active order blocks prior to mitigation projection.
        """
        index = market.get_column(self.OPEN_COLUMN).index
        opens = market.get_column(self.OPEN_COLUMN)
        highs = market.get_column(self.HIGH_COLUMN)
        lows = market.get_column(self.LOW_COLUMN)
        closes = market.get_column(self.CLOSE_COLUMN)
        bullish_bos = market.get_column(self.BULLISH_BOS_COLUMN)
        bearish_bos = market.get_column(self.BEARISH_BOS_COLUMN)

        average_body = self._rolling_average_body(opens, closes)
        average_range = self._rolling_average_range(highs, lows)

        active_blocks: list[_ActiveOrderBlock] = []

        for position, row_index in enumerate(index):
            if pd.notna(bullish_bos.loc[row_index]):
                block = self._build_bullish_order_block(
                    bos_position=position,
                    bos_index=row_index,
                    opens=opens,
                    highs=highs,
                    lows=lows,
                    closes=closes,
                    average_body=average_body,
                    average_range=average_range,
                    active_blocks=active_blocks,
                )
                if block is not None:
                    active_blocks.append(block)

            if pd.notna(bearish_bos.loc[row_index]):
                block = self._build_bearish_order_block(
                    bos_position=position,
                    bos_index=row_index,
                    opens=opens,
                    highs=highs,
                    lows=lows,
                    closes=closes,
                    average_body=average_body,
                    average_range=average_range,
                    active_blocks=active_blocks,
                )
                if block is not None:
                    active_blocks.append(block)

        return active_blocks

    def _build_bullish_order_block(
        self,
        bos_position: int,
        bos_index: object,
        opens: pd.Series,
        highs: pd.Series,
        lows: pd.Series,
        closes: pd.Series,
        average_body: pd.Series,
        average_range: pd.Series,
        active_blocks: list[_ActiveOrderBlock],
    ) -> _ActiveOrderBlock | None:
        """
        Build a bullish order block candidate from a bullish BOS event.

        Returns
        -------
        _ActiveOrderBlock | None
            Active order block when valid, otherwise ``None``.
        """
        if not self._is_impulsive_displacement(
            position=bos_position,
            direction=OrderBlockDirection.BULLISH,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            average_body=average_body,
            average_range=average_range,
        ):
            return None

        displacement_start = self._find_displacement_start(
            bos_position=bos_position,
            direction=OrderBlockDirection.BULLISH,
            closes=closes,
        )
        if displacement_start <= 0:
            return None

        order_block_position = self._find_origin_candle_position(
            start_position=displacement_start - 1,
            direction=OrderBlockDirection.BULLISH,
            opens=opens,
            closes=closes,
        )
        if order_block_position is None:
            return None

        if not self._is_valid_origin_candle(
            position=order_block_position,
            direction=OrderBlockDirection.BULLISH,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            average_body=average_body,
        ):
            return None

        if self._has_equal_level(
            position=order_block_position,
            level_kind="high",
            highs=highs,
            lows=lows,
        ):
            return None

        record = self._create_record(
            direction=OrderBlockDirection.BULLISH,
            order_block_position=order_block_position,
            bos_index=bos_index,
            bos_position=bos_position,
            index=opens.index,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
        )

        if self._has_overlapping_duplicate(record, active_blocks):
            return None

        return _ActiveOrderBlock(record=record)

    def _build_bearish_order_block(
        self,
        bos_position: int,
        bos_index: object,
        opens: pd.Series,
        highs: pd.Series,
        lows: pd.Series,
        closes: pd.Series,
        average_body: pd.Series,
        average_range: pd.Series,
        active_blocks: list[_ActiveOrderBlock],
    ) -> _ActiveOrderBlock | None:
        """
        Build a bearish order block candidate from a bearish BOS event.

        Returns
        -------
        _ActiveOrderBlock | None
            Active order block when valid, otherwise ``None``.
        """
        if not self._is_impulsive_displacement(
            position=bos_position,
            direction=OrderBlockDirection.BEARISH,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            average_body=average_body,
            average_range=average_range,
        ):
            return None

        displacement_start = self._find_displacement_start(
            bos_position=bos_position,
            direction=OrderBlockDirection.BEARISH,
            closes=closes,
        )
        if displacement_start <= 0:
            return None

        order_block_position = self._find_origin_candle_position(
            start_position=displacement_start - 1,
            direction=OrderBlockDirection.BEARISH,
            opens=opens,
            closes=closes,
        )
        if order_block_position is None:
            return None

        if not self._is_valid_origin_candle(
            position=order_block_position,
            direction=OrderBlockDirection.BEARISH,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            average_body=average_body,
        ):
            return None

        if self._has_equal_level(
            position=order_block_position,
            level_kind="low",
            highs=highs,
            lows=lows,
        ):
            return None

        record = self._create_record(
            direction=OrderBlockDirection.BEARISH,
            order_block_position=order_block_position,
            bos_index=bos_index,
            bos_position=bos_position,
            index=opens.index,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
        )

        if self._has_overlapping_duplicate(record, active_blocks):
            return None

        return _ActiveOrderBlock(record=record)

    def _find_displacement_start(
        self,
        bos_position: int,
        direction: OrderBlockDirection,
        closes: pd.Series,
    ) -> int:
        """
        Locate the first candle of the displacement leg ending at BOS.

        Parameters
        ----------
        bos_position : int
            Integer position of the BOS confirmation candle.
        direction : OrderBlockDirection
            Expected displacement direction.
        closes : pd.Series
            Close price series.

        Returns
        -------
        int
            Integer position where displacement begins.
        """
        displacement_start = bos_position
        previous_position = bos_position - 1

        while previous_position >= 0:
            current_close = float(closes.iloc[bos_position])
            previous_close = float(closes.iloc[previous_position])
            next_close = float(closes.iloc[previous_position + 1])

            if direction == OrderBlockDirection.BULLISH:
                continues_move = next_close >= previous_close
            else:
                continues_move = next_close <= previous_close

            if not continues_move:
                break

            displacement_start = previous_position
            previous_position -= 1

            if displacement_start == 0:
                break

        return displacement_start

    def _find_origin_candle_position(
        self,
        start_position: int,
        direction: OrderBlockDirection,
        opens: pd.Series,
        closes: pd.Series,
    ) -> int | None:
        """
        Find the final opposing candle immediately before displacement.

        Parameters
        ----------
        start_position : int
            Position to begin scanning backwards from.
        direction : OrderBlockDirection
            Expected order block side.
        opens : pd.Series
            Open price series.
        closes : pd.Series
            Close price series.

        Returns
        -------
        int | None
            Origin candle position when found.
        """
        is_target_candle = self._origin_candle_predicate(direction)

        for position in range(start_position, -1, -1):
            if is_target_candle(position, opens, closes):
                return position

        return None

    @staticmethod
    def _origin_candle_predicate(
        direction: OrderBlockDirection,
    ) -> Callable[[int, pd.Series, pd.Series], bool]:
        """
        Build a predicate that identifies the origin candle for a direction.

        Parameters
        ----------
        direction : OrderBlockDirection
            Expected order block side.

        Returns
        -------
        Callable[[int, pd.Series, pd.Series], bool]
            Predicate returning ``True`` for a valid origin candle.
        """

        def _predicate(position: int, opens: pd.Series, closes: pd.Series) -> bool:
            open_price = float(opens.iloc[position])
            close_price = float(closes.iloc[position])

            if direction == OrderBlockDirection.BULLISH:
                return close_price < open_price

            return close_price > open_price

        return _predicate

    def _is_valid_origin_candle(
        self,
        position: int,
        direction: OrderBlockDirection,
        opens: pd.Series,
        highs: pd.Series,
        lows: pd.Series,
        closes: pd.Series,
        average_body: pd.Series,
    ) -> bool:
        """
        Validate that an origin candle is not weak or doji-like.

        Parameters
        ----------
        position : int
            Candle position to validate.
        direction : OrderBlockDirection
            Expected order block side.
        opens : pd.Series
            Open price series.
        highs : pd.Series
            High price series.
        lows : pd.Series
            Low price series.
        closes : pd.Series
            Close price series.
        average_body : pd.Series
            Rolling average candle body series.

        Returns
        -------
        bool
            ``True`` when the candle passes validation.
        """
        open_price = float(opens.iloc[position])
        close_price = float(closes.iloc[position])
        high_price = float(highs.iloc[position])
        low_price = float(lows.iloc[position])
        candle_range = high_price - low_price
        body = abs(close_price - open_price)

        if candle_range <= 0.0:
            return False

        body_ratio = body / candle_range
        if body_ratio < self.min_body_ratio:
            return False

        rolling_body = float(average_body.iloc[position])
        if rolling_body > 0.0 and body < rolling_body * 0.75:
            return False

        if direction == OrderBlockDirection.BULLISH and close_price >= open_price:
            return False

        if direction == OrderBlockDirection.BEARISH and close_price <= open_price:
            return False

        return True

    def _is_impulsive_displacement(
        self,
        position: int,
        direction: OrderBlockDirection,
        opens: pd.Series,
        highs: pd.Series,
        lows: pd.Series,
        closes: pd.Series,
        average_body: pd.Series,
        average_range: pd.Series,
    ) -> bool:
        """
        Validate that the BOS candle represents impulsive displacement.

        Parameters
        ----------
        position : int
            BOS candle position.
        direction : OrderBlockDirection
            Expected displacement direction.
        opens : pd.Series
            Open price series.
        highs : pd.Series
            High price series.
        lows : pd.Series
            Low price series.
        closes : pd.Series
            Close price series.
        average_body : pd.Series
            Rolling average candle body series.
        average_range : pd.Series
            Rolling average candle range series.

        Returns
        -------
        bool
            ``True`` when displacement is impulsive.
        """
        open_price = float(opens.iloc[position])
        close_price = float(closes.iloc[position])
        high_price = float(highs.iloc[position])
        low_price = float(lows.iloc[position])
        candle_range = high_price - low_price
        body = abs(close_price - open_price)

        if candle_range <= 0.0:
            return False

        body_ratio = body / candle_range
        if body_ratio < self.min_displacement_body_ratio:
            return False

        if direction == OrderBlockDirection.BULLISH and close_price <= open_price:
            return False

        if direction == OrderBlockDirection.BEARISH and close_price >= open_price:
            return False

        rolling_body = float(average_body.iloc[position])
        rolling_range = float(average_range.iloc[position])

        if rolling_body > 0.0 and body < rolling_body * self.min_displacement_multiplier:
            return False

        if rolling_range > 0.0 and candle_range < rolling_range * 0.90:
            return False

        return True

    def _has_equal_level(
        self,
        position: int,
        level_kind: str,
        highs: pd.Series,
        lows: pd.Series,
    ) -> bool:
        """
        Detect equal highs or equal lows around an origin candle.

        Parameters
        ----------
        position : int
            Origin candle position.
        level_kind : str
            ``\"high\"`` or ``\"low\"`` comparison mode.
        highs : pd.Series
            High price series.
        lows : pd.Series
            Low price series.

        Returns
        -------
        bool
            ``True`` when an equal level is detected.
        """
        if level_kind == "high":
            reference = float(highs.iloc[position])
            candle_range = float(highs.iloc[position] - lows.iloc[position])
        else:
            reference = float(lows.iloc[position])
            candle_range = float(highs.iloc[position] - lows.iloc[position])

        if candle_range <= 0.0:
            return False

        tolerance = candle_range * self.equal_level_tolerance_ratio
        comparison_positions = (
            position - 2,
            position - 1,
            position + 1,
            position + 2,
        )

        for compare_position in comparison_positions:
            if compare_position < 0 or compare_position >= len(highs):
                continue

            if level_kind == "high":
                compare_value = float(highs.iloc[compare_position])
            else:
                compare_value = float(lows.iloc[compare_position])

            if abs(reference - compare_value) <= tolerance:
                return True

        return False

    def _has_overlapping_duplicate(
        self,
        candidate: OrderBlockRecord,
        active_blocks: list[_ActiveOrderBlock],
    ) -> bool:
        """
        Prevent overlapping duplicate order blocks of the same direction.

        Parameters
        ----------
        candidate : OrderBlockRecord
            Candidate order block record.
        active_blocks : list[_ActiveOrderBlock]
            Already accepted blocks.

        Returns
        -------
        bool
            ``True`` when the candidate overlaps an existing block excessively.
        """
        candidate_range = candidate.high - candidate.low
        if candidate_range <= 0.0:
            return True

        for active_block in active_blocks:
            existing = active_block.record
            if existing.direction != candidate.direction:
                continue

            overlap_low = max(existing.low, candidate.low)
            overlap_high = min(existing.high, candidate.high)
            overlap = overlap_high - overlap_low

            if overlap <= 0.0:
                continue

            overlap_ratio = overlap / candidate_range
            if overlap_ratio >= self.overlap_threshold:
                return True

        return False

    def _create_record(
        self,
        direction: OrderBlockDirection,
        order_block_position: int,
        bos_index: object,
        bos_position: int,
        index: pd.Index,
        opens: pd.Series,
        highs: pd.Series,
        lows: pd.Series,
        closes: pd.Series,
    ) -> OrderBlockRecord:
        """
        Create an immutable order block record.

        Parameters
        ----------
        direction : OrderBlockDirection
            Order block side.
        order_block_position : int
            Origin candle position.
        bos_index : object
            Confirming BOS index label.
        bos_position : int
            Confirming BOS integer position.
        index : pd.Index
            Market index labels.
        opens : pd.Series
            Open price series.
        highs : pd.Series
            High price series.
        lows : pd.Series
            Low price series.
        closes : pd.Series
            Close price series.

        Returns
        -------
        OrderBlockRecord
            Immutable order block record.
        """
        row_index = index[order_block_position]

        return OrderBlockRecord(
            direction=direction,
            index=row_index,
            position=order_block_position,
            open=float(opens.iloc[order_block_position]),
            high=float(highs.iloc[order_block_position]),
            low=float(lows.iloc[order_block_position]),
            close=float(closes.iloc[order_block_position]),
            bos_index=bos_index,
            bos_position=bos_position,
        )

    def _apply_mitigation(
        self,
        active_blocks: list[_ActiveOrderBlock],
        market: MarketData,
    ) -> None:
        """
        Mark order blocks mitigated when price trades through the zone.

        Parameters
        ----------
        active_blocks : list[_ActiveOrderBlock]
            Detected order blocks to evaluate.
        market : MarketData
            Market data containing OHLC columns.
        """
        highs = market.get_column(self.HIGH_COLUMN)
        lows = market.get_column(self.LOW_COLUMN)
        index = highs.index

        for block in active_blocks:
            start_position = block.record.bos_position + 1
            if start_position >= len(index):
                continue

            for position in range(start_position, len(index)):
                row_index = index[position]

                if block.record.direction == OrderBlockDirection.BULLISH:
                    if float(lows.iloc[position]) < block.record.low:
                        block.mitigated = True
                        block.mitigation_index = row_index
                        break
                elif float(highs.iloc[position]) > block.record.high:
                    block.mitigated = True
                    block.mitigation_index = row_index
                    break

    @staticmethod
    def _finalize_record(block: _ActiveOrderBlock) -> OrderBlockRecord:
        """
        Convert an active order block into an immutable finalized record.

        Parameters
        ----------
        block : _ActiveOrderBlock
            Active order block with mitigation state.

        Returns
        -------
        OrderBlockRecord
            Final immutable order block record.
        """
        return OrderBlockRecord(
            direction=block.record.direction,
            index=block.record.index,
            position=block.record.position,
            open=block.record.open,
            high=block.record.high,
            low=block.record.low,
            close=block.record.close,
            bos_index=block.record.bos_index,
            bos_position=block.record.bos_position,
            mitigated=block.mitigated,
            mitigation_index=block.mitigation_index,
        )

    def _write_output_columns(
        self,
        market: MarketData,
        active_blocks: list[_ActiveOrderBlock],
    ) -> None:
        """
        Project detected order blocks onto market data columns.

        Parameters
        ----------
        market : MarketData
            Market data to update.
        active_blocks : list[_ActiveOrderBlock]
            Finalized active order blocks.
        """
        index = market.get_column(self.OPEN_COLUMN).index

        bullish_high = self._empty_price_series(index)
        bullish_low = self._empty_price_series(index)
        bearish_high = self._empty_price_series(index)
        bearish_low = self._empty_price_series(index)
        bullish_mitigated = pd.Series(pd.NA, index=index, dtype="boolean")
        bearish_mitigated = pd.Series(pd.NA, index=index, dtype="boolean")

        for block in active_blocks:
            record = self._finalize_record(block)
            row_index = record.index
            is_mitigated = bool(record.mitigated)

            if record.direction == OrderBlockDirection.BULLISH:
                bullish_high.loc[row_index] = record.high
                bullish_low.loc[row_index] = record.low
                bullish_mitigated.loc[row_index] = is_mitigated
                continue

            bearish_high.loc[row_index] = record.high
            bearish_low.loc[row_index] = record.low
            bearish_mitigated.loc[row_index] = is_mitigated

        market.add_column(self.BULLISH_OB_HIGH_COLUMN, bullish_high)
        market.add_column(self.BULLISH_OB_LOW_COLUMN, bullish_low)
        market.add_column(self.BEARISH_OB_HIGH_COLUMN, bearish_high)
        market.add_column(self.BEARISH_OB_LOW_COLUMN, bearish_low)
        market.add_column(self.BULLISH_OB_MITIGATED_COLUMN, bullish_mitigated)
        market.add_column(self.BEARISH_OB_MITIGATED_COLUMN, bearish_mitigated)

    def _rolling_average_body(
        self,
        opens: pd.Series,
        closes: pd.Series,
    ) -> pd.Series:
        """
        Compute rolling average absolute candle body.

        Parameters
        ----------
        opens : pd.Series
            Open price series.
        closes : pd.Series
            Close price series.

        Returns
        -------
        pd.Series
            Rolling average body series.
        """
        body = (closes.astype(float) - opens.astype(float)).abs()
        return body.rolling(
            window=self.rolling_window,
            min_periods=1,
        ).mean()

    def _rolling_average_range(
        self,
        highs: pd.Series,
        lows: pd.Series,
    ) -> pd.Series:
        """
        Compute rolling average candle range.

        Parameters
        ----------
        highs : pd.Series
            High price series.
        lows : pd.Series
            Low price series.

        Returns
        -------
        pd.Series
            Rolling average range series.
        """
        candle_range = highs.astype(float) - lows.astype(float)
        return candle_range.rolling(
            window=self.rolling_window,
            min_periods=1,
        ).mean()

    def _validate_market(self, market: MarketData) -> None:
        """
        Validate required input columns.

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
                    "Run BreakOfStructure before OrderBlockDetector."
                )

    @staticmethod
    def _empty_price_series(index: pd.Index) -> pd.Series:
        """
        Create an empty float price series.

        Parameters
        ----------
        index : pd.Index
            Index aligned with market data.

        Returns
        -------
        pd.Series
            Empty float series.
        """
        return pd.Series(pd.NA, index=index, dtype="Float64")
