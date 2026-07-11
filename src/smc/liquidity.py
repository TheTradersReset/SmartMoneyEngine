from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from src.models.market_data import MarketData
from src.smc.base_smc import BaseSMC


class LiquiditySide(str, Enum):
    """Liquidity pool side."""

    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class SwingPoint:
    """
    Confirmed swing observation extracted from market structure.

    Attributes
    ----------
    index : object
        Index label of the swing candle.
    position : int
        Integer position of the swing candle.
    price : float
        Swing price value.
    """

    index: object
    position: int
    price: float


@dataclass(frozen=True)
class LiquidityCluster:
    """
    Equal-high or equal-low liquidity cluster.

    Attributes
    ----------
    side : LiquiditySide
        Cluster side. ``BUY`` for equal highs, ``SELL`` for equal lows.
    level : float
        Institutional liquidity pool level.
    touches : tuple[SwingPoint, ...]
        Swing points forming the cluster.
    strength : int
        Liquidity strength score from touch count.
    confirmed_position : int
        Integer position where the cluster became valid.
    confirmed_index : object
        Index label where the cluster became valid.
    """

    side: LiquiditySide
    level: float
    touches: tuple[SwingPoint, ...]
    strength: int
    confirmed_position: int
    confirmed_index: object


@dataclass(frozen=True)
class LiquidityPoolRecord:
    """
    Immutable liquidity pool record for downstream consumption.

    Attributes
    ----------
    side : LiquiditySide
        Pool side.
    level : float
        Pool price level.
    strength : int
        Liquidity strength score.
    confirmed_index : object
        Index where the pool became active.
    confirmed_position : int
        Position where the pool became active.
    swept : bool
        Whether the pool has been swept.
    sweep_index : object | None
        Index where sweep occurred, if swept.
    sweep_price : float | None
        Sweep price when swept.
    """

    side: LiquiditySide
    level: float
    strength: int
    confirmed_index: object
    confirmed_position: int
    swept: bool = False
    sweep_index: object | None = None
    sweep_price: float | None = None


@dataclass
class _LiquidityOutput:
    """Container for projected liquidity column data."""

    equal_high: pd.Series
    equal_low: pd.Series
    buy_side_liquidity: pd.Series
    sell_side_liquidity: pd.Series
    buy_liquidity_sweep: pd.Series
    sell_liquidity_sweep: pd.Series
    liquidity_strength: pd.Series


class LiquidityDetector(BaseSMC):
    """
    Detect institutional liquidity from confirmed swing points.

    Equal highs and equal lows are formed exclusively from ``Swing_High``
    and ``Swing_Low`` observations. Buy-side liquidity pools form at
    equal-high clusters; sell-side liquidity pools form at equal-low
    clusters. Sweeps are detected when price wicks through a pool but
    closes back on the opposing side of the level.
    """

    SWING_HIGH_COLUMN = "Swing_High"
    SWING_LOW_COLUMN = "Swing_Low"
    HIGH_COLUMN = "High"
    LOW_COLUMN = "Low"
    CLOSE_COLUMN = "Close"

    EQUAL_HIGH_COLUMN = "Equal_High"
    EQUAL_LOW_COLUMN = "Equal_Low"
    BUY_SIDE_LIQUIDITY_COLUMN = "Buy_Side_Liquidity"
    SELL_SIDE_LIQUIDITY_COLUMN = "Sell_Side_Liquidity"
    BUY_LIQUIDITY_SWEEP_COLUMN = "Buy_Liquidity_Sweep"
    SELL_LIQUIDITY_SWEEP_COLUMN = "Sell_Liquidity_Sweep"
    LIQUIDITY_STRENGTH_COLUMN = "Liquidity_Strength"

    REQUIRED_COLUMNS = (SWING_HIGH_COLUMN, SWING_LOW_COLUMN, HIGH_COLUMN, LOW_COLUMN, CLOSE_COLUMN)
    DEFAULT_TOLERANCE_RATIO = 0.001

    STRENGTH_TWO_TOUCHES = 1
    STRENGTH_THREE_TOUCHES = 2
    STRENGTH_FOUR_OR_MORE = 3

    def __init__(self, tolerance_ratio: float = DEFAULT_TOLERANCE_RATIO) -> None:
        if tolerance_ratio <= 0.0:
            raise ValueError("tolerance_ratio must be greater than zero.")

        super().__init__("Liquidity")
        self.tolerance_ratio = tolerance_ratio
        self._liquidity_pools: tuple[LiquidityPoolRecord, ...] = ()

    @property
    def liquidity_pools(self) -> tuple[LiquidityPoolRecord, ...]:
        """
        Return detected liquidity pools for downstream module consumption.

        Returns
        -------
        tuple[LiquidityPoolRecord, ...]
            Immutable liquidity pool records from the last run.
        """
        return self._liquidity_pools

    def detect(self, market: MarketData) -> MarketData:
        """
        Detect equal highs/lows, liquidity pools, sweeps, and strength.

        Parameters
        ----------
        market : MarketData
            Market data containing swing and OHLC columns.

        Returns
        -------
        MarketData
            Same instance with liquidity columns added.

        Raises
        ------
        ValueError
            If required columns are missing.
        """
        self.log_start()
        self._validate_market(market)

        index = market.get_column(self.SWING_HIGH_COLUMN).index
        swing_highs = self._extract_swing_points(
            market.get_column(self.SWING_HIGH_COLUMN),
        )
        swing_lows = self._extract_swing_points(
            market.get_column(self.SWING_LOW_COLUMN),
        )

        equal_high_clusters = self._cluster_swings(
            swings=swing_highs,
            side=LiquiditySide.BUY,
        )
        equal_low_clusters = self._cluster_swings(
            swings=swing_lows,
            side=LiquiditySide.SELL,
        )

        output = self._initialize_output(index)
        self._project_equal_levels(output, equal_high_clusters, equal_low_clusters)
        pool_records = self._project_liquidity_pools(output, equal_high_clusters, equal_low_clusters)
        self._detect_sweeps(output, pool_records, market)

        self._liquidity_pools = tuple(pool_records)
        self._write_output_columns(market, output)

        self.log_finish()
        return market

    def _extract_swing_points(self, swing_series: pd.Series) -> list[SwingPoint]:
        """
        Extract confirmed swing observations from a swing price series.

        Parameters
        ----------
        swing_series : pd.Series
            Swing price series containing ``NaN`` on non-swing candles.

        Returns
        -------
        list[SwingPoint]
            Chronologically ordered swing points.
        """
        swings: list[SwingPoint] = []

        for position, (row_index, value) in enumerate(swing_series.items()):
            if pd.notna(value):
                swings.append(
                    SwingPoint(
                        index=row_index,
                        position=position,
                        price=float(value),
                    )
                )

        return swings

    def _cluster_swings(
        self,
        swings: list[SwingPoint],
        side: LiquiditySide,
    ) -> list[LiquidityCluster]:
        """
        Group swing points into equal-high or equal-low clusters.

        Parameters
        ----------
        swings : list[SwingPoint]
            Chronologically ordered swing points.
        side : LiquiditySide
            Cluster side.

        Returns
        -------
        list[LiquidityCluster]
            Valid clusters with two or more touches.
        """
        raw_clusters: list[list[SwingPoint]] = []

        for swing in swings:
            matched_cluster: list[SwingPoint] | None = None

            for cluster in raw_clusters:
                if self._is_within_tolerance(swing.price, cluster):
                    matched_cluster = cluster
                    break

            if matched_cluster is None:
                raw_clusters.append([swing])
                continue

            matched_cluster.append(swing)

        clusters: list[LiquidityCluster] = []

        for cluster_swings in raw_clusters:
            if len(cluster_swings) < 2:
                continue

            level = self._cluster_level(cluster_swings, side)
            strength = self._calculate_strength(len(cluster_swings))
            confirmed = cluster_swings[1]

            clusters.append(
                LiquidityCluster(
                    side=side,
                    level=level,
                    touches=tuple(cluster_swings),
                    strength=strength,
                    confirmed_position=confirmed.position,
                    confirmed_index=confirmed.index,
                )
            )

        return clusters

    def _is_within_tolerance(
        self,
        price: float,
        cluster: list[SwingPoint],
    ) -> bool:
        """
        Check whether a price belongs to an existing swing cluster.

        Parameters
        ----------
        price : float
            Candidate swing price.
        cluster : list[SwingPoint]
            Existing cluster members.

        Returns
        -------
        bool
            ``True`` when the price is within tolerance of any member.
        """
        for member in cluster:
            midpoint = (price + member.price) / 2.0
            if midpoint <= 0.0:
                continue

            if abs(price - member.price) / midpoint <= self.tolerance_ratio:
                return True

        return False

    @staticmethod
    def _cluster_level(cluster: list[SwingPoint], side: LiquiditySide) -> float:
        """
        Derive the institutional liquidity level for a cluster.

        Parameters
        ----------
        cluster : list[SwingPoint]
            Cluster members.
        side : LiquiditySide
            Cluster side.

        Returns
        -------
        float
            Liquidity pool level.
        """
        prices = [swing.price for swing in cluster]

        if side == LiquiditySide.BUY:
            return max(prices)

        return min(prices)

    @classmethod
    def _calculate_strength(cls, touch_count: int) -> int:
        """
        Convert touch count into liquidity strength.

        Parameters
        ----------
        touch_count : int
            Number of swing touches in the cluster.

        Returns
        -------
        int
            Strength score between 1 and 3.
        """
        if touch_count >= 4:
            return cls.STRENGTH_FOUR_OR_MORE

        if touch_count == 3:
            return cls.STRENGTH_THREE_TOUCHES

        return cls.STRENGTH_TWO_TOUCHES

    def _initialize_output(self, index: pd.Index) -> _LiquidityOutput:
        """
        Initialize empty liquidity output series.

        Parameters
        ----------
        index : pd.Index
            Market index.

        Returns
        -------
        _LiquidityOutput
            Empty output container.
        """
        return _LiquidityOutput(
            equal_high=self._empty_price_series(index),
            equal_low=self._empty_price_series(index),
            buy_side_liquidity=self._empty_price_series(index),
            sell_side_liquidity=self._empty_price_series(index),
            buy_liquidity_sweep=self._empty_price_series(index),
            sell_liquidity_sweep=self._empty_price_series(index),
            liquidity_strength=pd.Series(pd.NA, index=index, dtype="Int64"),
        )

    def _project_equal_levels(
        self,
        output: _LiquidityOutput,
        equal_high_clusters: list[LiquidityCluster],
        equal_low_clusters: list[LiquidityCluster],
    ) -> None:
        """
        Project equal-high and equal-low values onto swing candles.

        Parameters
        ----------
        output : _LiquidityOutput
            Output container updated in place.
        equal_high_clusters : list[LiquidityCluster]
            Equal-high clusters.
        equal_low_clusters : list[LiquidityCluster]
            Equal-low clusters.
        """
        for cluster in equal_high_clusters:
            for touch in cluster.touches:
                output.equal_high.loc[touch.index] = cluster.level

        for cluster in equal_low_clusters:
            for touch in cluster.touches:
                output.equal_low.loc[touch.index] = cluster.level

    def _project_liquidity_pools(
        self,
        output: _LiquidityOutput,
        equal_high_clusters: list[LiquidityCluster],
        equal_low_clusters: list[LiquidityCluster],
    ) -> list[LiquidityPoolRecord]:
        """
        Project active buy-side and sell-side liquidity pools.

        Parameters
        ----------
        output : _LiquidityOutput
            Output container updated in place.
        equal_high_clusters : list[LiquidityCluster]
            Equal-high clusters.
        equal_low_clusters : list[LiquidityCluster]
            Equal-low clusters.

        Returns
        -------
        list[LiquidityPoolRecord]
            Active pool records prior to sweep evaluation.
        """
        pool_records: list[LiquidityPoolRecord] = []

        for cluster in equal_high_clusters:
            pool_records.append(
                LiquidityPoolRecord(
                    side=LiquiditySide.BUY,
                    level=cluster.level,
                    strength=cluster.strength,
                    confirmed_index=cluster.confirmed_index,
                    confirmed_position=cluster.confirmed_position,
                )
            )

            for position in range(cluster.confirmed_position, len(output.buy_side_liquidity)):
                row_index = output.buy_side_liquidity.index[position]
                output.buy_side_liquidity.loc[row_index] = cluster.level
                output.liquidity_strength.loc[row_index] = cluster.strength

        for cluster in equal_low_clusters:
            pool_records.append(
                LiquidityPoolRecord(
                    side=LiquiditySide.SELL,
                    level=cluster.level,
                    strength=cluster.strength,
                    confirmed_index=cluster.confirmed_index,
                    confirmed_position=cluster.confirmed_position,
                )
            )

            for position in range(cluster.confirmed_position, len(output.sell_side_liquidity)):
                row_index = output.sell_side_liquidity.index[position]
                output.sell_side_liquidity.loc[row_index] = cluster.level

                current_strength = output.liquidity_strength.loc[row_index]
                if pd.isna(current_strength):
                    output.liquidity_strength.loc[row_index] = cluster.strength
                else:
                    output.liquidity_strength.loc[row_index] = max(
                        int(current_strength),
                        cluster.strength,
                    )

        return pool_records

    def _detect_sweeps(
        self,
        output: _LiquidityOutput,
        pool_records: list[LiquidityPoolRecord],
        market: MarketData,
    ) -> None:
        """
        Detect liquidity sweeps against active pool levels.

        Parameters
        ----------
        output : _LiquidityOutput
            Output container updated in place.
        pool_records : list[LiquidityPoolRecord]
            Active liquidity pools.
        market : MarketData
            Market data containing OHLC columns.
        """
        highs = market.get_column(self.HIGH_COLUMN)
        lows = market.get_column(self.LOW_COLUMN)
        closes = market.get_column(self.CLOSE_COLUMN)
        index = highs.index

        finalized_pools: list[LiquidityPoolRecord] = []

        for pool in pool_records:
            swept = False
            sweep_index: object | None = None
            sweep_price: float | None = None

            for position in range(pool.confirmed_position, len(index)):
                row_index = index[position]
                high_price = float(highs.iloc[position])
                low_price = float(lows.iloc[position])
                close_price = float(closes.iloc[position])

                if pool.side == LiquiditySide.BUY and not swept:
                    if high_price > pool.level and close_price < pool.level:
                        output.buy_liquidity_sweep.loc[row_index] = high_price
                        swept = True
                        sweep_index = row_index
                        sweep_price = high_price

                if pool.side == LiquiditySide.SELL and not swept:
                    if low_price < pool.level and close_price > pool.level:
                        output.sell_liquidity_sweep.loc[row_index] = low_price
                        swept = True
                        sweep_index = row_index
                        sweep_price = low_price

            finalized_pools.append(
                LiquidityPoolRecord(
                    side=pool.side,
                    level=pool.level,
                    strength=pool.strength,
                    confirmed_index=pool.confirmed_index,
                    confirmed_position=pool.confirmed_position,
                    swept=swept,
                    sweep_index=sweep_index,
                    sweep_price=sweep_price,
                )
            )

        pool_records[:] = finalized_pools

    def _write_output_columns(
        self,
        market: MarketData,
        output: _LiquidityOutput,
    ) -> None:
        """
        Write liquidity output columns onto market data.

        Parameters
        ----------
        market : MarketData
            Market data to update.
        output : _LiquidityOutput
            Computed liquidity output.
        """
        market.add_column(self.EQUAL_HIGH_COLUMN, output.equal_high)
        market.add_column(self.EQUAL_LOW_COLUMN, output.equal_low)
        market.add_column(self.BUY_SIDE_LIQUIDITY_COLUMN, output.buy_side_liquidity)
        market.add_column(self.SELL_SIDE_LIQUIDITY_COLUMN, output.sell_side_liquidity)
        market.add_column(self.BUY_LIQUIDITY_SWEEP_COLUMN, output.buy_liquidity_sweep)
        market.add_column(self.SELL_LIQUIDITY_SWEEP_COLUMN, output.sell_liquidity_sweep)
        market.add_column(self.LIQUIDITY_STRENGTH_COLUMN, output.liquidity_strength)

    def _validate_market(self, market: MarketData) -> None:
        """
        Validate required swing and OHLC columns.

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
                    "Run SwingDetector before LiquidityDetector."
                )

    @staticmethod
    def _empty_price_series(index: pd.Index) -> pd.Series:
        """
        Create an empty float price series.

        Parameters
        ----------
        index : pd.Index
            Market index.

        Returns
        -------
        pd.Series
            Empty float series.
        """
        return pd.Series(pd.NA, index=index, dtype="Float64")
