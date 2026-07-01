from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from src.models.market_data import MarketData
from src.smc.base_smc import BaseSMC


class TrendDirection(str, Enum):
    """Supported trend direction values."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"


class StructureLabel(str, Enum):
    """Market structure label types consumed by the trend engine."""

    HH = "HH"
    HL = "HL"
    LH = "LH"
    LL = "LL"


@dataclass
class _TrendState:
    """
    Mutable trend state used while scanning structure events.

    Attributes
    ----------
    direction : TrendDirection
        Current confirmed trend direction.
    hh_count : int
        Higher-high labels observed in the active trend phase.
    hl_count : int
        Higher-low labels observed in the active trend phase.
    lh_count : int
        Lower-high labels observed in the active trend phase.
    ll_count : int
        Lower-low labels observed in the active trend phase.
    bearish_flip_lh : bool
        Whether a lower high has been seen while awaiting bearish flip.
    bearish_flip_ll : bool
        Whether a lower low has been seen while awaiting bearish flip.
    bullish_flip_hh : bool
        Whether a higher high has been seen while awaiting bullish flip.
    bullish_flip_hl : bool
        Whether a higher low has been seen while awaiting bullish flip.
    sideways_hh : bool
        Higher-high seen while trend is unconfirmed.
    sideways_hl : bool
        Higher-low seen while trend is unconfirmed.
    sideways_lh : bool
        Lower-high seen while trend is unconfirmed.
    sideways_ll : bool
        Lower-low seen while trend is unconfirmed.
    event_count : int
        Total structure labels processed so far.
    """

    direction: TrendDirection = TrendDirection.SIDEWAYS
    hh_count: int = 0
    hl_count: int = 0
    lh_count: int = 0
    ll_count: int = 0
    bearish_flip_lh: bool = False
    bearish_flip_ll: bool = False
    bullish_flip_hh: bool = False
    bullish_flip_hl: bool = False
    sideways_hh: bool = False
    sideways_hl: bool = False
    sideways_lh: bool = False
    sideways_ll: bool = False
    event_count: int = 0


class TrendEngine(BaseSMC):
    """
    Determine market trend from classified structure labels.

    Consumes ``HH``, ``HL``, ``LH``, and ``LL`` columns produced by
    ``MarketStructure`` and derives a stable trend regime with strength.

    Bullish trend requires both a higher high and a higher low.
    Bearish trend requires both a lower high and a lower low.
    Sideways is returned when structure is mixed or not yet confirmed.

    Trend changes require full opposing structural confirmation and
    do not flip on isolated counter-structure labels.
    """

    HH_COLUMN = "HH"
    HL_COLUMN = "HL"
    LH_COLUMN = "LH"
    LL_COLUMN = "LL"
    TREND_COLUMN = "Trend"
    TREND_STRENGTH_COLUMN = "Trend_Strength"
    REQUIRED_COLUMNS = (HH_COLUMN, HL_COLUMN, LH_COLUMN, LL_COLUMN)

    STRENGTH_UNKNOWN = 0
    STRENGTH_WEAK = 1
    STRENGTH_MEDIUM = 2
    STRENGTH_STRONG = 3

    _LABEL_COLUMN_MAP = {
        HH_COLUMN: StructureLabel.HH,
        HL_COLUMN: StructureLabel.HL,
        LH_COLUMN: StructureLabel.LH,
        LL_COLUMN: StructureLabel.LL,
    }

    def __init__(self) -> None:
        super().__init__("Trend Engine")

    def detect(self, market: MarketData) -> MarketData:
        """
        Compute trend direction and strength for every candle.

        Parameters
        ----------
        market : MarketData
            Market data containing structure label columns.

        Returns
        -------
        MarketData
            Same instance with ``Trend`` and ``Trend_Strength`` columns added.

        Raises
        ------
        ValueError
            If required structure columns are missing.
        """
        self.log_start()
        self._validate_market(market)

        trend_series, strength_series = self._compute_trend_series(market)

        market.add_column(self.TREND_COLUMN, trend_series)
        market.add_column(self.TREND_STRENGTH_COLUMN, strength_series)

        self.log_finish()

        return market

    def _compute_trend_series(
        self,
        market: MarketData,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Build per-row trend direction and strength series.

        Parameters
        ----------
        market : MarketData
            Market data containing structure label columns.

        Returns
        -------
        tuple[pd.Series, pd.Series]
            Trend direction strings and integer strength values.
        """
        index = market.get_column(self.HH_COLUMN).index
        trend_values: list[str] = []
        strength_values: list[int] = []

        state = _TrendState()
        structure_events = self._extract_structure_events(market)

        events_by_index: dict[object, list[StructureLabel]] = {}
        for row_index, label in structure_events:
            events_by_index.setdefault(row_index, []).append(label)

        for row_index in index:
            for label in events_by_index.get(row_index, []):
                self._apply_structure_label(state, label)

            trend_values.append(state.direction.value)
            strength_values.append(self._calculate_strength(state))

        trend_series = pd.Series(trend_values, index=index, dtype="string")
        strength_series = pd.Series(strength_values, index=index, dtype="int64")

        return trend_series, strength_series

    def _extract_structure_events(
        self,
        market: MarketData,
    ) -> list[tuple[object, StructureLabel]]:
        """
        Extract chronological structure label events from market data.

        Parameters
        ----------
        market : MarketData
            Market data containing structure label columns.

        Returns
        -------
        list[tuple[object, StructureLabel]]
            Ordered structure events as ``(index, label)`` pairs.
        """
        events: list[tuple[object, StructureLabel]] = []

        index = market.get_column(self.HH_COLUMN).index
        index_positions = {
            row_index: position for position, row_index in enumerate(index)
        }

        for column_name, label in self._LABEL_COLUMN_MAP.items():
            series = market.get_column(column_name)
            for row_index, value in series.items():
                if pd.notna(value):
                    events.append((row_index, label))

        events.sort(
            key=lambda item: (index_positions[item[0]], item[1].value)
        )

        return events

    def _apply_structure_label(
        self,
        state: _TrendState,
        label: StructureLabel,
    ) -> None:
        """
        Apply one structure label to the running trend state.

        Parameters
        ----------
        state : _TrendState
            Mutable trend state updated in place.
        label : StructureLabel
            Structure label to process.
        """
        state.event_count += 1

        if state.direction == TrendDirection.BULLISH:
            self._apply_label_in_bullish_trend(state, label)
            return

        if state.direction == TrendDirection.BEARISH:
            self._apply_label_in_bearish_trend(state, label)
            return

        self._apply_label_in_sideways_trend(state, label)

    def _apply_label_in_bullish_trend(
        self,
        state: _TrendState,
        label: StructureLabel,
    ) -> None:
        """
        Update state for a label observed during a bullish trend.

        Parameters
        ----------
        state : _TrendState
            Mutable trend state updated in place.
        label : StructureLabel
            Structure label to process.
        """
        if label == StructureLabel.HH:
            state.hh_count += 1
            return

        if label == StructureLabel.HL:
            state.hl_count += 1
            return

        if label == StructureLabel.LH:
            state.bearish_flip_lh = True
        elif label == StructureLabel.LL:
            state.bearish_flip_ll = True

        if state.bearish_flip_lh and state.bearish_flip_ll:
            self._set_bearish_trend(state, from_opposing_flip=True)

    def _apply_label_in_bearish_trend(
        self,
        state: _TrendState,
        label: StructureLabel,
    ) -> None:
        """
        Update state for a label observed during a bearish trend.

        Parameters
        ----------
        state : _TrendState
            Mutable trend state updated in place.
        label : StructureLabel
            Structure label to process.
        """
        if label == StructureLabel.LH:
            state.lh_count += 1
            return

        if label == StructureLabel.LL:
            state.ll_count += 1
            return

        if label == StructureLabel.HH:
            state.bullish_flip_hh = True
        elif label == StructureLabel.HL:
            state.bullish_flip_hl = True

        if state.bullish_flip_hh and state.bullish_flip_hl:
            self._set_bullish_trend(state, from_opposing_flip=True)

    def _apply_label_in_sideways_trend(
        self,
        state: _TrendState,
        label: StructureLabel,
    ) -> None:
        """
        Update state for a label observed while trend is unconfirmed.

        Parameters
        ----------
        state : _TrendState
            Mutable trend state updated in place.
        label : StructureLabel
            Structure label to process.
        """
        if label == StructureLabel.HH:
            state.sideways_hh = True
        elif label == StructureLabel.HL:
            state.sideways_hl = True
        elif label == StructureLabel.LH:
            state.sideways_lh = True
        elif label == StructureLabel.LL:
            state.sideways_ll = True

        bullish_ready = state.sideways_hh and state.sideways_hl
        bearish_ready = state.sideways_lh and state.sideways_ll
        has_bullish_evidence = state.sideways_hh or state.sideways_hl
        has_bearish_evidence = state.sideways_lh or state.sideways_ll

        if bullish_ready and not has_bearish_evidence:
            self._set_bullish_trend(state)
            return

        if bearish_ready and not has_bullish_evidence:
            self._set_bearish_trend(state)
            return

        state.direction = TrendDirection.SIDEWAYS

    def _set_bullish_trend(
        self,
        state: _TrendState,
        from_opposing_flip: bool = False,
    ) -> None:
        """
        Transition state to a confirmed bullish trend.

        Parameters
        ----------
        state : _TrendState
            Mutable trend state updated in place.
        from_opposing_flip : bool, default=False
            Whether the transition originated from a bearish trend flip.
        """
        if from_opposing_flip:
            hh_count = 1
            hl_count = 1
        else:
            hh_count = int(state.sideways_hh)
            hl_count = int(state.sideways_hl)

        state.direction = TrendDirection.BULLISH
        state.hh_count = max(hh_count, 1)
        state.hl_count = max(hl_count, 1)
        state.lh_count = 0
        state.ll_count = 0
        state.bearish_flip_lh = False
        state.bearish_flip_ll = False
        state.bullish_flip_hh = False
        state.bullish_flip_hl = False
        state.sideways_hh = False
        state.sideways_hl = False
        state.sideways_lh = False
        state.sideways_ll = False

    def _set_bearish_trend(
        self,
        state: _TrendState,
        from_opposing_flip: bool = False,
    ) -> None:
        """
        Transition state to a confirmed bearish trend.

        Parameters
        ----------
        state : _TrendState
            Mutable trend state updated in place.
        from_opposing_flip : bool, default=False
            Whether the transition originated from a bullish trend flip.
        """
        if from_opposing_flip:
            lh_count = 1
            ll_count = 1
        else:
            lh_count = int(state.sideways_lh)
            ll_count = int(state.sideways_ll)

        state.direction = TrendDirection.BEARISH
        state.lh_count = max(lh_count, 1)
        state.ll_count = max(ll_count, 1)
        state.hh_count = 0
        state.hl_count = 0
        state.bearish_flip_lh = False
        state.bearish_flip_ll = False
        state.bullish_flip_hh = False
        state.bullish_flip_hl = False
        state.sideways_hh = False
        state.sideways_hl = False
        state.sideways_lh = False
        state.sideways_ll = False

    def _calculate_strength(self, state: _TrendState) -> int:
        """
        Calculate trend strength from the current state.

        Parameters
        ----------
        state : _TrendState
            Current trend state.

        Returns
        -------
        int
            Strength value between 0 and 3.
        """
        if state.event_count == 0:
            return self.STRENGTH_UNKNOWN

        if state.direction == TrendDirection.SIDEWAYS:
            return self.STRENGTH_WEAK

        if state.direction == TrendDirection.BULLISH:
            return self._directional_strength(state.hh_count, state.hl_count)

        return self._directional_strength(state.lh_count, state.ll_count)

    @classmethod
    def _directional_strength(cls, primary_count: int, secondary_count: int) -> int:
        """
        Map paired structure counts to a directional strength score.

        Parameters
        ----------
        primary_count : int
            Count of primary labels such as HH or LH.
        secondary_count : int
            Count of secondary labels such as HL or LL.

        Returns
        -------
        int
            Strength value between 1 and 3.
        """
        if primary_count >= 2 and secondary_count >= 2:
            return cls.STRENGTH_STRONG

        if primary_count >= 1 and secondary_count >= 1:
            return cls.STRENGTH_MEDIUM

        return cls.STRENGTH_WEAK

    def _validate_market(self, market: MarketData) -> None:
        """
        Validate that required structure columns are present.

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
                    "Run MarketStructure before TrendEngine."
                )
