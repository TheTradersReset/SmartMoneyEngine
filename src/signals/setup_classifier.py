"""
Institutional setup classification for SmartMoneyEngine.

Classifies structural setup categories independently from the Decision Engine
and backtests each setup type in isolation for strategy validation research.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from src.signals.decision_engine import DecisionEngine, TradeDecision

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_CSV = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "signals" / "setup_classification_report.json"

SETUP_LOOKBACK_BARS = 20
MAX_FORWARD_BARS = 400
ENTRY_TOLERANCE = 0.05
SL_BUFFER_POINTS = 5.0
MIN_RISK_POINTS = 1.0
TIMEFRAME_MINUTES = 5


class SetupClassifierError(Exception):
    """Raised when setup classification fails."""


class SetupType(str, Enum):
    """Institutional setup categories."""

    LIQUIDITY_SWEEP_BOS = "Liquidity Sweep + BOS"
    CHOCH_FVG = "CHOCH + FVG"
    FRESH_OB_RETEST = "Fresh Order Block Retest"
    CONTINUATION_BOS = "Continuation BOS"
    LIQUIDITY_GRAB_FVG_RECLAIM = "Liquidity Grab + FVG Reclaim"


class SetupDirection(str, Enum):
    """Directional bias for a classified setup."""

    BULLISH = "bullish"
    BEARISH = "bearish"


class _ExitReason(str, Enum):
    NO_ENTRY = "No Entry"
    STOP_LOSS = "Stop Loss"
    TARGET_1 = "Target 1"
    TARGET_2 = "Target 2"
    OPEN = "Open"


@dataclass(frozen=True)
class SetupClassification:
    """One classified institutional setup."""

    setup_type: str
    direction: str
    confidence: float
    quality_score: int
    trigger_bar: int
    trigger_timestamp: str
    entry: float
    stop_loss: float
    target_1: float
    target_2: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SetupBacktestResult:
    """Backtest outcome for one classified setup."""

    setup_type: str
    direction: str
    trigger_bar: int
    trigger_timestamp: str
    entry_hit: bool
    outcome: str
    exit_reason: str
    realized_pnl_points: float
    realized_rr: float
    trade_duration_bars: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SetupTypeMetrics:
    """Aggregate metrics for one setup type."""

    setup_type: str
    frequency: int
    entries: int
    wins: int
    losses: int
    win_rate_pct: float
    average_rr: float
    profit_factor: float | None
    expectancy: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SetupClassificationReport:
    """Master setup classification and per-setup backtest report."""

    symbol: str
    timeframe: str
    source_csv: str
    total_candles: int
    total_setups: int
    setups: list[dict[str, Any]] = field(default_factory=list)
    setup_metrics: list[dict[str, Any]] = field(default_factory=list)
    backtest_results: list[dict[str, Any]] = field(default_factory=list)
    execution_time_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SetupClassifier:
    """
    Classify institutional setup categories from SMC pipeline output.

    Parameters
    ----------
    lookback_bars : int, optional
        Bars to search for supporting structural context.
    """

    REQUIRED_COLUMNS = (
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Trend",
        "Trend_Strength",
        "Bullish_BOS",
        "Bearish_BOS",
        "Bullish_CHOCH",
        "Bearish_CHOCH",
        "Bullish_FVG_Top",
        "Bullish_FVG_Bottom",
        "Bearish_FVG_Top",
        "Bearish_FVG_Bottom",
        "Bullish_OB_High",
        "Bullish_OB_Low",
        "Bearish_OB_High",
        "Bearish_OB_Low",
        "Bullish_OB_Mitigated",
        "Bearish_OB_Mitigated",
        "Buy_Liquidity_Sweep",
        "Sell_Liquidity_Sweep",
        "Liquidity_Strength",
    )

    def __init__(self, lookback_bars: int = SETUP_LOOKBACK_BARS) -> None:
        self.lookback_bars = lookback_bars

    @staticmethod
    def _is_active(value: Any) -> bool:
        return DecisionEngine._is_active(value)

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None or pd.isna(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_trend(value: Any) -> str:
        return str(value).strip().upper()

    def _validate_frame(self, frame: pd.DataFrame) -> None:
        missing = [column for column in self.REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise SetupClassifierError(f"Pipeline frame missing setup columns: {missing}")

    def _window(self, frame: pd.DataFrame, index: int) -> pd.DataFrame:
        start = max(0, index - self.lookback_bars)
        return frame.iloc[start : index + 1]

    def _column_active_in_window(self, window: pd.DataFrame, column: str) -> bool:
        if column not in window.columns:
            return False
        return any(self._is_active(value) for value in window[column])

    def _column_active_on_bar(self, row: pd.Series, column: str) -> bool:
        return self._is_active(row.get(column))

    def _fresh_fvg_bullish(self, window: pd.DataFrame) -> bool:
        tops = window["Bullish_FVG_Top"]
        bottoms = window["Bullish_FVG_Bottom"]
        return bool((tops.notna() & bottoms.notna()).any())

    def _fresh_fvg_bearish(self, window: pd.DataFrame) -> bool:
        tops = window["Bearish_FVG_Top"]
        bottoms = window["Bearish_FVG_Bottom"]
        return bool((tops.notna() & bottoms.notna()).any())

    def _latest_fvg_bullish(self, window: pd.DataFrame) -> tuple[float, float] | None:
        for offset in range(len(window) - 1, -1, -1):
            row = window.iloc[offset]
            top = self._to_float(row.get("Bullish_FVG_Top"))
            bottom = self._to_float(row.get("Bullish_FVG_Bottom"))
            if top is not None and bottom is not None:
                return bottom, top
        return None

    def _latest_fvg_bearish(self, window: pd.DataFrame) -> tuple[float, float] | None:
        for offset in range(len(window) - 1, -1, -1):
            row = window.iloc[offset]
            top = self._to_float(row.get("Bearish_FVG_Top"))
            bottom = self._to_float(row.get("Bearish_FVG_Bottom"))
            if top is not None and bottom is not None:
                return bottom, top
        return None

    def _nearest_swing_low(self, frame: pd.DataFrame, index: int) -> float | None:
        if "Swing_Low" not in frame.columns:
            return self._to_float(frame.iloc[index]["Low"])
        window = self._window(frame, index)
        values = [self._to_float(value) for value in window["Swing_Low"] if self._is_active(value)]
        return values[-1] if values else self._to_float(frame.iloc[index]["Low"])

    def _nearest_swing_high(self, frame: pd.DataFrame, index: int) -> float | None:
        if "Swing_High" not in frame.columns:
            return self._to_float(frame.iloc[index]["High"])
        window = self._window(frame, index)
        values = [self._to_float(value) for value in window["Swing_High"] if self._is_active(value)]
        return values[-1] if values else self._to_float(frame.iloc[index]["High"])

    def _quality_score(self, row: pd.Series, base: int, bonuses: dict[str, int]) -> int:
        score = base + sum(bonuses.values())
        strength = self._to_float(row.get("Trend_Strength")) or 0.0
        score += int(min(strength, 3) * 3)
        liquidity_strength = self._to_float(row.get("Liquidity_Strength")) or 0.0
        score += int(min(liquidity_strength, 3) * 2)
        return int(min(max(score, 0), 100))

    def _confidence(self, quality_score: int) -> float:
        return round(min(max(quality_score / 100.0, 0.05), 0.99), 3)

    def _build_buy_levels(
        self,
        frame: pd.DataFrame,
        index: int,
        row: pd.Series,
        sl_anchor: float | None = None,
    ) -> tuple[float, float, float, float]:
        close = self._to_float(row.get("Close")) or 0.0
        entry = round(close, 2)
        anchor = sl_anchor if sl_anchor is not None else self._nearest_swing_low(frame, index)
        anchor = anchor if anchor is not None else close * 0.995
        stop_loss = round(anchor - SL_BUFFER_POINTS, 2)
        risk = max(entry - stop_loss, MIN_RISK_POINTS)
        target_1 = round(entry + risk, 2)
        target_2 = round(entry + risk * 2.0, 2)
        return entry, stop_loss, target_1, target_2

    def _build_sell_levels(
        self,
        frame: pd.DataFrame,
        index: int,
        row: pd.Series,
        sl_anchor: float | None = None,
    ) -> tuple[float, float, float, float]:
        close = self._to_float(row.get("Close")) or 0.0
        entry = round(close, 2)
        anchor = sl_anchor if sl_anchor is not None else self._nearest_swing_high(frame, index)
        anchor = anchor if anchor is not None else close * 1.005
        stop_loss = round(anchor + SL_BUFFER_POINTS, 2)
        risk = max(stop_loss - entry, MIN_RISK_POINTS)
        target_1 = round(entry - risk, 2)
        target_2 = round(entry - risk * 2.0, 2)
        return entry, stop_loss, target_1, target_2

    def _detect_liquidity_sweep_bos(
        self,
        frame: pd.DataFrame,
        index: int,
        row: pd.Series,
        window: pd.DataFrame,
    ) -> SetupClassification | None:
        if self._column_active_on_bar(row, "Bullish_BOS") and self._column_active_in_window(
            window, "Sell_Liquidity_Sweep"
        ):
            entry, stop_loss, target_1, target_2 = self._build_buy_levels(frame, index, row)
            quality = self._quality_score(
                row,
                base=62,
                bonuses={"sweep": 10, "bos": 12},
            )
            return SetupClassification(
                setup_type=SetupType.LIQUIDITY_SWEEP_BOS.value,
                direction=SetupDirection.BULLISH.value,
                confidence=self._confidence(quality),
                quality_score=quality,
                trigger_bar=index,
                trigger_timestamp=str(row.get("Date")),
                entry=entry,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                reason="Sell-side liquidity sweep followed by bullish BOS.",
            )

        if self._column_active_on_bar(row, "Bearish_BOS") and self._column_active_in_window(
            window, "Buy_Liquidity_Sweep"
        ):
            entry, stop_loss, target_1, target_2 = self._build_sell_levels(frame, index, row)
            quality = self._quality_score(
                row,
                base=62,
                bonuses={"sweep": 10, "bos": 12},
            )
            return SetupClassification(
                setup_type=SetupType.LIQUIDITY_SWEEP_BOS.value,
                direction=SetupDirection.BEARISH.value,
                confidence=self._confidence(quality),
                quality_score=quality,
                trigger_bar=index,
                trigger_timestamp=str(row.get("Date")),
                entry=entry,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                reason="Buy-side liquidity sweep followed by bearish BOS.",
            )
        return None

    def _detect_choch_fvg(
        self,
        frame: pd.DataFrame,
        index: int,
        row: pd.Series,
        window: pd.DataFrame,
    ) -> SetupClassification | None:
        if self._column_active_on_bar(row, "Bullish_CHOCH") and self._fresh_fvg_bullish(window):
            fvg = self._latest_fvg_bullish(window)
            sl_anchor = fvg[0] if fvg is not None else None
            entry, stop_loss, target_1, target_2 = self._build_buy_levels(
                frame, index, row, sl_anchor=sl_anchor
            )
            quality = self._quality_score(row, base=58, bonuses={"choch": 10, "fvg": 10})
            return SetupClassification(
                setup_type=SetupType.CHOCH_FVG.value,
                direction=SetupDirection.BULLISH.value,
                confidence=self._confidence(quality),
                quality_score=quality,
                trigger_bar=index,
                trigger_timestamp=str(row.get("Date")),
                entry=entry,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                reason="Bullish CHOCH with fresh bullish FVG confluence.",
            )

        if self._column_active_on_bar(row, "Bearish_CHOCH") and self._fresh_fvg_bearish(window):
            fvg = self._latest_fvg_bearish(window)
            sl_anchor = fvg[1] if fvg is not None else None
            entry, stop_loss, target_1, target_2 = self._build_sell_levels(
                frame, index, row, sl_anchor=sl_anchor
            )
            quality = self._quality_score(row, base=58, bonuses={"choch": 10, "fvg": 10})
            return SetupClassification(
                setup_type=SetupType.CHOCH_FVG.value,
                direction=SetupDirection.BEARISH.value,
                confidence=self._confidence(quality),
                quality_score=quality,
                trigger_bar=index,
                trigger_timestamp=str(row.get("Date")),
                entry=entry,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                reason="Bearish CHOCH with fresh bearish FVG confluence.",
            )
        return None

    def _detect_fresh_ob_retest(
        self,
        frame: pd.DataFrame,
        index: int,
        row: pd.Series,
    ) -> SetupClassification | None:
        low = self._to_float(row.get("Low")) or 0.0
        close = self._to_float(row.get("Close")) or 0.0
        ob_high = self._to_float(row.get("Bullish_OB_High"))
        ob_low = self._to_float(row.get("Bullish_OB_Low"))
        ob_mitigated = self._is_active(row.get("Bullish_OB_Mitigated"))

        if (
            ob_high is not None
            and ob_low is not None
            and not ob_mitigated
            and ob_low <= low <= ob_high
            and close >= ob_low
        ):
            entry = round((ob_low + ob_high) / 2.0, 2)
            entry, stop_loss, target_1, target_2 = self._build_buy_levels(
                frame, index, row, sl_anchor=ob_low
            )
            entry = round((entry + close) / 2.0, 2)
            quality = self._quality_score(row, base=60, bonuses={"ob_retest": 14})
            return SetupClassification(
                setup_type=SetupType.FRESH_OB_RETEST.value,
                direction=SetupDirection.BULLISH.value,
                confidence=self._confidence(quality),
                quality_score=quality,
                trigger_bar=index,
                trigger_timestamp=str(row.get("Date")),
                entry=entry,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                reason="Fresh bullish order block retest holding above block low.",
            )

        ob_high = self._to_float(row.get("Bearish_OB_High"))
        ob_low = self._to_float(row.get("Bearish_OB_Low"))
        ob_mitigated = self._is_active(row.get("Bearish_OB_Mitigated"))
        high = self._to_float(row.get("High")) or 0.0

        if (
            ob_high is not None
            and ob_low is not None
            and not ob_mitigated
            and ob_low <= high <= ob_high
            and close <= ob_high
        ):
            entry = round((ob_low + ob_high) / 2.0, 2)
            entry, stop_loss, target_1, target_2 = self._build_sell_levels(
                frame, index, row, sl_anchor=ob_high
            )
            entry = round((entry + close) / 2.0, 2)
            quality = self._quality_score(row, base=60, bonuses={"ob_retest": 14})
            return SetupClassification(
                setup_type=SetupType.FRESH_OB_RETEST.value,
                direction=SetupDirection.BEARISH.value,
                confidence=self._confidence(quality),
                quality_score=quality,
                trigger_bar=index,
                trigger_timestamp=str(row.get("Date")),
                entry=entry,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                reason="Fresh bearish order block retest holding below block high.",
            )
        return None

    def _detect_continuation_bos(
        self,
        frame: pd.DataFrame,
        index: int,
        row: pd.Series,
        window: pd.DataFrame,
    ) -> SetupClassification | None:
        trend = self._normalize_trend(row.get("Trend"))

        if (
            trend == "BULLISH"
            and self._column_active_on_bar(row, "Bullish_BOS")
            and not self._column_active_in_window(window, "Bearish_CHOCH")
        ):
            entry, stop_loss, target_1, target_2 = self._build_buy_levels(frame, index, row)
            quality = self._quality_score(row, base=55, bonuses={"continuation": 12, "trend": 8})
            return SetupClassification(
                setup_type=SetupType.CONTINUATION_BOS.value,
                direction=SetupDirection.BULLISH.value,
                confidence=self._confidence(quality),
                quality_score=quality,
                trigger_bar=index,
                trigger_timestamp=str(row.get("Date")),
                entry=entry,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                reason="Bullish continuation BOS aligned with established bullish trend.",
            )

        if (
            trend == "BEARISH"
            and self._column_active_on_bar(row, "Bearish_BOS")
            and not self._column_active_in_window(window, "Bullish_CHOCH")
        ):
            entry, stop_loss, target_1, target_2 = self._build_sell_levels(frame, index, row)
            quality = self._quality_score(row, base=55, bonuses={"continuation": 12, "trend": 8})
            return SetupClassification(
                setup_type=SetupType.CONTINUATION_BOS.value,
                direction=SetupDirection.BEARISH.value,
                confidence=self._confidence(quality),
                quality_score=quality,
                trigger_bar=index,
                trigger_timestamp=str(row.get("Date")),
                entry=entry,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                reason="Bearish continuation BOS aligned with established bearish trend.",
            )
        return None

    def _detect_liquidity_grab_fvg_reclaim(
        self,
        frame: pd.DataFrame,
        index: int,
        row: pd.Series,
        window: pd.DataFrame,
    ) -> SetupClassification | None:
        close = self._to_float(row.get("Close")) or 0.0

        if self._column_active_in_window(window, "Sell_Liquidity_Sweep") and self._fresh_fvg_bullish(
            window
        ):
            fvg = self._latest_fvg_bullish(window)
            if fvg is not None and close > fvg[0]:
                entry, stop_loss, target_1, target_2 = self._build_buy_levels(
                    frame, index, row, sl_anchor=fvg[0]
                )
                quality = self._quality_score(row, base=64, bonuses={"grab": 10, "reclaim": 12})
                return SetupClassification(
                    setup_type=SetupType.LIQUIDITY_GRAB_FVG_RECLAIM.value,
                    direction=SetupDirection.BULLISH.value,
                    confidence=self._confidence(quality),
                    quality_score=quality,
                    trigger_bar=index,
                    trigger_timestamp=str(row.get("Date")),
                    entry=entry,
                    stop_loss=stop_loss,
                    target_1=target_1,
                    target_2=target_2,
                    reason="Sell-side liquidity grab with bullish FVG reclaim.",
                )

        if self._column_active_in_window(window, "Buy_Liquidity_Sweep") and self._fresh_fvg_bearish(
            window
        ):
            fvg = self._latest_fvg_bearish(window)
            if fvg is not None and close < fvg[1]:
                entry, stop_loss, target_1, target_2 = self._build_sell_levels(
                    frame, index, row, sl_anchor=fvg[1]
                )
                quality = self._quality_score(row, base=64, bonuses={"grab": 10, "reclaim": 12})
                return SetupClassification(
                    setup_type=SetupType.LIQUIDITY_GRAB_FVG_RECLAIM.value,
                    direction=SetupDirection.BEARISH.value,
                    confidence=self._confidence(quality),
                    quality_score=quality,
                    trigger_bar=index,
                    trigger_timestamp=str(row.get("Date")),
                    entry=entry,
                    stop_loss=stop_loss,
                    target_1=target_1,
                    target_2=target_2,
                    reason="Buy-side liquidity grab with bearish FVG reclaim.",
                )
        return None

    def classify_bar(
        self,
        frame: pd.DataFrame,
        index: int,
    ) -> list[SetupClassification]:
        """Return all setup types detected on one trigger bar."""
        row = frame.iloc[index]
        window = self._window(frame, index)
        candidates = (
            self._detect_liquidity_sweep_bos(frame, index, row, window),
            self._detect_choch_fvg(frame, index, row, window),
            self._detect_fresh_ob_retest(frame, index, row),
            self._detect_continuation_bos(frame, index, row, window),
            self._detect_liquidity_grab_fvg_reclaim(frame, index, row, window),
        )
        return [setup for setup in candidates if setup is not None]

    def classify(self, frame: pd.DataFrame) -> list[SetupClassification]:
        """Classify setups across the full pipeline dataframe."""
        self._validate_frame(frame)
        working = frame.reset_index(drop=True)
        setups: list[SetupClassification] = []
        for index in range(len(working)):
            setups.extend(self.classify_bar(working, index))
        return setups


class SetupBacktestSimulator:
    """Independent setup backtester that mirrors BacktestEngine exit priority."""

    def __init__(self, max_forward_bars: int = MAX_FORWARD_BARS) -> None:
        self.max_forward_bars = max_forward_bars

    @staticmethod
    def _risk_points(entry_price: float, stop_loss: float) -> float:
        risk = abs(entry_price - stop_loss)
        return risk if risk > 0 else 1e-9

    @staticmethod
    def _entry_hit(direction: str, bar: pd.Series, entry_price: float) -> bool:
        if direction == SetupDirection.BULLISH.value:
            return float(bar["Low"]) <= entry_price + ENTRY_TOLERANCE
        if direction == SetupDirection.BEARISH.value:
            return float(bar["High"]) >= entry_price - ENTRY_TOLERANCE
        return False

    @staticmethod
    def _resolve_long_exit(
        bar: pd.Series,
        stop_loss: float,
        target_1: float,
        target_2: float,
    ) -> _ExitReason | None:
        if float(bar["Low"]) <= stop_loss:
            return _ExitReason.STOP_LOSS
        if float(bar["High"]) >= target_2:
            return _ExitReason.TARGET_2
        if float(bar["High"]) >= target_1:
            return _ExitReason.TARGET_1
        return None

    @staticmethod
    def _resolve_short_exit(
        bar: pd.Series,
        stop_loss: float,
        target_1: float,
        target_2: float,
    ) -> _ExitReason | None:
        if float(bar["High"]) >= stop_loss:
            return _ExitReason.STOP_LOSS
        if float(bar["Low"]) <= target_2:
            return _ExitReason.TARGET_2
        if float(bar["Low"]) <= target_1:
            return _ExitReason.TARGET_1
        return None

    @staticmethod
    def _realized_pnl(
        direction: str,
        entry_price: float,
        stop_loss: float,
        target_1: float,
        target_2: float,
        exit_reason: _ExitReason,
    ) -> float:
        if exit_reason in {_ExitReason.NO_ENTRY, _ExitReason.OPEN}:
            return 0.0
        risk = SetupBacktestSimulator._risk_points(entry_price, stop_loss)
        if exit_reason == _ExitReason.STOP_LOSS:
            return -risk

        if direction == SetupDirection.BULLISH.value:
            if exit_reason == _ExitReason.TARGET_2:
                return target_2 - entry_price
            if exit_reason == _ExitReason.TARGET_1:
                return target_1 - entry_price
        else:
            if exit_reason == _ExitReason.TARGET_2:
                return entry_price - target_2
            if exit_reason == _ExitReason.TARGET_1:
                return entry_price - target_1
        return 0.0

    @staticmethod
    def _classify_outcome(exit_reason: _ExitReason, realized_pnl: float) -> str:
        if exit_reason == _ExitReason.NO_ENTRY:
            return "No Entry"
        if exit_reason == _ExitReason.OPEN:
            return "Open"
        if exit_reason in {_ExitReason.TARGET_1, _ExitReason.TARGET_2}:
            return "Win"
        if exit_reason == _ExitReason.STOP_LOSS:
            return "Loss" if realized_pnl < 0 else "Breakeven"
        return "Breakeven"

    def simulate(self, frame: pd.DataFrame, setup: SetupClassification) -> SetupBacktestResult:
        """Simulate one classified setup against OHLCV candles."""
        start_idx = setup.trigger_bar
        end_idx = min(len(frame) - 1, start_idx + self.max_forward_bars)
        entry_idx: int | None = None

        for idx in range(start_idx, end_idx + 1):
            if self._entry_hit(setup.direction, frame.iloc[idx], setup.entry):
                entry_idx = idx
                break

        if entry_idx is None:
            return SetupBacktestResult(
                setup_type=setup.setup_type,
                direction=setup.direction,
                trigger_bar=setup.trigger_bar,
                trigger_timestamp=setup.trigger_timestamp,
                entry_hit=False,
                outcome="No Entry",
                exit_reason=_ExitReason.NO_ENTRY.value,
                realized_pnl_points=0.0,
                realized_rr=0.0,
                trade_duration_bars=0,
            )

        exit_reason = _ExitReason.OPEN
        exit_idx = end_idx
        for idx in range(entry_idx, end_idx + 1):
            bar = frame.iloc[idx]
            if setup.direction == SetupDirection.BULLISH.value:
                bar_exit = self._resolve_long_exit(
                    bar, setup.stop_loss, setup.target_1, setup.target_2
                )
            else:
                bar_exit = self._resolve_short_exit(
                    bar, setup.stop_loss, setup.target_1, setup.target_2
                )

            if bar_exit == _ExitReason.STOP_LOSS:
                exit_reason = _ExitReason.STOP_LOSS
                exit_idx = idx
                break
            if bar_exit == _ExitReason.TARGET_2:
                exit_reason = _ExitReason.TARGET_2
                exit_idx = idx
                break
            if bar_exit == _ExitReason.TARGET_1:
                exit_reason = _ExitReason.TARGET_1
                exit_idx = idx
                break

        realized_pnl = self._realized_pnl(
            setup.direction,
            setup.entry,
            setup.stop_loss,
            setup.target_1,
            setup.target_2,
            exit_reason,
        )
        risk = self._risk_points(setup.entry, setup.stop_loss)
        realized_rr = (
            round(realized_pnl / risk, 2)
            if exit_reason not in {_ExitReason.NO_ENTRY, _ExitReason.OPEN}
            else 0.0
        )

        return SetupBacktestResult(
            setup_type=setup.setup_type,
            direction=setup.direction,
            trigger_bar=setup.trigger_bar,
            trigger_timestamp=setup.trigger_timestamp,
            entry_hit=True,
            outcome=self._classify_outcome(exit_reason, realized_pnl),
            exit_reason=exit_reason.value,
            realized_pnl_points=round(realized_pnl, 2),
            realized_rr=realized_rr,
            trade_duration_bars=max(0, exit_idx - entry_idx),
        )


class SetupClassificationEngine:
    """Orchestrate setup classification and independent backtesting."""

    def __init__(
        self,
        symbol: str = "NIFTY50",
        timeframe: str = "5",
        lookback_bars: int = SETUP_LOOKBACK_BARS,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.classifier = SetupClassifier(lookback_bars=lookback_bars)
        self.simulator = SetupBacktestSimulator()

    @staticmethod
    def _profit_factor(closed_pnls: list[float]) -> float | None:
        gross_profit = sum(pnl for pnl in closed_pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in closed_pnls if pnl < 0))
        if gross_loss == 0:
            return None if gross_profit == 0 else float("inf")
        return round(gross_profit / gross_loss, 2)

    def _aggregate_metrics(
        self,
        setup_type: str,
        results: list[SetupBacktestResult],
    ) -> SetupTypeMetrics:
        frequency = len(results)
        entries = sum(1 for result in results if result.entry_hit)
        wins = sum(1 for result in results if result.outcome == "Win")
        losses = sum(1 for result in results if result.outcome == "Loss")
        closed_pnls = [
            result.realized_pnl_points
            for result in results
            if result.entry_hit and result.outcome not in {"Open", "No Entry"}
        ]
        realized_rr = [
            result.realized_rr
            for result in results
            if result.entry_hit and result.outcome not in {"Open", "No Entry"}
        ]
        win_rate = round((wins / entries) * 100, 2) if entries else 0.0
        average_rr = round(sum(realized_rr) / len(realized_rr), 2) if realized_rr else 0.0
        expectancy = round(sum(closed_pnls) / entries, 2) if entries else 0.0
        return SetupTypeMetrics(
            setup_type=setup_type,
            frequency=frequency,
            entries=entries,
            wins=wins,
            losses=losses,
            win_rate_pct=win_rate,
            average_rr=average_rr,
            profit_factor=self._profit_factor(closed_pnls),
            expectancy=expectancy,
        )

    def run(self, frame: pd.DataFrame, source_csv: str = "") -> SetupClassificationReport:
        """Classify setups and backtest each setup type independently."""
        started = time.perf_counter()
        working = frame.reset_index(drop=True)
        setups = self.classifier.classify(working)
        backtest_results = [self.simulator.simulate(working, setup) for setup in setups]

        metrics: list[SetupTypeMetrics] = []
        for setup_type in SetupType:
            type_results = [
                result
                for result in backtest_results
                if result.setup_type == setup_type.value
            ]
            metrics.append(self._aggregate_metrics(setup_type.value, type_results))

        return SetupClassificationReport(
            symbol=self.symbol,
            timeframe=self.timeframe,
            source_csv=source_csv,
            total_candles=len(working),
            total_setups=len(setups),
            setups=[setup.as_dict() for setup in setups],
            setup_metrics=[metric.as_dict() for metric in metrics],
            backtest_results=[result.as_dict() for result in backtest_results],
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )

    def run_from_csv(self, pipeline_csv: Path | str) -> SetupClassificationReport:
        """Load pipeline CSV and run setup classification."""
        csv_path = Path(pipeline_csv)
        engine = DecisionEngine(symbol=self.symbol, timeframe=self.timeframe)
        frame = engine.load_pipeline_csv(csv_path)
        return self.run(frame, source_csv=str(csv_path))


def generate_setup_classification_report(
    pipeline_csv: Path | str | None = None,
    report_path: Path | str | None = None,
    symbol: str = "NIFTY50",
    timeframe: str = "5",
) -> SetupClassificationReport:
    """Run setup classification and export JSON report."""
    classifier_engine = SetupClassificationEngine(symbol=symbol, timeframe=timeframe)
    csv_path = Path(pipeline_csv) if pipeline_csv is not None else DEFAULT_PIPELINE_CSV
    report = classifier_engine.run_from_csv(csv_path)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(report.as_dict(), handle, indent=2)

    logger.info(
        "Setup classification completed: setups=%s types=%s",
        report.total_setups,
        len(report.setup_metrics),
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_setup_classification_report()
        print("Setup Classification Summary")
        print(f"Symbol: {report.symbol} | Timeframe: {report.timeframe}")
        print(f"Candles: {report.total_candles}")
        print(f"Total Setups: {report.total_setups}")
        print("Setup Metrics:")
        for metric in report.setup_metrics:
            print(
                f"  - {metric['setup_type']}: freq={metric['frequency']} "
                f"win_rate={metric['win_rate_pct']}% avg_rr={metric['average_rr']} "
                f"pf={metric['profit_factor']} expectancy={metric['expectancy']}"
            )
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except SetupClassifierError as exc:
        logger.error("Setup classification error: %s", exc)
        print(f"Setup classification error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected setup classification failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
