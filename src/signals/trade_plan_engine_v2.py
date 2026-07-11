"""
SmartMoneyEngine trade plan engine V2.

Institutional trade management with multi-target RR enforcement, HTF alignment
grading, and backward-compatible JSON output for the backtesting layer.
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

from src.signals.decision_engine import DecisionEngineError, TradeDecision, evaluate_pipeline

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_CSV = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
DEFAULT_MTF_REPORT = PROJECT_ROOT / "outputs" / "signals" / "multi_timeframe_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "signals" / "trade_plan_report_v2.json"

MIN_RR_T2 = 1.5
SL_BUFFER_POINTS = 5.0
LOOKBACK_BARS = 200
FORWARD_SCAN_BARS = 400
HTF_FORWARD_SCAN_BARS = 800
FRESH_ZONE_BARS = 20
MIN_RISK_POINTS = 1.0


class TradePlanEngineV2Error(Exception):
    """Raised when trade plan V2 generation fails."""


class TradeGrade(str, Enum):
    """Institutional trade quality grade."""

    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    REJECT = "Reject"


class EntrySource(str, Enum):
    """How the entry price was determined."""

    ORDER_BLOCK = "Fresh Order Block retest"
    FVG = "Fresh FVG retest"
    CLOSE = "Market close price"


@dataclass
class TradePlanV2:
    """Institutional trade plan with three targets and grading."""

    signal_index: int
    signal_date: str
    decision: str
    entry: float
    stop_loss: float
    target_1: float
    target_2: float
    target_3: float
    risk_reward_t1: float
    risk_reward_t2: float
    risk_reward_t3: float
    trade_grade: str
    confidence: float
    reason: str
    entry_source: str
    trade_validity: bool
    invalid_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """
        Return a serializable plan with V2 fields and V1 backtest aliases.

        BacktestEngine expects ``entry_price``, ``target_1``, ``target_2``,
        ``risk_reward_ratio``, ``confidence_pct``, and ``trade_validity``.
        """
        payload = asdict(self)
        payload.update(
            {
                "entry_price": self.entry,
                "risk_reward_ratio": self.risk_reward_t2,
                "confidence_pct": self.confidence,
            }
        )
        return payload


@dataclass
class TradePlanReportV2:
    """Aggregate V2 trade plan report."""

    symbol: str
    timeframe: str
    source_csv: str
    source_mtf_report: str
    engine_version: str
    total_signals: int
    valid_trade_plans: int
    invalid_trade_plans: int
    average_rr_t1: float
    average_rr_t2: float
    average_rr_t3: float
    average_confidence: float
    grade_distribution: dict[str, int]
    execution_time_seconds: float
    trade_plans: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable report dictionary."""
        return asdict(self)


class TradePlanEngineV2:
    """
    Build institutional SMC trade plans from decision and pipeline outputs.

    Parameters
    ----------
    symbol : str, optional
        Symbol label for reporting.
    timeframe : str, optional
        Timeframe label for reporting.
    min_rr_t2 : float, optional
        Minimum acceptable T2 risk-reward for validity.
    sl_buffer_points : float, optional
        Stop-loss buffer below/above structure in index points.
    lookback_bars : int, optional
        Bars to search backward for OB/FVG/swing context.
    """

    def __init__(
        self,
        symbol: str = "NIFTY50",
        timeframe: str = "5",
        min_rr_t2: float = MIN_RR_T2,
        sl_buffer_points: float = SL_BUFFER_POINTS,
        lookback_bars: int = LOOKBACK_BARS,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.min_rr_t2 = min_rr_t2
        self.sl_buffer_points = sl_buffer_points
        self.lookback_bars = lookback_bars
        self._mtf_report: dict[str, Any] | None = None

    @staticmethod
    def _is_active(value: Any) -> bool:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return not pd.isna(value)
        return bool(str(value).strip())

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if not TradePlanEngineV2._is_active(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _window_start(self, index: int) -> int:
        return max(0, index - self.lookback_bars)

    def load_mtf_report(self, path: Path | str | None = None) -> dict[str, Any]:
        """Load multi-timeframe alignment report."""
        report_path = Path(path) if path is not None else DEFAULT_MTF_REPORT
        if not report_path.exists():
            logger.warning("MTF report not found at %s; using neutral HTF defaults.", report_path)
            return {
                "overall_bias": "Neutral",
                "alignment_score": 0,
                "timeframes": [],
            }
        with report_path.open("r", encoding="utf-8") as handle:
            self._mtf_report = json.load(handle)
        return self._mtf_report

    def _find_fresh_bullish_ob(
        self,
        frame: pd.DataFrame,
        index: int,
    ) -> tuple[float, float, int] | None:
        """Return fresh bullish OB zone (low, high, bar_index)."""
        start = max(0, index - FRESH_ZONE_BARS)
        window = frame.iloc[start:index + 1]
        candidates = window[
            window["Bullish_OB_High"].notna() & window["Bullish_OB_Low"].notna()
        ]
        if candidates.empty:
            return None

        for offset in range(len(candidates) - 1, -1, -1):
            row = candidates.iloc[offset]
            if self._is_active(row.get("Bullish_OB_Mitigated")):
                continue
            low = self._to_float(row.get("Bullish_OB_Low"))
            high = self._to_float(row.get("Bullish_OB_High"))
            if low is not None and high is not None:
                bar_index = int(candidates.index[offset])
                return low, high, bar_index
        return None

    def _find_fresh_bearish_ob(
        self,
        frame: pd.DataFrame,
        index: int,
    ) -> tuple[float, float, int] | None:
        """Return fresh bearish OB zone (low, high, bar_index)."""
        start = max(0, index - FRESH_ZONE_BARS)
        window = frame.iloc[start:index + 1]
        candidates = window[
            window["Bearish_OB_High"].notna() & window["Bearish_OB_Low"].notna()
        ]
        if candidates.empty:
            return None

        for offset in range(len(candidates) - 1, -1, -1):
            row = candidates.iloc[offset]
            if self._is_active(row.get("Bearish_OB_Mitigated")):
                continue
            low = self._to_float(row.get("Bearish_OB_Low"))
            high = self._to_float(row.get("Bearish_OB_High"))
            if low is not None and high is not None:
                bar_index = int(candidates.index[offset])
                return low, high, bar_index
        return None

    def _find_fresh_bullish_fvg(
        self,
        frame: pd.DataFrame,
        index: int,
    ) -> tuple[float, float] | None:
        """Return fresh bullish FVG zone (bottom, top)."""
        start = max(0, index - FRESH_ZONE_BARS)
        window = frame.iloc[start:index + 1]
        candidates = window[
            window["Bullish_FVG_Top"].notna() & window["Bullish_FVG_Bottom"].notna()
        ]
        if candidates.empty:
            return None
        row = candidates.iloc[-1]
        bottom = self._to_float(row.get("Bullish_FVG_Bottom"))
        top = self._to_float(row.get("Bullish_FVG_Top"))
        if bottom is not None and top is not None:
            return bottom, top
        return None

    def _find_fresh_bearish_fvg(
        self,
        frame: pd.DataFrame,
        index: int,
    ) -> tuple[float, float] | None:
        """Return fresh bearish FVG zone (bottom, top)."""
        start = max(0, index - FRESH_ZONE_BARS)
        window = frame.iloc[start:index + 1]
        candidates = window[
            window["Bearish_FVG_Top"].notna() & window["Bearish_FVG_Bottom"].notna()
        ]
        if candidates.empty:
            return None
        row = candidates.iloc[-1]
        bottom = self._to_float(row.get("Bearish_FVG_Bottom"))
        top = self._to_float(row.get("Bearish_FVG_Top"))
        if bottom is not None and top is not None:
            return bottom, top
        return None

    def _nearest_swing_low(self, frame: pd.DataFrame, index: int) -> float | None:
        start = self._window_start(index)
        values = [
            self._to_float(value)
            for value in frame.iloc[start:index + 1]["Swing_Low"]
            if self._to_float(value) is not None
        ]
        return values[-1] if values else None

    def _nearest_swing_high(self, frame: pd.DataFrame, index: int) -> float | None:
        start = self._window_start(index)
        values = [
            self._to_float(value)
            for value in frame.iloc[start:index + 1]["Swing_High"]
            if self._to_float(value) is not None
        ]
        return values[-1] if values else None

    def _liquidity_levels_above(
        self,
        frame: pd.DataFrame,
        index: int,
        entry: float,
        max_bars: int = FORWARD_SCAN_BARS,
        min_distance: float = 0.0,
    ) -> list[float]:
        end = min(len(frame), index + max_bars)
        levels: set[float] = set()
        for value in frame.iloc[index:end]["Buy_Side_Liquidity"]:
            level = self._to_float(value)
            if level is not None and level > entry + min_distance:
                levels.add(level)
        for value in frame.iloc[index:end]["Swing_High"]:
            level = self._to_float(value)
            if level is not None and level > entry + min_distance:
                levels.add(level)
        return sorted(levels)

    def _liquidity_levels_below(
        self,
        frame: pd.DataFrame,
        index: int,
        entry: float,
        max_bars: int = FORWARD_SCAN_BARS,
        min_distance: float = 0.0,
    ) -> list[float]:
        end = min(len(frame), index + max_bars)
        levels: set[float] = set()
        for value in frame.iloc[index:end]["Sell_Side_Liquidity"]:
            level = self._to_float(value)
            if level is not None and level < entry - min_distance:
                levels.add(level)
        for value in frame.iloc[index:end]["Swing_Low"]:
            level = self._to_float(value)
            if level is not None and level < entry - min_distance:
                levels.add(level)
        return sorted(levels, reverse=True)

    @staticmethod
    def _compute_rr(entry: float, stop_loss: float, target: float, decision: str) -> float:
        risk = abs(entry - stop_loss)
        if risk <= 0:
            return 0.0
        if decision == TradeDecision.BUY.value:
            reward = target - entry
        else:
            reward = entry - target
        return round(reward / risk, 2) if reward > 0 else 0.0

    def _resolve_buy_targets(
        self,
        frame: pd.DataFrame,
        index: int,
        entry: float,
        risk: float,
    ) -> tuple[float, float, float]:
        liquidity = self._liquidity_levels_above(frame, index, entry)
        htf_liquidity = self._liquidity_levels_above(
            frame,
            index,
            entry,
            max_bars=HTF_FORWARD_SCAN_BARS,
            min_distance=risk * 2.0,
        )

        min_t1 = entry + risk * 1.0
        min_t2 = entry + risk * 2.0
        min_t3 = entry + risk * 3.0

        target_1 = max(liquidity[0], min_t1) if liquidity else min_t1

        remaining = [level for level in liquidity if level > target_1 + risk * 0.25]
        target_2 = max(remaining[0], min_t2) if remaining else min_t2
        target_2 = max(target_2, target_1 + risk * 0.5)

        htf_remaining = [level for level in htf_liquidity if level > target_2 + risk * 0.25]
        target_3 = max(htf_remaining[0], min_t3) if htf_remaining else min_t3
        target_3 = max(target_3, target_2 + risk * 0.5)

        return round(target_1, 2), round(target_2, 2), round(target_3, 2)

    def _resolve_sell_targets(
        self,
        frame: pd.DataFrame,
        index: int,
        entry: float,
        risk: float,
    ) -> tuple[float, float, float]:
        liquidity = self._liquidity_levels_below(frame, index, entry)
        htf_liquidity = self._liquidity_levels_below(
            frame,
            index,
            entry,
            max_bars=HTF_FORWARD_SCAN_BARS,
            min_distance=risk * 2.0,
        )

        min_t1 = entry - risk * 1.0
        min_t2 = entry - risk * 2.0
        min_t3 = entry - risk * 3.0

        target_1 = min(liquidity[0], min_t1) if liquidity else min_t1

        remaining = [level for level in liquidity if level < target_1 - risk * 0.25]
        target_2 = min(remaining[0], min_t2) if remaining else min_t2
        target_2 = min(target_2, target_1 - risk * 0.5)

        htf_remaining = [level for level in htf_liquidity if level < target_2 - risk * 0.25]
        target_3 = min(htf_remaining[0], min_t3) if htf_remaining else min_t3
        target_3 = min(target_3, target_2 - risk * 0.5)

        return round(target_1, 2), round(target_2, 2), round(target_3, 2)

    def _htf_alignment_score(self, decision: str) -> tuple[int, str]:
        """Score HTF alignment from the multi-timeframe report (0-25)."""
        if self._mtf_report is None:
            return 12, "Neutral HTF (no MTF report)"

        trends = {
            item["timeframe"]: item.get("trend", "Neutral")
            for item in self._mtf_report.get("timeframes", [])
        }
        htf_trends = [trends.get("1D", "Neutral"), trends.get("4H", "Neutral")]
        overall = str(self._mtf_report.get("overall_bias", "Neutral"))

        if decision == TradeDecision.BUY.value:
            aligned = sum(1 for trend in htf_trends if trend == "Bullish")
            opposed = sum(1 for trend in htf_trends if trend == "Bearish")
        else:
            aligned = sum(1 for trend in htf_trends if trend == "Bearish")
            opposed = sum(1 for trend in htf_trends if trend == "Bullish")

        if aligned == 2 or overall in {"Strong Bullish", "Strong Bearish"} and aligned >= 1:
            return 25, f"HTF aligned ({aligned}/2)"
        if aligned == 1 and opposed == 0:
            return 18, f"Partial HTF alignment ({aligned}/2)"
        if opposed >= 1 and aligned == 0:
            return 0, f"HTF misaligned ({opposed}/2 opposed)"
        return 12, "Neutral HTF"

    def _liquidity_strength_score(self, row: pd.Series) -> int:
        strength = self._to_float(row.get("Liquidity_Strength"))
        if strength is None:
            return 5
        if strength >= 0.75:
            return 15
        if strength >= 0.5:
            return 10
        if strength >= 0.25:
            return 6
        return 3

    def _entry_quality_score(self, entry_source: EntrySource) -> int:
        if entry_source == EntrySource.ORDER_BLOCK:
            return 15
        if entry_source == EntrySource.FVG:
            return 10
        return 5

    def _distance_score(self, risk: float, target_1: float, target_2: float, decision: str) -> int:
        if risk <= 0:
            return 0
        if decision == TradeDecision.BUY.value:
            spacing = (target_2 - target_1) / risk
        else:
            spacing = (target_1 - target_2) / risk
        if spacing >= 1.0:
            return 15
        if spacing >= 0.5:
            return 10
        return 5

    def _rr_quality_score(self, rr_t1: float, rr_t2: float, rr_t3: float) -> int:
        score = 0
        if rr_t1 >= 1.0:
            score += 5
        if rr_t2 >= 1.5:
            score += 8
        elif rr_t2 >= 1.0:
            score += 4
        if rr_t3 >= 3.0:
            score += 7
        elif rr_t3 >= 2.0:
            score += 4
        return min(score, 20)

    def _assign_grade(self, total_score: int, validation_passed: bool) -> TradeGrade:
        if not validation_passed:
            return TradeGrade.REJECT
        if total_score >= 85:
            return TradeGrade.A_PLUS
        if total_score >= 70:
            return TradeGrade.A
        if total_score >= 55:
            return TradeGrade.B
        if total_score >= 40:
            return TradeGrade.C
        return TradeGrade.REJECT

    def _validate_buy_plan(
        self,
        entry: float,
        stop_loss: float,
        target_1: float,
        target_2: float,
        target_3: float,
        rr_t2: float,
    ) -> tuple[bool, str | None]:
        if entry <= 0 or stop_loss <= 0:
            return False, "Invalid entry or stop loss."
        if not (stop_loss < entry < target_1 < target_2 < target_3):
            return False, "Invalid BUY target hierarchy."
        if stop_loss >= entry:
            return False, "Invalid BUY stop loss placement."
        if rr_t2 < self.min_rr_t2:
            return False, f"Risk-reward T2 {rr_t2} below minimum {self.min_rr_t2}."
        return True, None

    def _validate_sell_plan(
        self,
        entry: float,
        stop_loss: float,
        target_1: float,
        target_2: float,
        target_3: float,
        rr_t2: float,
    ) -> tuple[bool, str | None]:
        if entry <= 0 or stop_loss <= 0:
            return False, "Invalid entry or stop loss."
        if not (target_3 < target_2 < target_1 < entry < stop_loss):
            return False, "Invalid SELL target hierarchy."
        if stop_loss <= entry:
            return False, "Invalid SELL stop loss placement."
        if rr_t2 < self.min_rr_t2:
            return False, f"Risk-reward T2 {rr_t2} below minimum {self.min_rr_t2}."
        return True, None

    def _build_buy_plan(self, frame: pd.DataFrame, index: int, row: pd.Series) -> TradePlanV2:
        close = self._to_float(row.get("Close")) or 0.0
        ob = self._find_fresh_bullish_ob(frame, index)
        fvg = self._find_fresh_bullish_fvg(frame, index)
        swing_low = self._nearest_swing_low(frame, index)

        if ob is not None:
            ob_low, ob_high, _ = ob
            entry = round((ob_low + ob_high) / 2, 2)
            entry_source = EntrySource.ORDER_BLOCK
            sl_anchor = ob_low
        elif fvg is not None:
            fvg_bottom, fvg_top = fvg
            entry = round((fvg_bottom + fvg_top) / 2, 2)
            entry_source = EntrySource.FVG
            sl_anchor = fvg_bottom
        else:
            entry = round(close, 2)
            entry_source = EntrySource.CLOSE
            sl_anchor = swing_low if swing_low is not None else close * 0.995

        if swing_low is not None:
            sl_anchor = min(sl_anchor, swing_low)
        stop_loss = round(sl_anchor - self.sl_buffer_points, 2)
        risk = max(entry - stop_loss, MIN_RISK_POINTS)

        target_1, target_2, target_3 = self._resolve_buy_targets(frame, index, entry, risk)
        rr_t1 = self._compute_rr(entry, stop_loss, target_1, TradeDecision.BUY.value)
        rr_t2 = self._compute_rr(entry, stop_loss, target_2, TradeDecision.BUY.value)
        rr_t3 = self._compute_rr(entry, stop_loss, target_3, TradeDecision.BUY.value)

        valid, invalid_reason = self._validate_buy_plan(
            entry, stop_loss, target_1, target_2, target_3, rr_t2
        )

        htf_score, htf_note = self._htf_alignment_score(TradeDecision.BUY.value)
        total_score = (
            htf_score
            + self._liquidity_strength_score(row)
            + self._entry_quality_score(entry_source)
            + self._distance_score(risk, target_1, target_2, TradeDecision.BUY.value)
            + self._rr_quality_score(rr_t1, rr_t2, rr_t3)
        )
        grade = self._assign_grade(total_score, valid)
        if grade == TradeGrade.REJECT and valid:
            valid = False
            invalid_reason = invalid_reason or f"Trade grade {grade.value} below acceptance threshold."

        confidence_pct = round(float(row.get("Confidence", 0.0)) * 100, 1)
        reason = (
            f"BUY V2 from {entry_source.value}; SL {stop_loss:.2f}; "
            f"T1 {target_1:.2f} (RR {rr_t1}); T2 {target_2:.2f} (RR {rr_t2}); "
            f"T3 {target_3:.2f} (RR {rr_t3}); Grade {grade.value}; {htf_note}; "
            f"{row.get('Reason', '')}"
        )

        return TradePlanV2(
            signal_index=index,
            signal_date=str(row.get("Date")),
            decision=TradeDecision.BUY.value,
            entry=entry,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            target_3=target_3,
            risk_reward_t1=rr_t1,
            risk_reward_t2=rr_t2,
            risk_reward_t3=rr_t3,
            trade_grade=grade.value,
            confidence=confidence_pct,
            reason=reason,
            entry_source=entry_source.value,
            trade_validity=valid and grade != TradeGrade.REJECT,
            invalid_reason=invalid_reason,
        )

    def _build_sell_plan(self, frame: pd.DataFrame, index: int, row: pd.Series) -> TradePlanV2:
        close = self._to_float(row.get("Close")) or 0.0
        ob = self._find_fresh_bearish_ob(frame, index)
        fvg = self._find_fresh_bearish_fvg(frame, index)
        swing_high = self._nearest_swing_high(frame, index)

        if ob is not None:
            ob_low, ob_high, _ = ob
            entry = round((ob_low + ob_high) / 2, 2)
            entry_source = EntrySource.ORDER_BLOCK
            sl_anchor = ob_high
        elif fvg is not None:
            fvg_bottom, fvg_top = fvg
            entry = round((fvg_bottom + fvg_top) / 2, 2)
            entry_source = EntrySource.FVG
            sl_anchor = fvg_top
        else:
            entry = round(close, 2)
            entry_source = EntrySource.CLOSE
            sl_anchor = swing_high if swing_high is not None else close * 1.005

        if swing_high is not None:
            sl_anchor = max(sl_anchor, swing_high)
        stop_loss = round(sl_anchor + self.sl_buffer_points, 2)
        risk = max(stop_loss - entry, MIN_RISK_POINTS)

        target_1, target_2, target_3 = self._resolve_sell_targets(frame, index, entry, risk)
        rr_t1 = self._compute_rr(entry, stop_loss, target_1, TradeDecision.SELL.value)
        rr_t2 = self._compute_rr(entry, stop_loss, target_2, TradeDecision.SELL.value)
        rr_t3 = self._compute_rr(entry, stop_loss, target_3, TradeDecision.SELL.value)

        valid, invalid_reason = self._validate_sell_plan(
            entry, stop_loss, target_1, target_2, target_3, rr_t2
        )

        htf_score, htf_note = self._htf_alignment_score(TradeDecision.SELL.value)
        total_score = (
            htf_score
            + self._liquidity_strength_score(row)
            + self._entry_quality_score(entry_source)
            + self._distance_score(risk, target_1, target_2, TradeDecision.SELL.value)
            + self._rr_quality_score(rr_t1, rr_t2, rr_t3)
        )
        grade = self._assign_grade(total_score, valid)
        if grade == TradeGrade.REJECT and valid:
            valid = False
            invalid_reason = invalid_reason or f"Trade grade {grade.value} below acceptance threshold."

        confidence_pct = round(float(row.get("Confidence", 0.0)) * 100, 1)
        reason = (
            f"SELL V2 from {entry_source.value}; SL {stop_loss:.2f}; "
            f"T1 {target_1:.2f} (RR {rr_t1}); T2 {target_2:.2f} (RR {rr_t2}); "
            f"T3 {target_3:.2f} (RR {rr_t3}); Grade {grade.value}; {htf_note}; "
            f"{row.get('Reason', '')}"
        )

        return TradePlanV2(
            signal_index=index,
            signal_date=str(row.get("Date")),
            decision=TradeDecision.SELL.value,
            entry=entry,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            target_3=target_3,
            risk_reward_t1=rr_t1,
            risk_reward_t2=rr_t2,
            risk_reward_t3=rr_t3,
            trade_grade=grade.value,
            confidence=confidence_pct,
            reason=reason,
            entry_source=entry_source.value,
            trade_validity=valid and grade != TradeGrade.REJECT,
            invalid_reason=invalid_reason,
        )

    def build_plans(
        self,
        evaluated: pd.DataFrame,
        mtf_report: dict[str, Any] | None = None,
    ) -> list[TradePlanV2]:
        """Build V2 trade plans for all BUY/SELL signals."""
        required = {"Decision", "Close", "Confidence", "Reason"}
        missing = required - set(evaluated.columns)
        if missing:
            raise TradePlanEngineV2Error(
                f"Evaluated dataframe missing columns: {sorted(missing)}"
            )

        if mtf_report is not None:
            self._mtf_report = mtf_report
        elif self._mtf_report is None:
            self.load_mtf_report()

        plans: list[TradePlanV2] = []
        for index, row in evaluated.iterrows():
            decision = str(row.get("Decision"))
            if decision == TradeDecision.BUY.value:
                plans.append(self._build_buy_plan(evaluated, int(index), row))
            elif decision == TradeDecision.SELL.value:
                plans.append(self._build_sell_plan(evaluated, int(index), row))
        return plans

    def build_report(
        self,
        plans: list[TradePlanV2],
        source_csv: Path | str,
        source_mtf_report: Path | str,
        execution_time_seconds: float,
    ) -> TradePlanReportV2:
        """Build aggregate V2 trade plan report."""
        valid_plans = [plan for plan in plans if plan.trade_validity]
        grade_distribution: dict[str, int] = {
            grade.value: 0 for grade in TradeGrade
        }
        for plan in plans:
            grade_distribution[plan.trade_grade] = grade_distribution.get(plan.trade_grade, 0) + 1

        def _avg(values: list[float]) -> float:
            return round(sum(values) / len(values), 2) if values else 0.0

        return TradePlanReportV2(
            symbol=self.symbol,
            timeframe=self.timeframe,
            source_csv=str(source_csv),
            source_mtf_report=str(source_mtf_report),
            engine_version="v2",
            total_signals=len(plans),
            valid_trade_plans=len(valid_plans),
            invalid_trade_plans=len(plans) - len(valid_plans),
            average_rr_t1=_avg([plan.risk_reward_t1 for plan in valid_plans]),
            average_rr_t2=_avg([plan.risk_reward_t2 for plan in valid_plans]),
            average_rr_t3=_avg([plan.risk_reward_t3 for plan in valid_plans]),
            average_confidence=_avg([plan.confidence for plan in plans]),
            grade_distribution=grade_distribution,
            execution_time_seconds=execution_time_seconds,
            trade_plans=[plan.as_dict() for plan in plans],
        )


def generate_trade_plans_v2(
    pipeline_csv: Path | str | None = None,
    mtf_report_path: Path | str | None = None,
    report_path: Path | str | None = None,
    symbol: str = "NIFTY50",
    timeframe: str = "5",
) -> tuple[list[TradePlanV2], TradePlanReportV2]:
    """
    Evaluate decisions and generate V2 trade plans from pipeline and MTF data.

    Returns
    -------
    tuple[list[TradePlanV2], TradePlanReportV2]
        Trade plans and aggregate report.
    """
    csv_path = Path(pipeline_csv) if pipeline_csv is not None else DEFAULT_PIPELINE_CSV
    mtf_path = Path(mtf_report_path) if mtf_report_path is not None else DEFAULT_MTF_REPORT
    json_path = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH

    started = time.perf_counter()
    evaluated, _ = evaluate_pipeline(pipeline_csv=csv_path)
    engine = TradePlanEngineV2(symbol=symbol, timeframe=timeframe)
    mtf_report = engine.load_mtf_report(mtf_path)
    plans = engine.build_plans(evaluated, mtf_report=mtf_report)
    elapsed = time.perf_counter() - started
    report = engine.build_report(plans, csv_path, mtf_path, elapsed)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report.as_dict(), handle, indent=2)

    logger.info(
        "Trade plan V2 generation completed in %.3fs: signals=%s valid=%s avg_rr_t2=%s",
        elapsed,
        report.total_signals,
        report.valid_trade_plans,
        report.average_rr_t2,
    )
    return plans, report


def main() -> int:
    """CLI entry point."""
    try:
        _, report = generate_trade_plans_v2()
        print("Trade Plan Engine V2 Summary")
        print(f"Total Signals: {report.total_signals}")
        print(f"Valid Trade Plans: {report.valid_trade_plans}")
        print(f"Average RR T1: {report.average_rr_t1}")
        print(f"Average RR T2: {report.average_rr_t2}")
        print(f"Average RR T3: {report.average_rr_t3}")
        print(f"Grade Distribution: {report.grade_distribution}")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except (TradePlanEngineV2Error, DecisionEngineError) as exc:
        logger.error("Trade plan engine V2 error: %s", exc)
        print(f"Trade plan engine V2 error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected trade plan engine V2 failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
