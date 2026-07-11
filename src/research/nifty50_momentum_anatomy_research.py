"""
NIFTY50 Momentum Anatomy research for SmartMoneyEngine.

Understands how NIFTY50 creates real momentum moves over the last 120 calendar days.
Research-only; no production modifications.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd

from src.context.institutional_liquidity_map_engine import InstitutionalLiquidityMapEngine
from src.research.filter_research_engine import FilterResearchEngine, _json_safe
from src.research.institutional_expansion_trigger_discovery_research import (
    BLUEPRINT_ARROW,
    PRE_EXPANSION_LOOKBACK,
    InstitutionalExpansionTriggerDiscoveryResearch,
)
from src.research.liquidity_move_reconstruction_research import (
    FORWARD_BARS,
    LiquidityMoveReconstructionResearch,
    MIN_MOVE_SEPARATION_BARS,
    TIMEFRAME_MINUTES,
    _CheapMoveCandidate,
)
from src.research.smartmoneyengine_reality_check_validation_research import (
    SmartMoneyEngineRealityCheckValidationResearch,
)
from src.research.tiered_signal_framework_research import TieredSignalFrameworkResearch
from src.research.trade_construction_validation_research import TradeConstructionValidationResearch

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_FILTER_REPORT_PATH = RESEARCH_DIR / "filter_research_report.json"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "nifty50_momentum_anatomy_120d.json"

DEFAULT_SYMBOL = "NIFTY50"
RESEARCH_WINDOW_DAYS = 120
TIMEFRAMES = ("1D", "1H", "15M", "5M")
PIPELINE_TIMEFRAMES = ("5M", "15M", "1H")
MOVE_DETECTION_TIMEFRAME = "5M"
MOVE_THRESHOLDS = (100, 200, 300, 500)
ANATOMY_TIMELINE_OFFSETS_MINUTES = (60, 30, 15, 10, 5, 0)
EXTENDED_TIMEFRAME_MINUTES = {**TIMEFRAME_MINUTES, "1D": 1440}
MAX_MOVES_EXPORT = 300
MAX_BREAKOUT_EXPORT = 500
MIN_BLUEPRINT_SAMPLES = 3

MOVE_ORIGIN_TRIGGERS = (
    "Liquidity Grab",
    "Failed Breakdown",
    "Failed Breakout",
    "BOS",
    "CHOCH",
    "FVG Reclaim",
    "OB Reclaim",
    "VWAP Reclaim",
    "EMA Reclaim",
    "PDH Break",
    "PDL Break",
    "Range Expansion",
    "Compression Breakout",
    "Gap Continuation",
    "Gap Reversal",
)


class Nifty50MomentumAnatomyError(Exception):
    """Raised when NIFTY50 momentum anatomy research fails."""


@dataclass
class Nifty50MomentumAnatomyReport:
    """Full NIFTY50 momentum anatomy research output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    move_thresholds_points: list[int]
    completed_moves: dict[str, list[dict[str, Any]]]
    move_anatomy_records: list[dict[str, Any]]
    move_origin_classification: list[dict[str, Any]]
    origin_frequency_ranking: list[dict[str, Any]]
    breakout_analysis: list[dict[str, Any]]
    momentum_blueprint_discovery: dict[str, Any]
    engine_comparison: list[dict[str, Any]]
    final_questions: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Nifty50MomentumAnatomyResearch(SmartMoneyEngineRealityCheckValidationResearch):
    """NIFTY50-only momentum anatomy research with multi-timeframe context."""

    def __init__(self) -> None:
        super().__init__(
            symbols=(DEFAULT_SYMBOL,),
            research_days=RESEARCH_WINDOW_DAYS,
            timeframes=PIPELINE_TIMEFRAMES,
        )
        self.move_engine = LiquidityMoveReconstructionResearch(
            symbol=DEFAULT_SYMBOL,
            research_days=RESEARCH_WINDOW_DAYS,
            timeframes=PIPELINE_TIMEFRAMES,
        )

    @staticmethod
    def _resample_daily(frame_1h: pd.DataFrame) -> pd.DataFrame:
        working = frame_1h.copy()
        working["Date"] = pd.to_datetime(working["Date"])
        working = working.set_index("Date")
        daily = working.resample("1D").agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            },
        )
        daily = daily.dropna(subset=["Open", "Close"]).reset_index()
        daily["Date"] = daily["Date"].astype(str)
        return daily

    @staticmethod
    def _bar_for_timestamp(frame: pd.DataFrame, timestamp: pd.Timestamp) -> int:
        dates = pd.to_datetime(frame["Date"])
        eligible = dates[dates <= timestamp]
        if eligible.empty:
            return 0
        return int(eligible.index[-1])

    @staticmethod
    def _timeline_label(minutes: int) -> str:
        return "Move Start" if minutes == 0 else f"T-{minutes} minutes"

    @staticmethod
    def _bar_minutes_before(start_bar: int, minutes: int, timeframe: str) -> int:
        bar_offset = max(1, int(round(minutes / EXTENDED_TIMEFRAME_MINUTES.get(timeframe, 5))))
        return max(PRE_EXPANSION_LOOKBACK, start_bar - bar_offset)

    @staticmethod
    def _move_duration_minutes(start_bar: int, expansion_bar: int, timeframe: str) -> int:
        bar_minutes = EXTENDED_TIMEFRAME_MINUTES.get(timeframe, 5)
        return max(0, (expansion_bar - start_bar) * bar_minutes)

    @staticmethod
    def _threshold_tiers(magnitude: float) -> list[int]:
        return [threshold for threshold in MOVE_THRESHOLDS if magnitude >= threshold]

    @staticmethod
    def _levels_context(
        measurements: dict[str, Any],
        tags: tuple[str, ...],
        reasons: dict[str, Any],
    ) -> dict[str, Any]:
        sr = measurements.get("support_resistance", {})
        structure = measurements.get("structure", {})
        zone = structure.get("premium_discount", "Unknown")
        return {
            "support": sr.get("major_support"),
            "resistance": sr.get("major_resistance"),
            "demand_zone": zone in {"Discount", "Equilibrium"} or "Order Block" in tags,
            "supply_zone": zone in {"Premium", "Equilibrium"} or "Order Block" in tags,
            "pdh": sr.get("pdh_interactions", 0) > 0,
            "pdl": sr.get("pdl_interactions", 0) > 0,
            "pwh": sr.get("pwh_interactions", 0) > 0,
            "pwl": sr.get("pwl_interactions", 0) > 0,
            "round_numbers": bool(sr.get("round_number_proximity")),
            "market_location": reasons.get("major_level_context"),
        }

    @staticmethod
    def _reason_stack_extended(reasons: dict[str, Any], levels: dict[str, Any]) -> dict[str, Any]:
        return {
            "htf_trend": reasons.get("htf_trend"),
            "market_structure": reasons.get("market_structure"),
            "choch": reasons.get("choch"),
            "bos": reasons.get("bos"),
            "liquidity_grab": reasons.get("liquidity_grab"),
            "false_breakout": reasons.get("false_breakout"),
            "false_breakdown": reasons.get("false_breakdown"),
            "fvg": reasons.get("fvg"),
            "order_block": reasons.get("order_block"),
            "vwap": reasons.get("vwap"),
            "ema_structure": reasons.get("ema_structure"),
            "rsi": reasons.get("rsi"),
            "volume_expansion": reasons.get("volume_expansion", reasons.get("volume_spike")),
            "support": levels.get("support"),
            "resistance": levels.get("resistance"),
            "demand_zone": levels.get("demand_zone"),
            "supply_zone": levels.get("supply_zone"),
            "pdh": levels.get("pdh"),
            "pdl": levels.get("pdl"),
            "pwh": levels.get("pwh"),
            "pwl": levels.get("pwl"),
            "round_numbers": levels.get("round_numbers"),
        }

    @staticmethod
    def _classify_move_origin(
        *,
        direction: str,
        tags: tuple[str, ...],
        measurements: dict[str, Any],
        reasons: dict[str, Any],
        flags: dict[str, bool],
    ) -> str:
        sr = measurements.get("support_resistance", {})
        liquidity = measurements.get("liquidity", {})
        compression = measurements.get("compression", {})
        trigger = measurements.get("expansion_trigger_candle", {})
        tag_set = set(tags)

        gap_up = flags.get("gap_up", False)
        gap_down = flags.get("gap_down", False)
        if gap_up and direction == "bullish":
            return "Gap Continuation"
        if gap_down and direction == "bearish":
            return "Gap Continuation"
        if gap_up and direction == "bearish":
            return "Gap Reversal"
        if gap_down and direction == "bullish":
            return "Gap Reversal"

        if "Liquidity Grab" in tag_set or liquidity.get("liquidity_grab_count", 0) >= 1:
            return "Liquidity Grab"
        if direction == "bullish" and (
            "Failed Breakdown" in tag_set or sr.get("failed_breakdown_count", 0) >= 1
        ):
            return "Failed Breakdown"
        if direction == "bearish" and (
            "Failed Breakout" in tag_set or sr.get("failed_breakout_count", 0) >= 1
        ):
            return "Failed Breakout"
        if direction == "bullish" and sr.get("pdl_interactions", 0) >= 1:
            return "PDL Break"
        if direction == "bearish" and sr.get("pdh_interactions", 0) >= 1:
            return "PDH Break"
        if reasons.get("bos") or "BOS" in tag_set:
            return "BOS"
        if reasons.get("choch") or "CHOCH" in tag_set:
            return "CHOCH"
        if reasons.get("fvg") or "FVG" in tag_set:
            return "FVG Reclaim"
        if reasons.get("order_block") or "Order Block" in tag_set:
            return "OB Reclaim"
        if direction == "bullish" and flags.get("above_vwap"):
            return "VWAP Reclaim"
        if direction == "bearish" and flags.get("below_vwap"):
            return "VWAP Reclaim"
        if direction == "bullish" and flags.get("ema_bull_stack"):
            return "EMA Reclaim"
        if direction == "bearish" and flags.get("ema_bear_stack"):
            return "EMA Reclaim"
        if compression.get("volatility_compression_score", 0) >= 50:
            return "Compression Breakout"
        if trigger.get("volume_expansion_ratio", 0) >= 1.5:
            return "Range Expansion"
        return "Range Expansion"

    def _analyze_breakouts(
        self,
        frame: pd.DataFrame,
        start_bar: int,
        end_bar: int,
        direction: str,
        levels: dict[str, Any],
        atr: float,
    ) -> list[dict[str, Any]]:
        support = levels.get("major_support")
        resistance = levels.get("major_resistance")
        rows: list[dict[str, Any]] = []

        for index in range(start_bar, end_bar + 1):
            row = frame.iloc[index]
            high = float(row["High"])
            low = float(row["Low"])
            close = float(row["Close"])
            open_price = float(row["Open"])
            volume = float(row.get("Volume", 0.0))
            body = abs(close - open_price)
            displacement = round(body / max(atr, 0.01), 2)

            if resistance is not None and high > resistance:
                follow_through = close > resistance
                retracement = round(max(resistance - close, 0.0), 2)
                classification = "Real Breakout" if follow_through else "Fake Breakout"
                rows.append(
                    {
                        "bar": index,
                        "timestamp": str(row.get("Date", "")),
                        "event_type": "breakout",
                        "classification": classification,
                        "volume": volume,
                        "displacement": displacement,
                        "follow_through": follow_through,
                        "retracement": retracement,
                    },
                )

            if support is not None and low < support:
                follow_through = close < support
                retracement = round(max(close - support, 0.0), 2)
                classification = "Real Breakdown" if follow_through else "Fake Breakdown"
                rows.append(
                    {
                        "bar": index,
                        "timestamp": str(row.get("Date", "")),
                        "event_type": "breakdown",
                        "classification": classification,
                        "volume": volume,
                        "displacement": displacement,
                        "follow_through": follow_through,
                        "retracement": retracement,
                    },
                )

        return rows

    def _snapshot_at_bar(
        self,
        *,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        calendar: pd.DataFrame,
        intel: pd.DataFrame,
        prechecks: dict[str, np.ndarray],
        bar: int,
        direction: str,
        timeframe_label: str,
    ) -> dict[str, Any]:
        tags, measurements = self.discovery_engine.tags_at_bar(
            frame,
            enriched,
            calendar,
            intel,
            bar,
            direction,
        )
        flags = self._feature_flags_at_bar(frame, enriched, bar)
        reasons = self._reasons_at_bar(
            frame,
            enriched,
            intel,
            bar,
            direction,
            prechecks,
            tags,
            measurements,
        )
        reasons["volume_expansion"] = bool(measurements.get("volume_spike", False))
        levels = self._levels_context(measurements, tags, reasons)
        return {
            "timeframe": timeframe_label,
            "bar": bar,
            "timestamp": str(frame.iloc[bar].get("Date", "")),
            "tags": list(tags),
            "reason_stack": self._reason_stack_extended(reasons, levels),
            "feature_flags": flags,
            "levels": levels,
        }

    def _build_timeline(
        self,
        *,
        move: _CheapMoveCandidate,
        frames: dict[str, pd.DataFrame],
        enriched_map: dict[str, pd.DataFrame],
        calendar_map: dict[str, pd.DataFrame],
        intel_map: dict[str, pd.DataFrame],
        prechecks_map: dict[str, dict[str, np.ndarray]],
    ) -> list[dict[str, Any]]:
        trigger_frame = frames[MOVE_DETECTION_TIMEFRAME]
        steps: list[dict[str, Any]] = []

        for offset in ANATOMY_TIMELINE_OFFSETS_MINUTES:
            eval_bar = (
                move.start_bar
                if offset == 0
                else self._bar_minutes_before(move.start_bar, offset, MOVE_DETECTION_TIMEFRAME)
            )
            timestamp = pd.to_datetime(trigger_frame.iloc[eval_bar]["Date"])
            context_by_timeframe: dict[str, Any] = {}

            for timeframe_label in TIMEFRAMES:
                frame = frames[timeframe_label]
                mapped_bar = (
                    eval_bar
                    if timeframe_label == MOVE_DETECTION_TIMEFRAME
                    else self._bar_for_timestamp(frame, timestamp)
                )
                mapped_bar = min(max(PRE_EXPANSION_LOOKBACK, mapped_bar), len(frame) - 1)
                context_by_timeframe[timeframe_label] = self._snapshot_at_bar(
                    frame=frame,
                    enriched=enriched_map[timeframe_label],
                    calendar=calendar_map[timeframe_label],
                    intel=intel_map[timeframe_label],
                    prechecks=prechecks_map[timeframe_label],
                    bar=mapped_bar,
                    direction=move.direction,
                    timeframe_label=timeframe_label,
                )

            steps.append(
                {
                    "timeline_step": self._timeline_label(offset),
                    "minutes_before_move": offset,
                    "timestamp": str(timestamp),
                    "context_by_timeframe": context_by_timeframe,
                },
            )

        return steps

    def _rank_blueprints(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            blueprint = record.get("blueprint_pattern", "Unknown")
            grouped[blueprint].append(record)

        def build_rank(direction: str, *, sort_key: str) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            for blueprint, items in grouped.items():
                directional = [item for item in items if item["direction"] == direction]
                if len(directional) < MIN_BLUEPRINT_SAMPLES:
                    continue
                magnitudes = [float(item["move_size_points"]) for item in directional]
                rows.append(
                    {
                        "blueprint": blueprint,
                        "sample_size": len(directional),
                        "frequency_pct": round(len(directional) / max(len(records), 1) * 100, 2),
                        "average_move_size": round(mean(magnitudes), 2),
                        "reliability_score": round(
                            sum(1 for value in magnitudes if value >= 200) / len(magnitudes) * 100,
                            2,
                        ),
                    },
                )
            if sort_key == "frequency":
                rows.sort(key=lambda row: row["sample_size"], reverse=True)
            elif sort_key == "profitability":
                rows.sort(key=lambda row: row["average_move_size"], reverse=True)
            else:
                rows.sort(key=lambda row: row["reliability_score"], reverse=True)
            return rows[:20]

        bullish = [record for record in records if record["direction"] == "bullish"]
        bearish = [record for record in records if record["direction"] == "bearish"]
        return {
            "most_common_bullish": build_rank("bullish", sort_key="frequency"),
            "most_common_bearish": build_rank("bearish", sort_key="frequency"),
            "most_profitable_bullish": build_rank("bullish", sort_key="profitability"),
            "most_profitable_bearish": build_rank("bearish", sort_key="profitability"),
            "most_reliable_bullish": build_rank("bullish", sort_key="reliability"),
            "most_reliable_bearish": build_rank("bearish", sort_key="reliability"),
            "bullish_move_count": len(bullish),
            "bearish_move_count": len(bearish),
        }

    def _compare_engine_for_move(
        self,
        *,
        move: _CheapMoveCandidate,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        calendar: pd.DataFrame,
        intel: pd.DataFrame,
        prechecks: dict[str, np.ndarray],
        tier2_by_bar: dict[int, Any],
        buy_keys: tuple[str, ...],
        sell_keys: tuple[str, ...],
        no_trade_rules: list[str],
        archetypes: list[dict[str, Any]],
        trade_engine: TradeConstructionValidationResearch,
    ) -> dict[str, Any]:
        move_side = "BUY" if move.direction == "bullish" else "SELL"
        timeline_signals: list[dict[str, Any]] = []
        best_signal: dict[str, Any] | None = None

        for offset in ANATOMY_TIMELINE_OFFSETS_MINUTES:
            eval_bar = (
                move.start_bar
                if offset == 0
                else self._bar_minutes_before(move.start_bar, offset, MOVE_DETECTION_TIMEFRAME)
            )
            state = self._evaluate_at_bar(
                bar=eval_bar,
                symbol=DEFAULT_SYMBOL,
                timeframe_label=MOVE_DETECTION_TIMEFRAME,
                frame=frame,
                enriched=enriched,
                calendar=calendar,
                intel=intel,
                prechecks=prechecks,
                tier2_by_bar=tier2_by_bar,
                buy_keys=buy_keys,
                sell_keys=sell_keys,
                no_trade_rules=no_trade_rules,
                archetypes=archetypes,
                trade_engine=trade_engine,
            )
            signal = state.get("signal_direction", "NO_TRADE")
            points_captured = 0.0
            if signal == move_side:
                outcome = state.get("outcome", {})
                points_captured = float(outcome.get("realized_pnl_points") or 0.0)
                if best_signal is None or offset > int(best_signal.get("minutes_before_move", -1)):
                    best_signal = {
                        "timeline_step": self._timeline_label(offset),
                        "minutes_before_move": offset,
                        "signal_direction": signal,
                        "entry": state.get("entry"),
                        "stop_loss": state.get("stop_loss"),
                        "target_1": state.get("target_1"),
                        "target_2": state.get("target_2"),
                        "target_3": state.get("target_3"),
                        "points_captured": points_captured,
                    }

            timeline_signals.append(
                {
                    "timeline_step": self._timeline_label(offset),
                    "minutes_before_move": offset,
                    "timestamp": str(frame.iloc[eval_bar].get("Date", "")),
                    "engine_state": state.get("engine_state"),
                    "signal_direction": signal,
                    "signal_generated": signal in {"BUY", "SELL"},
                    "points_captured": points_captured,
                    "missing_conditions": state.get("missing_conditions", []),
                },
            )

        start_state = timeline_signals[-1]
        missed_reasons = (
            []
            if best_signal
            else start_state.get("missing_conditions", ["No aligned signal before move"])
        )
        return {
            "move_date": str(frame.iloc[move.start_bar].get("Date", "")),
            "direction": move.direction,
            "move_size_points": round(move.magnitude, 2),
            "engine_signal_at_move_start": start_state.get("signal_direction", "NO_TRADE"),
            "signal_existed": best_signal is not None,
            "minutes_early": best_signal.get("minutes_before_move") if best_signal else None,
            "points_captured": best_signal.get("points_captured", 0.0) if best_signal else 0.0,
            "capture_pct": round(
                (best_signal.get("points_captured", 0.0) / move.magnitude * 100) if best_signal else 0.0,
                2,
            ),
            "missed_reasons": missed_reasons,
            "timeline_signals": timeline_signals,
            "best_signal": best_signal,
        }

    @staticmethod
    def _condition_frequency(
        records: list[dict[str, Any]],
        *,
        min_move_size: float,
        step_label: str = "Move Start",
    ) -> list[dict[str, Any]]:
        counter: Counter[str] = Counter()
        considered = 0
        for record in records:
            if float(record["move_size_points"]) < min_move_size:
                continue
            considered += 1
            for step in record.get("timeline", []):
                if step.get("timeline_step") != step_label:
                    continue
                context = step.get("context_by_timeframe", {}).get(MOVE_DETECTION_TIMEFRAME, {})
                stack = context.get("reason_stack", {})
                for key, value in stack.items():
                    if isinstance(value, bool) and value:
                        counter[key] += 1
                    elif isinstance(value, str) and value not in {"", "Unknown", "No BOS", "Mixed"}:
                        counter[f"{key}:{value}"] += 1
                for tag in context.get("tags", []):
                    counter[f"tag:{tag}"] += 1
        return [
            {
                "condition": condition,
                "occurrences": count,
                "frequency_pct": round(count / max(considered, 1) * 100, 2),
            }
            for condition, count in counter.most_common(25)
        ]

    def _build_final_questions(
        self,
        *,
        completed_moves: dict[str, list[dict[str, Any]]],
        anatomy_records: list[dict[str, Any]],
        origin_ranking: list[dict[str, Any]],
        blueprint_discovery: dict[str, Any],
        engine_comparisons: list[dict[str, Any]],
        breakout_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        top_origin = origin_ranking[0]["origin_trigger"] if origin_ranking else "Unknown"
        level_counter: Counter[str] = Counter()
        ignored_counter: Counter[str] = Counter()
        restrictive_counter: Counter[str] = Counter()

        for record in anatomy_records:
            start_step = next(
                (step for step in record["timeline"] if step["timeline_step"] == "Move Start"),
                None,
            )
            if not start_step:
                continue
            stack = (
                start_step.get("context_by_timeframe", {})
                .get(MOVE_DETECTION_TIMEFRAME, {})
                .get("reason_stack", {})
            )
            if stack.get("pdh"):
                level_counter["PDH"] += 1
            if stack.get("pdl"):
                level_counter["PDL"] += 1
            if stack.get("pwh"):
                level_counter["PWH"] += 1
            if stack.get("pwl"):
                level_counter["PWL"] += 1
            if stack.get("round_numbers"):
                level_counter["Round Numbers"] += 1
            if stack.get("support"):
                level_counter["Support"] += 1
            if stack.get("resistance"):
                level_counter["Resistance"] += 1

        for comparison in engine_comparisons:
            if comparison.get("signal_existed"):
                continue
            for reason in comparison.get("missed_reasons", []):
                restrictive_counter[reason] += 1
                if "V2 Filter" in reason or "Confirmation" in reason or "Displacement" in reason:
                    ignored_counter[reason] += 1

        fake_breakouts = sum(1 for row in breakout_rows if "Fake" in row.get("classification", ""))
        real_breakouts = sum(1 for row in breakout_rows if "Real" in row.get("classification", ""))
        detected = sum(1 for row in engine_comparisons if row.get("signal_existed"))
        major_total = len(engine_comparisons)

        return {
            "1_how_nifty50_creates_major_momentum": (
                f"Major NIFTY50 momentum most often originates from {top_origin}, "
                f"with liquidity/structure confirmation preceding expansion."
            ),
            "2_most_important_liquidity_events": [
                item["origin_trigger"] for item in origin_ranking[:5]
            ],
            "3_most_important_support_resistance_levels": [
                level for level, _ in level_counter.most_common(5)
            ],
            "4_levels_frequently_ignored_by_engine": [
                reason for reason, _ in ignored_counter.most_common(5)
            ],
            "5_conditions_before_200_plus_moves": self._condition_frequency(
                anatomy_records,
                min_move_size=200,
            ),
            "6_conditions_before_300_plus_moves": self._condition_frequency(
                anatomy_records,
                min_move_size=300,
            ),
            "7_conditions_before_500_plus_moves": self._condition_frequency(
                anatomy_records,
                min_move_size=500,
            ),
            "8_why_engine_misses_major_moves": [
                reason for reason, _ in restrictive_counter.most_common(5)
            ],
            "9_restrictive_current_filters": [
                reason for reason, _ in restrictive_counter.most_common(5)
            ],
            "10_biggest_improvement_opportunity": (
                restrictive_counter.most_common(1)[0][0]
                if restrictive_counter
                else "Broaden pre-move context capture beyond Tier-2-only triggers"
            ),
            "supporting_metrics": {
                "major_move_engine_detection_rate_pct": round(
                    detected / max(major_total, 1) * 100,
                    2,
                ),
                "real_breakout_events": real_breakouts,
                "fake_breakout_events": fake_breakouts,
                "top_bullish_blueprint": (
                    blueprint_discovery.get("most_common_bullish", [{}])[0].get("blueprint")
                ),
                "top_bearish_blueprint": (
                    blueprint_discovery.get("most_common_bearish", [{}])[0].get("blueprint")
                ),
            },
        }

    def _prepare_frames(
        self,
        *,
        start: date,
        end: date,
    ) -> dict[str, dict[str, Any]]:
        filter_engine = FilterResearchEngine(
            symbol=DEFAULT_SYMBOL,
            research_days=RESEARCH_WINDOW_DAYS,
            timeframes=PIPELINE_TIMEFRAMES,
        )
        liquidity_map = InstitutionalLiquidityMapEngine(symbol=DEFAULT_SYMBOL)
        prepared: dict[str, dict[str, Any]] = {}

        raw_frames: dict[str, pd.DataFrame] = {}
        for timeframe_label in PIPELINE_TIMEFRAMES:
            path = filter_engine._ensure_pipeline(timeframe_label, start, end)
            raw_frames[timeframe_label] = pd.read_csv(path).reset_index(drop=True)

        raw_frames["1D"] = self._resample_daily(raw_frames["1H"])

        for timeframe_label in TIMEFRAMES:
            frame = raw_frames[timeframe_label]
            enriched = self.context_builder.enrich(frame)
            calendar = liquidity_map._attach_calendar_levels(frame)
            intel = self.discovery_engine.intelligence_engine.enrich(frame)
            prepared[timeframe_label] = {
                "frame": frame,
                "enriched": enriched,
                "calendar": calendar,
                "intel": intel,
                "prechecks": self._build_prechecks(frame),
            }
        return prepared

    def run(self, metadata: dict[str, Any]) -> Nifty50MomentumAnatomyReport:
        started = time.perf_counter()
        frozen = self._load_frozen_exports()
        v2_card = frozen["v2_production_card"]
        archetypes = frozen["production_candidate_archetypes"] or frozen["top_50_archetypes"]
        buy_keys = self._keys_from_labels(v2_card["buy_rules"]["filter_stack"])
        sell_keys = self._keys_from_labels(v2_card["sell_rules"]["filter_stack"])
        no_trade_rules = list(v2_card.get("no_trade_rules", []))

        end = date.fromisoformat(metadata["end_date"])
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=RESEARCH_WINDOW_DAYS)
        )

        prepared = self._prepare_frames(start=start, end=end)
        frames = {tf: payload["frame"] for tf, payload in prepared.items()}
        enriched_map = {tf: payload["enriched"] for tf, payload in prepared.items()}
        calendar_map = {tf: payload["calendar"] for tf, payload in prepared.items()}
        intel_map = {tf: payload["intel"] for tf, payload in prepared.items()}
        prechecks_map = {tf: payload["prechecks"] for tf, payload in prepared.items()}

        trigger_payload = prepared[MOVE_DETECTION_TIMEFRAME]
        trigger_frame = trigger_payload["frame"]
        highs = trigger_frame["High"].astype(float)
        lows = trigger_frame["Low"].astype(float)
        moves = self.move_engine._dedupe_cheap_moves(
            self.move_engine._detect_moves_cheap(highs, lows, MOVE_THRESHOLDS[0]),
        )
        moves = sorted(moves, key=lambda item: -item.magnitude)

        tier_engine = TieredSignalFrameworkResearch(
            symbol=DEFAULT_SYMBOL,
            research_days=RESEARCH_WINDOW_DAYS,
            timeframes=(MOVE_DETECTION_TIMEFRAME,),
        )
        trade_engine = TradeConstructionValidationResearch(
            symbol=DEFAULT_SYMBOL,
            research_days=RESEARCH_WINDOW_DAYS,
            timeframes=(MOVE_DETECTION_TIMEFRAME,),
        )
        tier2_by_bar = {
            signal.bos_bar: signal
            for signal in tier_engine._detect_tier2(trigger_frame, MOVE_DETECTION_TIMEFRAME)
        }

        completed_moves: dict[str, list[dict[str, Any]]] = {
            str(threshold): [] for threshold in MOVE_THRESHOLDS
        }
        anatomy_records: list[dict[str, Any]] = []
        origin_rows: list[dict[str, Any]] = []
        breakout_rows: list[dict[str, Any]] = []
        engine_comparisons: list[dict[str, Any]] = []
        blueprint_records: list[dict[str, Any]] = []

        for move in moves:
            if len(anatomy_records) >= MAX_MOVES_EXPORT:
                break

            start_ts = str(trigger_frame.iloc[move.start_bar].get("Date", ""))
            duration_minutes = self._move_duration_minutes(
                move.start_bar,
                move.expansion_bar,
                MOVE_DETECTION_TIMEFRAME,
            )
            move_entry = {
                "date": start_ts,
                "direction": move.direction,
                "move_size_points": round(move.magnitude, 2),
                "duration_minutes": duration_minutes,
                "start_bar": move.start_bar,
                "expansion_bar": move.expansion_bar,
            }
            for threshold in self._threshold_tiers(move.magnitude):
                completed_moves[str(threshold)].append(move_entry)

            timeline = self._build_timeline(
                move=move,
                frames=frames,
                enriched_map=enriched_map,
                calendar_map=calendar_map,
                intel_map=intel_map,
                prechecks_map=prechecks_map,
            )
            start_context = timeline[-1]["context_by_timeframe"][MOVE_DETECTION_TIMEFRAME]
            tags = tuple(start_context.get("tags", []))
            measurements = self.discovery_engine._combined_pre_expansion_measurements(
                trigger_frame,
                enriched_map[MOVE_DETECTION_TIMEFRAME],
                calendar_map[MOVE_DETECTION_TIMEFRAME],
                intel_map[MOVE_DETECTION_TIMEFRAME],
                max(0, move.start_bar - PRE_EXPANSION_LOOKBACK),
                move.start_bar,
                move.start_bar,
                move.direction,
            )
            reasons = start_context.get("reason_stack", {})
            flags = start_context.get("feature_flags", {})
            origin_trigger = self._classify_move_origin(
                direction=move.direction,
                tags=tags,
                measurements=measurements,
                reasons=reasons,
                flags=flags,
            )
            blueprint_pattern = BLUEPRINT_ARROW.join(tags) if tags else "No Context"

            pre_start = max(0, move.start_bar - PRE_EXPANSION_LOOKBACK)
            levels = self.discovery_engine._market_levels(trigger_frame, move.start_bar)
            atr = self.discovery_engine._atr(trigger_frame, move.start_bar)
            breakouts = self._analyze_breakouts(
                trigger_frame,
                pre_start,
                move.start_bar,
                move.direction,
                levels,
                atr,
            )
            if len(breakout_rows) < MAX_BREAKOUT_EXPORT:
                breakout_rows.extend(breakouts[: max(0, MAX_BREAKOUT_EXPORT - len(breakout_rows))])

            anatomy_records.append(
                {
                    "date": start_ts,
                    "direction": move.direction,
                    "move_size_points": round(move.magnitude, 2),
                    "duration_minutes": duration_minutes,
                    "origin_trigger": origin_trigger,
                    "blueprint_pattern": blueprint_pattern,
                    "timeline": timeline,
                },
            )
            origin_rows.append(
                {
                    "date": start_ts,
                    "direction": move.direction,
                    "move_size_points": round(move.magnitude, 2),
                    "origin_trigger": origin_trigger,
                    "blueprint_pattern": blueprint_pattern,
                },
            )
            blueprint_records.append(
                {
                    "direction": move.direction,
                    "move_size_points": round(move.magnitude, 2),
                    "blueprint_pattern": blueprint_pattern,
                },
            )

            if move.magnitude >= 200:
                engine_comparisons.append(
                    self._compare_engine_for_move(
                        move=move,
                        frame=trigger_frame,
                        enriched=trigger_payload["enriched"],
                        calendar=trigger_payload["calendar"],
                        intel=trigger_payload["intel"],
                        prechecks=trigger_payload["prechecks"],
                        tier2_by_bar=tier2_by_bar,
                        buy_keys=buy_keys,
                        sell_keys=sell_keys,
                        no_trade_rules=no_trade_rules,
                        archetypes=archetypes,
                        trade_engine=trade_engine,
                    ),
                )

        origin_counter = Counter(row["origin_trigger"] for row in origin_rows)
        origin_ranking = [
            {
                "origin_trigger": trigger,
                "occurrences": count,
                "frequency_pct": round(count / max(len(origin_rows), 1) * 100, 2),
            }
            for trigger, count in origin_counter.most_common()
        ]
        blueprint_discovery = self._rank_blueprints(blueprint_records)
        final_questions = self._build_final_questions(
            completed_moves=completed_moves,
            anatomy_records=anatomy_records,
            origin_ranking=origin_ranking,
            blueprint_discovery=blueprint_discovery,
            engine_comparisons=engine_comparisons,
            breakout_rows=breakout_rows,
        )

        conclusions = [
            "NIFTY50 Momentum Anatomy: 120-day multi-timeframe reconstruction complete.",
            f"Detected {len(moves)} completed moves on 5M (>= {MOVE_THRESHOLDS[0]} pts).",
            f"Top origin trigger: {origin_ranking[0]['origin_trigger'] if origin_ranking else 'N/A'}.",
            (
                f"Engine aligned before major move: "
                f"{final_questions['supporting_metrics']['major_move_engine_detection_rate_pct']}%."
            ),
            f"Biggest improvement opportunity: {final_questions['10_biggest_improvement_opportunity']}.",
        ]

        return Nifty50MomentumAnatomyReport(
            symbol=DEFAULT_SYMBOL,
            research_window_days=RESEARCH_WINDOW_DAYS,
            start_date=metadata.get("start_date", start.isoformat()),
            end_date=metadata.get("end_date", end.isoformat()),
            timeframes_analyzed=list(TIMEFRAMES),
            move_thresholds_points=list(MOVE_THRESHOLDS),
            completed_moves=completed_moves,
            move_anatomy_records=anatomy_records,
            move_origin_classification=origin_rows,
            origin_frequency_ranking=origin_ranking,
            breakout_analysis=breakout_rows,
            momentum_blueprint_discovery=blueprint_discovery,
            engine_comparison=engine_comparisons,
            final_questions=final_questions,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )


def generate_nifty50_momentum_anatomy_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> Nifty50MomentumAnatomyReport:
    """Run NIFTY50 momentum anatomy research and export JSON."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise Nifty50MomentumAnatomyError(f"Filter research report not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    metadata = {
        **metadata,
        "research_window_days": RESEARCH_WINDOW_DAYS,
        "start_date": (
            date.fromisoformat(metadata["end_date"]) - timedelta(days=RESEARCH_WINDOW_DAYS)
        ).isoformat(),
    }

    engine = Nifty50MomentumAnatomyResearch()
    report = engine.run(metadata)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info("NIFTY50 momentum anatomy report exported: %s", destination)
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        report = generate_nifty50_momentum_anatomy_report()
    except Nifty50MomentumAnatomyError as exc:
        logger.error("NIFTY50 momentum anatomy error: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected NIFTY50 momentum anatomy error")
        return 1

    detected = report.final_questions["supporting_metrics"]["major_move_engine_detection_rate_pct"]
    print("NIFTY50 Momentum Anatomy Research Summary")
    print(f"Moves analyzed: {sum(len(v) for v in report.completed_moves.values())}")
    print(f"Top origin: {report.origin_frequency_ranking[0]['origin_trigger'] if report.origin_frequency_ranking else 'N/A'}")
    print(f"Engine detection on 200+ moves: {detected}%")
    print(f"Report: {DEFAULT_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
