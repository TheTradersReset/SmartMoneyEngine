"""
SmartMoneyEngine Reality Check Validation V1 research.

Validates whether the CURRENT frozen production blueprint can detect real
market momentum before it happens. Strict no-look-ahead bar replay.
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

import pandas as pd

from src.research.filter_research_engine import RESEARCH_DAYS, FilterResearchEngine, _json_safe
from src.research.institutional_expansion_trigger_discovery_research import PRE_EXPANSION_LOOKBACK
from src.research.liquidity_move_reconstruction_research import FORWARD_BARS, TIMEFRAME_MINUTES
from src.research.smartmoneyengine_realtime_replay_validation_research import (
    DEFAULT_SYMBOLS,
    DEFAULT_TIMEFRAMES,
    MAX_SIGNAL_LOG,
    MIN_SIGNAL_SEPARATION_BARS,
    MOMENTUM_THRESHOLDS,
    SCAN_PROGRESS_INTERVAL,
    SmartMoneyEngineRealtimeReplayValidationResearch,
)
from src.research.smartmoneyengine_walkforward_validation_research import (
    SmartMoneyEngineWalkForwardValidationResearch,
)
from src.research.tiered_signal_framework_research import TierSignal, TieredSignalFrameworkResearch
from src.research.trade_construction_validation_research import TradeConstructionValidationResearch

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_FILTER_REPORT_PATH = RESEARCH_DIR / "filter_research_report.json"
DEFAULT_V1_VALIDATION_PATH = RESEARCH_DIR / "smartmoneyengine_final_production_validation.json"
DEFAULT_V2_OPTIMIZATION_PATH = RESEARCH_DIR / "smartmoneyengine_v2_frequency_optimization.json"
DEFAULT_V2_RANKING_PATH = RESEARCH_DIR / "smartmoneyengine_v2_signal_ranking.json"
DEFAULT_ARCHETYPE_WALKFORWARD_PATH = RESEARCH_DIR / "smartmoneyengine_archetype_walkforward.json"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "smartmoneyengine_reality_check_validation.json"

MISSED_THRESHOLDS = (100, 200, 300, 500)
MAJOR_TIMELINE_THRESHOLDS = (200, 300, 500)
TIMELINE_OFFSETS_MINUTES = (30, 15, 10, 5, 0)
MAX_TIMELINE_EXPORT = 150
MAX_MISSED_EXPORT = 400
PRODUCTION_READY_CAPTURE_PCT = 25.0
PRODUCTION_READY_MONTHLY_SIGNALS = 10.0


class RealityCheckValidationError(Exception):
    """Raised when reality check validation fails."""


@dataclass
class RealityCheckValidationReport:
    """Full reality check validation output."""

    symbols_analyzed: list[str]
    timeframes_analyzed: list[str]
    research_window_days: int
    start_date: str
    end_date: str
    frozen_exports_loaded: dict[str, Any]
    v1_production_card: dict[str, Any]
    v2_production_card: dict[str, Any]
    validated_archetypes: list[dict[str, Any]]
    replay_rules: dict[str, Any]
    overall_statistics: dict[str, Any]
    signal_by_signal_log: list[dict[str, Any]]
    major_move_replay_timeline: list[dict[str, Any]]
    missed_move_report: list[dict[str, Any]]
    missed_move_reason_ranking: list[dict[str, Any]]
    false_signal_report: dict[str, Any]
    frequency_report: dict[str, Any]
    top_performing_archetypes: list[dict[str, Any]]
    worst_performing_archetypes: list[dict[str, Any]]
    final_production_verdict: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SmartMoneyEngineRealityCheckValidationResearch(SmartMoneyEngineRealtimeReplayValidationResearch):
    """Reality-check validation of frozen SmartMoneyEngine blueprint."""

    def __init__(
        self,
        symbols: tuple[str, ...] | None = None,
        research_days: int = RESEARCH_DAYS,
        timeframes: tuple[str, ...] = DEFAULT_TIMEFRAMES,
        v1_validation_path: Path | str | None = None,
        v2_optimization_path: Path | str | None = None,
        v2_ranking_path: Path | str | None = None,
        archetype_walkforward_path: Path | str | None = None,
    ) -> None:
        super().__init__(
            symbols=symbols,
            research_days=research_days,
            timeframes=timeframes,
            v2_optimization_path=v2_optimization_path,
            research_dir=RESEARCH_DIR,
        )
        self.v1_validation_path = Path(v1_validation_path or DEFAULT_V1_VALIDATION_PATH)
        self.v2_ranking_path = Path(v2_ranking_path or DEFAULT_V2_RANKING_PATH)
        self.archetype_walkforward_path = Path(
            archetype_walkforward_path or DEFAULT_ARCHETYPE_WALKFORWARD_PATH,
        )

    def _load_json(self, path: Path, key: str | None = None) -> dict[str, Any]:
        if not path.exists():
            raise RealityCheckValidationError(f"Required export not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if key:
            section = payload.get(key)
            if section is None:
                raise RealityCheckValidationError(f"Missing section '{key}' in {path.name}")
            return section
        return payload

    def _load_frozen_exports(self) -> dict[str, Any]:
        v1_payload = self._load_json(self.v1_validation_path)
        v2_payload = self._load_json(self.v2_optimization_path)
        ranking_payload = self._load_json(self.v2_ranking_path)
        archetype_payload = self._load_json(self.archetype_walkforward_path)
        return {
            "v1_production_card": v1_payload.get("smartmoneyengine_v1_final_production_card", {}),
            "v2_production_card": v2_payload.get("smartmoneyengine_v2_production_card", {}),
            "top_50_archetypes": ranking_payload.get("top_50_signal_archetypes", []),
            "production_candidate_archetypes": archetype_payload.get("production_candidate_list", []),
            "archetype_walkforward_summary": archetype_payload.get("classification_summary", {}),
        }

    @staticmethod
    def _parse_archetype_key(key: str) -> dict[str, str]:
        criteria: dict[str, str] = {}
        for part in key.split(" | "):
            if "=" in part:
                dim, value = part.split("=", 1)
                criteria[dim.strip()] = value.strip()
        return criteria

    def _dimension_values(
        self,
        *,
        symbol: str,
        timeframe: str,
        side: str,
        tags: tuple[str, ...],
        measurements: dict[str, Any],
        reasons: dict[str, Any],
    ) -> dict[str, str]:
        session = measurements.get("session", "Unknown")
        for tag in tags:
            if tag.startswith("Session:"):
                session = tag.split(": ", 1)[-1]
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": side,
            "session": session,
            "vwap_state": reasons.get("vwap", "Unknown"),
            "rsi_bucket": reasons.get("rsi", "Unknown"),
            "ema_structure": reasons.get("ema_structure", "Unknown"),
            "choch_bos_timing": measurements.get("choch_to_bos_timing", "Unknown"),
            "displacement_strength": reasons.get("displacement", "Unknown"),
            "level_context": reasons.get("major_level_context", "Unknown"),
            "liquidity_context": measurements.get("liquidity_distance", "Unknown"),
            "confirmation_candle": (
                "Strong Confirmation" if measurements.get("strong_confirmation") else "Weak"
            ),
        }

    def _match_archetype(
        self,
        archetypes: list[dict[str, Any]],
        dimensions: dict[str, str],
    ) -> str | None:
        for item in archetypes:
            key = item.get("archetype_key", "")
            criteria = self._parse_archetype_key(key)
            if criteria and all(dimensions.get(dim) == val for dim, val in criteria.items()):
                return key
        return None

    def _evaluate_at_bar(
        self,
        *,
        bar: int,
        symbol: str,
        timeframe_label: str,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        calendar: pd.DataFrame,
        intel: pd.DataFrame,
        prechecks: dict[str, Any],
        tier2_by_bar: dict[int, TierSignal],
        buy_keys: tuple[str, ...],
        sell_keys: tuple[str, ...],
        no_trade_rules: list[str],
        archetypes: list[dict[str, Any]],
        trade_engine: TradeConstructionValidationResearch,
    ) -> dict[str, Any]:
        tier2 = tier2_by_bar.get(bar)
        if tier2 is None:
            return {
                "engine_state": "NO_TIER2_SETUP",
                "signal_generated": False,
                "signal_direction": "NO_TRADE",
                "missing_conditions": ["No Tier-2 Setup"],
            }

        direction = tier2.direction
        side = "BUY" if direction == "bullish" else "SELL"
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

        blocked = SmartMoneyEngineWalkForwardValidationResearch._no_trade_blocked(tags, no_trade_rules)
        required = buy_keys if side == "BUY" else sell_keys
        filters_pass = all(flags.get(key, False) for key in required)
        dimensions = self._dimension_values(
            symbol=symbol,
            timeframe=timeframe_label,
            side=side,
            tags=tags,
            measurements=measurements,
            reasons=reasons,
        )
        archetype = self._match_archetype(archetypes, dimensions)

        missing: list[str] = []
        if blocked:
            decision = "NO_TRADE"
            missing.append("NO-TRADE Rule Matched")
        elif filters_pass:
            decision = side
            missing = []
            if archetype is None and archetypes:
                missing.append("No Valid Archetype (informational)")
        else:
            decision = "NO_TRADE"
            missing.extend(self._diagnose_missed(frame, bar, direction, prechecks, flags))

        outcome: dict[str, Any] = {}
        if decision in {"BUY", "SELL"}:
            outcome = self._forward_outcome(frame, bar, direction, trade_engine)

        return {
            "engine_state": "SIGNAL_READY" if decision in {"BUY", "SELL"} else "BLOCKED",
            "signal_generated": decision in {"BUY", "SELL"},
            "signal_direction": decision,
            "signal_archetype": archetype,
            "signal_score": self._signal_score(decision, reasons, flags),
            "reason_stack": reasons,
            "missing_conditions": missing,
            "entry": outcome.get("entry"),
            "stop_loss": outcome.get("stop_loss"),
            "target_1": outcome.get("target_1"),
            "target_2": outcome.get("target_2"),
            "target_3": outcome.get("target_3"),
            "outcome": outcome,
        }

    @staticmethod
    def _bar_minutes_before(start_bar: int, minutes: int, timeframe: str) -> int:
        bar_offset = max(1, int(round(minutes / TIMEFRAME_MINUTES.get(timeframe, 5))))
        return max(PRE_EXPANSION_LOOKBACK, start_bar - bar_offset)

    def _build_major_move_timelines(
        self,
        *,
        symbol: str,
        timeframe_label: str,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        calendar: pd.DataFrame,
        intel: pd.DataFrame,
        prechecks: dict[str, Any],
        tier2_by_bar: dict[int, TierSignal],
        buy_keys: tuple[str, ...],
        sell_keys: tuple[str, ...],
        no_trade_rules: list[str],
        archetypes: list[dict[str, Any]],
        trade_engine: TradeConstructionValidationResearch,
    ) -> list[dict[str, Any]]:
        timelines: list[dict[str, Any]] = []
        highs = frame["High"].astype(float)
        lows = frame["Low"].astype(float)

        moves_by_bar: dict[int, Any] = {}
        for move in self.move_engine._dedupe_cheap_moves(
            self.move_engine._detect_moves_cheap(highs, lows, 200),
        ):
            moves_by_bar[move.expansion_bar] = move

        for move in sorted(moves_by_bar.values(), key=lambda item: -item.magnitude):
            if len(timelines) >= MAX_TIMELINE_EXPORT:
                break
            move = move
            threshold = (
                500
                if move.magnitude >= 500
                else 300
                if move.magnitude >= 300
                else 200
            )
            if move.magnitude < 200:
                continue

            steps: list[dict[str, Any]] = []
            signal_before_move = False
            best_capture = 0.0

            for offset in TIMELINE_OFFSETS_MINUTES:
                eval_bar = (
                    move.start_bar
                    if offset == 0
                    else self._bar_minutes_before(move.start_bar, offset, timeframe_label)
                )
                state = self._evaluate_at_bar(
                    bar=eval_bar,
                    symbol=symbol,
                    timeframe_label=timeframe_label,
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
                points_captured = 0.0
                capture_pct = 0.0
                if state["signal_generated"]:
                    expected_side = "BUY" if move.direction == "bullish" else "SELL"
                    if state["signal_direction"] == expected_side:
                        signal_before_move = True
                        mfe = float(state.get("outcome", {}).get("mfe_points", 0.0))
                        points_captured = round(min(mfe, move.magnitude), 2)
                        capture_pct = round(points_captured / move.magnitude * 100, 2) if move.magnitude else 0.0
                        best_capture = max(best_capture, points_captured)

                steps.append(
                    {
                        "timeline_step": f"T-{offset} minutes" if offset else "T-0",
                        "timestamp": str(frame.iloc[eval_bar].get("Date", "")),
                        "engine_state": state["engine_state"],
                        "signal_generated": state["signal_generated"],
                        "signal_direction": state["signal_direction"],
                        "entry": state.get("entry"),
                        "stop_loss": state.get("stop_loss"),
                        "target_1": state.get("target_1"),
                        "target_2": state.get("target_2"),
                        "target_3": state.get("target_3"),
                        "points_captured": points_captured,
                        "capture_pct": capture_pct,
                        "missing_conditions": state.get("missing_conditions", []),
                    },
                )

            timelines.append(
                {
                    "move_date": str(frame.iloc[move.expansion_bar].get("Date", "")),
                    "symbol": symbol,
                    "timeframe": timeframe_label,
                    "direction": move.direction,
                    "total_move_size_points": round(move.magnitude, 2),
                    "threshold_points": threshold,
                    "start_timestamp": str(frame.iloc[move.start_bar].get("Date", "")),
                    "could_enter_before_move": signal_before_move,
                    "best_points_captured": best_capture,
                    "best_capture_pct": round(best_capture / move.magnitude * 100, 2) if move.magnitude else 0.0,
                    "points_missed": round(max(move.magnitude - best_capture, 0.0), 2),
                    "timeline": steps,
                },
            )
        return timelines

    @staticmethod
    def _missed_classification(detected: bool, partial: bool) -> str:
        if detected:
            return "Detected"
        if partial:
            return "Partially Detected"
        return "Missed"

    def _build_missed_moves(
        self,
        *,
        symbol: str,
        timeframe_label: str,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        prechecks: dict[str, Any],
        trade_signals: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        highs = frame["High"].astype(float)
        lows = frame["Low"].astype(float)
        for threshold in MISSED_THRESHOLDS:
            moves = self.move_engine._dedupe_cheap_moves(
                self.move_engine._detect_moves_cheap(highs, lows, threshold),
            )
            for move in moves:
                if len(rows) >= MAX_MISSED_EXPORT:
                    return rows
                move_side = "BUY" if move.direction == "bullish" else "SELL"
                start_ts = pd.to_datetime(frame.iloc[move.start_bar].get("Date", ""))
                matching = [
                    item
                    for item in trade_signals
                    if item.get("signal_direction") == move_side
                    and item.get("timestamp")
                    and pd.to_datetime(item["timestamp"]) <= start_ts
                ]
                tier2_at_start = move.start_bar in {
                    int(item.get("bar", -1)) for item in trade_signals
                }
                partial = bool(
                    not matching
                    and any(
                        item.get("tier2_direction") == move.direction for item in trade_signals
                    ),
                )
                classification = self._missed_classification(bool(matching), partial)
                flags = self._feature_flags_at_bar(frame, enriched, move.start_bar)
                rows.append(
                    {
                        "threshold_points": threshold,
                        "symbol": symbol,
                        "timeframe": timeframe_label,
                        "direction": move.direction,
                        "move_magnitude_points": round(move.magnitude, 2),
                        "classification": classification,
                        "engine_generated_signal": bool(matching),
                        "points_captured": round(move.magnitude, 2) if matching else 0.0,
                        "points_missed": 0.0 if matching else round(move.magnitude, 2),
                        "missed_reasons": []
                        if matching
                        else self._diagnose_missed(
                            frame,
                            move.start_bar,
                            move.direction,
                            prechecks,
                            flags,
                        ),
                    },
                )
        return rows

    def _replay_frame_v1(
        self,
        *,
        symbol: str,
        timeframe_label: str,
        frame: pd.DataFrame,
        buy_keys: tuple[str, ...],
        sell_keys: tuple[str, ...],
        no_trade_rules: list[str],
        archetypes: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        tier_engine = TieredSignalFrameworkResearch(
            symbol=symbol,
            research_days=self.research_days,
            timeframes=(timeframe_label,),
        )
        trade_engine = TradeConstructionValidationResearch(
            symbol=symbol,
            research_days=self.research_days,
            timeframes=(timeframe_label,),
        )
        from src.context.institutional_liquidity_map_engine import InstitutionalLiquidityMapEngine

        liquidity_map = InstitutionalLiquidityMapEngine(symbol=symbol)
        enriched = self.context_builder.enrich(frame)
        calendar = liquidity_map._attach_calendar_levels(frame)
        intel = self.discovery_engine.intelligence_engine.enrich(frame)
        prechecks = self._build_prechecks(frame)

        tier2_by_bar: dict[int, TierSignal] = {}
        for signal in tier_engine._detect_tier2(frame, timeframe_label):
            tier2_by_bar[signal.bos_bar] = signal

        signals: list[dict[str, Any]] = []
        freq = Counter()
        monthly: dict[str, Counter] = defaultdict(Counter)
        last_signal_bar: dict[str, int] = {}
        scan_end = len(frame) - FORWARD_BARS

        for bar in range(PRE_EXPANSION_LOOKBACK, scan_end):
            if bar % SCAN_PROGRESS_INTERVAL == 0:
                logger.info(
                    "Reality check %s/%s bar=%s/%s signals=%s",
                    symbol,
                    timeframe_label,
                    bar,
                    scan_end,
                    len(signals),
                )

            tier2 = tier2_by_bar.get(bar)
            if tier2 is None:
                freq["NO_TRADE"] += 1
                continue

            state = self._evaluate_at_bar(
                bar=bar,
                symbol=symbol,
                timeframe_label=timeframe_label,
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
            decision = state["signal_direction"]
            freq[decision] += 1
            month_key = str(frame.iloc[bar].get("Date", ""))[:7]
            monthly[month_key][decision] += 1

            if decision in {"BUY", "SELL"}:
                prev = last_signal_bar.get(decision)
                if prev is not None and bar - prev < MIN_SIGNAL_SEPARATION_BARS:
                    freq["NO_TRADE"] += 1
                    continue
                last_signal_bar[decision] = bar

            if len(signals) < MAX_SIGNAL_LOG:
                outcome = state.get("outcome", {})
                signals.append(
                    {
                        "timestamp": str(frame.iloc[bar].get("Date", "")),
                        "symbol": symbol,
                        "timeframe": timeframe_label,
                        "bar": bar,
                        "signal_direction": decision,
                        "signal_archetype": state.get("signal_archetype"),
                        "entry": state.get("entry"),
                        "stop_loss": state.get("stop_loss"),
                        "target_1": state.get("target_1"),
                        "target_2": state.get("target_2"),
                        "target_3": state.get("target_3"),
                        "signal_score": state.get("signal_score"),
                        "reason_stack": state.get("reason_stack"),
                        "tier2_direction": tier2.direction,
                        "mfe_points": outcome.get("mfe_points"),
                        "mae_points": outcome.get("mae_points"),
                        "momentum_capture": outcome.get("momentum_capture"),
                        "hit_1r": outcome.get("hit_1r"),
                        "hit_2r": outcome.get("hit_2r"),
                        "hit_3r": outcome.get("hit_3r"),
                        "points_captured": outcome.get("realized_pnl_points"),
                        "realized_rr": outcome.get("realized_rr"),
                        "win": outcome.get("win"),
                    },
                )

        frequency_stats = {
            "total_bars_scanned": scan_end - PRE_EXPANSION_LOOKBACK,
            "buy_count": freq["BUY"],
            "sell_count": freq["SELL"],
            "no_trade_count": freq["NO_TRADE"],
            "by_symbol_timeframe": {f"{symbol}_{timeframe_label}": dict(freq)},
        }
        monthly_report = [
            {
                "month": month,
                "buy_count": counts["BUY"],
                "sell_count": counts["SELL"],
                "no_trade_count": counts["NO_TRADE"],
            }
            for month, counts in sorted(monthly.items())
        ]
        timelines = self._build_major_move_timelines(
            symbol=symbol,
            timeframe_label=timeframe_label,
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
        missed = self._build_missed_moves(
            symbol=symbol,
            timeframe_label=timeframe_label,
            frame=frame,
            enriched=enriched,
            prechecks=prechecks,
            trade_signals=[item for item in signals if item["signal_direction"] in {"BUY", "SELL"}],
        )
        return signals, frequency_stats, monthly_report, timelines, missed

    @staticmethod
    def _rank_missed_reasons(missed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        counter: Counter[str] = Counter()
        for row in missed_rows:
            if row.get("classification") == "Detected":
                continue
            for reason in row.get("missed_reasons", []):
                counter[reason] += 1
        return [
            {"reason": reason, "occurrences": count}
            for reason, count in counter.most_common(20)
        ]

    @staticmethod
    def _archetype_performance(signals: list[dict[str, Any]], months: float) -> list[dict[str, Any]]:
        buckets: dict[str, list[float]] = defaultdict(list)
        for item in signals:
            if item.get("signal_direction") not in {"BUY", "SELL"}:
                continue
            key = item.get("signal_archetype") or "Unmatched"
            pnl = float(item.get("points_captured") or 0.0)
            buckets[key].append(pnl)
        rows = []
        for key, pnls in buckets.items():
            if len(pnls) < 3:
                continue
            rows.append(
                {
                    "archetype": key,
                    "sample_size": len(pnls),
                    "signals_per_month": round(len(pnls) / max(months, 1.0), 2),
                    "win_rate_pct": round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 2),
                    "expectancy": round(mean(pnls), 2),
                    "profit_factor": SmartMoneyEngineRealityCheckValidationResearch._profit_factor(pnls),
                },
            )
        rows.sort(key=lambda row: row["expectancy"], reverse=True)
        return rows

    def _build_final_verdict(
        self,
        *,
        timelines: list[dict[str, Any]],
        missed_rows: list[dict[str, Any]],
        overall: dict[str, Any],
        frequency: dict[str, Any],
        reason_ranking: list[dict[str, Any]],
        archetype_summary: dict[str, Any],
    ) -> dict[str, Any]:
        def capture_rate(threshold: int) -> float:
            subset = [row for row in missed_rows if row["threshold_points"] == threshold]
            if not subset:
                return 0.0
            detected = sum(1 for row in subset if row["classification"] == "Detected")
            return round(detected / len(subset) * 100, 2)

        captured = [row["best_points_captured"] for row in timelines if row["could_enter_before_move"]]
        missed = [row["points_missed"] for row in timelines]
        avg_captured = round(mean(captured), 2) if captured else 0.0
        avg_missed = round(mean(missed), 2) if missed else 0.0

        can_detect = capture_rate(200) >= PRODUCTION_READY_CAPTURE_PCT
        monthly_signals = overall.get("signals_per_month", 0.0)
        production_ready = (
            can_detect
            and monthly_signals >= PRODUCTION_READY_MONTHLY_SIGNALS
            and overall.get("profit_factor", 0) is not None
            and (overall.get("profit_factor") or 0) >= 1.2
        )

        return {
            "can_detect_major_momentum_before_move": can_detect,
            "pct_200_plus_moves_detected": capture_rate(200),
            "pct_300_plus_moves_detected": capture_rate(300),
            "pct_500_plus_moves_detected": capture_rate(500),
            "average_points_captured_per_detected_move": avg_captured,
            "average_points_missed_per_move": avg_missed,
            "weekly_signal_frequency": frequency.get("signals_per_week", 0.0),
            "monthly_signal_frequency": monthly_signals,
            "is_engine_production_ready": production_ready,
            "production_readiness_verdict": "READY" if production_ready else "NOT READY",
            "biggest_failure_reasons": [item["reason"] for item in reason_ranking[:5]],
            "archetype_walkforward_reference": archetype_summary,
            "explicit_answers": {
                "1_can_detect_major_momentum": can_detect,
                "2_pct_200_plus_detected": capture_rate(200),
                "3_pct_300_plus_detected": capture_rate(300),
                "4_pct_500_plus_detected": capture_rate(500),
                "5_avg_points_captured": avg_captured,
                "6_avg_points_missed": avg_missed,
                "7_weekly_signal_frequency": frequency.get("signals_per_week", 0.0),
                "8_monthly_signal_frequency": monthly_signals,
                "9_production_ready": production_ready,
                "10_biggest_failure_reasons": [item["reason"] for item in reason_ranking[:5]],
            },
        }

    def run(self, metadata: dict[str, Any]) -> RealityCheckValidationReport:
        started = time.perf_counter()
        frozen = self._load_frozen_exports()
        v2_card = frozen["v2_production_card"]
        v1_card = frozen["v1_production_card"]
        archetypes = frozen["production_candidate_archetypes"] or frozen["top_50_archetypes"]

        buy_keys = self._keys_from_labels(v2_card["buy_rules"]["filter_stack"])
        sell_keys = self._keys_from_labels(v2_card["sell_rules"]["filter_stack"])
        no_trade_rules = list(v2_card.get("no_trade_rules", []))

        end = date.fromisoformat(metadata["end_date"])
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=self.research_days)
        )

        all_signals: list[dict[str, Any]] = []
        all_monthly: dict[str, Counter] = defaultdict(Counter)
        all_timelines: list[dict[str, Any]] = []
        all_missed: list[dict[str, Any]] = []
        aggregate = Counter()

        for symbol in self.symbols:
            filter_engine = FilterResearchEngine(
                symbol=symbol,
                research_days=self.research_days,
                timeframes=self.timeframes,
            )
            for timeframe_label in self.timeframes:
                try:
                    path = filter_engine._ensure_pipeline(timeframe_label, start, end)
                except Exception as exc:
                    logger.warning("Skipping %s/%s: %s", symbol, timeframe_label, exc)
                    continue
                frame = pd.read_csv(path).reset_index(drop=True)
                if len(frame) <= PRE_EXPANSION_LOOKBACK + FORWARD_BARS:
                    continue

                signals, freq_stats, monthly, timelines, missed = self._replay_frame_v1(
                    symbol=symbol,
                    timeframe_label=timeframe_label,
                    frame=frame,
                    buy_keys=buy_keys,
                    sell_keys=sell_keys,
                    no_trade_rules=no_trade_rules,
                    archetypes=archetypes,
                )
                all_signals.extend(signals)
                aggregate["total_bars_scanned"] += freq_stats.get("total_bars_scanned", 0)
                aggregate["buy_count"] += freq_stats.get("buy_count", 0)
                aggregate["sell_count"] += freq_stats.get("sell_count", 0)
                aggregate["no_trade_count"] += freq_stats.get("no_trade_count", 0)
                for row in monthly:
                    bucket = all_monthly[row["month"]]
                    bucket["BUY"] += row["buy_count"]
                    bucket["SELL"] += row["sell_count"]
                    bucket["NO_TRADE"] += row["no_trade_count"]
                all_timelines.extend(timelines)
                all_missed.extend(missed)

        trade_signals = [item for item in all_signals if item["signal_direction"] in {"BUY", "SELL"}]
        pnls = [float(item.get("points_captured") or 0.0) for item in trade_signals]
        months = max(metadata.get("research_window_days", self.research_days) / 30.4375, 1.0)
        weeks = max(metadata.get("research_window_days", self.research_days) / 7.0, 1.0)

        overall = {
            "total_bars_replayed": aggregate.get("total_bars_scanned", 0),
            "total_signals": len(trade_signals),
            "buy_count": aggregate.get("buy_count", 0),
            "sell_count": aggregate.get("sell_count", 0),
            "no_trade_count": aggregate.get("no_trade_count", 0),
            "signals_per_month": round(len(trade_signals) / months, 2),
            "signals_per_week": round(len(trade_signals) / weeks, 2),
            "win_rate_pct": round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 2) if pnls else 0.0,
            "profit_factor": self._profit_factor(pnls),
            "expectancy": round(mean(pnls), 2) if pnls else 0.0,
            "hit_1r_rate_pct": round(
                sum(1 for item in trade_signals if item.get("hit_1r")) / len(trade_signals) * 100,
                2,
            )
            if trade_signals
            else 0.0,
        }

        false_report = {
            "total_signals": len(trade_signals),
            "winning_signals": sum(1 for item in trade_signals if item.get("win")),
            "losing_signals": sum(1 for item in trade_signals if item.get("win") is False),
            "false_signals": sum(
                1
                for item in trade_signals
                if item.get("win") is False and not item.get("hit_1r")
            ),
            "early_signals": 0,
            "late_signals": 0,
            "average_drawdown_points": round(
                mean(float(item.get("mae_points") or 0.0) for item in trade_signals),
                2,
            )
            if trade_signals
            else 0.0,
            "average_holding_minutes": 0.0,
        }

        frequency_report = {
            "signals_per_week": overall["signals_per_week"],
            "signals_per_month": overall["signals_per_month"],
            "buy_count": overall["buy_count"],
            "sell_count": overall["sell_count"],
            "no_trade_count": overall["no_trade_count"],
            "per_symbol": {symbol: {"buy": 0, "sell": 0, "no_trade": 0} for symbol in self.symbols},
            "per_timeframe": {tf: {"buy": 0, "sell": 0, "no_trade": 0} for tf in self.timeframes},
            "monthly_breakdown": [
                {
                    "month": month,
                    "buy_count": counts["BUY"],
                    "sell_count": counts["SELL"],
                    "no_trade_count": counts["NO_TRADE"],
                }
                for month, counts in sorted(all_monthly.items())
            ],
        }
        for item in trade_signals:
            frequency_report["per_symbol"].setdefault(item["symbol"], {"buy": 0, "sell": 0, "no_trade": 0})
            frequency_report["per_timeframe"].setdefault(
                item["timeframe"],
                {"buy": 0, "sell": 0, "no_trade": 0},
            )
            side = item["signal_direction"].lower()
            frequency_report["per_symbol"][item["symbol"]][side] += 1
            frequency_report["per_timeframe"][item["timeframe"]][side] += 1

        reason_ranking = self._rank_missed_reasons(all_missed)
        archetype_rows = self._archetype_performance(trade_signals, months)
        verdict = self._build_final_verdict(
            timelines=all_timelines,
            missed_rows=all_missed,
            overall=overall,
            frequency=frequency_report,
            reason_ranking=reason_ranking,
            archetype_summary=frozen.get("archetype_walkforward_summary", {}),
        )

        conclusions = [
            "Reality Check V1: frozen V2 production blueprint replay with strict no-look-ahead.",
            f"Replayed {overall['total_bars_replayed']} bars; generated {overall['total_signals']} BUY/SELL signals.",
            f"200+ move detection rate: {verdict['pct_200_plus_moves_detected']}%.",
            f"300+ move detection rate: {verdict['pct_300_plus_moves_detected']}%.",
            f"500+ move detection rate: {verdict['pct_500_plus_moves_detected']}%.",
            f"Production readiness verdict: {verdict['production_readiness_verdict']}.",
            f"Top missed reason: {reason_ranking[0]['reason'] if reason_ranking else 'N/A'}.",
        ]

        return RealityCheckValidationReport(
            symbols_analyzed=list(self.symbols),
            timeframes_analyzed=list(self.timeframes),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            frozen_exports_loaded={
                "v1": str(self.v1_validation_path),
                "v2": str(self.v2_optimization_path),
                "ranking": str(self.v2_ranking_path),
                "archetype_walkforward": str(self.archetype_walkforward_path),
            },
            v1_production_card=v1_card,
            v2_production_card=v2_card,
            validated_archetypes=archetypes[:50],
            replay_rules={
                "no_future_leakage": True,
                "no_new_rules": True,
                "blueprint": "V2 production card",
                "mandatory_core": ["Displacement", "CHOCH", "BOS", "FVG Reclaim"],
                "buy_filters": v2_card["buy_rules"]["filter_stack"],
                "sell_filters": v2_card["sell_rules"]["filter_stack"],
                "no_trade_rules": no_trade_rules,
            },
            overall_statistics=overall,
            signal_by_signal_log=all_signals[:MAX_SIGNAL_LOG],
            major_move_replay_timeline=all_timelines[:MAX_TIMELINE_EXPORT],
            missed_move_report=all_missed[:MAX_MISSED_EXPORT],
            missed_move_reason_ranking=reason_ranking,
            false_signal_report=false_report,
            frequency_report=frequency_report,
            top_performing_archetypes=archetype_rows[:20],
            worst_performing_archetypes=list(reversed(archetype_rows[-20:])),
            final_production_verdict=verdict,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )


def generate_reality_check_validation_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
    symbols: tuple[str, ...] | None = None,
) -> RealityCheckValidationReport:
    """Run reality check validation and export JSON."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise RealityCheckValidationError(f"Filter research report not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = SmartMoneyEngineRealityCheckValidationResearch(symbols=symbols)
    report = engine.run(metadata)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info(
        "Reality check completed: verdict=%s 200+=%s%%",
        report.final_production_verdict.get("production_readiness_verdict"),
        report.final_production_verdict.get("pct_200_plus_moves_detected"),
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_reality_check_validation_report()
        verdict = report.final_production_verdict
        print("SmartMoneyEngine Reality Check Validation V1 Summary")
        print(f"Total signals: {report.overall_statistics['total_signals']}")
        print(f"200+ detection: {verdict['pct_200_plus_moves_detected']}%")
        print(f"300+ detection: {verdict['pct_300_plus_moves_detected']}%")
        print(f"500+ detection: {verdict['pct_500_plus_moves_detected']}%")
        print(f"Verdict: {verdict['production_readiness_verdict']}")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except RealityCheckValidationError as exc:
        logger.error("Reality check error: %s", exc)
        print(f"Reality check error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected reality check error")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
