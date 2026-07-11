"""
SmartMoneyEngine V5 Candidate Validation — research only.

Compares frozen V3, V4 Candidate, and V5 Candidate on 120-day NIFTY50 5M replay.
V5 change: VWAP gate accepts Below OR Rejected (V4 requires Below only).
All other V4 rules unchanged (EMA22+EMA200, optional confirmation).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd

from src.context.institutional_liquidity_map_engine import InstitutionalLiquidityMapEngine
from src.research.filter_research_engine import FilterResearchEngine, _json_safe
from src.research.institutional_expansion_trigger_discovery_research import PRE_EXPANSION_LOOKBACK
from src.research.liquidity_move_reconstruction_research import (
    FORWARD_BARS,
    LiquidityMoveReconstructionResearch,
    _CheapMoveCandidate,
)
from src.research.nifty50_trap_to_momentum_validation_research import DEFAULT_SYMBOL
from src.research.smartmoneyengine_v3_implementation_validation_research import (
    ALLOWED_VOLUME_BUCKETS,
    MOVE_DETECTION_TIMEFRAME,
    PIPELINE_TIMEFRAMES,
    SmartMoneyEngineV3Engine,
)
from src.research.smartmoneyengine_v4_candidate_validation_research import (
    V4CandidateEngine,
    V4_EMA22_RULE,
    V4_EMA_BEAR_CONTEXT,
    V4_EMA_BULL_CONTEXT,
    _attach_ema22,
    _build_statistics,
    _last_n_trading_day_set,
    _point_capture,
    _profit_factor,
    _signal_before_move,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_FILTER_REPORT_PATH = RESEARCH_DIR / "filter_research_report.json"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "smartmoneyengine_v5_candidate_validation.json"
DEFAULT_V4_REPORT_PATH = RESEARCH_DIR / "smartmoneyengine_v4_candidate_validation.json"

TRADING_DAYS_REPLAY = 120
POINT_CAPTURE_THRESHOLDS = (40, 60, 100, 200, 300, 500)
MAJOR_MOVE_THRESHOLDS = (100, 200, 300)
MOVE_DETECTION_THRESHOLD = 40

# VWAP states from Nifty50LiquidityDirectionDecisionMatrixResearch._vwap_state:
# "Above" | "Below" | "Reclaimed" | "Rejected"
# Context key at bar: context["vwap"] (string label, not vwap_state)
V5_VWAP_GATE_RULE = "VWAP Below OR VWAP Rejected"
V5_ALLOWED_VWAP_STATES = frozenset({"Below", "Rejected"})


class SmartMoneyEngineV5CandidateValidationError(Exception):
    """Raised when V5 candidate validation fails."""


def _v5_vwap_gate_passes(vwap_state: str | None) -> bool:
    return vwap_state in V5_ALLOWED_VWAP_STATES


@dataclass
class SmartMoneyEngineV5CandidateValidationReport:
    """V3 vs V4 vs V5 Candidate comparison validation output."""

    report_type: str
    engine_versions_compared: list[str]
    symbol: str
    timeframe: str
    trading_days_replayed: int
    replay_start_date: str
    replay_end_date: str
    v5_change_summary: dict[str, Any]
    methodology: dict[str, Any]
    replay_rules: dict[str, Any]
    comparison: dict[str, Any]
    incremental_vs_v4: dict[str, Any]
    point_capture: dict[str, Any]
    missed_move_recovery: dict[str, Any]
    final_questions: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float


class V5CandidateEngine(V4CandidateEngine):
    """V5 Candidate: V4 stack with relaxed VWAP gate (Below OR Rejected)."""

    MODEL_ID = "LDM-SELL-V5"

    def _layer2_directional_filter(self, context: dict[str, str]) -> dict[str, Any]:
        ema_bearish = context.get("v4_ema_bearish") == "True"
        vwap = context.get("vwap")
        vwap_ok = _v5_vwap_gate_passes(vwap)
        aligned = (
            context.get("htf_trend") == "Bearish"
            and vwap_ok
            and ema_bearish
        )
        return {
            "direction": "SELL" if aligned else "NO_TRADE",
            "htf_trend": context.get("htf_trend"),
            "vwap_state": vwap,
            "vwap_gate_rule": V5_VWAP_GATE_RULE,
            "vwap_gate_passes": vwap_ok,
            "ema_structure": context.get("v4_ema_structure"),
            "v4_ema_rule": V4_EMA22_RULE,
            "v4_ema_bearish": ema_bearish,
            "aligned": aligned,
        }

    def _layer5_no_trade_filters(
        self,
        *,
        layer1: dict[str, Any],
        layer2: dict[str, Any],
        layer3: dict[str, Any],
        context: dict[str, str],
        bar: int,
        emitted_bars: set[int],
    ) -> dict[str, Any]:
        reasons: list[str] = []
        if not layer1["active"]:
            reasons.append("NO_EARLY_WARNING")
        if not layer1["failed_breakout_present"]:
            reasons.append("NO_FAILED_BREAKOUT")
        if layer2.get("htf_trend") == "Bullish":
            reasons.append("HTF_CONFLICT")
        if not layer2.get("vwap_gate_passes"):
            reasons.append("VWAP_MISMATCH")
        if layer2.get("ema_structure") == V4_EMA_BULL_CONTEXT:
            reasons.append("EMA_MISMATCH")
        if not layer2.get("aligned"):
            reasons.append("DIRECTION_NOT_ALIGNED")
        if not layer3.get("confirmed"):
            reasons.append("VOLUME_FAILED")
        if context.get("location") == "Mid Range":
            reasons.append("LOCATION_MID_RANGE")
        if bar in emitted_bars:
            reasons.append("DUPLICATE_BAR")
        return {"pass": not reasons, "reason_codes": reasons}


class SmartMoneyEngineV5CandidateValidationResearch:
    """Compare V3 vs V4 vs V5 Candidate on 120-day NIFTY50 replay."""

    def __init__(self) -> None:
        self.v3_engine = SmartMoneyEngineV3Engine()
        self.v4_engine = V4CandidateEngine()
        self.v5_engine = V5CandidateEngine()
        self.move_engine = LiquidityMoveReconstructionResearch(
            symbol=DEFAULT_SYMBOL,
            research_days=120,
            timeframes=(MOVE_DETECTION_TIMEFRAME,),
        )

    def _build_signal(
        self,
        evaluation: dict[str, Any],
        *,
        engine_version: str,
    ) -> dict[str, Any]:
        layer4 = evaluation["layer4"]
        forward = dict(layer4.get("forward_outcome") or {})
        return {
            "timestamp": evaluation["timestamp"],
            "bar": evaluation["bar"],
            "symbol": DEFAULT_SYMBOL,
            "timeframe": MOVE_DETECTION_TIMEFRAME,
            "direction": "SELL",
            "engine_version": engine_version,
            "model_id": layer4.get("model_id"),
            "entry": layer4.get("entry"),
            "stop_loss": layer4.get("stop_loss"),
            "target_1": layer4.get("target_1"),
            "target_2": layer4.get("target_2"),
            "target_3": layer4.get("target_3"),
            "signal_reason_stack": layer4.get("signal_reason_stack"),
            "realized_pnl_points": forward.get("realized_pnl_points"),
            "mfe_points": forward.get("mfe_points"),
            "mae_points": forward.get("mae_points"),
            "hit_1r": forward.get("hit_1r"),
            "hit_2r": forward.get("hit_2r"),
            "hit_3r": forward.get("hit_3r"),
            "win": forward.get("win"),
            "layers": {
                "layer1": evaluation["layer1"],
                "layer2": evaluation["layer2"],
                "layer3": evaluation["layer3"],
                "layer5": evaluation["layer5"],
            },
        }

    def _replay_combined(
        self,
        *,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        calendar: pd.DataFrame,
        intel_frames: dict[str, pd.DataFrame],
        replay_bars: list[int],
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, int],
        dict[str, int],
        dict[str, int],
    ]:
        v3_signals: list[dict[str, Any]] = []
        v4_signals: list[dict[str, Any]] = []
        v5_signals: list[dict[str, Any]] = []
        v3_emitted_bars: set[int] = set()
        v4_emitted_bars: set[int] = set()
        v5_emitted_bars: set[int] = set()
        v3_rejections: dict[str, int] = {}
        v4_rejections: dict[str, int] = {}
        v5_rejections: dict[str, int] = {}
        total = len(replay_bars)
        log_every = max(total // 20, 1)
        started = time.perf_counter()

        for index, bar in enumerate(replay_bars):
            if index > 0 and index % log_every == 0:
                elapsed = time.perf_counter() - started
                logger.info(
                    "Replay progress: %s/%s bars (%.0f%%) elapsed %.0fs",
                    index,
                    total,
                    index / total * 100,
                    elapsed,
                )

            if bar < PRE_EXPANSION_LOOKBACK or bar >= len(frame) - FORWARD_BARS:
                continue

            v3_eval = self.v3_engine.evaluate_bar(
                frame=frame,
                enriched=enriched,
                calendar=calendar,
                intel_frames=intel_frames,
                bar=bar,
                emitted_bars=v3_emitted_bars,
            )
            v4_eval = self.v4_engine.evaluate_bar(
                frame=frame,
                enriched=enriched,
                calendar=calendar,
                intel_frames=intel_frames,
                bar=bar,
                emitted_bars=v4_emitted_bars,
            )
            v5_eval = self.v5_engine.evaluate_bar(
                frame=frame,
                enriched=enriched,
                calendar=calendar,
                intel_frames=intel_frames,
                bar=bar,
                emitted_bars=v5_emitted_bars,
            )

            if v3_eval["verdict"] == "SELL":
                v3_signals.append(self._build_signal(v3_eval, engine_version="V3"))
                v3_emitted_bars.add(bar)
            else:
                for reason in v3_eval["layer5"]["reason_codes"]:
                    v3_rejections[reason] = v3_rejections.get(reason, 0) + 1

            if v4_eval["verdict"] == "SELL":
                v4_signals.append(self._build_signal(v4_eval, engine_version="V4 Candidate"))
                v4_emitted_bars.add(bar)
            else:
                for reason in v4_eval["layer5"]["reason_codes"]:
                    v4_rejections[reason] = v4_rejections.get(reason, 0) + 1

            if v5_eval["verdict"] == "SELL":
                v5_signals.append(self._build_signal(v5_eval, engine_version="V5 Candidate"))
                v5_emitted_bars.add(bar)
            else:
                for reason in v5_eval["layer5"]["reason_codes"]:
                    v5_rejections[reason] = v5_rejections.get(reason, 0) + 1

        logger.info(
            "Replay complete: V3=%s V4=%s V5=%s signals in %.0fs",
            len(v3_signals),
            len(v4_signals),
            len(v5_signals),
            time.perf_counter() - started,
        )
        return (
            v3_signals,
            v4_signals,
            v5_signals,
            v3_rejections,
            v4_rejections,
            v5_rejections,
        )

    def _missed_move_recovery(
        self,
        moves: list[_CheapMoveCandidate],
        v4_signals: list[dict[str, Any]],
        v5_signals: list[dict[str, Any]],
        replay_dates: set[date],
        frame: pd.DataFrame,
    ) -> dict[str, Any]:
        by_threshold: dict[str, Any] = {}
        recovered_details: list[dict[str, Any]] = []

        for threshold in MAJOR_MOVE_THRESHOLDS:
            bearish = [
                move
                for move in moves
                if move.direction == "bearish"
                and move.magnitude >= threshold
                and pd.to_datetime(frame.iloc[move.start_bar]["Date"]).date() in replay_dates
            ]
            v4_missed_v5_captured = 0
            both_captured = 0
            v4_only = 0
            neither = 0

            for move in bearish:
                v4_entry = _signal_before_move(v4_signals, move.start_bar)
                v5_entry = _signal_before_move(v5_signals, move.start_bar)
                v4_ok = v4_entry is not None
                v5_ok = v5_entry is not None

                if v4_ok and v5_ok:
                    both_captured += 1
                elif v5_ok and not v4_ok:
                    v4_missed_v5_captured += 1
                    recovered_details.append(
                        {
                            "threshold_points": threshold,
                            "move_start_bar": move.start_bar,
                            "move_start_time": str(frame.iloc[move.start_bar]["Date"]),
                            "move_magnitude_points": round(move.magnitude, 2),
                            "v5_entry_bar": v5_entry["bar"],
                            "v5_entry_time": v5_entry["timestamp"],
                            "v5_entry_price": v5_entry.get("entry"),
                            "v5_vwap_state": v5_entry.get("layers", {})
                            .get("layer2", {})
                            .get("vwap_state"),
                            "v5_mfe_points": v5_entry.get("mfe_points"),
                        }
                    )
                elif v4_ok and not v5_ok:
                    v4_only += 1
                else:
                    neither += 1

            total = len(bearish)
            v4_missed = total - sum(
                1 for move in bearish if _signal_before_move(v4_signals, move.start_bar) is not None
            )
            by_threshold[str(threshold)] = {
                "total_bearish_moves": total,
                "v4_missed_moves": v4_missed,
                "v4_missed_v5_captured": v4_missed_v5_captured,
                "recovery_rate_pct": round(v4_missed_v5_captured / max(v4_missed, 1) * 100, 2),
                "both_captured": both_captured,
                "v4_only_captured": v4_only,
                "neither_captured": neither,
            }

        return {
            "by_threshold": by_threshold,
            "recovered_move_details": recovered_details[:100],
            "total_v4_missed_v5_recovered_200_plus": by_threshold.get("200", {}).get(
                "v4_missed_v5_captured", 0
            ),
            "summary": (
                f"V5 recovered {by_threshold.get('200', {}).get('v4_missed_v5_captured', 0)} "
                f"previously V4-missed 200+ bearish moves."
            ),
        }

    def _incremental_vs_v4(
        self,
        v4_stats: dict[str, Any],
        v5_stats: dict[str, Any],
        v4_capture: dict[str, Any],
        v5_capture: dict[str, Any],
        recovery: dict[str, Any],
    ) -> dict[str, Any]:
        additional_signals = (v5_stats.get("signals_emitted") or 0) - (v4_stats.get("signals_emitted") or 0)
        pf_v4 = v4_stats.get("profit_factor") or 0
        pf_v5 = v5_stats.get("profit_factor") or 0
        wr_v4 = v4_stats.get("win_rate_pct") or 0
        wr_v5 = v5_stats.get("win_rate_pct") or 0

        additional_moves: dict[str, int] = {}
        for threshold in POINT_CAPTURE_THRESHOLDS:
            key = str(threshold)
            v4_captured = v4_capture.get(key, {}).get("signals_before_move", 0)
            v5_captured = v5_capture.get(key, {}).get("signals_before_move", 0)
            additional_moves[key] = v5_captured - v4_captured

        return {
            "additional_signals_emitted": additional_signals,
            "additional_moves_captured_by_threshold": additional_moves,
            "additional_moves_200_plus": additional_moves.get("200", 0),
            "pf_delta_v5_minus_v4": round(pf_v5 - pf_v4, 2) if pf_v4 and pf_v5 else None,
            "pf_lost_vs_v4": round(pf_v4 - pf_v5, 2) if pf_v4 and pf_v5 and pf_v5 < pf_v4 else 0.0,
            "wr_delta_pp_v5_minus_v4": round(wr_v5 - wr_v4, 2),
            "wr_lost_vs_v4_pp": round(wr_v4 - wr_v5, 2) if wr_v5 < wr_v4 else 0.0,
            "expectancy_delta": round(
                (v5_stats.get("expectancy") or 0) - (v4_stats.get("expectancy") or 0),
                2,
            ),
            "signals_per_month_delta": round(
                (v5_stats.get("signals_per_month") or 0) - (v4_stats.get("signals_per_month") or 0),
                2,
            ),
            "v4_missed_v5_recovered_200_plus": recovery.get("total_v4_missed_v5_recovered_200_plus", 0),
            "headline": (
                f"V5 adds {additional_signals} signals vs V4; "
                f"recovers {recovery.get('total_v4_missed_v5_recovered_200_plus', 0)} V4-missed 200+ moves; "
                f"PF delta {round(pf_v5 - pf_v4, 2) if pf_v4 and pf_v5 else 'N/A'}; "
                f"WR delta {round(wr_v5 - wr_v4, 2)}pp."
            ),
        }

    def _final_questions(
        self,
        v4_stats: dict[str, Any],
        v5_stats: dict[str, Any],
        v4_capture: dict[str, Any],
        v5_capture: dict[str, Any],
        incremental: dict[str, Any],
    ) -> dict[str, Any]:
        capture_improved = any(
            v5_capture.get(str(threshold), {}).get("capture_rate_pct", 0)
            > v4_capture.get(str(threshold), {}).get("capture_rate_pct", 0)
            for threshold in (100, 200, 300, 500)
        ) or (incremental.get("additional_moves_200_plus") or 0) > 0

        pf_ok = (v5_stats.get("profit_factor") or 0) >= 3.0
        wr_ok = (v5_stats.get("win_rate_pct") or 0) >= 65.0
        superior = (
            capture_improved
            and pf_ok
            and wr_ok
            and (v5_stats.get("expectancy") or 0) >= (v4_stats.get("expectancy") or 0) * 0.90
        )

        def _answer(yes: bool, partial: bool, evidence: str) -> dict[str, Any]:
            verdict = "YES" if yes else ("PARTIAL" if partial else "NO")
            return {"answer": verdict, "evidence": evidence}

        recovered_200 = incremental.get("v4_missed_v5_recovered_200_plus", 0)
        pf_v4 = v4_stats.get("profit_factor")
        pf_v5 = v5_stats.get("profit_factor")
        wr_v4 = v4_stats.get("win_rate_pct")
        wr_v5 = v5_stats.get("win_rate_pct")

        return {
            "1_does_v5_improve_capture": _answer(
                capture_improved,
                capture_improved
                and not all(
                    v5_capture.get(str(t), {}).get("capture_rate_pct", 0)
                    >= v4_capture.get(str(t), {}).get("capture_rate_pct", 0)
                    for t in POINT_CAPTURE_THRESHOLDS
                ),
                f"200+ capture V4 {v4_capture.get('200', {}).get('capture_rate_pct')}% vs "
                f"V5 {v5_capture.get('200', {}).get('capture_rate_pct')}%; "
                f"additional 200+ moves {incremental.get('additional_moves_200_plus', 0)}; "
                f"recovered {recovered_200} V4-missed 200+ moves.",
            ),
            "2_does_pf_remain_above_3": _answer(
                pf_ok,
                (v5_stats.get("profit_factor") or 0) >= 2.5,
                f"V5 PF {pf_v5} vs V4 {pf_v4}; delta {incremental.get('pf_delta_v5_minus_v4')}.",
            ),
            "3_does_wr_remain_above_65": _answer(
                wr_ok,
                (v5_stats.get("win_rate_pct") or 0) >= 60.0,
                f"V5 WR {wr_v5}% vs V4 {wr_v4}%; lost {incremental.get('wr_lost_vs_v4_pp')}pp vs V4.",
            ),
            "4_is_v5_superior_to_v4": _answer(
                superior,
                capture_improved or pf_ok,
                f"Expectancy V5 {v5_stats.get('expectancy')} vs V4 {v4_stats.get('expectancy')}; "
                f"signals/month V5 {v5_stats.get('signals_per_month')} vs V4 {v4_stats.get('signals_per_month')}; "
                f"{incremental.get('headline')}",
            ),
        }

    def run(self, metadata: dict[str, Any]) -> SmartMoneyEngineV5CandidateValidationReport:
        started = time.perf_counter()
        end = date.fromisoformat(metadata["end_date"])
        start = end - timedelta(days=120)

        logger.info("V5 validation starting: %s trading days, %s 5M", TRADING_DAYS_REPLAY, DEFAULT_SYMBOL)

        filter_engine = FilterResearchEngine(
            symbol=DEFAULT_SYMBOL,
            research_days=120,
            timeframes=PIPELINE_TIMEFRAMES,
        )
        path = filter_engine._ensure_pipeline(MOVE_DETECTION_TIMEFRAME, start, end)
        frame = pd.read_csv(path).reset_index(drop=True)
        replay_dates = _last_n_trading_day_set(frame, TRADING_DAYS_REPLAY)
        bar_dates = pd.to_datetime(frame["Date"]).dt.date
        replay_bars = [index for index, day in enumerate(bar_dates) if day in replay_dates]

        logger.info("Loading enriched context and intel frames...")
        enriched = _attach_ema22(self.v3_engine.context_builder.enrich(frame))
        calendar = InstitutionalLiquidityMapEngine(symbol=DEFAULT_SYMBOL)._attach_calendar_levels(frame)
        intel_frames: dict[str, pd.DataFrame] = {"5M": self.v3_engine.intelligence.enrich(frame)}
        for tf in ("15M", "1H"):
            tf_path = filter_engine._ensure_pipeline(tf, start, end)
            tf_frame = pd.read_csv(tf_path).reset_index(drop=True)
            intel_frames[tf] = self.v3_engine.intelligence.enrich(tf_frame)
        intel_frames["1D"] = self.v3_engine.intelligence.enrich(
            self.v3_engine._resample_daily(intel_frames["1H"]),
        )

        logger.info("Detecting bearish moves (threshold=%s)...", MOVE_DETECTION_THRESHOLD)
        highs = frame["High"].astype(float)
        lows = frame["Low"].astype(float)
        moves = self.move_engine._dedupe_cheap_moves(
            self.move_engine._detect_moves_cheap(highs, lows, MOVE_DETECTION_THRESHOLD),
        )
        logger.info("Detected %s deduped moves >= %s pts", len(moves), MOVE_DETECTION_THRESHOLD)

        v3_signals, v4_signals, v5_signals, v3_rej, v4_rej, v5_rej = self._replay_combined(
            frame=frame,
            enriched=enriched,
            calendar=calendar,
            intel_frames=intel_frames,
            replay_bars=replay_bars,
        )

        v3_stats = _build_statistics(v3_signals, trading_days=TRADING_DAYS_REPLAY)
        v4_stats = _build_statistics(v4_signals, trading_days=TRADING_DAYS_REPLAY)
        v5_stats = _build_statistics(v5_signals, trading_days=TRADING_DAYS_REPLAY)
        v3_capture = _point_capture(moves, v3_signals, replay_dates, frame, POINT_CAPTURE_THRESHOLDS)
        v4_capture = _point_capture(moves, v4_signals, replay_dates, frame, POINT_CAPTURE_THRESHOLDS)
        v5_capture = _point_capture(moves, v5_signals, replay_dates, frame, POINT_CAPTURE_THRESHOLDS)
        recovery = self._missed_move_recovery(moves, v4_signals, v5_signals, replay_dates, frame)
        incremental = self._incremental_vs_v4(v4_stats, v5_stats, v4_capture, v5_capture, recovery)

        replay_start = min(replay_dates).isoformat() if replay_dates else ""
        replay_end = max(replay_dates).isoformat() if replay_dates else ""

        final_questions = self._final_questions(v4_stats, v5_stats, v4_capture, v5_capture, incremental)

        reference_context: dict[str, Any] = {}
        if DEFAULT_V4_REPORT_PATH.exists():
            v4_ref = json.loads(DEFAULT_V4_REPORT_PATH.read_text(encoding="utf-8"))
            reference_context["v4_baseline"] = {
                "source": str(DEFAULT_V4_REPORT_PATH.name),
                "v4_signals": v4_ref.get("comparison", {})
                .get("v4_candidate", {})
                .get("signals_emitted_count"),
                "v4_pf": v4_ref.get("comparison", {})
                .get("v4_candidate", {})
                .get("overall_statistics", {})
                .get("profit_factor"),
            }

        return SmartMoneyEngineV5CandidateValidationReport(
            report_type="SmartMoneyEngine V5 Candidate Validation",
            engine_versions_compared=[
                "SmartMoneyEngine V3",
                "SmartMoneyEngine V4 Candidate",
                "SmartMoneyEngine V5 Candidate",
            ],
            symbol=DEFAULT_SYMBOL,
            timeframe=MOVE_DETECTION_TIMEFRAME,
            trading_days_replayed=TRADING_DAYS_REPLAY,
            replay_start_date=replay_start,
            replay_end_date=replay_end,
            v5_change_summary={
                "unchanged_from_v4": [
                    "Failed Breakout",
                    "HTF Bearish",
                    "EMA22 + EMA200 Context (no EMA50)",
                    "Location filters",
                    "Confirmation optional",
                    "Volume bucket gate",
                ],
                "modified": {
                    "vwap_gate": {
                        "v3": "VWAP Below required",
                        "v4": "VWAP Below required",
                        "v5": V5_VWAP_GATE_RULE,
                        "allowed_vwap_states": sorted(V5_ALLOWED_VWAP_STATES),
                        "rejected_vwap_states": ["Above", "Reclaimed"],
                    },
                },
            },
            methodology={
                "research_only": True,
                "single_pass_replay": True,
                "v3_engine": "SmartMoneyEngineV3Engine (frozen)",
                "v4_engine": "V4CandidateEngine (frozen baseline)",
                "v5_engine": "V5CandidateEngine",
                "vwap_state_source": (
                    "Nifty50LiquidityDirectionDecisionMatrixResearch._vwap_state() "
                    "exposed as context['vwap'] at bar evaluation"
                ),
                "vwap_state_values": {
                    "Above": "close >= vwap (no cross/rejection pattern)",
                    "Below": "close < vwap (no cross/rejection pattern)",
                    "Reclaimed": "VWAP crossed (prev_close vs vwap cross on current bar)",
                    "Rejected": (
                        "price wicked through VWAP but closed on opposite side: "
                        "(high >= vwap and close < vwap) OR (low <= vwap and close > vwap)"
                    ),
                },
                "v5_vwap_gate_rule": V5_VWAP_GATE_RULE,
                "v5_vwap_gate_implementation": (
                    "layer2 aligned when context['vwap'] in {'Below', 'Rejected'}; "
                    "layer5 VWAP_MISMATCH when vwap not in allowed set"
                ),
                "v4_vwap_gate_rule": "context['vwap'] == 'Below' only",
                "move_detection_threshold": MOVE_DETECTION_THRESHOLD,
                "point_capture_thresholds": list(POINT_CAPTURE_THRESHOLDS),
                "point_capture_note": (
                    "40+ and 60+ computed from move engine at detection threshold 40; "
                    "same LiquidityMoveReconstructionResearch._detect_moves_cheap as V4 family"
                ),
                "missed_move_recovery_method": (
                    "Bearish move >= threshold in replay window; V4 missed = no signal within "
                    f"PRE_EXPANSION_LOOKBACK ({PRE_EXPANSION_LOOKBACK}) bars before move start; "
                    "V5 captured = signal present in same window when V4 absent"
                ),
                "reference_exports": reference_context,
            },
            replay_rules={
                "symbol": DEFAULT_SYMBOL,
                "timeframe": MOVE_DETECTION_TIMEFRAME,
                "trading_days": TRADING_DAYS_REPLAY,
                "candle_by_candle": True,
                "no_look_ahead": True,
                "research_only": True,
                "combined_replay": "V3 + V4 + V5 evaluated once per bar",
            },
            comparison={
                "v3": {
                    "overall_statistics": v3_stats,
                    "point_capture": v3_capture,
                    "layer_rejection_summary": v3_rej,
                    "signals_emitted_count": len(v3_signals),
                },
                "v4_candidate": {
                    "overall_statistics": v4_stats,
                    "point_capture": v4_capture,
                    "layer_rejection_summary": v4_rej,
                    "signals_emitted_count": len(v4_signals),
                },
                "v5_candidate": {
                    "overall_statistics": v5_stats,
                    "point_capture": v5_capture,
                    "layer_rejection_summary": v5_rej,
                    "signals_emitted_count": len(v5_signals),
                },
            },
            incremental_vs_v4=incremental,
            point_capture={
                "v3": {str(k): v3_capture[str(k)] for k in POINT_CAPTURE_THRESHOLDS},
                "v4_candidate": {str(k): v4_capture[str(k)] for k in POINT_CAPTURE_THRESHOLDS},
                "v5_candidate": {str(k): v5_capture[str(k)] for k in POINT_CAPTURE_THRESHOLDS},
            },
            missed_move_recovery=recovery,
            final_questions=final_questions,
            conclusions=[
                f"V3={len(v3_signals)} V4={len(v4_signals)} V5={len(v5_signals)} signals over {TRADING_DAYS_REPLAY} days.",
                f"WR V4 {v4_stats.get('win_rate_pct')}% vs V5 {v5_stats.get('win_rate_pct')}% "
                f"(lost {incremental.get('wr_lost_vs_v4_pp')}pp).",
                f"PF V4 {v4_stats.get('profit_factor')} vs V5 {v5_stats.get('profit_factor')} "
                f"(delta {incremental.get('pf_delta_v5_minus_v4')}).",
                f"200+ capture V4 {v4_capture.get('200', {}).get('capture_rate_pct')}% vs "
                f"V5 {v5_capture.get('200', {}).get('capture_rate_pct')}% "
                f"(+{incremental.get('additional_moves_200_plus', 0)} moves).",
                incremental.get("headline", ""),
                f"Final: V5 superior to V4 — {final_questions['4_is_v5_superior_to_v4']['answer']}.",
            ],
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )

    def export(self, report: SmartMoneyEngineV5CandidateValidationReport, report_path: Path) -> Path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(_json_safe(asdict(report)), indent=2), encoding="utf-8")
        logger.info("V5 candidate validation exported: %s", report_path)
        return report_path


def generate_smartmoneyengine_v5_candidate_validation_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> SmartMoneyEngineV5CandidateValidationReport:
    """Run V3 vs V4 vs V5 Candidate validation and export JSON."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise SmartMoneyEngineV5CandidateValidationError(
            f"Filter research report not found: {metadata_path}",
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    research = SmartMoneyEngineV5CandidateValidationResearch()
    report = research.run(metadata)
    destination = Path(report_path) if report_path else DEFAULT_REPORT_PATH
    research.export(report, destination)
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        report = generate_smartmoneyengine_v5_candidate_validation_report()
    except SmartMoneyEngineV5CandidateValidationError as exc:
        logger.error("V5 candidate validation error: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected V5 candidate validation error")
        return 1

    v4 = report.comparison["v4_candidate"]["overall_statistics"]
    v5 = report.comparison["v5_candidate"]["overall_statistics"]
    inc = report.incremental_vs_v4
    print("SmartMoneyEngine V5 Candidate Validation Summary")
    print(f"V4 signals: {v4['signals_emitted']} | V5 signals: {v5['signals_emitted']}")
    print(f"V4 WR: {v4['win_rate_pct']}% | V5 WR: {v5['win_rate_pct']}%")
    print(f"V4 PF: {v4['profit_factor']} | V5 PF: {v5['profit_factor']}")
    print(f"Additional 200+ moves vs V4: {inc.get('additional_moves_200_plus')}")
    for key in report.final_questions:
        print(f"{key}: {report.final_questions[key]['answer']}")
    print(f"Report: {DEFAULT_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
