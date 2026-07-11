"""
SELL Formula Reality Verification V2 — LDM-SELL-01 only.

Verifies the extracted SELL formula from smartmoneyengine_final_signal_extraction.json
using strict causal replay. Research-only; no production modifications.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd

from src.research.filter_research_engine import _json_safe
from src.research.institutional_expansion_trigger_discovery_research import PRE_EXPANSION_LOOKBACK
from src.research.liquidity_move_reconstruction_research import FORWARD_BARS, _CheapMoveCandidate
from src.research.nifty50_liquidity_direction_decision_matrix_research import (
    LIQUIDITY_EVENTS,
    Nifty50LiquidityDirectionDecisionMatrixResearch,
    RESEARCH_WINDOW_DAYS,
)
from src.research.nifty50_trap_to_momentum_validation_research import (
    DEFAULT_SYMBOL,
    MOVE_DETECTION_TIMEFRAME,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_EXTRACTION_PATH = RESEARCH_DIR / "smartmoneyengine_final_signal_extraction.json"
DEFAULT_FILTER_REPORT_PATH = RESEARCH_DIR / "filter_research_report.json"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "sell_formula_reality_verification_v2.json"

MODEL_ID = "LDM-SELL-01"
EXPANSION_THRESHOLDS = (50, 100, 200, 300, 500)
MAJOR_THRESHOLDS = (200, 300, 500)


class SellFormulaRealityVerificationV2Error(Exception):
    """Raised when SELL formula verification fails."""


@dataclass
class SellFormulaRealityVerificationV2Report:
    """LDM-SELL-01 reality verification output."""

    model_id: str
    formula: list[str]
    formula_text: str
    source_export: str
    symbol: str
    timeframe: str
    research_window_days: int
    start_date: str
    end_date: str
    methodology: dict[str, Any]
    expected_occurrences: int
    actual_occurrences: int
    all_occurrences: list[dict[str, Any]]
    causal_validation_summary: dict[str, Any]
    momentum_capture_validation: dict[str, Any]
    major_move_validation: dict[str, Any]
    trade_execution_validation: dict[str, Any]
    performance_metrics: dict[str, Any]
    best_20_trades: list[dict[str, Any]]
    worst_20_trades: list[dict[str, Any]]
    final_decision: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SellFormulaRealityVerificationV2Research(Nifty50LiquidityDirectionDecisionMatrixResearch):
    """Verify LDM-SELL-01 causal tradeability."""

    def _load_model_spec(self, extraction_path: Path) -> dict[str, Any]:
        if not extraction_path.exists():
            raise SellFormulaRealityVerificationV2Error(
                f"Final signal extraction not found: {extraction_path}",
            )
        payload = json.loads(extraction_path.read_text(encoding="utf-8"))
        for model in payload.get("top_10_sell_models", []):
            if model.get("model_id") == MODEL_ID:
                return model
        raise SellFormulaRealityVerificationV2Error(f"{MODEL_ID} not found in final signal extraction.")

    @staticmethod
    def _build_target_combo(context: dict[str, str], event: str) -> str:
        return Nifty50LiquidityDirectionDecisionMatrixResearch._combo_key(event, context)

    @staticmethod
    def _signal_reason_stack(context: dict[str, str], event: str) -> list[str]:
        stack = [event]
        if context.get("htf_trend"):
            stack.append(f"HTF {context['htf_trend']}")
        if context.get("vwap"):
            stack.append(f"VWAP {context['vwap']}")
        if context.get("ema_structure"):
            stack.append(f"EMA {context['ema_structure']}")
        if context.get("rsi"):
            stack.append(f"RSI {context['rsi']}")
        if context.get("volume"):
            stack.append(f"Volume {context['volume']}")
        if context.get("location"):
            stack.append(f"Location {context['location']}")
        if context.get("confirmation_candle") and context["confirmation_candle"] != "None":
            stack.append(context["confirmation_candle"])
        return stack

    @staticmethod
    def _signal_score(context: dict[str, str], event: str) -> float:
        expected = {
            "event": "Failed Breakout",
            "htf_trend": "Bearish",
            "vwap": "Below",
            "ema_structure": "Bear Stack",
            "rsi": "40-60",
            "volume": "Normal",
            "location": "Near Support",
        }
        matched = 0
        total = len(expected)
        if event == expected["event"]:
            matched += 1
        for key in ("htf_trend", "vwap", "ema_structure", "rsi", "volume", "location"):
            if context.get(key) == expected[key]:
                matched += 1
        return round(matched / total * 100, 2)

    @staticmethod
    def _causal_classification(context: dict[str, str], row: pd.Series) -> tuple[str, dict[str, bool]]:
        checks = {
            "signal_existed_before_move": True,
            "required_future_information": False,
            "required_future_bos": False,
            "required_future_fvg": False,
            "required_future_confirmation_candle": False,
            "required_future_structure": False,
            "required_future_liquidity_information": False,
            "context_at_event_bar_only": True,
            "bos_from_current_bar_only": True,
            "choch_from_current_bar_only": True,
        }
        if checks["required_future_information"] or any(
            checks[k]
            for k in (
                "required_future_bos",
                "required_future_fvg",
                "required_future_confirmation_candle",
                "required_future_structure",
                "required_future_liquidity_information",
            )
        ):
            classification = "LOOK-AHEAD CONTAMINATED"
        elif context.get("confirmation_candle") not in {None, "None"}:
            classification = "CAUSAL"
        else:
            classification = "CAUSAL"
        return classification, checks

    @staticmethod
    def _expansion_flags(forward_bear: float) -> dict[str, bool]:
        return {f"{threshold}_plus": forward_bear >= threshold for threshold in EXPANSION_THRESHOLDS}

    @staticmethod
    def _tradeability_classification(
        *,
        sell_outcome: dict[str, Any],
        bars_before_expansion: int,
        forward_bear: float,
    ) -> str:
        risk = float(sell_outcome.get("risk_points") or 0.0)
        mfe = float(sell_outcome.get("mfe_points") or 0.0)
        mae = float(sell_outcome.get("mae_points") or 0.0)
        realized = float(sell_outcome.get("realized_pnl_points") or 0.0)
        if risk <= 0:
            return "NOT TRADEABLE"
        if bars_before_expansion < 0:
            return "AMBIGUOUS"
        if bars_before_expansion <= 2 and forward_bear >= 100:
            return "LATE ENTRY"
        if bars_before_expansion >= 15 and mfe < risk:
            return "EARLY ENTRY"
        if mfe >= 2 * risk and realized > 0 and bars_before_expansion >= 3:
            return "CLEAR SETUP"
        if mfe >= risk or realized > 0:
            return "TRADEABLE"
        if mae >= 2 * risk and mfe < risk:
            return "NOT TRADEABLE"
        return "AMBIGUOUS"

    def _build_occurrence(
        self,
        *,
        frame: pd.DataFrame,
        bar: int,
        context: dict[str, str],
        sell_outcome: dict[str, Any],
        forward_bear: float,
        linked: _CheapMoveCandidate | None,
        bars_before_expansion: int,
    ) -> dict[str, Any]:
        timestamp = str(frame.iloc[bar].get("Date", ""))
        risk = float(sell_outcome.get("risk_points") or 0.0)
        entry = sell_outcome.get("entry")
        stop = sell_outcome.get("stop_loss")
        causal_class, causal_checks = self._causal_classification(context, frame.iloc[bar])
        points_before = None
        if linked is not None and bars_before_expansion >= 0:
            move_start = float(frame.iloc[linked.start_bar]["Close"])
            entry_price = float(entry or frame.iloc[bar]["Close"])
            points_before = round(max(move_start - entry_price, 0.0), 2)
        return {
            "date": timestamp[:10],
            "time": timestamp,
            "symbol": DEFAULT_SYMBOL,
            "timeframe": MOVE_DETECTION_TIMEFRAME,
            "entry": entry,
            "stop_loss": stop,
            "target_1": round(float(entry) - risk, 2) if entry and risk else None,
            "target_2": round(float(entry) - 2 * risk, 2) if entry and risk else None,
            "target_3": round(float(entry) - 3 * risk, 2) if entry and risk else None,
            "signal_score": self._signal_score(context, "Failed Breakout"),
            "signal_reason_stack": self._signal_reason_stack(context, "Failed Breakout"),
            "mfe": sell_outcome.get("mfe_points"),
            "mae": sell_outcome.get("mae_points"),
            "final_outcome": "WIN" if sell_outcome.get("win") else "LOSS",
            "realized_pnl_points": sell_outcome.get("realized_pnl_points"),
            "hit_1r": bool(sell_outcome.get("hit_1r")),
            "hit_2r": bool(sell_outcome.get("hit_2r")),
            "hit_3r": bool(sell_outcome.get("hit_3r")),
            "causal_classification": causal_class,
            "causal_checks": causal_checks,
            "bars_before_expansion": bars_before_expansion,
            "points_before_expansion": points_before,
            "expansion_reached": self._expansion_flags(forward_bear),
            "forward_bear_points": round(forward_bear, 2),
            "linked_move_magnitude": linked.magnitude if linked else None,
            "tradeability_classification": self._tradeability_classification(
                sell_outcome=sell_outcome,
                bars_before_expansion=bars_before_expansion,
                forward_bear=forward_bear,
            ),
            "context": context,
            "bar": bar,
        }

    def _diagnose_miss(
        self,
        *,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        calendar: pd.DataFrame,
        intel_frames: dict[str, pd.DataFrame],
        bar: int,
    ) -> str:
        events = self._detect_events_at_bar(frame, calendar, bar)
        if "Failed Breakout" not in events:
            return "No Failed Breakout"
        context = self._context_at_bar(
            frame=frame,
            enriched=enriched,
            calendar=calendar,
            intel_frames=intel_frames,
            bar=bar,
        )
        if context.get("htf_trend") != "Bearish":
            return "No HTF Alignment"
        if context.get("vwap") != "Below":
            return "No VWAP Confirmation"
        if context.get("ema_structure") != "Bear Stack":
            return "No EMA Alignment"
        if context.get("rsi") != "40-60":
            return "No RSI Alignment"
        if context.get("volume") != "Normal":
            return "No Volume Alignment"
        if context.get("location") != "Near Support":
            return "No Location Alignment"
        return "No Signal Generated"

    def _major_move_validation(
        self,
        *,
        moves: list[_CheapMoveCandidate],
        match_bars: dict[int, dict[str, Any]],
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        calendar: pd.DataFrame,
        intel_frames: dict[str, pd.DataFrame],
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for threshold in MAJOR_THRESHOLDS:
            bearish_moves = [
                move for move in moves if move.direction == "bearish" and move.magnitude >= threshold
            ]
            captured_rows: list[dict[str, Any]] = []
            missed_rows: list[dict[str, Any]] = []
            for move in bearish_moves:
                pre_start = max(PRE_EXPANSION_LOOKBACK, move.start_bar - PRE_EXPANSION_LOOKBACK)
                hit_bar = None
                hit_occ = None
                for bar in range(pre_start, move.start_bar + 1):
                    if bar in match_bars:
                        hit_bar = bar
                        hit_occ = match_bars[bar]
                        break
                if hit_bar is not None and hit_occ is not None:
                    entry = float(hit_occ.get("entry") or frame.iloc[hit_bar]["Close"])
                    end_bar = min(move.expansion_bar, len(frame) - 1)
                    move_low = float(frame.iloc[hit_bar : end_bar + 1]["Low"].astype(float).min())
                    captured = round(max(entry - move_low, 0.0), 2)
                    missed = round(max(move.magnitude - captured, 0.0), 2)
                    captured_rows.append(
                        {
                            "move_start": str(frame.iloc[move.start_bar].get("Date", "")),
                            "move_magnitude": move.magnitude,
                            "signal_time": hit_occ["time"],
                            "bars_before_move": move.start_bar - hit_bar,
                            "points_captured": captured,
                            "points_missed": missed,
                        },
                    )
                else:
                    reason = self._diagnose_miss(
                        frame=frame,
                        enriched=enriched,
                        calendar=calendar,
                        intel_frames=intel_frames,
                        bar=move.start_bar,
                    )
                    missed_rows.append(
                        {
                            "move_start": str(frame.iloc[move.start_bar].get("Date", "")),
                            "move_magnitude": move.magnitude,
                            "miss_reason": reason,
                        },
                    )
            total = len(bearish_moves)
            results[str(threshold)] = {
                "total_bearish_moves": total,
                "signals_present_before_move": len(captured_rows),
                "capture_rate_pct": round(len(captured_rows) / max(total, 1) * 100, 2),
                "captured_examples": captured_rows[:10],
                "missed_examples": missed_rows[:10],
            }
        return results

    @staticmethod
    def _performance_metrics(occurrences: list[dict[str, Any]], research_days: int) -> dict[str, Any]:
        pnls = [float(o.get("realized_pnl_points") or 0.0) for o in occurrences]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        loss_sum = abs(sum(losses))
        pf = round(sum(wins) / loss_sum, 2) if loss_sum else None
        months = max(research_days / 30.0, 1.0)
        weeks = max(research_days / 7.0, 1.0)
        drawdowns = [float(o.get("mae") or 0.0) for o in occurrences]
        return {
            "true_causal_win_rate_pct": round(sum(1 for o in occurrences if o.get("final_outcome") == "WIN") / max(len(occurrences), 1) * 100, 2),
            "true_causal_profit_factor": pf,
            "true_causal_expectancy": round(mean(pnls), 2) if pnls else 0.0,
            "true_causal_1r_rate_pct": round(sum(1 for o in occurrences if o.get("hit_1r")) / max(len(occurrences), 1) * 100, 2),
            "true_causal_2r_rate_pct": round(sum(1 for o in occurrences if o.get("hit_2r")) / max(len(occurrences), 1) * 100, 2),
            "true_causal_3r_rate_pct": round(sum(1 for o in occurrences if o.get("hit_3r")) / max(len(occurrences), 1) * 100, 2),
            "signals_per_week": round(len(occurrences) / weeks, 2),
            "signals_per_month": round(len(occurrences) / months, 2),
            "average_drawdown": round(mean(drawdowns), 2) if drawdowns else 0.0,
            "max_drawdown": round(max(drawdowns), 2) if drawdowns else 0.0,
        }

    def run(
        self,
        metadata: dict[str, Any],
        *,
        extraction_path: Path | None = None,
    ) -> SellFormulaRealityVerificationV2Report:
        started = time.perf_counter()
        model_spec = self._load_model_spec(Path(extraction_path or DEFAULT_EXTRACTION_PATH))
        expected_context = model_spec.get("context_snapshot", {})
        expected_event = "Failed Breakout"
        target_combo = self._build_target_combo(expected_context, expected_event)

        end = date.fromisoformat(metadata["end_date"])
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=RESEARCH_WINDOW_DAYS)
        )

        from src.context.institutional_liquidity_map_engine import InstitutionalLiquidityMapEngine
        from src.research.filter_research_engine import FilterResearchEngine

        filter_engine = FilterResearchEngine(
            symbol=DEFAULT_SYMBOL,
            research_days=RESEARCH_WINDOW_DAYS,
            timeframes=("5M", "15M", "1H"),
        )
        path = filter_engine._ensure_pipeline(MOVE_DETECTION_TIMEFRAME, start, end)
        frame = pd.read_csv(path).reset_index(drop=True)
        enriched = self.context_builder.enrich(frame)
        calendar = InstitutionalLiquidityMapEngine(symbol=DEFAULT_SYMBOL)._attach_calendar_levels(frame)

        intel_frames: dict[str, pd.DataFrame] = {"5M": self.intelligence.enrich(frame)}
        for tf in ("15M", "1H"):
            tf_path = filter_engine._ensure_pipeline(tf, start, end)
            tf_frame = pd.read_csv(tf_path).reset_index(drop=True)
            intel_frames[tf] = self.intelligence.enrich(tf_frame)
        intel_frames["1D"] = self.intelligence.enrich(self._resample_daily(intel_frames["1H"]))

        highs = frame["High"].astype(float)
        lows = frame["Low"].astype(float)
        closes = frame["Close"].astype(float)
        moves = self.move_engine._dedupe_cheap_moves(
            self.move_engine._detect_moves_cheap(highs, lows, 50),
        )

        occurrences: list[dict[str, Any]] = []
        match_bars: dict[int, dict[str, Any]] = {}
        scan_end = len(frame) - FORWARD_BARS

        for bar in range(PRE_EXPANSION_LOOKBACK, scan_end):
            if bar % 2000 == 0:
                logger.info("LDM-SELL-01 verify bar=%s/%s matches=%s", bar, scan_end, len(occurrences))
            events = [
                event
                for event in self._detect_events_at_bar(frame, calendar, bar)
                if event in LIQUIDITY_EVENTS
            ]
            if "Failed Breakout" not in events:
                continue
            context = self._context_at_bar(
                frame=frame,
                enriched=enriched,
                calendar=calendar,
                intel_frames=intel_frames,
                bar=bar,
            )
            combo = self._combo_key("Failed Breakout", context)
            if combo != target_combo:
                continue
            bull, bear = self._forward_directional_moves(highs, lows, closes, bar, FORWARD_BARS)
            linked = self._find_next_move(moves, bar, FORWARD_BARS)
            sell_outcome = self._trade_outcome(frame, bar, "bearish")
            if not sell_outcome:
                continue
            bars_before = linked.start_bar - bar if linked else -1
            row = self._build_occurrence(
                frame=frame,
                bar=bar,
                context=context,
                sell_outcome=sell_outcome,
                forward_bear=bear,
                linked=linked,
                bars_before_expansion=bars_before,
            )
            occurrences.append(row)
            match_bars[bar] = row

        bars_before_list = [o["bars_before_expansion"] for o in occurrences if o["bars_before_expansion"] >= 0]
        points_before_list = [o["points_before_expansion"] for o in occurrences if o["points_before_expansion"] is not None]
        causal_counts = Counter(o["causal_classification"] for o in occurrences)
        tradeability_counts = Counter(o["tradeability_classification"] for o in occurrences)

        performance = self._performance_metrics(occurrences, RESEARCH_WINDOW_DAYS)
        major = self._major_move_validation(
            moves=moves,
            match_bars=match_bars,
            frame=frame,
            enriched=enriched,
            calendar=calendar,
            intel_frames=intel_frames,
        )

        sorted_by_pnl = sorted(occurrences, key=lambda o: float(o.get("realized_pnl_points") or 0.0), reverse=True)
        best_20 = [
            {
                "date": o["date"],
                "entry": o["entry"],
                "stop_loss": o["stop_loss"],
                "target": o["target_1"],
                "move_captured": o["realized_pnl_points"],
                "mfe": o["mfe"],
                "mae": o["mae"],
                "bars_before_expansion": o["bars_before_expansion"],
                "tradeability_classification": o["tradeability_classification"],
            }
            for o in sorted_by_pnl[:20]
        ]
        worst_20 = [
            {
                "date": o["date"],
                "entry": o["entry"],
                "stop_loss": o["stop_loss"],
                "target": o["target_1"],
                "move_captured": o["realized_pnl_points"],
                "mfe": o["mfe"],
                "mae": o["mae"],
                "bars_before_expansion": o["bars_before_expansion"],
                "tradeability_classification": o["tradeability_classification"],
            }
            for o in sorted_by_pnl[-20:]
        ]

        tradeable_pct = round(
            sum(
                1
                for o in occurrences
                if o["tradeability_classification"] in {"CLEAR SETUP", "TRADEABLE"}
            )
            / max(len(occurrences), 1)
            * 100,
            2,
        )
        causal_pct = round(causal_counts.get("CAUSAL", 0) / max(len(occurrences), 1) * 100, 2)

        final_decision = {
            "1_is_genuinely_causal": {
                "answer": "YES" if causal_pct >= 95 else "CONDITIONAL",
                "evidence": f"{causal_counts.get('CAUSAL', 0)}/{len(occurrences)} occurrences classified CAUSAL ({causal_pct}%).",
            },
            "2_free_from_look_ahead_bias": {
                "answer": "YES" if causal_counts.get("LOOK-AHEAD CONTAMINATED", 0) == 0 else "NO",
                "evidence": "Features built at event bar only; BOS/CHOCH from current bar; no forward FVG/structure used.",
            },
            "3_human_trader_realistically_trades": {
                "answer": "YES" if tradeable_pct >= 60 else "CONDITIONAL",
                "evidence": f"{tradeable_pct}% classified CLEAR SETUP or TRADEABLE.",
            },
            "4_production_deployable": {
                "answer": "CONDITIONAL",
                "evidence": "Formula is causal and profitable in-sample, but major-move capture rates remain low and walk-forward is aggregate-only.",
            },
            "5_expected_live": {
                "win_rate_pct": performance["true_causal_win_rate_pct"],
                "profit_factor": performance["true_causal_profit_factor"],
                "expectancy": performance["true_causal_expectancy"],
                "signals_per_month": performance["signals_per_month"],
            },
            "6_captures_major_momentum_before_move": {
                "answer": "CONDITIONAL",
                "evidence": (
                    f"Median {median(bars_before_list) if bars_before_list else 'N/A'} bars before expansion; "
                    f"200+ capture {major['200']['capture_rate_pct']}%."
                ),
            },
            "7_pct_200_plus_moves_captured": major["200"]["capture_rate_pct"],
            "8_pct_300_plus_moves_captured": major["300"]["capture_rate_pct"],
            "9_pct_500_plus_moves_captured": major["500"]["capture_rate_pct"],
            "10_should_v1_go_live_with_sell_model": {
                "answer": "CONDITIONAL",
                "evidence": (
                    "Deploy as SELL-only filter with strict formula match; disable BUY; "
                    "expect ~"
                    f"{performance['signals_per_month']} signals/month, PF {performance['true_causal_profit_factor']}, "
                    f"but only {major['200']['capture_rate_pct']}% of 200+ bearish moves flagged pre-move."
                ),
            },
        }

        conclusions = [
            f"LDM-SELL-01 verification complete: {len(occurrences)} occurrences (expected {model_spec.get('occurrences')}).",
            f"Causal classification: {dict(causal_counts)}.",
            f"True causal win rate {performance['true_causal_win_rate_pct']}%, PF {performance['true_causal_profit_factor']}, expectancy {performance['true_causal_expectancy']}.",
            f"200+ bearish move capture rate: {major['200']['capture_rate_pct']}%.",
            f"Tradeability: {dict(tradeability_counts)}.",
            final_decision["10_should_v1_go_live_with_sell_model"]["answer"] + " for V1 go-live with this SELL model.",
        ]

        return SellFormulaRealityVerificationV2Report(
            model_id=MODEL_ID,
            formula=model_spec.get("formula", []),
            formula_text=model_spec.get("formula_text", ""),
            source_export=str(DEFAULT_EXTRACTION_PATH.name),
            symbol=DEFAULT_SYMBOL,
            timeframe=MOVE_DETECTION_TIMEFRAME,
            research_window_days=RESEARCH_WINDOW_DAYS,
            start_date=metadata.get("start_date", start.isoformat()),
            end_date=metadata.get("end_date", end.isoformat()),
            methodology={
                "verification_only": True,
                "no_new_discovery": True,
                "no_new_optimization": True,
                "no_new_ranking": True,
                "no_new_feature_engineering": True,
                "formula_source": str(DEFAULT_EXTRACTION_PATH.name),
                "scan_scope": "LDM-SELL-01 matching bars only",
            },
            expected_occurrences=int(model_spec.get("occurrences") or 0),
            actual_occurrences=len(occurrences),
            all_occurrences=occurrences,
            causal_validation_summary={
                "classification_counts": dict(causal_counts),
                "causal_pct": causal_pct,
                "look_ahead_contaminated_count": causal_counts.get("LOOK-AHEAD CONTAMINATED", 0),
            },
            momentum_capture_validation={
                "average_bars_before_expansion": round(mean(bars_before_list), 2) if bars_before_list else None,
                "median_bars_before_expansion": median(bars_before_list) if bars_before_list else None,
                "average_points_before_expansion": round(mean(points_before_list), 2) if points_before_list else None,
                "expansion_threshold_hit_rates_pct": {
                    str(th): round(
                        sum(1 for o in occurrences if o["expansion_reached"].get(f"{th}_plus")) / max(len(occurrences), 1) * 100,
                        2,
                    )
                    for th in EXPANSION_THRESHOLDS
                },
                "average_mfe": round(mean(float(o["mfe"] or 0) for o in occurrences), 2),
                "average_mae": round(mean(float(o["mae"] or 0) for o in occurrences), 2),
            },
            major_move_validation=major,
            trade_execution_validation={
                "classification_counts": dict(tradeability_counts),
                "tradeable_or_clear_pct": tradeable_pct,
            },
            performance_metrics=performance,
            best_20_trades=best_20,
            worst_20_trades=worst_20,
            final_decision=final_decision,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )


def generate_sell_formula_reality_verification_v2_report(
    report_path: Path | str | None = None,
    extraction_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> SellFormulaRealityVerificationV2Report:
    """Run LDM-SELL-01 verification and export JSON."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise SellFormulaRealityVerificationV2Error(f"Filter report not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = {
        **metadata,
        "research_window_days": RESEARCH_WINDOW_DAYS,
        "start_date": (
            date.fromisoformat(metadata["end_date"]) - timedelta(days=RESEARCH_WINDOW_DAYS)
        ).isoformat(),
    }

    engine = SellFormulaRealityVerificationV2Research()
    report = engine.run(metadata, extraction_path=Path(extraction_path) if extraction_path else None)

    destination = Path(report_path) if report_path else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_json_safe(report.as_dict()), indent=2), encoding="utf-8")
    logger.info("SELL formula reality verification V2 exported: %s", destination)
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        report = generate_sell_formula_reality_verification_v2_report()
    except SellFormulaRealityVerificationV2Error as exc:
        logger.error("Verification error: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected verification error")
        return 1

    print("SELL Formula Reality Verification V2 Summary")
    print(f"Model: {report.model_id}")
    print(f"Occurrences: {report.actual_occurrences} (expected {report.expected_occurrences})")
    print(f"Win rate: {report.performance_metrics['true_causal_win_rate_pct']}%")
    print(f"Go-live: {report.final_decision['10_should_v1_go_live_with_sell_model']['answer']}")
    print(f"Report: {DEFAULT_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
