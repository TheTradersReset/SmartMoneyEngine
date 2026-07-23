"""
SmartMoneyEngine V3 Implementation Validation research.

Implements the frozen V3 five-layer architecture and replays the last 30
trading days bar-by-bar. Research-only; no production modifications.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.context.institutional_liquidity_map_engine import InstitutionalLiquidityMapEngine
from src.context.market_intelligence_engine import MarketIntelligenceEngine
from src.research.filter_research_engine import FilterContextBuilder, FilterResearchEngine, _json_safe
from src.research.institutional_expansion_trigger_discovery_research import PRE_EXPANSION_LOOKBACK
from src.research.liquidity_move_reconstruction_research import (
    FORWARD_BARS,
    LiquidityMoveReconstructionResearch,
    _CheapMoveCandidate,
)
from src.research.nifty50_liquidity_direction_decision_matrix_research import (
    MOVE_DETECTION_TIMEFRAME,
    Nifty50LiquidityDirectionDecisionMatrixResearch,
)
from src.research.nifty50_trap_to_momentum_validation_research import DEFAULT_SYMBOL
from src.research.trade_construction_validation_research import TradeConstructionValidationResearch

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_FILTER_REPORT_PATH = RESEARCH_DIR / "filter_research_report.json"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "smartmoneyengine_v3_implementation_validation.json"

TRADING_DAYS_REPLAY = 30
PIPELINE_TIMEFRAMES = ("5M", "15M", "1H")
CONTEXT_TIMEFRAMES = ("15M", "1H", "1D")
MAJOR_THRESHOLDS = (200, 300, 500)

LAYER1_EVENTS = frozenset(
    {
        "Gap Reversal",
        "Gap Continuation",
        "Liquidity Grab",
        "Failed Breakout",
        "Failed Breakdown",
    },
)

SELL_CONFIRMATION_CANDLES = frozenset(
    {
        "Shooting Star",
        "Bearish Engulfing",
        "Evening Star",
        "Marubozu",
    },
)

ALLOWED_VOLUME_BUCKETS = frozenset({"Normal", "Expanded"})


class SmartMoneyEngineV3ImplementationValidationError(Exception):
    """Raised when V3 implementation validation fails."""


@dataclass
class SmartMoneyEngineV3ImplementationValidationReport:
    """V3 implementation validation output."""

    engine_version: str
    symbol: str
    timeframe: str
    trading_days_replayed: int
    replay_start_date: str
    replay_end_date: str
    architecture: dict[str, Any]
    replay_rules: dict[str, Any]
    overall_statistics: dict[str, Any]
    major_move_capture: dict[str, Any]
    emitted_signals: list[dict[str, Any]]
    layer_rejection_summary: dict[str, int]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SmartMoneyEngineV3Engine(Nifty50LiquidityDirectionDecisionMatrixResearch):
    """Frozen V3 five-layer stack (research implementation)."""

    MODEL_ID = "LDM-SELL-01"

    def __init__(self) -> None:
        super().__init__()
        self.intelligence = MarketIntelligenceEngine(symbol=DEFAULT_SYMBOL)

    def _layer1_early_warning(self, events: tuple[str, ...]) -> dict[str, Any]:
        matched = [event for event in events if event in LAYER1_EVENTS]
        return {
            "active": bool(matched),
            "events_detected": matched,
            "primary_event": matched[0] if matched else None,
            "failed_breakout_present": "Failed Breakout" in matched,
        }

    def _layer2_directional_filter(self, context: dict[str, str]) -> dict[str, Any]:
        aligned = (
            context.get("htf_trend") == "Bearish"
            and context.get("vwap") == "Below"
            and context.get("ema_structure") == "Bear Stack"
        )
        return {
            "direction": "SELL" if aligned else "NO_TRADE",
            "htf_trend": context.get("htf_trend"),
            "vwap_state": context.get("vwap"),
            "ema_structure": context.get("ema_structure"),
            "aligned": aligned,
        }

    def _layer3_confirmation(self, context: dict[str, str]) -> dict[str, Any]:
        candle = context.get("confirmation_candle", "None")
        volume = context.get("volume", "Normal")
        candle_ok = candle == "None" or candle in SELL_CONFIRMATION_CANDLES
        volume_ok = volume in ALLOWED_VOLUME_BUCKETS
        return {
            "confirmation_candle": candle,
            "volume_bucket": volume,
            "confirmed": candle_ok and volume_ok,
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
        if layer2.get("vwap_state") != "Below":
            reasons.append("VWAP_MISMATCH")
        if layer2.get("ema_structure") == "Bull Stack":
            reasons.append("EMA_MISMATCH")
        if not layer2.get("aligned"):
            reasons.append("DIRECTION_NOT_ALIGNED")
        if not layer3.get("confirmed"):
            reasons.append("CONFIRMATION_FAILED")
        if context.get("location") == "Mid Range":
            reasons.append("LOCATION_MID_RANGE")
        if bar in emitted_bars:
            reasons.append("DUPLICATE_BAR")
        return {"pass": not reasons, "reason_codes": reasons}

    def _layer4_execution(
        self,
        frame: pd.DataFrame,
        bar: int,
        *,
        layer1: dict[str, Any],
        layer2: dict[str, Any],
        layer3: dict[str, Any],
        context: dict[str, str],
    ) -> dict[str, Any] | None:
        """
        Realtime execution plan for SELL.

        Uses structural stop/risk only. Forward ``_trade_outcome`` is optional
        enrichment when future bars already exist (research frames) and must
        never block emission on the live/replay latest bar.
        """
        from src.signals.signal_outcome import build_realtime_layer4_plan

        entry = round(float(frame.iloc[bar]["Close"]), 2)
        stop, risk = self.trade_engine._structural_stop(frame, bar, entry, "bearish")
        if risk <= 0:
            return None
        target_liq = self.trade_engine._opposite_liquidity_target(
            frame, bar, entry, "bearish", risk
        )
        # Optional: attach forward stats when available (never required).
        forward = self._trade_outcome(frame, bar, "bearish") or None
        return build_realtime_layer4_plan(
            model_id=self.MODEL_ID,
            direction="SELL",
            entry=entry,
            stop_loss=stop,
            risk_points=risk,
            liquidity_target=target_liq,
            signal_reason_stack={
                "layer1": layer1["events_detected"],
                "layer2": {
                    "htf_trend": layer2["htf_trend"],
                    "vwap": layer2["vwap_state"],
                    "ema_structure": layer2["ema_structure"],
                },
                "layer3": {
                    "confirmation_candle": layer3["confirmation_candle"],
                    "volume": layer3["volume_bucket"],
                },
                "location": context.get("location"),
            },
            forward_outcome=forward if forward else None,
        )

    def evaluate_bar(
        self,
        *,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        calendar: pd.DataFrame,
        intel_frames: dict[str, pd.DataFrame],
        bar: int,
        emitted_bars: set[int],
    ) -> dict[str, Any]:
        events = self._detect_events_at_bar(frame, calendar, bar)
        context = self._context_at_bar(
            frame=frame,
            enriched=enriched,
            calendar=calendar,
            intel_frames=intel_frames,
            bar=bar,
        )
        layer1 = self._layer1_early_warning(events)
        layer2 = self._layer2_directional_filter(context)
        layer3 = self._layer3_confirmation(context)
        layer5 = self._layer5_no_trade_filters(
            layer1=layer1,
            layer2=layer2,
            layer3=layer3,
            context=context,
            bar=bar,
            emitted_bars=emitted_bars,
        )
        timestamp = str(frame.iloc[bar].get("Date", ""))
        result: dict[str, Any] = {
            "timestamp": timestamp,
            "bar": bar,
            "verdict": "NO_TRADE",
            "layer1": layer1,
            "layer2": layer2,
            "layer3": layer3,
            "layer5": layer5,
            "context": context,
        }
        if layer5["pass"]:
            execution = self._layer4_execution(
                frame,
                bar,
                layer1=layer1,
                layer2=layer2,
                layer3=layer3,
                context=context,
            )
            if execution:
                result["verdict"] = "SELL"
                result["layer4"] = execution
        return result


class SmartMoneyEngineV3ImplementationValidationResearch:
    """Replay V3 over the last 30 trading days."""

    def __init__(self) -> None:
        self.engine = SmartMoneyEngineV3Engine()
        self.move_engine = LiquidityMoveReconstructionResearch(
            symbol=DEFAULT_SYMBOL,
            research_days=120,
            timeframes=(MOVE_DETECTION_TIMEFRAME,),
        )

    @staticmethod
    def _profit_factor(pnls: list[float]) -> float | None:
        wins = sum(p for p in pnls if p > 0)
        losses = abs(sum(p for p in pnls if p < 0))
        if losses == 0:
            return None if wins == 0 else round(wins, 2)
        return round(wins / losses, 2)

    @staticmethod
    def _last_n_trading_day_set(frame: pd.DataFrame, n: int) -> set[date]:
        dates = pd.to_datetime(frame["Date"]).dt.date
        unique = sorted(set(dates))
        return set(unique[-n:])

    def _major_move_capture(
        self,
        moves: list[_CheapMoveCandidate],
        signals: list[dict[str, Any]],
        replay_dates: set[date],
        frame: pd.DataFrame,
    ) -> dict[str, Any]:
        signal_by_bar = {signal["bar"]: signal for signal in signals}
        results: dict[str, Any] = {}
        for threshold in MAJOR_THRESHOLDS:
            bearish = [
                move
                for move in moves
                if move.direction == "bearish"
                and move.magnitude >= threshold
                and pd.to_datetime(frame.iloc[move.start_bar]["Date"]).date() in replay_dates
            ]
            captured = 0
            for move in bearish:
                pre_start = max(PRE_EXPANSION_LOOKBACK, move.start_bar - PRE_EXPANSION_LOOKBACK)
                for bar in range(pre_start, move.start_bar + 1):
                    if bar in signal_by_bar:
                        captured += 1
                        break
            total = len(bearish)
            results[str(threshold)] = {
                "total_bearish_moves": total,
                "signals_before_move": captured,
                "capture_rate_pct": round(captured / max(total, 1) * 100, 2),
            }
        return results

    def run(self, metadata: dict[str, Any]) -> SmartMoneyEngineV3ImplementationValidationReport:
        started = time.perf_counter()
        end = date.fromisoformat(metadata["end_date"])
        start = end - timedelta(days=120)

        filter_engine = FilterResearchEngine(
            symbol=DEFAULT_SYMBOL,
            research_days=120,
            timeframes=PIPELINE_TIMEFRAMES,
        )
        path = filter_engine._ensure_pipeline(MOVE_DETECTION_TIMEFRAME, start, end)
        frame = pd.read_csv(path).reset_index(drop=True)
        replay_dates = self._last_n_trading_day_set(frame, TRADING_DAYS_REPLAY)
        bar_dates = pd.to_datetime(frame["Date"]).dt.date
        replay_bars = [index for index, day in enumerate(bar_dates) if day in replay_dates]

        enriched = self.engine.context_builder.enrich(frame)
        calendar = InstitutionalLiquidityMapEngine(symbol=DEFAULT_SYMBOL)._attach_calendar_levels(frame)
        intel_frames: dict[str, pd.DataFrame] = {"5M": self.engine.intelligence.enrich(frame)}
        for tf in ("15M", "1H"):
            tf_path = filter_engine._ensure_pipeline(tf, start, end)
            tf_frame = pd.read_csv(tf_path).reset_index(drop=True)
            intel_frames[tf] = self.engine.intelligence.enrich(tf_frame)
        intel_frames["1D"] = self.engine.intelligence.enrich(
            self.engine._resample_daily(intel_frames["1H"]),
        )

        highs = frame["High"].astype(float)
        lows = frame["Low"].astype(float)
        moves = self.move_engine._dedupe_cheap_moves(
            self.move_engine._detect_moves_cheap(highs, lows, 50),
        )

        emitted_signals: list[dict[str, Any]] = []
        emitted_bars: set[int] = set()
        rejection_counter: dict[str, int] = {}

        for bar in replay_bars:
            if bar < PRE_EXPANSION_LOOKBACK or bar >= len(frame) - FORWARD_BARS:
                continue
            evaluation = self.engine.evaluate_bar(
                frame=frame,
                enriched=enriched,
                calendar=calendar,
                intel_frames=intel_frames,
                bar=bar,
                emitted_bars=emitted_bars,
            )
            if evaluation["verdict"] == "SELL":
                layer4 = evaluation["layer4"]
                forward = layer4.pop("forward_outcome", {})
                signal = {
                    "timestamp": evaluation["timestamp"],
                    "bar": bar,
                    "symbol": DEFAULT_SYMBOL,
                    "timeframe": MOVE_DETECTION_TIMEFRAME,
                    "direction": "SELL",
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
                emitted_signals.append(signal)
                emitted_bars.add(bar)
            else:
                for reason in evaluation["layer5"]["reason_codes"]:
                    rejection_counter[reason] = rejection_counter.get(reason, 0) + 1

        pnls = [float(s.get("realized_pnl_points") or 0.0) for s in emitted_signals]
        wins = [p for p in pnls if p > 0]
        trading_weeks = max(TRADING_DAYS_REPLAY / 5.0, 1.0)
        trading_months = max(TRADING_DAYS_REPLAY / 22.0, 1.0)
        major_capture = self._major_move_capture(moves, emitted_signals, replay_dates, frame)

        replay_start = min(replay_dates).isoformat() if replay_dates else ""
        replay_end = max(replay_dates).isoformat() if replay_dates else ""

        overall = {
            "bars_replayed": len(replay_bars),
            "signals_emitted": len(emitted_signals),
            "signals_per_week": round(len(emitted_signals) / trading_weeks, 2),
            "signals_per_month": round(len(emitted_signals) / trading_months, 2),
            "win_rate_pct": round(sum(1 for s in emitted_signals if s.get("win")) / max(len(emitted_signals), 1) * 100, 2),
            "profit_factor": self._profit_factor(pnls),
            "expectancy": round(mean(pnls), 2) if pnls else 0.0,
            "hit_1r_rate_pct": round(sum(1 for s in emitted_signals if s.get("hit_1r")) / max(len(emitted_signals), 1) * 100, 2),
            "average_mfe": round(mean(float(s.get("mfe_points") or 0) for s in emitted_signals), 2) if emitted_signals else 0.0,
            "average_mae": round(mean(float(s.get("mae_points") or 0) for s in emitted_signals), 2) if emitted_signals else 0.0,
        }

        architecture = {
            "layer1_early_warning": list(LAYER1_EVENTS),
            "layer2_directional_filter": ["HTF Trend", "VWAP", "EMA Structure"],
            "layer3_confirmation": ["Confirmation Candle", "Volume Expansion"],
            "layer4_execution": ["Entry", "SL", "T1", "T2", "T3"],
            "layer5_no_trade_filters": [
                "BUY disabled",
                "Requires Failed Breakout",
                "HTF Bearish + VWAP Below + EMA Bear Stack",
                "Confirmation pass",
                "Reject Mid Range location",
                "Duplicate bar guard",
            ],
            "production_formula": "Failed Breakout + HTF Bearish + VWAP Below + EMA Bear Stack + SELL confirmation",
        }

        conclusions = [
            f"V3 implementation replay complete over {TRADING_DAYS_REPLAY} trading days ({replay_start} to {replay_end}).",
            f"Signals emitted: {len(emitted_signals)}; win rate {overall['win_rate_pct']}%; PF {overall['profit_factor']}.",
            f"200+ bearish move capture: {major_capture.get('200', {}).get('capture_rate_pct', 0)}%.",
            "Frozen five-layer architecture enforced with SELL-only execution path.",
        ]

        return SmartMoneyEngineV3ImplementationValidationReport(
            engine_version="SmartMoneyEngine V3",
            symbol=DEFAULT_SYMBOL,
            timeframe=MOVE_DETECTION_TIMEFRAME,
            trading_days_replayed=TRADING_DAYS_REPLAY,
            replay_start_date=replay_start,
            replay_end_date=replay_end,
            architecture=architecture,
            replay_rules={
                "candle_by_candle": True,
                "no_look_ahead": True,
                "no_discovery": True,
                "no_optimization": True,
                "no_pattern_ranking": True,
            },
            overall_statistics=overall,
            major_move_capture=major_capture,
            emitted_signals=emitted_signals,
            layer_rejection_summary=rejection_counter,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )


def generate_smartmoneyengine_v3_implementation_validation_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> SmartMoneyEngineV3ImplementationValidationReport:
    """Run V3 implementation validation and export JSON."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise SmartMoneyEngineV3ImplementationValidationError(
            f"Filter research report not found: {metadata_path}",
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = {**metadata, "end_date": metadata["end_date"]}

    report = SmartMoneyEngineV3ImplementationValidationResearch().run(metadata)
    destination = Path(report_path) if report_path else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_json_safe(report.as_dict()), indent=2), encoding="utf-8")
    logger.info("V3 implementation validation exported: %s", destination)
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        report = generate_smartmoneyengine_v3_implementation_validation_report()
    except SmartMoneyEngineV3ImplementationValidationError as exc:
        logger.error("V3 validation error: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected V3 validation error")
        return 1

    print("SmartMoneyEngine V3 Implementation Validation Summary")
    print(f"Signals: {report.overall_statistics['signals_emitted']}")
    print(f"Win rate: {report.overall_statistics['win_rate_pct']}%")
    print(f"PF: {report.overall_statistics['profit_factor']}")
    print(f"Report: {DEFAULT_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
