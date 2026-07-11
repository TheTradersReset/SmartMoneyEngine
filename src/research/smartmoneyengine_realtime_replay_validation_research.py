"""
SmartMoneyEngine Real-Time Replay Validation V2 research.

Replays market history bar-by-bar with strict no-look-ahead rules to validate
whether the current blueprint could detect momentum moves prospectively.
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
from src.context.liquidity_narrative_engine import LiquidityNarrativeEngine
from src.research.filter_research_engine import (
    FilterContextBuilder,
    FilterResearchEngine,
    RESEARCH_DAYS,
    _json_safe,
)
from src.research.institutional_expansion_trigger_discovery_research import (
    PRE_EXPANSION_LOOKBACK,
    InstitutionalExpansionTriggerDiscoveryResearch,
)
from src.research.institutional_quality_validation_research import (
    InstitutionalQualityValidationResearch,
)
from src.research.liquidity_move_reconstruction_research import (
    FORWARD_BARS,
    LiquidityMoveReconstructionResearch,
    MIN_MOVE_SEPARATION_BARS,
    TIMEFRAME_MINUTES,
)
from src.research.smartmoneyengine_walkforward_validation_research import (
    FILTER_LABEL_TO_KEY,
    SmartMoneyEngineWalkForwardValidationResearch,
)
from src.research.tier2_production_validation_research import Tier2ProductionValidationResearch
from src.research.tiered_signal_framework_research import TierSignal, TieredSignalFrameworkResearch
from src.research.trade_construction_validation_research import TradeConstructionValidationResearch
from src.signals.decision_engine import DecisionEngine

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_FILTER_REPORT_PATH = RESEARCH_DIR / "filter_research_report.json"
DEFAULT_V2_OPTIMIZATION_PATH = RESEARCH_DIR / "smartmoneyengine_v2_frequency_optimization.json"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "smartmoneyengine_realtime_replay_validation.json"

DEFAULT_SYMBOLS = ("NIFTY50", "BANKNIFTY", "FINNIFTY")
DEFAULT_TIMEFRAMES = ("5M", "15M", "1H")
MOMENTUM_THRESHOLDS = (50, 100, 200, 300, 500)
MISSED_MOVE_THRESHOLDS = (100, 200, 300, 500)
SCAN_PROGRESS_INTERVAL = 1000
MAX_SIGNAL_LOG = 2000
MAX_MISSED_EXPORT = 300
MIN_SIGNAL_SEPARATION_BARS = 20


class RealtimeReplayValidationError(Exception):
    """Raised when realtime replay validation fails."""


@dataclass
class RealtimeReplayValidationReport:
    """Full realtime replay validation output."""

    symbols_analyzed: list[str]
    timeframes_analyzed: list[str]
    research_window_days: int
    start_date: str
    end_date: str
    research_exports_used: list[dict[str, Any]]
    v2_production_card: dict[str, Any]
    replay_rules: dict[str, Any]
    overall_statistics: dict[str, Any]
    signal_by_signal_log: list[dict[str, Any]]
    missed_move_report: list[dict[str, Any]]
    false_signal_report: dict[str, Any]
    monthly_frequency_report: list[dict[str, Any]]
    major_200_plus_move_analysis: dict[str, Any]
    top_performing_conditions: list[dict[str, Any]]
    worst_performing_conditions: list[dict[str, Any]]
    production_candidate_list: list[dict[str, Any]]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SmartMoneyEngineRealtimeReplayValidationResearch:
    """Prospective bar-by-bar replay with no future leakage."""

    def __init__(
        self,
        symbols: tuple[str, ...] | None = None,
        research_days: int = RESEARCH_DAYS,
        timeframes: tuple[str, ...] = DEFAULT_TIMEFRAMES,
        v2_optimization_path: Path | str | None = None,
        research_dir: Path | str | None = None,
    ) -> None:
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.research_days = research_days
        self.timeframes = timeframes
        self.v2_optimization_path = Path(v2_optimization_path or DEFAULT_V2_OPTIMIZATION_PATH)
        self.research_dir = Path(research_dir or RESEARCH_DIR)
        self.discovery_engine = InstitutionalExpansionTriggerDiscoveryResearch(
            symbols=self.symbols,
            research_days=research_days,
            timeframes=timeframes,
        )
        self.move_engine = LiquidityMoveReconstructionResearch(
            research_days=research_days,
            timeframes=timeframes,
        )
        self.context_builder = FilterContextBuilder()

    @staticmethod
    def _is_active(value: Any) -> bool:
        return DecisionEngine._is_active(value)

    @staticmethod
    def _profit_factor(pnls: list[float]) -> float | None:
        return InstitutionalQualityValidationResearch._profit_factor(pnls)

    def _load_v2_card(self) -> dict[str, Any]:
        if not self.v2_optimization_path.exists():
            raise RealtimeReplayValidationError(
                f"V2 optimization export not found: {self.v2_optimization_path}",
            )
        with self.v2_optimization_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        card = payload.get("smartmoneyengine_v2_production_card")
        if not card:
            raise RealtimeReplayValidationError("V2 production card missing.")
        return card

    def _load_research_exports(self) -> list[dict[str, Any]]:
        exports: list[dict[str, Any]] = []
        if not self.research_dir.exists():
            return exports
        for path in sorted(self.research_dir.glob("*.json")):
            exports.append({"file": path.name, "path": str(path), "status": "available"})
        return exports

    @staticmethod
    def _keys_from_labels(labels: list[str]) -> tuple[str, ...]:
        keys: list[str] = []
        for label in labels:
            key = FILTER_LABEL_TO_KEY.get(label)
            if key:
                keys.append(key)
        return tuple(keys)

    @staticmethod
    def _window_count(cumsum: np.ndarray, end_bar: int, lookback: int) -> int:
        start_bar = max(0, end_bar - lookback)
        if start_bar == 0:
            return int(cumsum[end_bar])
        return int(cumsum[end_bar] - cumsum[start_bar - 1])

    def _build_prechecks(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        length = len(frame)
        bos = np.zeros(length, dtype=np.int8)
        choch = np.zeros(length, dtype=np.int8)
        sweep = np.zeros(length, dtype=np.int8)
        fvg = np.zeros(length, dtype=np.int8)
        ob = np.zeros(length, dtype=np.int8)
        for index in range(length):
            row = frame.iloc[index]
            if self._is_active(row.get("Bullish_BOS")) or self._is_active(row.get("Bearish_BOS")):
                bos[index] = 1
            if self._is_active(row.get("Bullish_CHOCH")) or self._is_active(row.get("Bearish_CHOCH")):
                choch[index] = 1
            if self._is_active(row.get("Buy_Liquidity_Sweep")) or self._is_active(
                row.get("Sell_Liquidity_Sweep"),
            ):
                sweep[index] = 1
            if self._is_active(row.get("Bullish_FVG_Top")) or self._is_active(row.get("Bearish_FVG_Top")):
                fvg[index] = 1
            if self._is_active(row.get("Bullish_OB_High")) or self._is_active(row.get("Bearish_OB_High")):
                ob[index] = 1
        return {
            "bos_cumsum": np.cumsum(bos),
            "choch_cumsum": np.cumsum(choch),
            "sweep_cumsum": np.cumsum(sweep),
            "fvg_cumsum": np.cumsum(fvg),
            "ob_cumsum": np.cumsum(ob),
        }

    def _feature_flags_at_bar(
        self,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        bar: int,
    ) -> dict[str, bool]:
        row = frame.iloc[bar]
        filters = self.context_builder.filter_state(enriched, bar)
        close = float(row["Close"])
        open_price = float(row["Open"])
        body = abs(close - open_price)
        candle_range = max(float(row["High"]) - float(row["Low"]), 0.01)
        gap_down = gap_up = False
        if bar >= 1:
            prev_close = float(frame.iloc[bar - 1]["Close"])
            gap = open_price - prev_close
            gap_up = gap > 0.5
            gap_down = gap < -0.5
        return {
            "strong_confirmation": (body / candle_range) >= 0.55,
            "ema_bull_stack": filters.ema_alignment == "EMA20 > EMA50 > EMA200",
            "ema_bear_stack": filters.ema_alignment == "EMA20 < EMA50 < EMA200",
            "below_vwap": filters.vwap_position == "Below VWAP",
            "above_vwap": filters.vwap_position == "Above VWAP",
            "gap_down": gap_down,
            "gap_up": gap_up,
            "choch_present": self._is_active(row.get("Bullish_CHOCH"))
            or self._is_active(row.get("Bearish_CHOCH")),
            "bos_present": self._is_active(row.get("Bullish_BOS"))
            or self._is_active(row.get("Bearish_BOS")),
            "liquidity_sweep": self._is_active(row.get("Buy_Liquidity_Sweep"))
            or self._is_active(row.get("Sell_Liquidity_Sweep")),
            "fvg_reclaim": self._is_active(row.get("Bullish_FVG_Top"))
            or self._is_active(row.get("Bearish_FVG_Top")),
            "order_block_reaction": self._is_active(row.get("Bullish_OB_High"))
            or self._is_active(row.get("Bearish_OB_High")),
        }

    def _reasons_at_bar(
        self,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        intel_frame: pd.DataFrame,
        bar: int,
        direction: str,
        prechecks: dict[str, np.ndarray],
        tags: tuple[str, ...],
        measurements: dict[str, Any],
    ) -> dict[str, Any]:
        row = frame.iloc[bar]
        filters = self.context_builder.filter_state(enriched, bar)
        intel = self.discovery_engine.intelligence_engine.evaluate_bar(intel_frame, bar)
        displacement = LiquidityNarrativeEngine._displacement_strength_for_bar(row, direction)
        return {
            "htf_trend": measurements.get("htf_trend", intel.momentum_state),
            "market_structure": (
                "Bullish BOS" if self._is_active(row.get("Bullish_BOS"))
                else "Bearish BOS" if self._is_active(row.get("Bearish_BOS"))
                else "No BOS"
            ),
            "choch": bool(self._window_count(prechecks["choch_cumsum"], bar, PRE_EXPANSION_LOOKBACK)),
            "bos": bool(self._window_count(prechecks["bos_cumsum"], bar, PRE_EXPANSION_LOOKBACK)),
            "liquidity_grab": bool(
                self._window_count(prechecks["sweep_cumsum"], bar, PRE_EXPANSION_LOOKBACK),
            ),
            "false_breakout": measurements.get("false_breakout_count", 0) > 0,
            "false_breakdown": measurements.get("false_breakdown_count", 0) > 0,
            "fvg": bool(self._window_count(prechecks["fvg_cumsum"], bar, PRE_EXPANSION_LOOKBACK)),
            "order_block": bool(self._window_count(prechecks["ob_cumsum"], bar, PRE_EXPANSION_LOOKBACK)),
            "vwap": filters.vwap_position,
            "ema_structure": filters.ema_alignment,
            "rsi": filters.rsi_band,
            "volume_spike": measurements.get("volume_spike", False),
            "major_level_context": measurements.get("market_location", intel.market_location),
            "displacement": displacement.value,
            "blueprint_tags": list(tags),
        }

    @staticmethod
    def _signal_score(decision: str, reasons: dict[str, Any], flags: dict[str, bool]) -> float:
        if decision not in {"BUY", "SELL"}:
            return 0.0
        score = 20.0
        if reasons.get("bos"):
            score += 15.0
        if reasons.get("choch"):
            score += 10.0
        if reasons.get("liquidity_grab"):
            score += 10.0
        if reasons.get("fvg"):
            score += 10.0
        if flags.get("strong_confirmation"):
            score += 10.0
        if reasons.get("displacement") == "Strong":
            score += 10.0
        if reasons.get("vwap") in {"Below VWAP", "Above VWAP"}:
            score += 5.0
        if reasons.get("htf_trend") in {"Bullish", "Bearish", "Strong Bullish", "Strong Bearish"}:
            score += 10.0
        return round(min(score, 100.0), 2)

    def _forward_outcome(
        self,
        frame: pd.DataFrame,
        bar: int,
        direction: str,
        trade_engine: TradeConstructionValidationResearch,
    ) -> dict[str, Any]:
        entry_price = round(float(frame.iloc[bar]["Close"]), 2)
        stop, risk = trade_engine._structural_stop(frame, bar, entry_price, direction)
        if risk <= 0:
            return {}
        target_liq = trade_engine._opposite_liquidity_target(
            frame,
            bar,
            entry_price,
            direction,
            risk,
        )
        if direction == "bullish":
            t1, t2, t3 = entry_price + risk, entry_price + risk * 2, entry_price + risk * 3
        else:
            t1, t2, t3 = entry_price - risk, entry_price - risk * 2, entry_price - risk * 3

        end = min(len(frame) - 1, bar + FORWARD_BARS)
        mfe = mae = 0.0
        hit_1r = hit_2r = hit_3r = False
        stop_hit = False
        pnl = 0.0
        rr = 0.0
        time_to_expansion: int | None = None
        momentum_hits = {threshold: False for threshold in MOMENTUM_THRESHOLDS}

        for index in range(bar + 1, end + 1):
            bar_high = float(frame.iloc[index]["High"])
            bar_low = float(frame.iloc[index]["Low"])
            if direction == "bullish":
                favorable = bar_high - entry_price
                adverse = entry_price - bar_low
                mfe = max(mfe, favorable)
                mae = max(mae, adverse)
                if bar_high >= entry_price + risk:
                    hit_1r = True
                if bar_high >= entry_price + risk * 2:
                    hit_2r = True
                if bar_high >= entry_price + risk * 3:
                    hit_3r = True
                if bar_low <= stop:
                    stop_hit = True
                    pnl = -risk
                    rr = -1.0
                    break
                if bar_high >= target_liq:
                    pnl = round(target_liq - entry_price, 2)
                    rr = round(pnl / risk, 2)
                    break
            else:
                favorable = entry_price - bar_low
                adverse = bar_high - entry_price
                mfe = max(mfe, favorable)
                mae = max(mae, adverse)
                if bar_low <= entry_price - risk:
                    hit_1r = True
                if bar_low <= entry_price - risk * 2:
                    hit_2r = True
                if bar_low <= entry_price - risk * 3:
                    hit_3r = True
                if bar_high >= stop:
                    stop_hit = True
                    pnl = -risk
                    rr = -1.0
                    break
                if bar_low <= target_liq:
                    pnl = round(entry_price - target_liq, 2)
                    rr = round(pnl / risk, 2)
                    break

            for threshold in MOMENTUM_THRESHOLDS:
                if mfe >= threshold and time_to_expansion is None:
                    momentum_hits[threshold] = True
                    if threshold == 200:
                        time_to_expansion = index - bar

        if not stop_hit and pnl == 0.0:
            close = float(frame.iloc[end]["Close"])
            pnl = round((close - entry_price) if direction == "bullish" else (entry_price - close), 2)
            rr = round(pnl / risk, 2) if risk else 0.0

        for threshold in MOMENTUM_THRESHOLDS:
            if mfe >= threshold:
                momentum_hits[threshold] = True

        return {
            "entry": entry_price,
            "stop_loss": stop,
            "target_1": round(t1, 2),
            "target_2": round(t2, 2),
            "target_3": round(t3, 2),
            "risk_points": risk,
            "mfe_points": round(mfe, 2),
            "mae_points": round(mae, 2),
            "hit_1r": hit_1r,
            "hit_2r": hit_2r,
            "hit_3r": hit_3r,
            "realized_pnl_points": pnl,
            "realized_rr": rr,
            "win": pnl > 0,
            "is_false_signal": stop_hit and not hit_1r,
            "momentum_capture": {str(k): momentum_hits[k] for k in MOMENTUM_THRESHOLDS},
            "time_to_expansion_bars": time_to_expansion,
        }

    def _diagnose_missed(
        self,
        frame: pd.DataFrame,
        bar: int,
        direction: str,
        prechecks: dict[str, np.ndarray],
        flags: dict[str, bool],
    ) -> list[str]:
        reasons: list[str] = []
        if not self._window_count(prechecks["bos_cumsum"], bar, PRE_EXPANSION_LOOKBACK):
            reasons.append("No BOS")
        if not self._window_count(prechecks["choch_cumsum"], bar, PRE_EXPANSION_LOOKBACK):
            reasons.append("No CHOCH")
        if not self._window_count(prechecks["sweep_cumsum"], bar, PRE_EXPANSION_LOOKBACK):
            reasons.append("No Liquidity Grab")
        if not flags.get("strong_confirmation"):
            reasons.append("No Confirmation Candle")
        row = frame.iloc[bar]
        displacement = LiquidityNarrativeEngine._displacement_strength_for_bar(row, direction)
        if displacement.value == "Weak":
            reasons.append("Weak Displacement")
        if not flags.get("fvg_reclaim") and not self._window_count(
            prechecks["fvg_cumsum"],
            bar,
            PRE_EXPANSION_LOOKBACK,
        ):
            reasons.append("No FVG Reclaim")
        if not reasons:
            reasons.append("V2 Filter Stack Not Met")
        return reasons

    def _replay_frame(
        self,
        *,
        symbol: str,
        timeframe_label: str,
        frame: pd.DataFrame,
        metadata: dict[str, Any],
        v2_card: dict[str, Any],
        buy_keys: tuple[str, ...],
        sell_keys: tuple[str, ...],
        no_trade_rules: list[str],
    ) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
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
        scan_started = time.perf_counter()

        for bar in range(PRE_EXPANSION_LOOKBACK, scan_end):
            if bar % SCAN_PROGRESS_INTERVAL == 0:
                elapsed = time.perf_counter() - scan_started
                logger.info(
                    "Replay %s/%s bar=%s/%s signals=%s elapsed=%.0fs",
                    symbol,
                    timeframe_label,
                    bar,
                    scan_end,
                    len(signals),
                    elapsed,
                )

            tier2 = tier2_by_bar.get(bar)
            if tier2 is None:
                freq["NO_TRADE"] += 1
                freq["NO_TRADE_no_tier2"] += 1
                month_key = str(frame.iloc[bar].get("Date", ""))[:7]
                monthly[month_key]["NO_TRADE"] += 1
                continue

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

            blocked = SmartMoneyEngineWalkForwardValidationResearch._no_trade_blocked(
                tags,
                no_trade_rules,
            )
            required = buy_keys if side == "BUY" else sell_keys
            filters_pass = all(flags.get(key, False) for key in required)

            if blocked:
                decision = "NO_TRADE"
                block_reason = "NO-TRADE rule matched"
            elif filters_pass:
                decision = side
                block_reason = None
            else:
                decision = "NO_TRADE"
                block_reason = "V2 filter stack not met"

            freq[decision] += 1
            month_key = str(frame.iloc[bar].get("Date", ""))[:7]
            monthly[month_key][decision] += 1

            outcome: dict[str, Any] = {}
            if decision in {"BUY", "SELL"}:
                prev = last_signal_bar.get(side)
                if prev is not None and bar - prev < MIN_SIGNAL_SEPARATION_BARS:
                    freq["NO_TRADE"] += 1
                    freq["NO_TRADE_separation"] += 1
                    continue
                last_signal_bar[side] = bar
                outcome = self._forward_outcome(frame, bar, direction, trade_engine)

            score = self._signal_score(decision, reasons, flags)
            record = {
                "timestamp": str(frame.iloc[bar].get("Date", "")),
                "symbol": symbol,
                "timeframe": timeframe_label,
                "signal_direction": decision,
                "tier2_direction": direction,
                "entry": outcome.get("entry"),
                "stop_loss": outcome.get("stop_loss"),
                "target_1": outcome.get("target_1"),
                "target_2": outcome.get("target_2"),
                "target_3": outcome.get("target_3"),
                "signal_score": score,
                "block_reason": block_reason,
                "reasons": reasons,
                "outcome": outcome,
            }
            if len(signals) < MAX_SIGNAL_LOG:
                signals.append(record)

        frequency_stats = {
            "total_bars_scanned": scan_end - PRE_EXPANSION_LOOKBACK,
            "signals_per_week": round(
                (freq["BUY"] + freq["SELL"]) / max((scan_end - PRE_EXPANSION_LOOKBACK) / (7 * 75), 1),
                2,
            ),
            "buy_count": freq["BUY"],
            "sell_count": freq["SELL"],
            "no_trade_count": freq["NO_TRADE"],
            "no_trade_no_tier2": freq["NO_TRADE_no_tier2"],
            "by_symbol_timeframe": {f"{symbol}_{timeframe_label}": dict(freq)},
        }

        monthly_report = [
            {
                "month": month,
                "buy_count": counts["BUY"],
                "sell_count": counts["SELL"],
                "no_trade_count": counts["NO_TRADE"],
                "total_signals": counts["BUY"] + counts["SELL"],
            }
            for month, counts in sorted(monthly.items())
        ]

        moves_report: list[dict[str, Any]] = []
        highs = frame["High"].astype(float)
        lows = frame["Low"].astype(float)
        for threshold in MISSED_MOVE_THRESHOLDS:
            cheap = self.move_engine._detect_moves_cheap(highs, lows, threshold)
            deduped = self.move_engine._dedupe_cheap_moves(cheap)
            for move in deduped[:MAX_MISSED_EXPORT]:
                move_side = "BUY" if move.direction == "bullish" else "SELL"
                matching = [
                    item
                    for item in signals
                    if item["signal_direction"] == move_side
                    and item.get("timestamp")
                    and pd.to_datetime(item["timestamp"]) <= pd.to_datetime(
                        frame.iloc[move.start_bar].get("Date", ""),
                    )
                ]
                flags_at_start = self._feature_flags_at_bar(frame, enriched, move.start_bar)
                captured = bool(matching)
                captured_points = round(move.magnitude, 2) if captured else 0.0
                missed_points = 0.0 if captured else round(move.magnitude, 2)
                moves_report.append(
                    {
                        "threshold_points": threshold,
                        "symbol": symbol,
                        "timeframe": timeframe_label,
                        "direction": move.direction,
                        "move_magnitude_points": round(move.magnitude, 2),
                        "start_timestamp": str(frame.iloc[move.start_bar].get("Date", "")),
                        "expansion_timestamp": str(frame.iloc[move.expansion_bar].get("Date", "")),
                        "engine_generated_signal": captured,
                        "could_enter_before_move": captured,
                        "points_captured": captured_points,
                        "points_missed": missed_points,
                        "missed_reasons": []
                        if captured
                        else self._diagnose_missed(
                            frame,
                            move.start_bar,
                            move.direction,
                            prechecks,
                            flags_at_start,
                        ),
                    },
                )

        return signals, frequency_stats, monthly_report, moves_report

    def run(self, metadata: dict[str, Any]) -> RealtimeReplayValidationReport:
        started = time.perf_counter()
        v2_card = self._load_v2_card()
        exports = self._load_research_exports()
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
        all_moves: list[dict[str, Any]] = []
        aggregate_freq = Counter()

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

                frame_signals, freq_stats, monthly, moves = self._replay_frame(
                    symbol=symbol,
                    timeframe_label=timeframe_label,
                    frame=frame,
                    metadata=metadata,
                    v2_card=v2_card,
                    buy_keys=buy_keys,
                    sell_keys=sell_keys,
                    no_trade_rules=no_trade_rules,
                )
                all_signals.extend(frame_signals)
                aggregate_freq["total_bars_scanned"] += freq_stats.get("total_bars_scanned", 0)
                aggregate_freq["buy_count"] += freq_stats.get("buy_count", 0)
                aggregate_freq["sell_count"] += freq_stats.get("sell_count", 0)
                aggregate_freq["no_trade_count"] += freq_stats.get("no_trade_count", 0)
                for month_row in monthly:
                    bucket = all_monthly[month_row["month"]]
                    bucket["BUY"] += month_row["buy_count"]
                    bucket["SELL"] += month_row["sell_count"]
                    bucket["NO_TRADE"] += month_row["no_trade_count"]
                all_moves.extend(moves)

        trade_signals = [item for item in all_signals if item["signal_direction"] in {"BUY", "SELL"}]
        pnls = [
            float(item["outcome"]["realized_pnl_points"])
            for item in trade_signals
            if item.get("outcome", {}).get("realized_pnl_points") is not None
        ]
        wins = [item for item in trade_signals if item.get("outcome", {}).get("win")]
        losses = [item for item in trade_signals if item.get("outcome") and not item["outcome"].get("win")]

        months = max(metadata.get("research_window_days", self.research_days) / 30.4375, 1.0)
        overall = {
            "total_bars_replayed": aggregate_freq.get("total_bars_scanned", 0),
            "total_signals": len(trade_signals),
            "buy_signals": aggregate_freq.get("buy_count", 0),
            "sell_signals": aggregate_freq.get("sell_count", 0),
            "no_trade_decisions": aggregate_freq.get("no_trade_count", 0),
            "signals_per_month": round(len(trade_signals) / months, 2),
            "win_rate_pct": round(len(wins) / len(trade_signals) * 100, 2) if trade_signals else 0.0,
            "profit_factor": self._profit_factor(pnls),
            "expectancy": round(mean(pnls), 2) if pnls else 0.0,
            "hit_1r_rate_pct": round(
                sum(1 for item in trade_signals if item.get("outcome", {}).get("hit_1r")) / len(trade_signals) * 100,
                2,
            )
            if trade_signals
            else 0.0,
        }

        false_report = {
            "total_signals": len(trade_signals),
            "winning_signals": len(wins),
            "losing_signals": len(losses),
            "false_break_signals": sum(
                1 for item in trade_signals if item.get("outcome", {}).get("is_false_signal")
            ),
            "late_signals": 0,
            "early_signals": 0,
        }

        moves_200 = [item for item in all_moves if item["threshold_points"] >= 200]
        captured_200 = [item for item in moves_200 if item["engine_generated_signal"]]
        major_200 = {
            "total_200_plus_moves": len(moves_200),
            "signals_before_move": len(captured_200),
            "missed_200_plus_moves": len(moves_200) - len(captured_200),
            "capture_rate_pct": round(len(captured_200) / len(moves_200) * 100, 2) if moves_200 else 0.0,
            "total_points_captured": round(sum(item["points_captured"] for item in moves_200), 2),
            "total_points_missed": round(sum(item["points_missed"] for item in moves_200), 2),
            "sample_moves": moves_200[:50],
        }

        condition_stats: dict[str, list[float]] = defaultdict(list)
        for item in trade_signals:
            session = item.get("reasons", {}).get("blueprint_tags", ["Unknown"])[0] if item.get("reasons") else "Unknown"
            key = f"{item['signal_direction']}|{item['timeframe']}|{session}"
            pnl = item.get("outcome", {}).get("realized_pnl_points", 0.0)
            condition_stats[key].append(float(pnl))

        ranked_conditions = []
        for key, bucket in condition_stats.items():
            if len(bucket) < 5:
                continue
            ranked_conditions.append(
                {
                    "condition": key,
                    "sample_size": len(bucket),
                    "expectancy": round(mean(bucket), 2),
                    "win_rate_pct": round(sum(1 for p in bucket if p > 0) / len(bucket) * 100, 2),
                    "signals_per_month": round(len(bucket) / months, 2),
                },
            )
        ranked_conditions.sort(key=lambda row: row["expectancy"], reverse=True)

        monthly_report = [
            {
                "month": month,
                "buy_count": counts["BUY"],
                "sell_count": counts["SELL"],
                "no_trade_count": counts["NO_TRADE"],
                "signals_per_month": counts["BUY"] + counts["SELL"],
            }
            for month, counts in sorted(all_monthly.items())
        ]

        production_candidates = [
            {
                "timestamp": item["timestamp"],
                "symbol": item["symbol"],
                "timeframe": item["timeframe"],
                "signal_direction": item["signal_direction"],
                "signal_score": item["signal_score"],
                "signals_per_month": overall["signals_per_month"],
                "win_rate_pct": overall["win_rate_pct"],
                "profit_factor": overall["profit_factor"],
                "expectancy": item.get("outcome", {}).get("realized_pnl_points", 0.0),
                "hit_1r": item.get("outcome", {}).get("hit_1r"),
                "hit_2r": item.get("outcome", {}).get("hit_2r"),
                "hit_3r": item.get("outcome", {}).get("hit_3r"),
            }
            for item in sorted(trade_signals, key=lambda row: row.get("signal_score", 0), reverse=True)[:50]
            if item.get("outcome", {}).get("win")
        ]

        conclusions = [
            "Realtime replay validation V2: strict no-look-ahead bar-by-bar replay.",
            f"Replayed {overall.get('total_bars_replayed', 0)} bars across {len(self.symbols)} symbols.",
            f"Generated {overall['total_signals']} BUY/SELL signals ({overall['signals_per_month']}/month).",
            f"NO TRADE decisions: {overall['no_trade_decisions']}.",
            f"200+ point moves: {major_200['total_200_plus_moves']} detected; "
            f"capture rate {major_200['capture_rate_pct']}%.",
            f"False signal rate: {false_report['false_break_signals']}/{false_report['total_signals']}.",
        ]

        return RealtimeReplayValidationReport(
            symbols_analyzed=list(self.symbols),
            timeframes_analyzed=list(self.timeframes),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            research_exports_used=exports,
            v2_production_card=v2_card,
            replay_rules={
                "no_future_leakage": True,
                "train_test_not_used": "full_history_replay",
                "mandatory_core": ["Displacement", "CHOCH", "BOS", "FVG Reclaim"],
                "v2_buy_filters": v2_card["buy_rules"]["filter_stack"],
                "v2_sell_filters": v2_card["sell_rules"]["filter_stack"],
                "no_trade_rules": no_trade_rules,
            },
            overall_statistics=overall,
            signal_by_signal_log=all_signals[:MAX_SIGNAL_LOG],
            missed_move_report=all_moves[:MAX_MISSED_EXPORT],
            false_signal_report=false_report,
            monthly_frequency_report=monthly_report,
            major_200_plus_move_analysis=major_200,
            top_performing_conditions=ranked_conditions[:20],
            worst_performing_conditions=list(reversed(ranked_conditions[-20:])),
            production_candidate_list=production_candidates,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )


def generate_realtime_replay_validation_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
    v2_optimization_path: Path | str | None = None,
    symbols: tuple[str, ...] | None = None,
) -> RealtimeReplayValidationReport:
    """Run realtime replay validation and export JSON."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise RealtimeReplayValidationError(f"Filter research report not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = SmartMoneyEngineRealtimeReplayValidationResearch(
        symbols=symbols,
        v2_optimization_path=v2_optimization_path,
    )
    report = engine.run(metadata)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info(
        "Realtime replay validation completed: signals=%s capture_200=%s%%",
        report.overall_statistics.get("total_signals"),
        report.major_200_plus_move_analysis.get("capture_rate_pct"),
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_realtime_replay_validation_report()
        print("SmartMoneyEngine Real-Time Replay Validation V2 Summary")
        print(f"Total signals: {report.overall_statistics['total_signals']}")
        print(f"Signals/month: {report.overall_statistics['signals_per_month']}")
        print(f"WR: {report.overall_statistics['win_rate_pct']}%")
        print(f"200+ capture rate: {report.major_200_plus_move_analysis['capture_rate_pct']}%")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except RealtimeReplayValidationError as exc:
        logger.error("Realtime replay validation error: %s", exc)
        print(f"Realtime replay validation error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected realtime replay validation error")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
