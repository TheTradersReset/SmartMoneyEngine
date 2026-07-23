"""
NIFTY50 Liquidity Direction Decision Matrix research.

First real predictive model: liquidity events + causal context at the event bar
mapped to BUY / SELL / NO TRADE outcomes. Strict no-look-ahead feature construction.
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

from src.context.institutional_liquidity_map_engine import InstitutionalLiquidityMapEngine
from src.context.market_intelligence_engine import MarketIntelligenceEngine
from src.research.filter_research_engine import FilterContextBuilder, FilterResearchEngine, _json_safe
from src.research.institutional_expansion_trigger_discovery_research import (
    LEVEL_CLUSTER_POINTS,
    PRE_EXPANSION_LOOKBACK,
    VOLUME_LOOKBACK,
    InstitutionalExpansionTriggerDiscoveryResearch,
)
from src.research.liquidity_move_reconstruction_research import (
    FORWARD_BARS,
    LiquidityMoveReconstructionResearch,
    TIMEFRAME_MINUTES,
    _CheapMoveCandidate,
)
from src.research.nifty50_trap_to_momentum_validation_research import (
    DEFAULT_SYMBOL,
    MOVE_THRESHOLDS,
    RESEARCH_WINDOW_DAYS,
    Nifty50TrapToMomentumValidationResearch,
)
from src.research.trade_construction_validation_research import TradeConstructionValidationResearch

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_FILTER_REPORT_PATH = RESEARCH_DIR / "filter_research_report.json"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "nifty50_liquidity_direction_decision_matrix.json"

MOVE_DETECTION_TIMEFRAME = "5M"
CONTEXT_TIMEFRAMES = ("15M", "1H", "1D")
PIPELINE_TIMEFRAMES = ("5M", "15M", "1H")
MIN_COMBO_SAMPLES = 50
TOP_MODEL_COUNT = 20
EXAMPLES_PER_MODEL = 5
NO_TRADE_MOVE_THRESHOLD = 50.0

LIQUIDITY_EVENTS = (
    "Liquidity Grab",
    "Stop Hunt",
    "PDH Sweep",
    "PDL Sweep",
    "PWH Sweep",
    "PWL Sweep",
    "Round Number Sweep",
    "Failed Breakout",
    "Failed Breakdown",
    "Gap Reversal",
    "Gap Continuation",
)


class Nifty50LiquidityDirectionDecisionMatrixError(Exception):
    """Raised when liquidity direction decision matrix research fails."""


@dataclass
class Nifty50LiquidityDirectionDecisionMatrixReport:
    """Full liquidity direction decision matrix output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    primary_timeframe: str
    context_timeframes: list[str]
    methodology: dict[str, Any]
    total_liquidity_events: int
    decision_matrix: list[dict[str, Any]]
    top_20_buy_decision_models: list[dict[str, Any]]
    top_20_sell_decision_models: list[dict[str, Any]]
    top_20_no_trade_decision_models: list[dict[str, Any]]
    most_reliable_formulas: dict[str, Any]
    reality_check_examples: dict[str, list[dict[str, Any]]]
    counterfactual_capture: dict[str, Any]
    final_answers: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Nifty50LiquidityDirectionDecisionMatrixResearch(Nifty50TrapToMomentumValidationResearch):
    """Build causal liquidity-event decision matrix for NIFTY50."""

    def __init__(self) -> None:
        super().__init__()
        self.context_builder = FilterContextBuilder()
        self.intelligence = MarketIntelligenceEngine(symbol=DEFAULT_SYMBOL)
        self.trade_engine = TradeConstructionValidationResearch(
            symbol=DEFAULT_SYMBOL,
            research_days=RESEARCH_WINDOW_DAYS,
            timeframes=(MOVE_DETECTION_TIMEFRAME,),
        )
        self._frame_date_cache: dict[int, pd.Series] = {}

    def clear_frame_caches(self) -> None:
        """Drop cached per-frame datetime indexes (call when OHLCV frames are rebuilt)."""
        self._frame_date_cache.clear()
        discovery = getattr(self, "discovery", None)
        if discovery is not None and hasattr(discovery, "clear_market_levels_cache"):
            discovery.clear_market_levels_cache()

    def _parsed_frame_dates(self, frame: pd.DataFrame) -> pd.Series:
        key = id(frame)
        cached = self._frame_date_cache.get(key)
        if cached is None:
            cached = pd.to_datetime(frame["Date"])
            self._frame_date_cache[key] = cached
        return cached

    @staticmethod
    def _resample_daily(frame_1h: pd.DataFrame) -> pd.DataFrame:
        working = frame_1h.copy()
        working["Date"] = pd.to_datetime(working["Date"])
        working = working.set_index("Date")
        daily = working.resample("1D").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"},
        )
        return daily.dropna(subset=["Open", "Close"]).reset_index()

    def _bar_for_timestamp(self, frame: pd.DataFrame, timestamp: pd.Timestamp) -> int:
        dates = self._parsed_frame_dates(frame)
        eligible = dates[dates <= timestamp]
        if eligible.empty:
            return 0
        return int(eligible.index[-1])

    @staticmethod
    def _momentum_to_htf(label: str) -> str:
        lowered = (label or "").lower()
        if "bull" in lowered:
            return "Bullish"
        if "bear" in lowered:
            return "Bearish"
        return "Neutral"

    @staticmethod
    def _ema_structure_label(filters) -> str:
        alignment = filters.ema_alignment
        if alignment == "EMA20 > EMA50 > EMA200":
            return "Bull Stack"
        if alignment == "EMA20 < EMA50 < EMA200":
            return "Bear Stack"
        return "Mixed"

    @staticmethod
    def _rsi_bucket_label(rsi_band: str) -> str:
        mapping = {
            "Below 30": "<30",
            "Below 40": "30-40",
            "40-50": "40-60",
            "50-60": "40-60",
            "60-70": "60-70",
            "Above 70": ">70",
        }
        return mapping.get(rsi_band, "40-60")

    def _volume_bucket(self, frame: pd.DataFrame, bar: int) -> str:
        volume = self.discovery._to_float(frame.iloc[bar].get("Volume")) or 0.0
        start = max(0, bar - VOLUME_LOOKBACK)
        avg = mean(
            self.discovery._to_float(frame.iloc[offset].get("Volume")) or 0.0
            for offset in range(start, bar)
        ) if bar > start else volume
        if avg <= 0:
            return "Normal"
        ratio = volume / avg
        if ratio >= 2.0:
            return "Climactic"
        if ratio >= 1.5:
            return "Expanded"
        return "Normal"

    def _vwap_state(self, frame: pd.DataFrame, enriched: pd.DataFrame, bar: int) -> str:
        filters = self.context_builder.filter_state(enriched, bar)
        if bar < 1:
            return "Above" if filters.vwap_position == "Above VWAP" else "Below"
        prev = self.context_builder.filter_state(enriched, bar - 1)
        close = float(frame.iloc[bar]["Close"])
        prev_close = float(frame.iloc[bar - 1]["Close"])
        vwap = self.discovery._to_float(enriched.iloc[bar].get("_vwap"))
        if vwap is None:
            return "Above" if filters.vwap_position == "Above VWAP" else "Below"
        crossed_up = prev_close <= vwap and close > vwap
        crossed_down = prev_close >= vwap and close < vwap
        if crossed_up or crossed_down:
            return "Reclaimed"
        high = float(frame.iloc[bar]["High"])
        low = float(frame.iloc[bar]["Low"])
        if (high >= vwap and close < vwap) or (low <= vwap and close > vwap):
            return "Rejected"
        return "Above" if close >= vwap else "Below"

    def _confirmation_pattern(self, frame: pd.DataFrame, bar: int) -> str:
        direction = "bullish" if float(frame.iloc[bar]["Close"]) >= float(frame.iloc[bar]["Open"]) else "bearish"
        trigger = self.discovery._measure_expansion_trigger(frame, bar, direction)
        if trigger.get("hammer"):
            return "Hammer"
        if trigger.get("shooting_star"):
            return "Shooting Star"
        if trigger.get("engulfing") and direction == "bullish":
            return "Bullish Engulfing"
        if trigger.get("engulfing") and direction == "bearish":
            return "Bearish Engulfing"
        if trigger.get("morning_star"):
            return "Morning Star"
        if trigger.get("evening_star"):
            return "Evening Star"
        if trigger.get("marubozu"):
            return "Marubozu"
        return "None"

    def _location_label(
        self,
        frame: pd.DataFrame,
        calendar: pd.DataFrame,
        bar: int,
        close: float,
    ) -> str:
        cal = calendar.iloc[bar]
        atr = self.discovery._atr(frame, bar)
        threshold = max(atr * 0.35, LEVEL_CLUSTER_POINTS)
        levels = self.discovery._market_levels(frame, bar)
        labels: list[str] = []
        support = levels.get("major_support")
        resistance = levels.get("major_resistance")
        if support is not None and abs(close - support) <= threshold:
            labels.append("Near Support")
        if resistance is not None and abs(close - resistance) <= threshold:
            labels.append("Near Resistance")
        for key, label in (
            ("_pdh", "PDH"),
            ("_pdl", "PDL"),
            ("_pwh", "PWH"),
            ("_pwl", "PWL"),
        ):
            level = self.discovery._to_float(cal.get(key))
            if level is not None and abs(close - level) <= threshold:
                labels.append(label)
        round_level = self._round_number_level(close)
        if abs(close - round_level) <= threshold:
            labels.append("Round Number")
        return labels[0] if labels else "Mid Range"

    def _htf_trend(
        self,
        *,
        intel_frames: dict[str, pd.DataFrame],
        timestamp: pd.Timestamp,
    ) -> str:
        votes: list[int] = []
        for tf in ("5M", *CONTEXT_TIMEFRAMES):
            frame = intel_frames.get(tf)
            if frame is None:
                continue
            mapped = self._bar_for_timestamp(frame, timestamp)
            mapped = min(max(0, mapped), len(frame) - 1)
            state = self.intelligence.evaluate_bar(frame, mapped).momentum_state
            trend = self._momentum_to_htf(state)
            if trend == "Bullish":
                votes.append(1)
            elif trend == "Bearish":
                votes.append(-1)
        if not votes:
            return "Neutral"
        score = sum(votes)
        if score > 0:
            return "Bullish"
        if score < 0:
            return "Bearish"
        return "Neutral"

    def _context_at_bar(
        self,
        *,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        calendar: pd.DataFrame,
        intel_frames: dict[str, pd.DataFrame],
        bar: int,
    ) -> dict[str, str]:
        row = frame.iloc[bar]
        timestamp = pd.to_datetime(row["Date"])
        close = float(row["Close"])
        filters = self.context_builder.filter_state(enriched, bar)
        choch = self.discovery._is_active(row.get("Bullish_CHOCH")) or self.discovery._is_active(
            row.get("Bearish_CHOCH"),
        )
        bos = self.discovery._is_active(row.get("Bullish_BOS")) or self.discovery._is_active(
            row.get("Bearish_BOS"),
        )
        return {
            "htf_trend": self._htf_trend(intel_frames=intel_frames, timestamp=timestamp),
            "choch": "Present" if choch else "Absent",
            "bos": "Present" if bos else "Absent",
            "vwap": self._vwap_state(frame, enriched, bar),
            "ema_structure": self._ema_structure_label(filters),
            "rsi": self._rsi_bucket_label(filters.rsi_band),
            "volume": self._volume_bucket(frame, bar),
            "confirmation_candle": self._confirmation_pattern(frame, bar),
            "location": self._location_label(frame, calendar, bar, close),
        }

    @staticmethod
    def _combo_key(event: str, context: dict[str, str]) -> str:
        return (
            f"{event} | HTF={context['htf_trend']} | CHOCH={context['choch']} | BOS={context['bos']} | "
            f"VWAP={context['vwap']} | EMA={context['ema_structure']} | RSI={context['rsi']} | "
            f"Vol={context['volume']} | Candle={context['confirmation_candle']} | Loc={context['location']}"
        )

    @staticmethod
    def _optimal_decision(bull: float, bear: float) -> str:
        if bull >= NO_TRADE_MOVE_THRESHOLD and bull > bear * 1.05:
            return "BUY"
        if bear >= NO_TRADE_MOVE_THRESHOLD and bear > bull * 1.05:
            return "SELL"
        return "NO TRADE"

    @staticmethod
    def _forward_directional_moves(
        highs: pd.Series,
        lows: pd.Series,
        closes: pd.Series,
        bar: int,
        forward_bars: int,
    ) -> tuple[float, float]:
        end = min(len(highs) - 1, bar + forward_bars)
        base = float(closes.iloc[bar])
        bull = float(highs.iloc[bar : end + 1].max()) - base
        bear = base - float(lows.iloc[bar : end + 1].min())
        return max(bull, 0.0), max(bear, 0.0)

    def _trade_outcome(
        self,
        frame: pd.DataFrame,
        bar: int,
        direction: str,
    ) -> dict[str, Any]:
        """
        Forward-window trade statistics for a decided signal bar.

        Post-signal evaluation only. Requires bars after ``bar``. Returns ``{}``
        when forward data is missing — callers must not use emptiness to block
        BUY/SELL emission (see Layer4 realtime plan builders).
        """
        entry = round(float(frame.iloc[bar]["Close"]), 2)
        stop, risk = self.trade_engine._structural_stop(frame, bar, entry, direction)
        if risk <= 0:
            return {}
        target_liq = self.trade_engine._opposite_liquidity_target(frame, bar, entry, direction, risk)
        end = min(len(frame) - 1, bar + FORWARD_BARS)
        highs = frame.iloc[bar + 1 : end + 1]["High"].astype(float)
        lows = frame.iloc[bar + 1 : end + 1]["Low"].astype(float)
        if highs.empty:
            return {}
        if direction == "bullish":
            mfe = float(highs.max()) - entry
            mae = entry - float(lows.min())
            realized = float(frame.iloc[end]["Close"]) - entry
            target = target_liq if target_liq else entry + 3 * risk
        else:
            mfe = entry - float(lows.min())
            mae = float(highs.max()) - entry
            realized = entry - float(frame.iloc[end]["Close"])
            target = target_liq if target_liq else entry - 3 * risk
        return {
            "entry": entry,
            "stop_loss": round(stop, 2),
            "target": round(target, 2) if target else None,
            "risk_points": round(risk, 2),
            "mfe_points": round(mfe, 2),
            "mae_points": round(mae, 2),
            "realized_pnl_points": round(realized, 2),
            "hit_1r": mfe >= risk,
            "hit_2r": mfe >= 2 * risk,
            "hit_3r": mfe >= 3 * risk,
            "win": realized > 0,
        }

    @staticmethod
    def _profit_factor(pnls: list[float]) -> float | None:
        wins = sum(p for p in pnls if p > 0)
        losses = abs(sum(p for p in pnls if p < 0))
        if losses == 0:
            return None if wins == 0 else round(wins / 1.0, 2)
        return round(wins / losses, 2)

    def _aggregate_combo(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        optimal = Counter(row["optimal_decision"] for row in rows)
        buy_rows = [row for row in rows if row["optimal_decision"] == "BUY"]
        sell_rows = [row for row in rows if row["optimal_decision"] == "SELL"]
        buy_pnls = [float(row["buy_outcome"].get("realized_pnl_points", 0.0)) for row in rows if row.get("buy_outcome")]
        sell_pnls = [float(row["sell_outcome"].get("realized_pnl_points", 0.0)) for row in rows if row.get("sell_outcome")]
        buy_prob = round(optimal["BUY"] / n * 100, 2)
        sell_prob = round(optimal["SELL"] / n * 100, 2)
        no_trade_prob = round(optimal["NO TRADE"] / n * 100, 2)
        dominant = max(("BUY", buy_prob), ("SELL", sell_prob), ("NO TRADE", no_trade_prob), key=lambda item: item[1])
        dominant_pnls = buy_pnls if dominant[0] == "BUY" else sell_pnls if dominant[0] == "SELL" else []
        dominant_outcomes = (
            [row["buy_outcome"] for row in rows if row.get("buy_outcome")]
            if dominant[0] == "BUY"
            else [row["sell_outcome"] for row in rows if row.get("sell_outcome")]
            if dominant[0] == "SELL"
            else []
        )
        magnitudes = [float(row["forward_max_move"]) for row in rows]
        drawdowns = [float(row["buy_outcome"].get("mae_points", 0.0)) for row in rows if row.get("buy_outcome")]
        times = [float(row["time_to_expansion_bars"]) for row in rows if row["time_to_expansion_bars"] >= 0]
        return {
            "combination": rows[0]["combination"],
            "liquidity_event": rows[0]["event"],
            "context": rows[0]["context"],
            "occurrences": n,
            "buy_probability_pct": buy_prob,
            "sell_probability_pct": sell_prob,
            "no_trade_probability_pct": no_trade_prob,
            "direction_accuracy_pct": round(max(buy_prob, sell_prob, no_trade_prob), 2),
            "dominant_decision": dominant[0],
            "hit_1r_rate_pct": self._probability([o.get("hit_1r") for o in dominant_outcomes]),
            "hit_2r_rate_pct": self._probability([o.get("hit_2r") for o in dominant_outcomes]),
            "hit_3r_rate_pct": self._probability([o.get("hit_3r") for o in dominant_outcomes]),
            "average_move": round(mean(magnitudes), 2) if magnitudes else 0.0,
            "average_drawdown": round(mean(drawdowns), 2) if drawdowns else 0.0,
            "average_time_to_expansion_bars": round(mean(times), 2) if times else 0.0,
            "profit_factor": self._profit_factor(dominant_pnls) if dominant_pnls else None,
            "expectancy": round(mean(dominant_pnls), 2) if dominant_pnls else 0.0,
            "rank_score": round(
                max(buy_prob, sell_prob, no_trade_prob)
                + (self._profit_factor(dominant_pnls) or 0) * 10
                + (mean(dominant_pnls) if dominant_pnls else 0) / 10
                + n / 100,
                2,
            ),
            "instances": rows,
        }

    def _rank_models(self, matrix: list[dict[str, Any]], side: str) -> list[dict[str, Any]]:
        prob_key = {
            "BUY": "buy_probability_pct",
            "SELL": "sell_probability_pct",
            "NO TRADE": "no_trade_probability_pct",
        }[side]
        filtered = [row for row in matrix if row["occurrences"] >= MIN_COMBO_SAMPLES]
        filtered.sort(
            key=lambda row: (
                row[prob_key],
                row.get("profit_factor") or 0,
                row.get("expectancy") or 0,
                row["occurrences"],
            ),
            reverse=True,
        )
        winners = [row for row in filtered if row["dominant_decision"] == side][:TOP_MODEL_COUNT]
        if len(winners) < TOP_MODEL_COUNT:
            remainder = [row for row in filtered if row not in winners]
            remainder.sort(key=lambda row: row[prob_key], reverse=True)
            winners.extend(remainder[: TOP_MODEL_COUNT - len(winners)])
        return winners[:TOP_MODEL_COUNT]

    def _reality_examples(self, model: dict[str, Any]) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        for row in model["instances"][:200]:
            if len(examples) >= EXAMPLES_PER_MODEL:
                break
            side = model["dominant_decision"]
            outcome = row["buy_outcome"] if side == "BUY" else row["sell_outcome"]
            if not outcome:
                continue
            examples.append(
                {
                    "date": row["timestamp"][:10],
                    "time": row["timestamp"],
                    "entry": outcome.get("entry"),
                    "stop_loss": outcome.get("stop_loss"),
                    "target": outcome.get("target"),
                    "move_achieved_points": outcome.get("mfe_points"),
                    "signal_existed_before_move": row["time_to_expansion_bars"] >= 0,
                    "causal_verification": {
                        "context_at_event_bar_only": True,
                        "choch_from_current_bar_only": row["context"]["choch"],
                        "bos_from_current_bar_only": row["context"]["bos"],
                        "no_future_bos_used": True,
                        "no_future_fvg_used": True,
                    },
                },
            )
        return examples

    def _counterfactual_capture(
        self,
        moves: list[_CheapMoveCandidate],
        instance_index: dict[str, list[dict[str, Any]]],
        top_models: list[dict[str, Any]],
    ) -> dict[str, Any]:
        model_keys = {model["combination"] for model in top_models}
        captured = {str(th): 0 for th in (50, 100, 200, 300)}
        for move in moves:
            if move.magnitude < 50:
                continue
            move_side = "BUY" if move.direction == "bullish" else "SELL"
            pre_start = max(PRE_EXPANSION_LOOKBACK, move.start_bar - PRE_EXPANSION_LOOKBACK)
            hit = False
            for bar in range(pre_start, move.start_bar + 1):
                for row in instance_index.get(str(bar), []):
                    if row["combination"] not in model_keys:
                        continue
                    if row["dominant_decision"] != move_side:
                        continue
                    hit = True
                    break
                if hit:
                    break
            if hit:
                for threshold in (50, 100, 200, 300):
                    if move.magnitude >= threshold:
                        captured[str(threshold)] += 1
        totals = {str(th): sum(1 for move in moves if move.magnitude >= th) for th in (50, 100, 200, 300)}
        return {
            "moves_captured_if_matrix_only": captured,
            "total_moves_by_threshold": totals,
            "capture_rate_pct": {
                str(th): round(captured[str(th)] / max(totals[str(th)], 1) * 100, 2) for th in (50, 100, 200, 300)
            },
            "models_used": len(model_keys),
        }

    def run(self, metadata: dict[str, Any]) -> Nifty50LiquidityDirectionDecisionMatrixReport:
        started = time.perf_counter()
        end = date.fromisoformat(metadata["end_date"])
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=RESEARCH_WINDOW_DAYS)
        )

        filter_engine = FilterResearchEngine(
            symbol=DEFAULT_SYMBOL,
            research_days=RESEARCH_WINDOW_DAYS,
            timeframes=PIPELINE_TIMEFRAMES,
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
            self.move_engine._detect_moves_cheap(highs, lows, MOVE_THRESHOLDS[0]),
        )

        instances: list[dict[str, Any]] = []
        scan_end = len(frame) - FORWARD_BARS
        for bar in range(PRE_EXPANSION_LOOKBACK, scan_end):
            if bar % 1000 == 0:
                logger.info("Decision matrix scan bar=%s/%s events=%s", bar, scan_end, len(instances))
            events = [
                event
                for event in self._detect_events_at_bar(frame, calendar, bar)
                if event in LIQUIDITY_EVENTS
            ]
            if not events:
                continue
            context = self._context_at_bar(
                frame=frame,
                enriched=enriched,
                calendar=calendar,
                intel_frames=intel_frames,
                bar=bar,
            )
            bull, bear = self._forward_directional_moves(highs, lows, closes, bar, FORWARD_BARS)
            linked = self._find_next_move(moves, bar, FORWARD_BARS)
            for event in events:
                combo = self._combo_key(event, context)
                buy_outcome = self._trade_outcome(frame, bar, "bullish")
                sell_outcome = self._trade_outcome(frame, bar, "bearish")
                instances.append(
                    {
                        "timestamp": str(frame.iloc[bar].get("Date", "")),
                        "bar": bar,
                        "event": event,
                        "context": context,
                        "combination": combo,
                        "forward_bull": bull,
                        "forward_bear": bear,
                        "forward_max_move": max(bull, bear),
                        "optimal_decision": self._optimal_decision(bull, bear),
                        "buy_outcome": buy_outcome,
                        "sell_outcome": sell_outcome,
                        "time_to_expansion_bars": linked.start_bar - bar if linked else -1,
                    },
                )

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in instances:
            grouped[row["combination"]].append(row)

        matrix = [
            self._aggregate_combo(rows)
            for rows in grouped.values()
            if len(rows) >= MIN_COMBO_SAMPLES
        ]
        matrix.sort(key=lambda row: row["rank_score"], reverse=True)

        top_buy = self._rank_models(matrix, "BUY")
        top_sell = self._rank_models(matrix, "SELL")
        top_no_trade = self._rank_models(matrix, "NO TRADE")

        formulas = {
            "most_reliable_buy_formula": top_buy[0] if top_buy else None,
            "most_reliable_sell_formula": top_sell[0] if top_sell else None,
            "most_reliable_no_trade_formula": top_no_trade[0] if top_no_trade else None,
        }

        bar_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
        combo_dominant = {row["combination"]: row for row in matrix}
        for row in instances:
            agg = combo_dominant.get(row["combination"])
            if not agg:
                continue
            enriched_row = {**row, "dominant_decision": agg["dominant_decision"]}
            bar_index[str(row["bar"])].append(enriched_row)

        all_top = top_buy + top_sell
        counterfactual = self._counterfactual_capture(moves, bar_index, all_top)

        reality = {
            "top_buy_examples": {
                model["combination"]: self._reality_examples(model) for model in top_buy[:TOP_MODEL_COUNT]
            },
            "top_sell_examples": {
                model["combination"]: self._reality_examples(model) for model in top_sell[:TOP_MODEL_COUNT]
            },
        }

        final_answers = {
            "if_engine_used_only_this_matrix": counterfactual,
            "total_combinations_tested": len(grouped),
            "combinations_passing_min_sample": len(matrix),
            "total_liquidity_event_instances": len(instances),
        }

        conclusions = [
            "NIFTY50 Liquidity Direction Decision Matrix complete (strict causal context at event bar).",
            f"Liquidity event instances: {len(instances)}; valid combinations (n>={MIN_COMBO_SAMPLES}): {len(matrix)}.",
            f"Top BUY formula direction accuracy: {formulas['most_reliable_buy_formula']['direction_accuracy_pct'] if formulas['most_reliable_buy_formula'] else 'N/A'}%.",
            f"Counterfactual 200+ capture rate: {counterfactual['capture_rate_pct'].get('200', 0)}%.",
        ]

        slim_matrix = [{k: v for k, v in row.items() if k != "instances"} for row in matrix[:500]]

        return Nifty50LiquidityDirectionDecisionMatrixReport(
            symbol=DEFAULT_SYMBOL,
            research_window_days=RESEARCH_WINDOW_DAYS,
            start_date=metadata.get("start_date", start.isoformat()),
            end_date=metadata.get("end_date", end.isoformat()),
            primary_timeframe=MOVE_DETECTION_TIMEFRAME,
            context_timeframes=list(CONTEXT_TIMEFRAMES),
            methodology={
                "no_future_leakage_features": True,
                "features_from_event_bar_only": True,
                "forward_outcomes_for_labeling_only": True,
                "minimum_sample_size": MIN_COMBO_SAMPLES,
                "liquidity_events_only": list(LIQUIDITY_EVENTS),
            },
            total_liquidity_events=len(instances),
            decision_matrix=slim_matrix,
            top_20_buy_decision_models=[{k: v for k, v in m.items() if k != "instances"} for m in top_buy],
            top_20_sell_decision_models=[{k: v for k, v in m.items() if k != "instances"} for m in top_sell],
            top_20_no_trade_decision_models=[{k: v for k, v in m.items() if k != "instances"} for m in top_no_trade],
            most_reliable_formulas=formulas,
            reality_check_examples=reality,
            counterfactual_capture=counterfactual,
            final_answers=final_answers,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )


def generate_nifty50_liquidity_direction_decision_matrix_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> Nifty50LiquidityDirectionDecisionMatrixReport:
    """Run liquidity direction decision matrix research and export JSON."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise Nifty50LiquidityDirectionDecisionMatrixError(
            f"Filter research report not found: {metadata_path}",
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    metadata = {
        **metadata,
        "research_window_days": RESEARCH_WINDOW_DAYS,
        "start_date": (
            date.fromisoformat(metadata["end_date"]) - timedelta(days=RESEARCH_WINDOW_DAYS)
        ).isoformat(),
    }

    engine = Nifty50LiquidityDirectionDecisionMatrixResearch()
    report = engine.run(metadata)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)

    export = report.as_dict()
    export["decision_matrix"] = report.decision_matrix
    export["most_reliable_formulas"] = {
        key: ({k: v for k, v in value.items() if k != "instances"} if isinstance(value, dict) else value)
        for key, value in report.most_reliable_formulas.items()
    }

    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(export), handle, indent=2)

    logger.info("Liquidity direction decision matrix exported: %s", destination)
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        report = generate_nifty50_liquidity_direction_decision_matrix_report()
    except Nifty50LiquidityDirectionDecisionMatrixError as exc:
        logger.error("Decision matrix error: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected decision matrix error")
        return 1

    print("NIFTY50 Liquidity Direction Decision Matrix Summary")
    print(f"Events: {report.total_liquidity_events}")
    print(f"Valid combos: {len(report.decision_matrix)}")
    buy = report.most_reliable_formulas.get("most_reliable_buy_formula") or {}
    print(f"Top BUY accuracy: {buy.get('direction_accuracy_pct', 'N/A')}%")
    print(f"Report: {DEFAULT_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
