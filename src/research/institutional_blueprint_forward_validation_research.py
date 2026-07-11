"""
Institutional Blueprint Forward Validation research.

Prospectively scans market history for expansion blueprint matches and validates
whether blueprints predict future momentum before it happens. Research-only.
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
    BLUEPRINT_ARROW,
    PRE_EXPANSION_LOOKBACK,
    InstitutionalExpansionTriggerDiscoveryResearch,
)
from src.research.institutional_quality_validation_research import (
    InstitutionalQualityValidationResearch,
)
from src.research.liquidity_move_reconstruction_research import FORWARD_BARS, TIMEFRAME_MINUTES
from src.research.tier2_production_validation_research import Tier2ProductionValidationResearch
from src.research.trade_construction_validation_research import TradeConstructionValidationResearch
from src.signals.decision_engine import DecisionEngine

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_DISCOVERY_REPORT_PATH = (
    PROJECT_ROOT / "outputs" / "research" / "institutional_expansion_trigger_discovery.json"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "outputs" / "research" / "institutional_blueprint_forward_validation.json"
)

TOP_BLUEPRINT_COUNT = 10
MIN_SIGNAL_SAMPLES = 100
MIN_SIGNAL_SEPARATION_BARS = 20
MAX_EXPORT_SIGNALS = 200
MAX_SIGNALS_PER_BLUEPRINT = 5000

DEFAULT_SYMBOLS = ("NIFTY50", "BANKNIFTY", "FINNIFTY")
TIMEFRAMES = ("5M", "15M", "1H")


class BlueprintForwardValidationError(Exception):
    """Raised when blueprint forward validation fails."""


@dataclass(frozen=True)
class BlueprintSpec:
    """One blueprint loaded from discovery research."""

    blueprint_id: str
    blueprint: str
    direction: str
    signal_side: str
    required_tags: tuple[str, ...]
    blueprint_score: float
    discovery_rank: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BlueprintSignalOutcome:
    """Forward outcome for one prospective blueprint signal."""

    symbol: str
    timeframe: str
    timestamp: str
    signal_bar: int
    blueprint_id: str
    blueprint: str
    blueprint_score: float
    signal_side: str
    direction: str
    entry_price: float
    stop_price: float
    target_1r: float
    target_2r: float
    target_3r: float
    target_4_opposite_liquidity: float
    risk_points: float
    hit_1r: bool
    hit_2r: bool
    hit_3r: bool
    hit_4r: bool
    hit_5r: bool
    stop_hit: bool
    hit_opposite_liquidity: bool
    mfe_points: float
    mae_points: float
    time_to_1r_bars: int | None
    time_to_2r_bars: int | None
    time_to_3r_bars: int | None
    time_to_stop_bars: int | None
    time_to_target_bars: int | None
    realized_pnl_points: float
    realized_rr: float
    win: bool
    is_false_signal: bool
    filter_context: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BlueprintReliabilityMetrics:
    """Aggregate reliability metrics for one blueprint."""

    blueprint_id: str
    blueprint: str
    direction: str
    signal_side: str
    occurrences: int
    win_rate_pct: float
    hit_1r_rate_pct: float
    hit_2r_rate_pct: float
    hit_3r_rate_pct: float
    profit_factor: float | None
    expectancy: float
    average_rr: float
    net_points: float
    maximum_drawdown_points: float
    false_signal_rate_pct: float
    classification: str
    blueprint_score: float
    discovery_rank: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BlueprintForwardValidationReport:
    """Full blueprint forward validation output."""

    symbols_analyzed: list[str]
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    blueprints_validated: list[dict[str, Any]]
    total_signals: int
    blueprint_reliability: list[dict[str, Any]]
    production_ready_blueprints: list[dict[str, Any]]
    needs_validation_blueprints: list[dict[str, Any]]
    rejected_blueprints: list[dict[str, Any]]
    filter_discovery: dict[str, Any]
    sample_signals: list[dict[str, Any]]
    trade_construction: dict[str, str]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class InstitutionalBlueprintForwardValidationResearch:
    """Prospectively validate expansion blueprints on full market history."""

    def __init__(
        self,
        symbols: tuple[str, ...] | None = None,
        research_days: int = RESEARCH_DAYS,
        timeframes: tuple[str, ...] = TIMEFRAMES,
        discovery_report_path: Path | str | None = None,
    ) -> None:
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.research_days = research_days
        self.timeframes = timeframes
        self.discovery_report_path = Path(
            discovery_report_path or DEFAULT_DISCOVERY_REPORT_PATH,
        )
        self.discovery_engine = InstitutionalExpansionTriggerDiscoveryResearch(
            symbols=self.symbols,
            research_days=research_days,
            timeframes=timeframes,
        )
        self.trade_engine = TradeConstructionValidationResearch(
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

    @staticmethod
    def _maximum_drawdown(pnls: list[float]) -> float:
        return Tier2ProductionValidationResearch._maximum_drawdown(pnls)

    @staticmethod
    def _classify_blueprint(
        signals: int,
        win_rate_pct: float,
        expectancy: float,
        profit_factor: float | None,
    ) -> str:
        if signals < MIN_SIGNAL_SAMPLES:
            return "Reject"
        if expectancy < 0:
            return "Reject"
        if (
            win_rate_pct >= 40.0
            and expectancy >= 50.0
            and profit_factor is not None
            and profit_factor >= 1.5
        ):
            return "Production Ready"
        if expectancy > 0:
            return "Needs Validation"
        return "Reject"

    @staticmethod
    def _parse_blueprint(blueprint: str) -> tuple[str, ...]:
        return tuple(part.strip() for part in blueprint.split(BLUEPRINT_ARROW) if part.strip())

    @staticmethod
    def _matches_blueprint(required_tags: tuple[str, ...], active_tags: tuple[str, ...]) -> bool:
        active = set(active_tags)
        return all(tag in active for tag in required_tags)

    def _load_blueprints(self) -> tuple[list[BlueprintSpec], list[BlueprintSpec]]:
        if not self.discovery_report_path.exists():
            raise BlueprintForwardValidationError(
                f"Discovery report not found: {self.discovery_report_path}",
            )
        with self.discovery_report_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        bullish: list[BlueprintSpec] = []
        bearish: list[BlueprintSpec] = []
        for index, item in enumerate(payload.get("top_20_bullish_momentum_blueprints", [])[:TOP_BLUEPRINT_COUNT], start=1):
            blueprint = str(item["blueprint"])
            bullish.append(
                BlueprintSpec(
                    blueprint_id=f"bullish_bp_{index:02d}",
                    blueprint=blueprint,
                    direction="bullish",
                    signal_side="BUY",
                    required_tags=self._parse_blueprint(blueprint),
                    blueprint_score=float(item.get("reliability_score", 0.0)),
                    discovery_rank=int(item.get("rank", index)),
                ),
            )
        for index, item in enumerate(payload.get("top_20_bearish_momentum_blueprints", [])[:TOP_BLUEPRINT_COUNT], start=1):
            blueprint = str(item["blueprint"])
            bearish.append(
                BlueprintSpec(
                    blueprint_id=f"bearish_bp_{index:02d}",
                    blueprint=blueprint,
                    direction="bearish",
                    signal_side="SELL",
                    required_tags=self._parse_blueprint(blueprint),
                    blueprint_score=float(item.get("reliability_score", 0.0)),
                    discovery_rank=int(item.get("rank", index)),
                ),
            )
        return bullish, bearish

    def _filter_engine(self, symbol: str) -> FilterResearchEngine:
        return FilterResearchEngine(
            symbol=symbol,
            research_days=self.research_days,
            timeframes=self.timeframes,
        )

    def _build_filter_context(
        self,
        enriched: pd.DataFrame,
        measurements: dict[str, Any],
        bar: int,
        direction: str,
    ) -> dict[str, Any]:
        filters = self.context_builder.filter_state(enriched, bar)
        sr = measurements["support_resistance"]
        structure = measurements["structure"]
        liquidity = measurements["liquidity"]
        rsi_series = enriched["_rsi"] if "_rsi" in enriched.columns else enriched.get("RSI")
        rsi_value = (
            float(rsi_series.iloc[bar])
            if rsi_series is not None and pd.notna(rsi_series.iloc[bar])
            else 50.0
        )
        close = float(enriched.iloc[bar]["Close"])
        gap_up = gap_down = False
        if bar >= 1:
            gap = float(enriched.iloc[bar]["Open"]) - float(enriched.iloc[bar - 1]["Close"])
            gap_up = gap > 0.5
            gap_down = gap < -0.5

        return {
            "rsi": round(rsi_value, 2),
            "rsi_below_40": rsi_value < 40,
            "rsi_above_60": rsi_value > 60,
            "vwap_position": filters.vwap_position,
            "ema_alignment": filters.ema_alignment,
            "session": filters.session,
            "support_distance_points": sr.get("distance_from_level_points"),
            "resistance_distance_points": sr.get("distance_from_level_points"),
            "level_strength_category": sr.get("level_strength_category"),
            "level_strength_score": sr.get("level_strength_score"),
            "gap_up": gap_up,
            "gap_down": gap_down,
            "liquidity_sweep": liquidity.get("liquidity_grab_count", 0) > 0,
            "choch_present": structure.get("choch_count", 0) > 0,
            "bos_present": structure.get("bos_count", 0) > 0,
            "fvg_present": structure.get("fvg_count", 0) > 0,
            "premium_discount": structure.get("premium_discount"),
            "htf_alignment": structure.get("htf_alignment"),
        }

    def _forward_validate(
        self,
        frame: pd.DataFrame,
        signal_bar: int,
        direction: str,
    ) -> dict[str, Any]:
        entry_price = round(float(frame.iloc[signal_bar]["Close"]), 2)
        stop, risk = self.trade_engine._structural_stop(
            frame,
            signal_bar,
            entry_price,
            direction,
        )
        target_liq = self.trade_engine._opposite_liquidity_target(
            frame,
            signal_bar,
            entry_price,
            direction,
            risk,
        )
        t1 = round(entry_price + risk if direction == "bullish" else entry_price - risk, 2)
        t2 = round(entry_price + risk * 2 if direction == "bullish" else entry_price - risk * 2, 2)
        t3 = round(entry_price + risk * 3 if direction == "bullish" else entry_price - risk * 3, 2)
        t5 = round(entry_price + risk * 5 if direction == "bullish" else entry_price - risk * 5, 2)

        end = min(len(frame) - 1, signal_bar + FORWARD_BARS)
        mfe = mae = 0.0
        hit_1r = hit_2r = hit_3r = hit_4r = hit_5r = False
        stop_hit = hit_target = False
        time_to_1r = time_to_2r = time_to_3r = time_to_stop = time_to_target = None
        pnl = 0.0
        rr = 0.0

        for index in range(signal_bar + 1, end + 1):
            bar_high = float(frame.iloc[index]["High"])
            bar_low = float(frame.iloc[index]["Low"])

            if direction == "bullish":
                favorable_high = bar_high - entry_price
                adverse = entry_price - bar_low
                mfe = max(mfe, favorable_high)
                mae = max(mae, adverse)
                if not stop_hit:
                    if favorable_high >= risk:
                        hit_1r = True
                        if time_to_1r is None:
                            time_to_1r = index - signal_bar
                    if favorable_high >= risk * 2:
                        hit_2r = True
                        if time_to_2r is None:
                            time_to_2r = index - signal_bar
                    if favorable_high >= risk * 3:
                        hit_3r = True
                        if time_to_3r is None:
                            time_to_3r = index - signal_bar
                    if favorable_high >= risk * 4:
                        hit_4r = True
                    if bar_high >= t5:
                        hit_5r = True
                if bar_low <= stop:
                    stop_hit = True
                    pnl = -risk
                    rr = -1.0
                    time_to_stop = index - signal_bar
                    break
                if bar_high >= target_liq:
                    hit_target = True
                    pnl = round(target_liq - entry_price, 2)
                    rr = round(pnl / risk, 2) if risk > 0 else 0.0
                    time_to_target = index - signal_bar
                    break
            else:
                favorable_low = entry_price - bar_low
                adverse = bar_high - entry_price
                mfe = max(mfe, favorable_low)
                mae = max(mae, adverse)
                if not stop_hit:
                    if favorable_low >= risk:
                        hit_1r = True
                        if time_to_1r is None:
                            time_to_1r = index - signal_bar
                    if favorable_low >= risk * 2:
                        hit_2r = True
                        if time_to_2r is None:
                            time_to_2r = index - signal_bar
                    if favorable_low >= risk * 3:
                        hit_3r = True
                        if time_to_3r is None:
                            time_to_3r = index - signal_bar
                    if favorable_low >= risk * 4:
                        hit_4r = True
                    if bar_low <= t5:
                        hit_5r = True
                if bar_high >= stop:
                    stop_hit = True
                    pnl = -risk
                    rr = -1.0
                    time_to_stop = index - signal_bar
                    break
                if bar_low <= target_liq:
                    hit_target = True
                    pnl = round(entry_price - target_liq, 2)
                    rr = round(pnl / risk, 2) if risk > 0 else 0.0
                    time_to_target = index - signal_bar
                    break

        if not stop_hit and not hit_target:
            close = float(frame.iloc[end]["Close"])
            if direction == "bullish":
                pnl = round(close - entry_price, 2)
            else:
                pnl = round(entry_price - close, 2)
            rr = round(pnl / risk, 2) if risk > 0 else 0.0

        return {
            "entry_price": entry_price,
            "stop_price": stop,
            "target_1r": t1,
            "target_2r": t2,
            "target_3r": t3,
            "target_4_opposite_liquidity": target_liq,
            "risk_points": risk,
            "hit_1r": hit_1r,
            "hit_2r": hit_2r,
            "hit_3r": hit_3r,
            "hit_4r": hit_4r,
            "hit_5r": hit_5r,
            "stop_hit": stop_hit,
            "hit_opposite_liquidity": hit_target,
            "mfe_points": round(mfe, 2),
            "mae_points": round(mae, 2),
            "time_to_1r_bars": time_to_1r,
            "time_to_2r_bars": time_to_2r,
            "time_to_3r_bars": time_to_3r,
            "time_to_stop_bars": time_to_stop,
            "time_to_target_bars": time_to_target,
            "realized_pnl_points": pnl,
            "realized_rr": rr,
            "win": pnl > 0,
            "is_false_signal": stop_hit and not hit_1r,
        }

    @staticmethod
    def _displacement_tag(frame: pd.DataFrame, bar: int, direction: str) -> str:
        row = frame.iloc[bar]
        displacement = LiquidityNarrativeEngine._displacement_strength_for_bar(row, direction)
        return f"Displacement:{displacement.value}"

    @staticmethod
    def _build_frame_prechecks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
        length = len(frame)
        weak_bull = np.zeros(length, dtype=bool)
        weak_bear = np.zeros(length, dtype=bool)
        bos_active = np.zeros(length, dtype=np.int8)
        choch_active = np.zeros(length, dtype=np.int8)
        sweep_active = np.zeros(length, dtype=np.int8)
        fvg_active = np.zeros(length, dtype=np.int8)
        ob_active = np.zeros(length, dtype=np.int8)

        for index in range(length):
            row = frame.iloc[index]
            weak_bull[index] = (
                LiquidityNarrativeEngine._displacement_strength_for_bar(row, "bullish").value == "Weak"
            )
            weak_bear[index] = (
                LiquidityNarrativeEngine._displacement_strength_for_bar(row, "bearish").value == "Weak"
            )
            if InstitutionalBlueprintForwardValidationResearch._is_active(row.get("Bullish_BOS")) or InstitutionalBlueprintForwardValidationResearch._is_active(
                row.get("Bearish_BOS"),
            ):
                bos_active[index] = 1
            if InstitutionalBlueprintForwardValidationResearch._is_active(row.get("Bullish_CHOCH")) or InstitutionalBlueprintForwardValidationResearch._is_active(
                row.get("Bearish_CHOCH"),
            ):
                choch_active[index] = 1
            if InstitutionalBlueprintForwardValidationResearch._is_active(row.get("Buy_Liquidity_Sweep")) or InstitutionalBlueprintForwardValidationResearch._is_active(
                row.get("Sell_Liquidity_Sweep"),
            ):
                sweep_active[index] = 1
            if InstitutionalBlueprintForwardValidationResearch._is_active(row.get("Bullish_FVG_Top")) or InstitutionalBlueprintForwardValidationResearch._is_active(
                row.get("Bearish_FVG_Top"),
            ):
                fvg_active[index] = 1
            if InstitutionalBlueprintForwardValidationResearch._is_active(row.get("Bullish_OB_High")) or InstitutionalBlueprintForwardValidationResearch._is_active(
                row.get("Bearish_OB_High"),
            ):
                ob_active[index] = 1

        return {
            "weak_bull": weak_bull,
            "weak_bear": weak_bear,
            "bos_cumsum": np.cumsum(bos_active),
            "choch_cumsum": np.cumsum(choch_active),
            "sweep_cumsum": np.cumsum(sweep_active),
            "fvg_cumsum": np.cumsum(fvg_active),
            "ob_cumsum": np.cumsum(ob_active),
        }

    @staticmethod
    def _window_count(cumsum: np.ndarray, end_bar: int, lookback: int) -> int:
        start_bar = max(0, end_bar - lookback)
        if start_bar == 0:
            return int(cumsum[end_bar])
        return int(cumsum[end_bar] - cumsum[start_bar - 1])

    def _spec_precheck(
        self,
        spec: BlueprintSpec,
        bar: int,
        direction: str,
        frame: pd.DataFrame,
        prechecks: dict[str, np.ndarray],
    ) -> bool:
        displacement_tag = self._displacement_tag(frame, bar, direction)
        required = set(spec.required_tags)
        displacement_tags = {tag for tag in required if tag.startswith("Displacement:")}
        if displacement_tags and displacement_tag not in displacement_tags:
            return False
        if "BOS" in required and self._window_count(prechecks["bos_cumsum"], bar, PRE_EXPANSION_LOOKBACK) == 0:
            return False
        if "CHOCH" in required and self._window_count(prechecks["choch_cumsum"], bar, PRE_EXPANSION_LOOKBACK) == 0:
            return False
        if "Liquidity Grab" in required and self._window_count(prechecks["sweep_cumsum"], bar, PRE_EXPANSION_LOOKBACK) == 0:
            return False
        if "FVG" in required and self._window_count(prechecks["fvg_cumsum"], bar, PRE_EXPANSION_LOOKBACK) == 0:
            return False
        if "Order Block" in required and self._window_count(prechecks["ob_cumsum"], bar, PRE_EXPANSION_LOOKBACK) == 0:
            return False
        return True

    def _direction_candidate(
        self,
        bar: int,
        direction: str,
        specs: list[BlueprintSpec],
        frame: pd.DataFrame,
        prechecks: dict[str, np.ndarray],
    ) -> bool:
        return any(self._spec_precheck(spec, bar, direction, frame, prechecks) for spec in specs)

    def _scan_history(
        self,
        metadata: dict[str, Any],
        blueprints: list[BlueprintSpec],
    ) -> list[BlueprintSignalOutcome]:
        end = date.fromisoformat(metadata["end_date"])
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=self.research_days)
        )

        outcomes: list[BlueprintSignalOutcome] = []
        last_signal_bar: dict[tuple[str, str, str], int] = {}
        signal_counts: Counter[str] = Counter()

        for symbol in self.symbols:
            filter_engine = self._filter_engine(symbol)
            liquidity_map = InstitutionalLiquidityMapEngine(symbol=symbol)
            for timeframe_label in self.timeframes:
                path = filter_engine._pipeline_path(timeframe_label)
                if not path.exists():
                    try:
                        path = filter_engine._ensure_pipeline(timeframe_label, start, end)
                    except Exception as exc:
                        logger.warning("Skipping %s/%s pipeline: %s", symbol, timeframe_label, exc)
                        continue

                frame = pd.read_csv(path).reset_index(drop=True)
                if len(frame) <= PRE_EXPANSION_LOOKBACK + FORWARD_BARS:
                    continue

                enriched = self.context_builder.enrich(frame)
                calendar = liquidity_map._attach_calendar_levels(frame)
                intel = self.discovery_engine.intelligence_engine.enrich(frame)

                bullish_specs = [item for item in blueprints if item.direction == "bullish"]
                bearish_specs = [item for item in blueprints if item.direction == "bearish"]
                prechecks = self._build_frame_prechecks(frame)
                logger.info(
                    "Forward scan: %s/%s bars=%s",
                    symbol,
                    timeframe_label,
                    len(frame),
                )

                scan_end = len(frame) - FORWARD_BARS
                for bar in range(PRE_EXPANSION_LOOKBACK, scan_end):
                    if bar % 5000 == 0:
                        logger.info(
                            "Scan progress %s/%s bar=%s signals=%s",
                            symbol,
                            timeframe_label,
                            bar,
                            len(outcomes),
                        )
                    for direction, specs in (("bullish", bullish_specs), ("bearish", bearish_specs)):
                        if not specs:
                            continue
                        if not self._direction_candidate(bar, direction, specs, frame, prechecks):
                            continue
                        tags, measurements = self.discovery_engine.tags_at_bar(
                            frame,
                            enriched,
                            calendar,
                            intel,
                            bar,
                            direction,
                        )
                        filter_context = self._build_filter_context(
                            enriched,
                            measurements,
                            bar,
                            direction,
                        )
                        for spec in specs:
                            if signal_counts[spec.blueprint_id] >= MAX_SIGNALS_PER_BLUEPRINT:
                                continue
                            if not self._spec_precheck(spec, bar, direction, frame, prechecks):
                                continue
                            if not self._matches_blueprint(spec.required_tags, tags):
                                continue
                            key = (spec.blueprint_id, symbol, timeframe_label)
                            previous = last_signal_bar.get(key)
                            if previous is not None and bar - previous < MIN_SIGNAL_SEPARATION_BARS:
                                continue

                            forward = self._forward_validate(frame, bar, direction)
                            last_signal_bar[key] = bar
                            signal_counts[spec.blueprint_id] += 1
                            outcomes.append(
                                BlueprintSignalOutcome(
                                    symbol=symbol,
                                    timeframe=timeframe_label,
                                    timestamp=str(frame.iloc[bar]["Date"]),
                                    signal_bar=bar,
                                    blueprint_id=spec.blueprint_id,
                                    blueprint=spec.blueprint,
                                    blueprint_score=spec.blueprint_score,
                                    signal_side=spec.signal_side,
                                    direction=direction,
                                    filter_context=filter_context,
                                    **forward,
                                ),
                            )
        return outcomes

    @staticmethod
    def _aggregate_reliability(
        outcomes: list[BlueprintSignalOutcome],
        specs: list[BlueprintSpec],
    ) -> list[BlueprintReliabilityMetrics]:
        spec_lookup = {item.blueprint_id: item for item in specs}
        grouped: dict[str, list[BlueprintSignalOutcome]] = defaultdict(list)
        for outcome in outcomes:
            grouped[outcome.blueprint_id].append(outcome)

        metrics: list[BlueprintReliabilityMetrics] = []
        for blueprint_id, bucket in grouped.items():
            spec = spec_lookup[blueprint_id]
            pnls = [item.realized_pnl_points for item in bucket]
            wins = sum(1 for item in bucket if item.win)
            total = len(bucket)
            pf = InstitutionalBlueprintForwardValidationResearch._profit_factor(pnls)
            exp = round(mean(pnls), 2) if pnls else 0.0
            win_rate = round(wins / total * 100, 2) if total else 0.0
            metrics.append(
                BlueprintReliabilityMetrics(
                    blueprint_id=blueprint_id,
                    blueprint=spec.blueprint,
                    direction=spec.direction,
                    signal_side=spec.signal_side,
                    occurrences=total,
                    win_rate_pct=win_rate,
                    hit_1r_rate_pct=round(sum(1 for item in bucket if item.hit_1r) / total * 100, 2),
                    hit_2r_rate_pct=round(sum(1 for item in bucket if item.hit_2r) / total * 100, 2),
                    hit_3r_rate_pct=round(sum(1 for item in bucket if item.hit_3r) / total * 100, 2),
                    profit_factor=pf,
                    expectancy=exp,
                    average_rr=round(mean(item.realized_rr for item in bucket), 2),
                    net_points=round(sum(pnls), 2),
                    maximum_drawdown_points=InstitutionalBlueprintForwardValidationResearch._maximum_drawdown(pnls),
                    false_signal_rate_pct=round(
                        sum(1 for item in bucket if item.is_false_signal) / total * 100,
                        2,
                    ),
                    classification=InstitutionalBlueprintForwardValidationResearch._classify_blueprint(
                        total,
                        win_rate,
                        exp,
                        pf,
                    ),
                    blueprint_score=spec.blueprint_score,
                    discovery_rank=spec.discovery_rank,
                ),
            )
        return metrics

    @staticmethod
    def _filter_discovery(
        outcomes: list[BlueprintSignalOutcome],
    ) -> dict[str, Any]:
        winners = [item for item in outcomes if item.win]
        losers = [item for item in outcomes if not item.win]
        if not winners or not losers:
            return {"winning_signal_count": len(winners), "losing_signal_count": len(losers), "traits": []}

        trait_labels = {
            "rsi_below_40": "RSI < 40",
            "rsi_above_60": "RSI > 60",
            "gap_up": "Gap Up",
            "gap_down": "Gap Down",
            "liquidity_sweep": "Liquidity Sweep",
            "choch_present": "CHOCH Present",
            "bos_present": "BOS Present",
            "fvg_present": "FVG Present",
            "htf_alignment": "HTF Aligned",
        }
        rows: list[dict[str, Any]] = []
        for key, label in trait_labels.items():
            win_freq = sum(1 for item in winners if item.filter_context.get(key)) / len(winners) * 100
            lose_freq = sum(1 for item in losers if item.filter_context.get(key)) / len(losers) * 100
            edge = round(win_freq - lose_freq, 2)
            rows.append(
                {
                    "trait": label,
                    "winner_frequency_pct": round(win_freq, 2),
                    "loser_frequency_pct": round(lose_freq, 2),
                    "edge_pct": edge,
                },
            )

        categorical = ("vwap_position", "ema_alignment", "session", "level_strength_category", "premium_discount")
        for field in categorical:
            win_counter: Counter[str] = Counter()
            lose_counter: Counter[str] = Counter()
            for item in winners:
                win_counter[str(item.filter_context.get(field, "Unknown"))] += 1
            for item in losers:
                lose_counter[str(item.filter_context.get(field, "Unknown"))] += 1
            for value in set(win_counter) | set(lose_counter):
                win_freq = win_counter[value] / len(winners) * 100
                lose_freq = lose_counter[value] / len(losers) * 100
                edge = round(win_freq - lose_freq, 2)
                if abs(edge) >= 3.0:
                    rows.append(
                        {
                            "trait": f"{field}:{value}",
                            "winner_frequency_pct": round(win_freq, 2),
                            "loser_frequency_pct": round(lose_freq, 2),
                            "edge_pct": edge,
                        },
                    )

        rows.sort(key=lambda item: abs(item["edge_pct"]), reverse=True)
        return {
            "winning_signal_count": len(winners),
            "losing_signal_count": len(losers),
            "traits": rows[:30],
        }

    def run(self, metadata: dict[str, Any]) -> BlueprintForwardValidationReport:
        started = time.perf_counter()
        bullish_specs, bearish_specs = self._load_blueprints()
        all_specs = bullish_specs + bearish_specs
        outcomes = self._scan_history(metadata, all_specs)
        reliability = self._aggregate_reliability(outcomes, all_specs)
        filter_discovery = self._filter_discovery(outcomes)

        production_ready = [item.as_dict() for item in reliability if item.classification == "Production Ready"]
        needs_validation = [item.as_dict() for item in reliability if item.classification == "Needs Validation"]
        rejected = [item.as_dict() for item in reliability if item.classification == "Reject"]

        production_ready.sort(key=lambda item: item["expectancy"], reverse=True)
        needs_validation.sort(key=lambda item: item["expectancy"], reverse=True)
        rejected.sort(key=lambda item: item["occurrences"], reverse=True)

        ready_count = len(production_ready)
        validate_count = len(needs_validation)
        reject_count = len(rejected)

        top_ready = production_ready[0] if production_ready else None
        conclusions = [
            f"Prospectively scanned history for {len(all_specs)} expansion blueprints (10 bullish, 10 bearish).",
            f"Generated {len(outcomes)} blueprint signals across {self.symbols}.",
            f"Classification: {ready_count} Production Ready, {validate_count} Needs Validation, {reject_count} Reject.",
            (
                f"Top production-ready blueprint: {top_ready['blueprint_id']} "
                f"(Exp={top_ready['expectancy']}, 1R={top_ready['hit_1r_rate_pct']}%, n={top_ready['occurrences']})"
                if top_ready
                else "No blueprints reached Production Ready with n>=100."
            ),
            "Forward validation uses blueprint confirmation close entry and structural swing SL.",
        ]

        return BlueprintForwardValidationReport(
            symbols_analyzed=list(self.symbols),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            timeframes_analyzed=list(self.timeframes),
            blueprints_validated=[item.as_dict() for item in all_specs],
            total_signals=len(outcomes),
            blueprint_reliability=[item.as_dict() for item in sorted(reliability, key=lambda x: x.expectancy, reverse=True)],
            production_ready_blueprints=production_ready,
            needs_validation_blueprints=needs_validation,
            rejected_blueprints=rejected,
            filter_discovery=filter_discovery,
            sample_signals=[item.as_dict() for item in outcomes[:MAX_EXPORT_SIGNALS]],
            trade_construction={
                "entry": "Blueprint confirmation candle close",
                "stop_loss": "Structural swing SL",
                "t1": "1R",
                "t2": "2R",
                "t3": "3R",
                "t4": "Opposite liquidity pool",
            },
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_blueprint_forward_validation_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
    discovery_report_path: Path | str | None = None,
    symbols: tuple[str, ...] | None = None,
) -> BlueprintForwardValidationReport:
    """Run blueprint forward validation and export JSON."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise BlueprintForwardValidationError(f"Filter research report not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = InstitutionalBlueprintForwardValidationResearch(
        symbols=symbols,
        research_days=metadata.get("research_window_days", RESEARCH_DAYS),
        timeframes=tuple(metadata.get("timeframes_analyzed", TIMEFRAMES)),
        discovery_report_path=discovery_report_path,
    )
    report = engine.run(metadata)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info(
        "Blueprint forward validation completed: signals=%s ready=%s",
        report.total_signals,
        len(report.production_ready_blueprints),
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_blueprint_forward_validation_report()
        print("Institutional Blueprint Forward Validation Summary")
        print(f"Total signals: {report.total_signals}")
        print(f"Production Ready: {len(report.production_ready_blueprints)}")
        print(f"Needs Validation: {len(report.needs_validation_blueprints)}")
        print(f"Reject: {len(report.rejected_blueprints)}")
        if report.production_ready_blueprints:
            top = report.production_ready_blueprints[0]
            print(
                f"Top ready: {top['blueprint_id']} "
                f"(Exp={top['expectancy']}, 1R={top['hit_1r_rate_pct']}%)",
            )
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except BlueprintForwardValidationError as exc:
        logger.error("Blueprint forward validation error: %s", exc)
        print(f"Blueprint forward validation error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected blueprint forward validation error")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
