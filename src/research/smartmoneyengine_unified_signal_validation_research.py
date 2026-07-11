"""
SmartMoneyEngine Unified Signal Validation research.

Constructs one final BUY/SELL signal engine from completed research exports,
prospectively scans market history, and validates trade profitability and
signal frequency. Research-only; no production modifications.
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
from src.research.filter_research_engine import (
    FilterContextBuilder,
    FilterResearchEngine,
    RESEARCH_DAYS,
    _json_safe,
)
from src.research.institutional_blueprint_forward_validation_research import (
    InstitutionalBlueprintForwardValidationResearch,
)
from src.research.institutional_expansion_trigger_discovery_research import (
    BLUEPRINT_ARROW,
    PRE_EXPANSION_LOOKBACK,
    InstitutionalExpansionTriggerDiscoveryResearch,
)
from src.research.institutional_quality_validation_research import (
    InstitutionalQualityValidationResearch,
)
from src.research.liquidity_move_reconstruction_research import FORWARD_BARS
from src.research.tier2_production_validation_research import Tier2ProductionValidationResearch
from src.research.trade_construction_validation_research import TradeConstructionValidationResearch

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_FILTER_REPORT_PATH = RESEARCH_DIR / "filter_research_report.json"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "smartmoneyengine_unified_signal_validation.json"
FEATURE_CACHE_REPORT_PATH = RESEARCH_DIR / "smartmoneyengine_feature_cache_architecture.json"

# Empirical seconds/bar from optimized unified scan (NIFTY50/5M benchmark).
EMPIRICAL_PRECHECK_SECONDS_PER_BAR = 0.0065
EMPIRICAL_SCAN_SECONDS_PER_BAR = 0.24
EMPIRICAL_TAGS_AT_BAR_SECONDS = 0.08
EMPIRICAL_ENRICH_SECONDS_PER_FRAME = 35.0
ESTIMATED_PRECHECK_CANDIDATE_RATE = 0.18

MIN_SIGNAL_SEPARATION_BARS = 20
MAX_EXPORT_SIGNALS = 200
MIN_LEVEL_TESTS = 3
MIN_FALSE_MOVES = 3
MIN_REJECTION_WICKS = 5
SCAN_PROGRESS_INTERVAL = 1000
STALL_THRESHOLD_SECONDS = 600
SIGNAL_VOLUME_THRESHOLDS = (20, 30, 40, 50)
HIGH_QUALITY_MIN_HIT_1R_PCT = 60.0
HIGH_QUALITY_MIN_PROFIT_FACTOR = 1.5
HIGH_QUALITY_MIN_EXPECTANCY = 50.0

DEFAULT_SYMBOLS = ("NIFTY50", "BANKNIFTY", "FINNIFTY")
TIMEFRAMES = ("5M", "15M", "1H")

RESEARCH_INPUTS = (
    "institutional_expansion_trigger_discovery.json",
    "institutional_move_dna.json",
    "trigger_trade_validation.json",
    "trigger_entry_optimization.json",
    "smartmoneyengine_production_candidate.json",
    "tier2_composite_edge_validation.json",
)

BUY_BLUEPRINT_TEXT = (
    "Liquidity Grab"
    f"{BLUEPRINT_ARROW}Failed Breakdown"
    f"{BLUEPRINT_ARROW}Level Tests >= 3"
    f"{BLUEPRINT_ARROW}Displacement:Weak"
    f"{BLUEPRINT_ARROW}CHOCH"
    f"{BLUEPRINT_ARROW}BOS"
    f"{BLUEPRINT_ARROW}Zone:Discount"
    f"{BLUEPRINT_ARROW}Confirmation Candle"
)
SELL_BLUEPRINT_TEXT = (
    "False Moves x3+"
    f"{BLUEPRINT_ARROW}Heavy Rejection Wicks"
    f"{BLUEPRINT_ARROW}Displacement:Weak"
    f"{BLUEPRINT_ARROW}CHOCH"
    f"{BLUEPRINT_ARROW}BOS"
    f"{BLUEPRINT_ARROW}Zone:Premium"
    f"{BLUEPRINT_ARROW}Confirmation Candle"
)

BUY_REQUIRED_TAGS = (
    "Liquidity Grab",
    "Failed Breakdown",
    "Displacement:Weak",
    "CHOCH",
    "BOS",
    "Zone:Discount",
)
SELL_REQUIRED_TAGS = (
    "False Moves x3+",
    "Heavy Rejection Wicks",
    "Displacement:Weak",
    "CHOCH",
    "BOS",
    "Zone:Premium",
)


class UnifiedSignalValidationError(Exception):
    """Raised when unified signal validation fails."""


@dataclass(frozen=True)
class RuntimeEstimate:
    """Pre-run runtime projection for unified signal validation."""

    total_seconds: float
    total_frames: int
    total_scan_bars: int
    expected_signals_hint: int | None
    per_frame_estimates: list[dict[str, Any]]
    complexity_risks: list[dict[str, str]]
    export_reuse: list[dict[str, Any]]
    features_rebuilt: list[str]
    features_avoided: list[str]
    estimation_source: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UnifiedSignalBlueprint:
    """One unified signal engine blueprint."""

    blueprint_id: str
    blueprint: str
    direction: str
    signal_side: str
    required_tags: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UnifiedSignalOutcome:
    """Forward outcome for one unified engine signal."""

    symbol: str
    timeframe: str
    timestamp: str
    signal_bar: int
    blueprint_id: str
    blueprint: str
    signal_side: str
    direction: str
    entry_price: float
    stop_price: float
    target_1r: float
    target_2r: float
    target_3r: float
    risk_points: float
    hit_1r: bool
    hit_2r: bool
    hit_3r: bool
    stop_hit: bool
    realized_pnl_points: float
    realized_rr: float
    win: bool
    high_quality: bool
    filter_context: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UnifiedSignalMetrics:
    """Aggregate metrics for one scope (overall, symbol, side)."""

    scope: str
    signal_side: str | None
    total_signals: int
    signals_per_month: float
    signals_per_week: float
    hit_1r_rate_pct: float
    hit_2r_rate_pct: float
    hit_3r_rate_pct: float
    win_rate_pct: float
    profit_factor: float | None
    expectancy: float
    average_rr: float
    net_points: float
    maximum_drawdown_points: float
    high_quality_signals: int
    high_quality_rate_pct: float
    classification: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UnifiedSignalValidationReport:
    """Full unified signal validation output."""

    symbols_analyzed: list[str]
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    research_inputs_loaded: dict[str, Any]
    unified_signal_engine: dict[str, Any]
    buy_blueprint: dict[str, Any]
    sell_blueprint: dict[str, Any]
    total_signals: int
    overall_metrics: dict[str, Any]
    buy_metrics: dict[str, Any]
    sell_metrics: dict[str, Any]
    per_symbol_metrics: list[dict[str, Any]]
    per_symbol_buy_metrics: list[dict[str, Any]]
    per_symbol_sell_metrics: list[dict[str, Any]]
    monthly_signal_frequency: list[dict[str, Any]]
    weekly_signal_frequency: list[dict[str, Any]]
    signal_volume_assessment: dict[str, Any]
    trade_construction: dict[str, str]
    sample_signals: list[dict[str, Any]]
    research_synthesis: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SmartMoneyEngineUnifiedSignalValidationResearch:
    """Validate the unified BUY/SELL signal engine on full market history."""

    def __init__(
        self,
        symbols: tuple[str, ...] | None = None,
        research_days: int = RESEARCH_DAYS,
        timeframes: tuple[str, ...] = TIMEFRAMES,
        research_dir: Path | str | None = None,
    ) -> None:
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.research_days = research_days
        self.timeframes = timeframes
        self.research_dir = Path(research_dir or RESEARCH_DIR)
        self.discovery_engine = InstitutionalExpansionTriggerDiscoveryResearch(
            symbols=self.symbols,
            research_days=research_days,
            timeframes=timeframes,
        )
        self.trade_engine = TradeConstructionValidationResearch(
            research_days=research_days,
            timeframes=timeframes,
        )
        self.scan_helper = InstitutionalBlueprintForwardValidationResearch(
            symbols=self.symbols,
            research_days=research_days,
            timeframes=timeframes,
        )
        self.context_builder = FilterContextBuilder()
        self.buy_blueprint = UnifiedSignalBlueprint(
            blueprint_id="unified_buy",
            blueprint=BUY_BLUEPRINT_TEXT,
            direction="bullish",
            signal_side="BUY",
            required_tags=BUY_REQUIRED_TAGS,
        )
        self.sell_blueprint = UnifiedSignalBlueprint(
            blueprint_id="unified_sell",
            blueprint=SELL_BLUEPRINT_TEXT,
            direction="bearish",
            signal_side="SELL",
            required_tags=SELL_REQUIRED_TAGS,
        )

    @staticmethod
    def _profit_factor(pnls: list[float]) -> float | None:
        return InstitutionalQualityValidationResearch._profit_factor(pnls)

    @staticmethod
    def _maximum_drawdown(pnls: list[float]) -> float:
        return Tier2ProductionValidationResearch._maximum_drawdown(pnls)

    @staticmethod
    def _matches_tags(required_tags: tuple[str, ...], active_tags: tuple[str, ...]) -> bool:
        return all(tag in set(active_tags) for tag in required_tags)

    @staticmethod
    def _is_confirmation_candle(trigger: dict[str, Any], direction: str) -> bool:
        patterns = (
            trigger.get("engulfing"),
            trigger.get("marubozu"),
            trigger.get("hammer") and direction == "bullish",
            trigger.get("shooting_star") and direction == "bearish",
            trigger.get("morning_star") and direction == "bullish",
            trigger.get("evening_star") and direction == "bearish",
            trigger.get("bullish_harami") and direction == "bullish",
            trigger.get("bearish_harami") and direction == "bearish",
        )
        if any(patterns):
            return True
        body_pct = float(trigger.get("body_pct", 0.0))
        volume_ratio = float(trigger.get("volume_expansion_ratio", 1.0))
        if direction == "bullish":
            return body_pct >= 55.0 and volume_ratio >= 1.1
        return body_pct >= 55.0 and volume_ratio >= 1.1

    def _matches_unified_blueprint(
        self,
        blueprint: UnifiedSignalBlueprint,
        tags: tuple[str, ...],
        measurements: dict[str, Any],
    ) -> bool:
        if not self._matches_tags(blueprint.required_tags, tags):
            return False
        sr = measurements["support_resistance"]
        trigger = measurements["expansion_trigger_candle"]
        if blueprint.direction == "bullish":
            if int(sr.get("number_of_tests", 0)) < MIN_LEVEL_TESTS:
                return False
        if not self._is_confirmation_candle(trigger, blueprint.direction):
            return False
        return True

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(seconds, 0.0)
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes}m {secs}s"
        if minutes:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

    @staticmethod
    def _fast_frame_bar_count(path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as handle:
            return max(sum(1 for _ in handle) - 1, 0)

    def _load_feature_cache_benchmarks(self) -> dict[str, float]:
        defaults = {
            "market_level_per_bar": EMPIRICAL_PRECHECK_SECONDS_PER_BAR,
            "scan_per_bar": EMPIRICAL_SCAN_SECONDS_PER_BAR,
            "tags_at_bar": EMPIRICAL_TAGS_AT_BAR_SECONDS,
            "enrich_per_frame": EMPIRICAL_ENRICH_SECONDS_PER_FRAME,
        }
        if not FEATURE_CACHE_REPORT_PATH.exists():
            return defaults
        with FEATURE_CACHE_REPORT_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        benchmarks = {
            item["category"]: item for item in payload.get("category_benchmarks", [])
        }
        market = benchmarks.get("market_level", {})
        candle = benchmarks.get("candle_pattern", {})
        liquidity = benchmarks.get("liquidity", {})
        market_invocations = max(market.get("invocations_per_benchmark", 1), 1)
        candle_invocations = max(candle.get("invocations_per_benchmark", 1), 1)
        return {
            "market_level_per_bar": market.get("runtime_seconds", 0.0) / market_invocations,
            "scan_per_bar": EMPIRICAL_SCAN_SECONDS_PER_BAR,
            "tags_at_bar": candle.get("runtime_seconds", 0.0) / candle_invocations,
            "enrich_per_frame": (
                liquidity.get("runtime_seconds", 0.0) / max(liquidity.get("invocations_per_benchmark", 1), 1)
            )
            + EMPIRICAL_ENRICH_SECONDS_PER_FRAME * 0.35,
            "estimation_source": "smartmoneyengine_feature_cache_architecture.json",
        }

    def _complexity_risks(self, total_scan_bars: int) -> list[dict[str, str]]:
        return [
            {
                "pattern": "O(N) full-bar precheck build",
                "location": "_build_unified_extended_prechecks",
                "detail": (
                    f"Calls _market_levels once per bar ({total_scan_bars:,} bars projected). "
                    "Dominant setup cost."
                ),
            },
            {
                "pattern": "O(N) prospective bar scan",
                "location": "_scan_history",
                "detail": (
                    f"Iterates every bar across {len(self.symbols)} symbols × {len(self.timeframes)} timeframes."
                ),
            },
            {
                "pattern": "O(N × k) tags_at_bar",
                "location": "InstitutionalExpansionTriggerDiscoveryResearch.tags_at_bar",
                "detail": (
                    f"100-bar fused measurement loop on ~{int(ESTIMATED_PRECHECK_CANDIDATE_RATE * 100)}% "
                    "of bars passing blueprint precheck."
                ),
            },
            {
                "pattern": "Repeated full-bar enrich",
                "location": "FilterContextBuilder.enrich + calendar + intel",
                "detail": "Rebuilt per symbol/timeframe despite completed research exports.",
            },
        ]

    def _export_reuse_plan(self, loaded_inputs: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        reuse: list[dict[str, Any]] = []
        avoided: list[str] = []
        rebuilt: list[str] = [
            "pipeline_csv_load",
            "filter_context_enrich",
            "calendar_level_attach",
            "market_intelligence_enrich",
            "extended_precheck_arrays",
            "prospective_bar_scan",
        ]

        for filename, summary in loaded_inputs.items():
            if summary.get("status") != "loaded":
                continue
            if filename == "institutional_expansion_trigger_discovery.json":
                reuse.append(
                    {
                        "export": filename,
                        "reuse": "Blueprint tag vocabulary + top blueprint alignment metadata",
                        "avoids": "Re-running expansion discovery move analysis",
                    },
                )
                avoided.append("institutional_expansion_trigger_discovery_research full move scan")
            elif filename == "institutional_move_dna.json":
                reuse.append(
                    {
                        "export": filename,
                        "reuse": "Move DNA trait context for synthesis only",
                        "avoids": "Re-running move DNA measurement pass",
                    },
                )
                avoided.append("institutional_move_dna_research full move scan")
            elif filename == "trigger_trade_validation.json":
                reuse.append(
                    {
                        "export": filename,
                        "reuse": "Validated trigger trade construction benchmarks",
                        "avoids": "Re-simulating trigger trade matrix",
                    },
                )
                avoided.append("trigger_trade_validation_research full replay")
            elif filename == "trigger_entry_optimization.json":
                reuse.append(
                    {
                        "export": filename,
                        "reuse": f"Best entry method: {summary.get('best_overall_entry')}",
                        "avoids": "Re-running 8 entry resolver comparisons",
                    },
                )
                avoided.append("trigger_entry_optimization_research full replay")
            elif filename == "smartmoneyengine_production_candidate.json":
                reuse.append(
                    {
                        "export": filename,
                        "reuse": "Production candidate engine recommendation",
                        "avoids": "Re-synthesizing production candidate models",
                    },
                )
                avoided.append("smartmoneyengine_production_candidate_research")
            elif filename == "tier2_composite_edge_validation.json":
                reuse.append(
                    {
                        "export": filename,
                        "reuse": f"Best filter: {summary.get('best_production_ready_filter')}",
                        "avoids": "Re-running tier-2 composite trait replay",
                    },
                )
                avoided.append("tier2_composite_edge_validation_research")

        if FEATURE_CACHE_REPORT_PATH.exists():
            reuse.append(
                {
                    "export": FEATURE_CACHE_REPORT_PATH.name,
                    "reuse": "Per-category runtime benchmarks for pre-run estimation",
                    "avoids": "Blind runtime planning",
                },
            )

        return reuse, avoided, rebuilt

    def estimate_runtime(self, metadata: dict[str, Any]) -> RuntimeEstimate:
        """Estimate runtime and complexity before starting the prospective scan."""
        loaded = self._load_research_inputs()
        benchmarks = self._load_feature_cache_benchmarks()
        end = date.fromisoformat(metadata["end_date"])
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=self.research_days)
        )

        per_frame: list[dict[str, Any]] = []
        total_scan_bars = 0
        total_seconds = 0.0
        expected_signals_hint: int | None = None

        discovery_path = self.research_dir / "institutional_expansion_trigger_discovery.json"
        if discovery_path.exists():
            with discovery_path.open("r", encoding="utf-8") as handle:
                discovery = json.load(handle)
            expected_signals_hint = discovery.get("total_moves_analyzed")

        for symbol in self.symbols:
            filter_engine = FilterResearchEngine(
                symbol=symbol,
                research_days=self.research_days,
                timeframes=self.timeframes,
            )
            for timeframe_label in self.timeframes:
                path = filter_engine._pipeline_path(timeframe_label)
                pipeline_exists = path.exists()
                bars = self._fast_frame_bar_count(path)
                scan_bars = max(bars - PRE_EXPANSION_LOOKBACK - FORWARD_BARS, 0)
                total_scan_bars += scan_bars

                precheck_seconds = scan_bars * benchmarks["market_level_per_bar"]
                enrich_seconds = benchmarks["enrich_per_frame"] if pipeline_exists else benchmarks["enrich_per_frame"] + 120.0
                candidate_bars = int(scan_bars * ESTIMATED_PRECHECK_CANDIDATE_RATE)
                tags_seconds = candidate_bars * benchmarks["tags_at_bar"] * 2
                loop_seconds = scan_bars * benchmarks["scan_per_bar"]
                frame_seconds = precheck_seconds + enrich_seconds + loop_seconds + tags_seconds
                total_seconds += frame_seconds

                per_frame.append(
                    {
                        "symbol": symbol,
                        "timeframe": timeframe_label,
                        "bars": bars,
                        "scan_bars": scan_bars,
                        "pipeline_exists": pipeline_exists,
                        "estimated_seconds": round(frame_seconds, 1),
                        "estimated_duration": self._format_duration(frame_seconds),
                        "breakdown_seconds": {
                            "enrich": round(enrich_seconds, 1),
                            "precheck_build": round(precheck_seconds, 1),
                            "bar_scan_loop": round(loop_seconds, 1),
                            "tags_at_bar": round(tags_seconds, 1),
                        },
                    },
                )

        reuse, avoided, rebuilt = self._export_reuse_plan(loaded)
        source = benchmarks.get("estimation_source", "empirical_defaults")
        return RuntimeEstimate(
            total_seconds=round(total_seconds, 1),
            total_frames=len(per_frame),
            total_scan_bars=total_scan_bars,
            expected_signals_hint=expected_signals_hint,
            per_frame_estimates=per_frame,
            complexity_risks=self._complexity_risks(total_scan_bars),
            export_reuse=reuse,
            features_rebuilt=rebuilt,
            features_avoided=avoided,
            estimation_source=source,
        )

    def print_runtime_estimate(self, estimate: RuntimeEstimate) -> None:
        """Print pre-run runtime plan to stdout."""
        print("=" * 72)
        print("SmartMoneyEngine Unified Signal Validation — Runtime Estimate")
        print("=" * 72)
        print(f"Estimation source: {estimate.estimation_source}")
        print(
            f"Projected runtime: {self._format_duration(estimate.total_seconds)} "
            f"({estimate.total_seconds:.0f}s)",
        )
        print(
            f"Frames: {estimate.total_frames} | Scan bars: {estimate.total_scan_bars:,}",
        )
        if estimate.expected_signals_hint is not None:
            print(
                f"Expansion discovery reference moves: {estimate.expected_signals_hint} "
                "(export reused; not rescanned)",
            )
        print("\nPer frame:")
        for item in estimate.per_frame_estimates:
            print(
                f"  {item['symbol']}/{item['timeframe']}: {item['estimated_duration']} "
                f"({item['scan_bars']:,} bars, pipeline={'cached' if item['pipeline_exists'] else 'build'})",
            )
        print("\nComplexity risks:")
        for risk in estimate.complexity_risks:
            print(f"  [{risk['pattern']}] {risk['location']}")
            print(f"    {risk['detail']}")
        print("\nExports reused (no rebuild):")
        for item in estimate.export_reuse:
            print(f"  - {item['export']}: {item['avoids']}")
        print("\nFeatures still rebuilt this run:")
        for feature in estimate.features_rebuilt:
            print(f"  - {feature}")
        print("=" * 72, flush=True)

    def _build_unified_extended_prechecks(self, frame: pd.DataFrame) -> dict[str, Any]:
        """Precompute rolling counters to avoid expensive tags_at_bar on non-candidates."""
        length = len(frame)
        false_moves = np.zeros(length, dtype=np.int8)
        rejection_wicks = np.zeros(length, dtype=np.int8)
        failed_breakdowns = np.zeros(length, dtype=np.int8)
        zone_discount = np.zeros(length, dtype=bool)
        zone_premium = np.zeros(length, dtype=bool)

        discovery = self.discovery_engine
        precheck_started = time.perf_counter()
        for index in range(length):
            if index % SCAN_PROGRESS_INTERVAL == 0 and index > 0:
                elapsed = time.perf_counter() - precheck_started
                remaining = (elapsed / index) * (length - index) if index else 0.0
                message = (
                    f"Precheck build {index}/{length} ({round(index / length * 100, 1)}%) | "
                    f"elapsed={self._format_duration(elapsed)} | "
                    f"eta={self._format_duration(remaining)} | "
                    f"bottleneck=_market_levels"
                )
                logger.info(message)
                print(message, flush=True)
            row = frame.iloc[index]
            atr = discovery._atr(frame, index)
            parts = discovery._candle_parts(row)
            total_wick = parts["upper_wick"] + parts["lower_wick"]
            if total_wick >= atr * 0.35:
                rejection_wicks[index] = 1

            if index >= 20:
                window = frame.iloc[index - 20 : index]
                prior_high = float(window["High"].astype(float).max())
                prior_low = float(window["Low"].astype(float).min())
                high = float(row["High"])
                low = float(row["Low"])
                close = float(row["Close"])
                if high > prior_high and close < prior_high:
                    false_moves[index] = 1
                if low < prior_low and close > prior_low:
                    false_moves[index] = 1

            levels = discovery._market_levels(frame, index)
            support = levels.get("major_support")
            resistance = levels.get("major_resistance")
            close = float(row["Close"])
            if support is not None and resistance is not None:
                midpoint = (support + resistance) / 2
                zone_discount[index] = close <= midpoint
                zone_premium[index] = close > midpoint
            elif support is not None:
                zone_discount[index] = True
            elif resistance is not None:
                zone_premium[index] = True

            if support is not None:
                low = float(row["Low"])
                if low < support and close >= support:
                    failed_breakdowns[index] = 1

        prechecks = self.scan_helper._build_frame_prechecks(frame)
        prechecks["false_move_cumsum"] = np.cumsum(false_moves)
        prechecks["rejection_wick_cumsum"] = np.cumsum(rejection_wicks)
        prechecks["failed_breakdown_cumsum"] = np.cumsum(failed_breakdowns)
        prechecks["zone_discount"] = zone_discount
        prechecks["zone_premium"] = zone_premium
        return prechecks

    def _blueprint_precheck(
        self,
        blueprint: UnifiedSignalBlueprint,
        bar: int,
        frame: pd.DataFrame,
        prechecks: dict[str, Any],
    ) -> bool:
        direction = blueprint.direction
        required = set(blueprint.required_tags)
        displacement_tag = InstitutionalBlueprintForwardValidationResearch._displacement_tag(
            frame,
            bar,
            direction,
        )
        if "Displacement:Weak" in required and displacement_tag != "Displacement:Weak":
            return False
        window = PRE_EXPANSION_LOOKBACK
        if "BOS" in required and self.scan_helper._window_count(prechecks["bos_cumsum"], bar, window) == 0:
            return False
        if "CHOCH" in required and self.scan_helper._window_count(prechecks["choch_cumsum"], bar, window) == 0:
            return False
        if "Liquidity Grab" in required and self.scan_helper._window_count(prechecks["sweep_cumsum"], bar, window) == 0:
            return False
        if blueprint.direction == "bullish":
            if "Failed Breakdown" in required and self.scan_helper._window_count(
                prechecks["failed_breakdown_cumsum"],
                bar,
                window,
            ) < 1:
                return False
            if "Zone:Discount" in required and not bool(prechecks["zone_discount"][bar]):
                return False
        if blueprint.direction == "bearish":
            if "False Moves x3+" in required and self.scan_helper._window_count(
                prechecks["false_move_cumsum"],
                bar,
                window,
            ) < MIN_FALSE_MOVES:
                return False
            if "Heavy Rejection Wicks" in required and self.scan_helper._window_count(
                prechecks["rejection_wick_cumsum"],
                bar,
                window,
            ) < MIN_REJECTION_WICKS:
                return False
            if "Zone:Premium" in required and not bool(prechecks["zone_premium"][bar]):
                return False
        return True

    @staticmethod
    def _is_high_quality(outcome: UnifiedSignalOutcome) -> bool:
        return outcome.hit_1r and outcome.realized_pnl_points > 0

    @staticmethod
    def _classify_engine(
        signals: int,
        win_rate_pct: float,
        expectancy: float,
        profit_factor: float | None,
        hit_1r_rate_pct: float,
    ) -> str:
        if signals < 20:
            return "Reject"
        if expectancy < 0:
            return "Reject"
        if (
            win_rate_pct >= 40.0
            and expectancy >= HIGH_QUALITY_MIN_EXPECTANCY
            and profit_factor is not None
            and profit_factor >= HIGH_QUALITY_MIN_PROFIT_FACTOR
            and hit_1r_rate_pct >= HIGH_QUALITY_MIN_HIT_1R_PCT
        ):
            return "Production Ready"
        if expectancy > 0:
            return "Needs Validation"
        return "Reject"

    def _load_research_inputs(self) -> dict[str, Any]:
        loaded: dict[str, Any] = {}
        for filename in RESEARCH_INPUTS:
            path = self.research_dir / filename
            if not path.exists():
                loaded[filename] = {"status": "missing", "path": str(path)}
                continue
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            summary: dict[str, Any] = {"status": "loaded", "path": str(path)}
            if filename == "institutional_expansion_trigger_discovery.json":
                summary["total_moves_analyzed"] = payload.get("total_moves_analyzed")
                top_bull = payload.get("top_20_bullish_momentum_blueprints") or []
                top_bear = payload.get("top_20_bearish_momentum_blueprints") or []
                summary["top_bullish_blueprint"] = top_bull[0]["blueprint"] if top_bull else None
                summary["top_bearish_blueprint"] = top_bear[0]["blueprint"] if top_bear else None
            elif filename == "institutional_move_dna.json":
                summary["move_dna_records_total"] = payload.get("move_dna_records_total")
            elif filename == "trigger_trade_validation.json":
                summary["trades_simulated"] = payload.get("trades_simulated")
                matrix = payload.get("production_trigger_matrix") or []
                summary["production_ready_triggers"] = len(matrix)
            elif filename == "trigger_entry_optimization.json":
                summary["best_overall_entry"] = payload.get("best_overall_entry")
            elif filename == "smartmoneyengine_production_candidate.json":
                summary["eligible_models"] = len(payload.get("eligible_models") or [])
                summary["recommended_engine"] = payload.get("recommended_production_signal_engine")
            elif filename == "tier2_composite_edge_validation.json":
                summary["best_production_ready_filter"] = payload.get("best_production_ready_filter")
            loaded[filename] = summary
        return loaded

    def _synthesize_research_context(self, loaded: dict[str, Any]) -> dict[str, Any]:
        return {
            "inputs_available": sum(1 for item in loaded.values() if item.get("status") == "loaded"),
            "inputs_missing": [name for name, item in loaded.items() if item.get("status") == "missing"],
            "unified_buy_alignment": (
                "BUY blueprint aligns with top discovery patterns: Failed Breakdown, "
                "Liquidity Grab, Zone:Discount, Displacement:Weak."
            ),
            "unified_sell_alignment": (
                "SELL blueprint aligns with top discovery patterns: False Moves x3+, "
                "Heavy Rejection Wicks, Displacement:Weak."
            ),
            "trade_construction_source": "Structural swing SL + 1R/2R/3R targets (V1-aligned).",
            "entry_method": "Confirmation candle close at signal bar.",
        }

    def _forward_validate(
        self,
        frame: pd.DataFrame,
        signal_bar: int,
        direction: str,
    ) -> dict[str, Any]:
        forward = self.scan_helper._forward_validate(frame, signal_bar, direction)
        return {
            "entry_price": forward["entry_price"],
            "stop_price": forward["stop_price"],
            "target_1r": forward["target_1r"],
            "target_2r": forward["target_2r"],
            "target_3r": forward["target_3r"],
            "risk_points": forward["risk_points"],
            "hit_1r": forward["hit_1r"],
            "hit_2r": forward["hit_2r"],
            "hit_3r": forward["hit_3r"],
            "stop_hit": forward["stop_hit"],
            "realized_pnl_points": forward["realized_pnl_points"],
            "realized_rr": forward["realized_rr"],
            "win": forward["win"],
        }

    def _log_scan_progress(
        self,
        *,
        symbol: str,
        timeframe_label: str,
        bar: int,
        scan_end: int,
        signals: int,
        scan_started: float,
        profile_seconds: dict[str, float],
    ) -> None:
        elapsed = time.perf_counter() - scan_started
        bars_done = max(bar - PRE_EXPANSION_LOOKBACK, 1)
        bars_total = max(scan_end - PRE_EXPANSION_LOOKBACK, 1)
        remaining_bars = max(bars_total - bars_done, 0)
        eta_seconds = (elapsed / bars_done) * remaining_bars if bars_done else 0.0
        pct = round(bars_done / bars_total * 100, 1)
        bottleneck = max(profile_seconds, key=profile_seconds.get) if profile_seconds else "none"
        logger.info(
            "Unified progress %s/%s | bar=%s/%s (%s%%) | signals=%s | elapsed=%s | eta=%s | bottleneck=%s",
            symbol,
            timeframe_label,
            bar,
            scan_end,
            pct,
            signals,
            self._format_duration(elapsed),
            self._format_duration(eta_seconds),
            bottleneck,
        )
        print(
            f"Unified progress {symbol}/{timeframe_label} | "
            f"bar={bar}/{scan_end} ({pct}%) | signals={signals} | "
            f"elapsed={self._format_duration(elapsed)} | "
            f"eta={self._format_duration(eta_seconds)}",
            flush=True,
        )

    def _log_scan_stall(
        self,
        *,
        symbol: str,
        timeframe_label: str,
        bar: int,
        signals: int,
        stalled_seconds: float,
        profile_seconds: dict[str, float],
    ) -> None:
        bottleneck = max(profile_seconds, key=profile_seconds.get, default="tags_at_bar")
        logger.warning(
            "Unified scan stalled %s/%s bar=%s signals=%s stalled_for=%s bottleneck=%s profile=%s",
            symbol,
            timeframe_label,
            bar,
            signals,
            self._format_duration(stalled_seconds),
            bottleneck,
            {key: round(value, 2) for key, value in sorted(profile_seconds.items(), key=lambda item: item[1], reverse=True)},
        )
        print(
            f"STALL DETECTED {symbol}/{timeframe_label} | bar={bar} | signals={signals} | "
            f"stalled_for={self._format_duration(stalled_seconds)} | "
            f"bottleneck={bottleneck}",
            flush=True,
        )

    def _scan_history(self, metadata: dict[str, Any]) -> list[UnifiedSignalOutcome]:
        end = date.fromisoformat(metadata["end_date"])
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=self.research_days)
        )
        outcomes: list[UnifiedSignalOutcome] = []
        last_signal_bar: dict[tuple[str, str, str], int] = {}

        for symbol in self.symbols:
            filter_engine = FilterResearchEngine(
                symbol=symbol,
                research_days=self.research_days,
                timeframes=self.timeframes,
            )
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
                prechecks = self._build_unified_extended_prechecks(frame)
                logger.info("Unified scan: %s/%s bars=%s", symbol, timeframe_label, len(frame))
                print(f"Unified scan started: {symbol}/{timeframe_label} bars={len(frame)}", flush=True)

                scan_end = len(frame) - FORWARD_BARS
                tag_cache: dict[tuple[int, str], tuple[tuple[str, ...], dict[str, Any]]] = {}
                scan_started = time.perf_counter()
                last_activity = scan_started
                profile_seconds: dict[str, float] = defaultdict(float)
                for bar in range(PRE_EXPANSION_LOOKBACK, scan_end):
                    now = time.perf_counter()
                    if now - last_activity >= STALL_THRESHOLD_SECONDS:
                        self._log_scan_stall(
                            symbol=symbol,
                            timeframe_label=timeframe_label,
                            bar=bar,
                            signals=len(outcomes),
                            stalled_seconds=now - last_activity,
                            profile_seconds=profile_seconds,
                        )
                        last_activity = now
                        profile_seconds.clear()

                    if bar % SCAN_PROGRESS_INTERVAL == 0:
                        self._log_scan_progress(
                            symbol=symbol,
                            timeframe_label=timeframe_label,
                            bar=bar,
                            scan_end=scan_end,
                            signals=len(outcomes),
                            scan_started=scan_started,
                            profile_seconds=profile_seconds,
                        )
                        profile_seconds.clear()

                    precheck_started = time.perf_counter()
                    buy_ready = self._blueprint_precheck(self.buy_blueprint, bar, frame, prechecks)
                    sell_ready = self._blueprint_precheck(self.sell_blueprint, bar, frame, prechecks)
                    profile_seconds["blueprint_precheck"] += time.perf_counter() - precheck_started
                    if not buy_ready and not sell_ready:
                        last_activity = time.perf_counter()
                        continue
                    active_blueprints = (
                        [self.buy_blueprint] if buy_ready else []
                    ) + ([self.sell_blueprint] if sell_ready else [])
                    for blueprint in active_blueprints:
                        cache_key = (bar, blueprint.direction)
                        if cache_key not in tag_cache:
                            tags_started = time.perf_counter()
                            tag_cache[cache_key] = self.discovery_engine.tags_at_bar(
                                frame,
                                enriched,
                                calendar,
                                intel,
                                bar,
                                blueprint.direction,
                            )
                            profile_seconds["tags_at_bar"] += time.perf_counter() - tags_started
                        tags, measurements = tag_cache[cache_key]
                        match_started = time.perf_counter()
                        matched = self._matches_unified_blueprint(blueprint, tags, measurements)
                        profile_seconds["blueprint_match"] += time.perf_counter() - match_started
                        if not matched:
                            continue
                        key = (blueprint.blueprint_id, symbol, timeframe_label)
                        previous = last_signal_bar.get(key)
                        if previous is not None and bar - previous < MIN_SIGNAL_SEPARATION_BARS:
                            continue

                        forward_started = time.perf_counter()
                        forward = self._forward_validate(frame, bar, blueprint.direction)
                        profile_seconds["forward_validate"] += time.perf_counter() - forward_started
                        filter_started = time.perf_counter()
                        filter_context = self.scan_helper._build_filter_context(
                            enriched,
                            measurements,
                            bar,
                            blueprint.direction,
                        )
                        profile_seconds["build_filter_context"] += time.perf_counter() - filter_started
                        last_signal_bar[key] = bar
                        outcomes.append(
                            UnifiedSignalOutcome(
                                symbol=symbol,
                                timeframe=timeframe_label,
                                timestamp=str(frame.iloc[bar]["Date"]),
                                signal_bar=bar,
                                blueprint_id=blueprint.blueprint_id,
                                blueprint=blueprint.blueprint,
                                signal_side=blueprint.signal_side,
                                direction=blueprint.direction,
                                filter_context=filter_context,
                                high_quality=forward["hit_1r"] and forward["realized_pnl_points"] > 0,
                                **forward,
                            ),
                        )
                    last_activity = time.perf_counter()
                self._log_scan_progress(
                    symbol=symbol,
                    timeframe_label=timeframe_label,
                    bar=scan_end,
                    scan_end=scan_end,
                    signals=len(outcomes),
                    scan_started=scan_started,
                    profile_seconds=profile_seconds,
                )
        return outcomes

    @staticmethod
    def _research_months(start_date: str, end_date: str, fallback_days: int) -> float:
        if start_date and end_date:
            start = date.fromisoformat(start_date)
            end = date.fromisoformat(end_date)
            days = max((end - start).days, 1)
            return max(days / 30.4375, 1.0)
        return max(fallback_days / 30.4375, 1.0)

    @staticmethod
    def _research_weeks(start_date: str, end_date: str, fallback_days: int) -> float:
        if start_date and end_date:
            start = date.fromisoformat(start_date)
            end = date.fromisoformat(end_date)
            days = max((end - start).days, 1)
            return max(days / 7.0, 1.0)
        return max(fallback_days / 7.0, 1.0)

    @staticmethod
    def _timestamp_bucket(timestamps: list[str], freq: str) -> list[dict[str, Any]]:
        counter: Counter[str] = Counter()
        for value in timestamps:
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.isna(parsed):
                continue
            if freq == "month":
                counter[parsed.strftime("%Y-%m")] += 1
            else:
                iso = parsed.isocalendar()
                counter[f"{iso.year}-W{iso.week:02d}"] += 1
        return [
            {"period": period, "signals": count}
            for period, count in sorted(counter.items())
        ]

    def _aggregate_metrics(
        self,
        outcomes: list[UnifiedSignalOutcome],
        scope: str,
        signal_side: str | None,
        months: float,
        weeks: float,
    ) -> UnifiedSignalMetrics:
        if signal_side is not None:
            bucket = [item for item in outcomes if item.signal_side == signal_side]
        else:
            bucket = outcomes
        total = len(bucket)
        pnls = [item.realized_pnl_points for item in bucket]
        wins = sum(1 for item in bucket if item.win)
        high_quality = sum(1 for item in bucket if item.high_quality)
        pf = self._profit_factor(pnls)
        exp = round(mean(pnls), 2) if pnls else 0.0
        win_rate = round(wins / total * 100, 2) if total else 0.0
        hit_1r = round(sum(1 for item in bucket if item.hit_1r) / total * 100, 2) if total else 0.0
        hit_2r = round(sum(1 for item in bucket if item.hit_2r) / total * 100, 2) if total else 0.0
        hit_3r = round(sum(1 for item in bucket if item.hit_3r) / total * 100, 2) if total else 0.0
        return UnifiedSignalMetrics(
            scope=scope,
            signal_side=signal_side,
            total_signals=total,
            signals_per_month=round(total / months, 2) if total else 0.0,
            signals_per_week=round(total / weeks, 2) if total else 0.0,
            hit_1r_rate_pct=hit_1r,
            hit_2r_rate_pct=hit_2r,
            hit_3r_rate_pct=hit_3r,
            win_rate_pct=win_rate,
            profit_factor=pf,
            expectancy=exp,
            average_rr=round(mean(item.realized_rr for item in bucket), 2) if bucket else 0.0,
            net_points=round(sum(pnls), 2),
            maximum_drawdown_points=self._maximum_drawdown(pnls),
            high_quality_signals=high_quality,
            high_quality_rate_pct=round(high_quality / total * 100, 2) if total else 0.0,
            classification=self._classify_engine(total, win_rate, exp, pf, hit_1r),
        )

    def _assess_signal_volume(
        self,
        overall: UnifiedSignalMetrics,
        per_symbol: list[UnifiedSignalMetrics],
    ) -> dict[str, Any]:
        assessments: dict[str, Any] = {}
        for threshold in SIGNAL_VOLUME_THRESHOLDS:
            key = f"{threshold}_plus_per_month"
            overall_meets = overall.signals_per_month >= threshold
            profitable = (
                overall.expectancy >= HIGH_QUALITY_MIN_EXPECTANCY
                and overall.profit_factor is not None
                and overall.profit_factor >= HIGH_QUALITY_MIN_PROFIT_FACTOR
            )
            symbol_rates = {
                item.scope: item.signals_per_month
                for item in per_symbol
            }
            symbol_meets = {
                symbol: rate >= threshold
                for symbol, rate in symbol_rates.items()
            }
            assessments[key] = {
                "threshold_signals_per_month": threshold,
                "overall_signals_per_month": overall.signals_per_month,
                "overall_meets_threshold": overall_meets,
                "overall_profitable": profitable,
                "realistic": overall_meets and profitable and overall.classification != "Reject",
                "per_symbol_signals_per_month": symbol_rates,
                "per_symbol_meets_threshold": symbol_meets,
            }
        return assessments

    def run(self, metadata: dict[str, Any]) -> UnifiedSignalValidationReport:
        started = time.perf_counter()
        estimate = self.estimate_runtime(metadata)
        self.print_runtime_estimate(estimate)
        loaded = self._load_research_inputs()
        synthesis = self._synthesize_research_context(loaded)
        outcomes = self._scan_history(metadata)

        months = self._research_months(
            metadata.get("start_date", ""),
            metadata.get("end_date", ""),
            metadata.get("research_window_days", self.research_days),
        )
        weeks = self._research_weeks(
            metadata.get("start_date", ""),
            metadata.get("end_date", ""),
            metadata.get("research_window_days", self.research_days),
        )

        overall = self._aggregate_metrics(outcomes, "overall", None, months, weeks)
        buy_metrics = self._aggregate_metrics(outcomes, "buy", "BUY", months, weeks)
        sell_metrics = self._aggregate_metrics(outcomes, "sell", "SELL", months, weeks)

        per_symbol = [
            self._aggregate_metrics(
                [item for item in outcomes if item.symbol == symbol],
                symbol,
                None,
                months,
                weeks,
            )
            for symbol in self.symbols
        ]
        per_symbol_buy = [
            self._aggregate_metrics(
                [item for item in outcomes if item.symbol == symbol and item.signal_side == "BUY"],
                f"{symbol}_BUY",
                "BUY",
                months,
                weeks,
            )
            for symbol in self.symbols
        ]
        per_symbol_sell = [
            self._aggregate_metrics(
                [item for item in outcomes if item.symbol == symbol and item.signal_side == "SELL"],
                f"{symbol}_SELL",
                "SELL",
                months,
                weeks,
            )
            for symbol in self.symbols
        ]

        volume_assessment = self._assess_signal_volume(overall, per_symbol)
        monthly_freq = self._timestamp_bucket([item.timestamp for item in outcomes], "month")
        weekly_freq = self._timestamp_bucket([item.timestamp for item in outcomes], "week")

        realistic_20 = volume_assessment["20_plus_per_month"]["realistic"]
        realistic_30 = volume_assessment["30_plus_per_month"]["realistic"]
        realistic_40 = volume_assessment["40_plus_per_month"]["realistic"]
        realistic_50 = volume_assessment["50_plus_per_month"]["realistic"]

        conclusions = [
            (
                f"Unified engine generated {overall.total_signals} signals "
                f"({overall.signals_per_month:.1f}/month, {overall.signals_per_week:.1f}/week)."
            ),
            (
                f"BUY: n={buy_metrics.total_signals}, 1R={buy_metrics.hit_1r_rate_pct}%, "
                f"Exp={buy_metrics.expectancy}, PF={buy_metrics.profit_factor}, "
                f"class={buy_metrics.classification}."
            ),
            (
                f"SELL: n={sell_metrics.total_signals}, 1R={sell_metrics.hit_1r_rate_pct}%, "
                f"Exp={sell_metrics.expectancy}, PF={sell_metrics.profit_factor}, "
                f"class={sell_metrics.classification}."
            ),
            (
                f"20+ signals/month realistic: {'Yes' if realistic_20 else 'No'} "
                f"({overall.signals_per_month:.1f}/month at {overall.classification})."
            ),
            (
                f"30+ signals/month realistic: {'Yes' if realistic_30 else 'No'}."
            ),
            (
                f"40+ signals/month realistic: {'Yes' if realistic_40 else 'No'}."
            ),
            (
                f"50+ signals/month realistic: {'Yes' if realistic_50 else 'No'}."
            ),
            f"Research inputs loaded: {synthesis['inputs_available']}/{len(RESEARCH_INPUTS)}.",
        ]

        return UnifiedSignalValidationReport(
            symbols_analyzed=list(self.symbols),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            timeframes_analyzed=list(self.timeframes),
            research_inputs_loaded=loaded,
            unified_signal_engine={
                "buy_blueprint": BUY_BLUEPRINT_TEXT,
                "sell_blueprint": SELL_BLUEPRINT_TEXT,
                "classification": overall.classification,
                "signals_per_month": overall.signals_per_month,
                "signals_per_week": overall.signals_per_week,
            },
            buy_blueprint=self.buy_blueprint.as_dict(),
            sell_blueprint=self.sell_blueprint.as_dict(),
            total_signals=overall.total_signals,
            overall_metrics=overall.as_dict(),
            buy_metrics=buy_metrics.as_dict(),
            sell_metrics=sell_metrics.as_dict(),
            per_symbol_metrics=[item.as_dict() for item in per_symbol],
            per_symbol_buy_metrics=[item.as_dict() for item in per_symbol_buy],
            per_symbol_sell_metrics=[item.as_dict() for item in per_symbol_sell],
            monthly_signal_frequency=monthly_freq,
            weekly_signal_frequency=weekly_freq,
            signal_volume_assessment=volume_assessment,
            trade_construction={
                "entry": "Confirmation candle close (signal bar close)",
                "stop_loss": "Structural swing SL (20-bar lookback + buffer)",
                "t1": "1R",
                "t2": "2R",
                "t3": "3R",
            },
            sample_signals=[item.as_dict() for item in outcomes[:MAX_EXPORT_SIGNALS]],
            research_synthesis=synthesis,
            conclusions=conclusions + [
                f"Pre-run runtime estimate: {self._format_duration(estimate.total_seconds)}.",
            ],
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_unified_signal_validation_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
    symbols: tuple[str, ...] | None = None,
) -> UnifiedSignalValidationReport:
    """Run unified signal validation and export JSON."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise UnifiedSignalValidationError(f"Filter research report not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = SmartMoneyEngineUnifiedSignalValidationResearch(symbols=symbols)
    report = engine.run(metadata)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info(
        "Unified signal validation completed: signals=%s per_month=%.1f class=%s",
        report.total_signals,
        report.overall_metrics["signals_per_month"],
        report.overall_metrics["classification"],
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_unified_signal_validation_report()
        print("SmartMoneyEngine Unified Signal Validation Summary")
        print(f"Total signals: {report.total_signals}")
        print(f"Signals/month: {report.overall_metrics['signals_per_month']}")
        print(f"Signals/week: {report.overall_metrics['signals_per_week']}")
        print(f"1R rate: {report.overall_metrics['hit_1r_rate_pct']}%")
        print(f"Win rate: {report.overall_metrics['win_rate_pct']}%")
        print(f"Profit factor: {report.overall_metrics['profit_factor']}")
        print(f"Expectancy: {report.overall_metrics['expectancy']}")
        print(f"Classification: {report.overall_metrics['classification']}")
        for threshold in SIGNAL_VOLUME_THRESHOLDS:
            key = f"{threshold}_plus_per_month"
            realistic = report.signal_volume_assessment[key]["realistic"]
            print(f"{threshold}+ signals/month realistic: {'Yes' if realistic else 'No'}")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except UnifiedSignalValidationError as exc:
        logger.error("Unified signal validation error: %s", exc)
        print(f"Unified signal validation error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected unified signal validation error")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
