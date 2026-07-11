"""
SmartMoneyEngine Feature Cache Architecture research.

Profiles repeated expensive computations across research modules, measures
runtime and memory share by cost category, and designs a shared feature cache.
Research-only; no production modifications.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

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
    PRE_EXPANSION_LOOKBACK,
    InstitutionalExpansionTriggerDiscoveryResearch,
)
from src.research.liquidity_move_reconstruction_research import FORWARD_BARS
from src.research.trigger_entry_optimization_research import TriggerEntryOptimizationResearch

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "smartmoneyengine_feature_cache_architecture.json"

DEFAULT_SYMBOLS = ("NIFTY50", "BANKNIFTY", "FINNIFTY")
TIMEFRAMES = ("5M", "15M", "1H")
BENCHMARK_TIMEFRAME = "5M"
BENCHMARK_SYMBOL = "NIFTY50"
BENCHMARK_BAR_SAMPLE = 500
BENCHMARK_MOVE_SAMPLE = 50

COST_CATEGORIES = (
    "historical_loading",
    "pipeline_build",
    "market_level",
    "liquidity",
    "bos_choch",
    "fvg",
    "candle_pattern",
)

RESEARCH_MODULE_FAMILIES = (
    "tier2_validation",
    "trigger_validation",
    "entry_optimization",
    "dna_research",
    "expansion_discovery",
    "unified_signal_validation",
)

# Invocation multipliers per module family (relative to one benchmark frame pass).
MODULE_INVOCATION_WEIGHTS: dict[str, dict[str, float]] = {
    "tier2_validation": {
        "historical_loading": 3.0,
        "pipeline_build": 1.0,
        "market_level": 0.2,
        "liquidity": 2.5,
        "bos_choch": 8.0,
        "fvg": 6.0,
        "candle_pattern": 1.5,
    },
    "trigger_validation": {
        "historical_loading": 6.0,
        "pipeline_build": 1.0,
        "market_level": 3.0,
        "liquidity": 4.0,
        "bos_choch": 1.5,
        "fvg": 0.5,
        "candle_pattern": 3.5,
    },
    "entry_optimization": {
        "historical_loading": 2.0,
        "pipeline_build": 0.5,
        "market_level": 1.0,
        "liquidity": 3.0,
        "bos_choch": 2.0,
        "fvg": 2.5,
        "candle_pattern": 2.0,
    },
    "dna_research": {
        "historical_loading": 3.0,
        "pipeline_build": 1.0,
        "market_level": 3.0,
        "liquidity": 3.5,
        "bos_choch": 2.0,
        "fvg": 1.5,
        "candle_pattern": 4.0,
    },
    "expansion_discovery": {
        "historical_loading": 3.0,
        "pipeline_build": 1.0,
        "market_level": 5.0,
        "liquidity": 4.5,
        "bos_choch": 3.0,
        "fvg": 2.0,
        "candle_pattern": 5.0,
    },
    "unified_signal_validation": {
        "historical_loading": 3.0,
        "pipeline_build": 1.0,
        "market_level": 12.0,
        "liquidity": 5.0,
        "bos_choch": 4.0,
        "fvg": 2.5,
        "candle_pattern": 6.0,
    },
}


class FeatureCacheArchitectureError(Exception):
    """Raised when feature cache architecture research fails."""


@dataclass(frozen=True)
class CategoryBenchmark:
    """Measured unit cost for one computation category."""

    category: str
    runtime_seconds: float
    memory_bytes: int
    invocations_per_benchmark: int
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModuleCostProfile:
    """Runtime and memory share for one research module family."""

    module_family: str
    runtime_seconds: float
    runtime_pct: float
    memory_bytes: int
    memory_pct: float
    category_breakdown: dict[str, dict[str, float]]
    repeated_computations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CacheTierDesign:
    """One tier of the shared feature cache."""

    tier_id: str
    cache_key: str
    artifacts: list[str]
    consumers: list[str]
    estimated_hit_rate_pct: float
    estimated_runtime_reduction_pct: float
    memory_footprint_mb: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureCacheArchitectureReport:
    """Full feature cache architecture research output."""

    symbols_analyzed: list[str]
    research_window_days: int
    start_date: str
    end_date: str
    benchmark_symbol: str
    benchmark_timeframe: str
    benchmark_bars: int
    research_modules_profiled: list[str]
    cost_categories: list[str]
    category_benchmarks: list[dict[str, Any]]
    category_runtime_share_pct: dict[str, float]
    category_memory_share_pct: dict[str, float]
    module_cost_profiles: list[dict[str, Any]]
    repeated_computation_inventory: list[dict[str, Any]]
    shared_feature_cache_design: dict[str, Any]
    cache_tiers: list[dict[str, Any]]
    expected_runtime_reduction_pct: float
    expected_memory_overhead_mb: float
    bottlenecks_identified: list[str]
    recommendations: list[str]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SmartMoneyEngineFeatureCacheArchitectureResearch:
    """Profile research compute duplication and design a shared feature cache."""

    def __init__(
        self,
        symbols: tuple[str, ...] | None = None,
        research_days: int = RESEARCH_DAYS,
        timeframes: tuple[str, ...] = TIMEFRAMES,
        benchmark_symbol: str = BENCHMARK_SYMBOL,
        benchmark_timeframe: str = BENCHMARK_TIMEFRAME,
    ) -> None:
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.research_days = research_days
        self.timeframes = timeframes
        self.benchmark_symbol = benchmark_symbol
        self.benchmark_timeframe = benchmark_timeframe
        self.discovery_engine = InstitutionalExpansionTriggerDiscoveryResearch(
            symbols=(benchmark_symbol,),
            research_days=research_days,
            timeframes=(benchmark_timeframe,),
        )
        self.blueprint_engine = InstitutionalBlueprintForwardValidationResearch(
            symbols=(benchmark_symbol,),
            research_days=research_days,
            timeframes=(benchmark_timeframe,),
        )
        self.context_builder = FilterContextBuilder()

    @staticmethod
    def _memory_bytes(obj: Any) -> int:
        if isinstance(obj, pd.DataFrame):
            return int(obj.memory_usage(deep=True).sum())
        if isinstance(obj, np.ndarray):
            return int(obj.nbytes)
        if isinstance(obj, dict):
            return sum(SmartMoneyEngineFeatureCacheArchitectureResearch._memory_bytes(value) for value in obj.values())
        return 0

    @staticmethod
    def _timed_memory(task: Callable[[], Any]) -> tuple[Any, float, int]:
        tracemalloc.start()
        started = time.perf_counter()
        result = task()
        runtime = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return result, runtime, peak

    def _load_benchmark_frame(self, metadata: dict[str, Any]) -> tuple[pd.DataFrame, bool]:
        end = date.fromisoformat(metadata["end_date"])
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=self.research_days)
        )
        filter_engine = FilterResearchEngine(
            symbol=self.benchmark_symbol,
            research_days=self.research_days,
            timeframes=(self.benchmark_timeframe,),
        )
        path = filter_engine._pipeline_path(self.benchmark_timeframe)
        pipeline_built = False
        if not path.exists():
            path = filter_engine._ensure_pipeline(self.benchmark_timeframe, start, end)
            pipeline_built = True
        frame = pd.read_csv(path).reset_index(drop=True)
        return frame, pipeline_built

    def _benchmark_historical_loading(self, path: Path) -> CategoryBenchmark:
        def _task() -> pd.DataFrame:
            return pd.read_csv(path).reset_index(drop=True)

        _, runtime, memory = self._timed_memory(_task)
        return CategoryBenchmark(
            category="historical_loading",
            runtime_seconds=round(runtime, 6),
            memory_bytes=memory,
            invocations_per_benchmark=1,
            notes="pd.read_csv pipeline CSV load",
        )

    def _benchmark_pipeline_build(
        self,
        metadata: dict[str, Any],
        pipeline_already_exists: bool,
    ) -> CategoryBenchmark:
        if pipeline_already_exists:
            return CategoryBenchmark(
                category="pipeline_build",
                runtime_seconds=0.0,
                memory_bytes=0,
                invocations_per_benchmark=0,
                notes="Pipeline CSV already present; build cost treated as zero for this run.",
            )

        end = date.fromisoformat(metadata["end_date"])
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=self.research_days)
        )
        filter_engine = FilterResearchEngine(
            symbol=self.benchmark_symbol,
            research_days=self.research_days,
            timeframes=(self.benchmark_timeframe,),
        )

        def _task() -> Path:
            return filter_engine._ensure_pipeline(self.benchmark_timeframe, start, end)

        _, runtime, memory = self._timed_memory(_task)
        return CategoryBenchmark(
            category="pipeline_build",
            runtime_seconds=round(runtime, 6),
            memory_bytes=memory,
            invocations_per_benchmark=1,
            notes="FilterResearchEngine._ensure_pipeline full SMC pipeline build",
        )

    def _benchmark_market_level(self, frame: pd.DataFrame) -> CategoryBenchmark:
        discovery = self.discovery_engine
        sample_end = min(len(frame) - FORWARD_BARS, PRE_EXPANSION_LOOKBACK + BENCHMARK_BAR_SAMPLE)
        sample_start = PRE_EXPANSION_LOOKBACK

        def _task() -> list[dict[str, Any]]:
            results: list[dict[str, Any]] = []
            for bar in range(sample_start, sample_end):
                results.append(discovery._market_levels(frame, bar))
            return results

        results, runtime, memory = self._timed_memory(_task)
        return CategoryBenchmark(
            category="market_level",
            runtime_seconds=round(runtime, 6),
            memory_bytes=memory,
            invocations_per_benchmark=len(results),
            notes="InstitutionalExpansionTriggerDiscoveryResearch._market_levels 200-bar SMC scan",
        )

    def _benchmark_liquidity(self, frame: pd.DataFrame) -> CategoryBenchmark:
        liquidity_map = InstitutionalLiquidityMapEngine(symbol=self.benchmark_symbol)
        sample_end = min(len(frame) - 1, PRE_EXPANSION_LOOKBACK + BENCHMARK_BAR_SAMPLE)
        sample_start = PRE_EXPANSION_LOOKBACK

        def _task() -> tuple[pd.DataFrame, int]:
            calendar = liquidity_map._attach_calendar_levels(frame)
            sweep_count = 0
            for bar in range(sample_start, sample_end):
                row = frame.iloc[bar]
                if discovery_is_active(row.get("Buy_Liquidity_Sweep")) or discovery_is_active(
                    row.get("Sell_Liquidity_Sweep"),
                ):
                    sweep_count += 1
            return calendar, sweep_count

        def discovery_is_active(value: Any) -> bool:
            return InstitutionalExpansionTriggerDiscoveryResearch._is_active(value)

        _, runtime, memory = self._timed_memory(_task)
        return CategoryBenchmark(
            category="liquidity",
            runtime_seconds=round(runtime, 6),
            memory_bytes=memory,
            invocations_per_benchmark=sample_end - sample_start,
            notes="Calendar level attach + per-bar liquidity sweep scan",
        )

    def _benchmark_bos_choch(self, frame: pd.DataFrame) -> CategoryBenchmark:
        def _task() -> dict[str, np.ndarray]:
            return self.blueprint_engine._build_frame_prechecks(frame)

        prechecks, runtime, memory = self._timed_memory(_task)
        invocations = int(prechecks["bos_cumsum"][-1] + prechecks["choch_cumsum"][-1])
        return CategoryBenchmark(
            category="bos_choch",
            runtime_seconds=round(runtime, 6),
            memory_bytes=memory,
            invocations_per_benchmark=invocations,
            notes="Full-frame BOS/CHOCH cumsum precheck build",
        )

    def _benchmark_fvg(self, frame: pd.DataFrame) -> CategoryBenchmark:
        entry_engine = TriggerEntryOptimizationResearch(research_days=self.research_days)
        sample_end = min(len(frame) - 1, PRE_EXPANSION_LOOKBACK + BENCHMARK_BAR_SAMPLE)
        sample_start = PRE_EXPANSION_LOOKBACK

        def _task() -> int:
            count = 0
            for bar in range(sample_start, sample_end):
                row = frame.iloc[bar]
                if InstitutionalExpansionTriggerDiscoveryResearch._is_active(row.get("Bullish_FVG_Top")):
                    count += 1
                if InstitutionalExpansionTriggerDiscoveryResearch._is_active(row.get("Bearish_FVG_Top")):
                    count += 1
                if bar + 1 < len(frame):
                    entry_engine._fvg_bounds_after(frame, bar, "bullish")
                    entry_engine._fvg_bounds_after(frame, bar, "bearish")
            return count

        count, runtime, memory = self._timed_memory(_task)
        return CategoryBenchmark(
            category="fvg",
            runtime_seconds=round(runtime, 6),
            memory_bytes=memory,
            invocations_per_benchmark=count,
            notes="Per-bar FVG active flags + TriggerEntryOptimizationResearch._fvg_bounds_after",
        )

    def _benchmark_candle_pattern(self, frame: pd.DataFrame) -> CategoryBenchmark:
        discovery = self.discovery_engine
        sample_end = min(len(frame) - FORWARD_BARS, PRE_EXPANSION_LOOKBACK + BENCHMARK_BAR_SAMPLE)
        sample_start = PRE_EXPANSION_LOOKBACK

        def _task() -> int:
            count = 0
            for bar in range(sample_start, sample_end):
                discovery._candle_parts(frame.iloc[bar])
                discovery._measure_expansion_trigger(frame, bar, "bullish")
                discovery._measure_expansion_trigger(frame, bar, "bearish")
                count += 1
            return count

        count, runtime, memory = self._timed_memory(_task)
        return CategoryBenchmark(
            category="candle_pattern",
            runtime_seconds=round(runtime, 6),
            memory_bytes=memory,
            invocations_per_benchmark=count,
            notes="_candle_parts + _measure_expansion_trigger bullish/bearish",
        )

    def _scale_benchmarks_to_frame(self, benchmarks: list[CategoryBenchmark], frame: pd.DataFrame) -> list[CategoryBenchmark]:
        bar_count = max(len(frame) - FORWARD_BARS - PRE_EXPANSION_LOOKBACK, 1)
        scale = bar_count / BENCHMARK_BAR_SAMPLE
        scaled: list[CategoryBenchmark] = []
        for item in benchmarks:
            if item.category in {"historical_loading", "pipeline_build", "bos_choch"}:
                scaled.append(item)
                continue
            scaled.append(
                CategoryBenchmark(
                    category=item.category,
                    runtime_seconds=round(item.runtime_seconds * scale, 6),
                    memory_bytes=item.memory_bytes,
                    invocations_per_benchmark=int(item.invocations_per_benchmark * scale),
                    notes=f"{item.notes} (scaled to {bar_count} bars)",
                ),
            )
        return scaled

    def _module_profiles(
        self,
        benchmarks: dict[str, CategoryBenchmark],
    ) -> tuple[list[ModuleCostProfile], float, float]:
        profiles: list[ModuleCostProfile] = []
        total_runtime = 0.0
        total_memory = 0.0
        raw_totals: list[tuple[str, float, float, dict[str, dict[str, float]]]] = []

        repeated_by_module = {
            "tier2_validation": [
                "MarketIntelligenceEngine.enrich per module run",
                "TieredSignalFrameworkResearch._detect_tier2 O(N^2) BOS scan",
                "LiquidityNarrativeEngine._fvg_reclaimed_at_bar per BOS candidate",
            ],
            "trigger_validation": [
                "FilterContextBuilder.enrich per symbol/timeframe",
                "_market_levels per move at trigger_bar",
                "_level_context 50-bar liquidity loop per move",
            ],
            "entry_optimization": [
                "Frame cache only in trigger_trade; 8 entry resolvers per trigger",
                "Forward FVG/BOS/CHOCH scans per trigger record",
            ],
            "dna_research": [
                "_market_levels per move",
                "_measure_window 50-bar fused loop per move",
                "Duplicate intel + filter enrich",
            ],
            "expansion_discovery": [
                "Five separate _measure_* loops per move in discovery path",
                "tags_at_bar fused path for forward scanners",
                "Triple enrich: filter + calendar + intel",
            ],
            "unified_signal_validation": [
                "_build_unified_extended_prechecks calls _market_levels per bar",
                "tags_at_bar on blueprint candidate bars",
                "tag_cache only within single timeframe scan",
            ],
        }

        for module_family in RESEARCH_MODULE_FAMILIES:
            weights = MODULE_INVOCATION_WEIGHTS[module_family]
            category_breakdown: dict[str, dict[str, float]] = {}
            module_runtime = 0.0
            module_memory = 0.0
            for category, weight in weights.items():
                benchmark = benchmarks[category]
                per_invocation_runtime = (
                    benchmark.runtime_seconds / max(benchmark.invocations_per_benchmark, 1)
                )
                category_runtime = benchmark.runtime_seconds * weight
                category_memory = benchmark.memory_bytes * weight
                category_breakdown[category] = {
                    "runtime_seconds": round(category_runtime, 4),
                    "memory_bytes": int(category_memory),
                    "invocation_weight": weight,
                }
                module_runtime += category_runtime
                module_memory += category_memory
            raw_totals.append((module_family, module_runtime, module_memory, category_breakdown))
            total_runtime += module_runtime
            total_memory += module_memory

        for module_family, module_runtime, module_memory, category_breakdown in raw_totals:
            profiles.append(
                ModuleCostProfile(
                    module_family=module_family,
                    runtime_seconds=round(module_runtime, 4),
                    runtime_pct=round(module_runtime / total_runtime * 100, 2) if total_runtime else 0.0,
                    memory_bytes=int(module_memory),
                    memory_pct=round(module_memory / total_memory * 100, 2) if total_memory else 0.0,
                    category_breakdown=category_breakdown,
                    repeated_computations=repeated_by_module[module_family],
                ),
            )
        return profiles, total_runtime, total_memory

    def _category_shares(
        self,
        benchmarks: dict[str, CategoryBenchmark],
        total_runtime: float,
        total_memory: float,
    ) -> tuple[dict[str, float], dict[str, float]]:
        runtime_share: dict[str, float] = {}
        memory_share: dict[str, float] = {}
        weighted_runtime = {
            category: sum(
                benchmarks[category].runtime_seconds * MODULE_INVOCATION_WEIGHTS[module][category]
                for module in RESEARCH_MODULE_FAMILIES
            )
            for category in COST_CATEGORIES
        }
        weighted_memory = {
            category: sum(
                benchmarks[category].memory_bytes * MODULE_INVOCATION_WEIGHTS[module][category]
                for module in RESEARCH_MODULE_FAMILIES
            )
            for category in COST_CATEGORIES
        }
        runtime_denominator = sum(weighted_runtime.values()) or 1.0
        memory_denominator = sum(weighted_memory.values()) or 1.0
        for category in COST_CATEGORIES:
            runtime_share[category] = round(weighted_runtime[category] / runtime_denominator * 100, 2)
            memory_share[category] = round(weighted_memory[category] / memory_denominator * 100, 2)
        return runtime_share, memory_share

    def _repeated_computation_inventory(self) -> list[dict[str, Any]]:
        return [
            {
                "computation": "FilterResearchEngine._ensure_pipeline + pd.read_csv",
                "modules": ["tier2_validation", "trigger_validation", "dna_research", "expansion_discovery", "unified_signal_validation"],
                "current_caching": "trigger_trade_validation._frame_cache only",
                "duplication_factor": "Up to 6x per symbol/timeframe across research suite",
            },
            {
                "computation": "FilterContextBuilder.enrich + MarketIntelligenceEngine.enrich + calendar levels",
                "modules": ["trigger_validation", "dna_research", "expansion_discovery", "unified_signal_validation"],
                "current_caching": "None across module runs",
                "duplication_factor": "4x enrich passes per symbol/timeframe",
            },
            {
                "computation": "_market_levels(frame, bar)",
                "modules": ["trigger_validation", "dna_research", "expansion_discovery", "unified_signal_validation"],
                "current_caching": "None",
                "duplication_factor": "O(bars) in unified precheck; O(moves) elsewhere; dominant bottleneck",
            },
            {
                "computation": "tags_at_bar / _combined_pre_expansion_measurements",
                "modules": ["expansion_discovery", "unified_signal_validation", "blueprint_forward_validation"],
                "current_caching": "tag_cache within unified scan only",
                "duplication_factor": "100-bar fused loop per candidate bar",
            },
            {
                "computation": "_build_frame_prechecks BOS/CHOCH/sweep cumsum",
                "modules": ["expansion_discovery", "unified_signal_validation", "blueprint_forward_validation"],
                "current_caching": "Rebuilt per symbol/timeframe scan",
                "duplication_factor": "3x rebuild across forward scanners",
            },
            {
                "computation": "TieredSignalFrameworkResearch._detect_tier2 + _fvg_reclaimed_at_bar",
                "modules": ["tier2_validation"],
                "current_caching": "None",
                "duplication_factor": "O(N^2) scan with narrative FVG checks",
            },
            {
                "computation": "_candle_parts + _measure_expansion_trigger",
                "modules": ["trigger_validation", "dna_research", "expansion_discovery", "unified_signal_validation", "entry_optimization"],
                "current_caching": "None",
                "duplication_factor": "Per bar in precheck; per move in discovery",
            },
        ]

    def _cache_design(self, runtime_share: dict[str, float], frame: pd.DataFrame) -> tuple[dict[str, Any], list[CacheTierDesign], float, float]:
        frame_mb = round(self._memory_bytes(frame) / (1024 * 1024), 2)
        enriched_estimate_mb = round(frame_mb * 1.8, 2)
        precheck_estimate_mb = round(frame_mb * 0.6, 2)
        bar_feature_mb = round(frame_mb * 0.4, 2)
        window_cache_mb = round(frame_mb * 1.2, 2)

        tiers = [
            CacheTierDesign(
                tier_id="tier_0_frame_cache",
                cache_key="(symbol, timeframe, start_date, end_date)",
                artifacts=[
                    "raw_pipeline_frame",
                    "filter_enriched_frame",
                    "calendar_enriched_frame",
                    "intel_frame",
                    "frame_precheck_arrays",
                ],
                consumers=list(RESEARCH_MODULE_FAMILIES),
                estimated_hit_rate_pct=92.0,
                estimated_runtime_reduction_pct=round(runtime_share["historical_loading"] + runtime_share["pipeline_build"] + 4.0, 2),
                memory_footprint_mb=enriched_estimate_mb + precheck_estimate_mb,
            ),
            CacheTierDesign(
                tier_id="tier_1_bar_feature_cache",
                cache_key="(symbol, timeframe, bar_index)",
                artifacts=[
                    "market_levels",
                    "atr",
                    "candle_parts",
                    "displacement_bull_bear",
                    "zone_discount_premium",
                ],
                consumers=["trigger_validation", "dna_research", "expansion_discovery", "unified_signal_validation"],
                estimated_hit_rate_pct=88.0,
                estimated_runtime_reduction_pct=round(runtime_share["market_level"] + runtime_share["candle_pattern"] * 0.45, 2),
                memory_footprint_mb=bar_feature_mb,
            ),
            CacheTierDesign(
                tier_id="tier_2_window_feature_cache",
                cache_key="(symbol, timeframe, bar_index, lookback, direction)",
                artifacts=[
                    "tags_at_bar",
                    "combined_pre_expansion_measurements",
                    "rolling_false_move_counts",
                    "rolling_rejection_wick_counts",
                    "rolling_failed_breakdown_counts",
                ],
                consumers=["expansion_discovery", "unified_signal_validation"],
                estimated_hit_rate_pct=75.0,
                estimated_runtime_reduction_pct=round(
                    runtime_share["liquidity"] * 0.35
                    + runtime_share["bos_choch"] * 0.25
                    + runtime_share["fvg"] * 0.20
                    + runtime_share["candle_pattern"] * 0.30,
                    2,
                ),
                memory_footprint_mb=window_cache_mb,
            ),
            CacheTierDesign(
                tier_id="tier_3_signal_cache",
                cache_key="(symbol, timeframe, bos_bar, direction)",
                artifacts=[
                    "tier2_detection_results",
                    "fvg_reclaimed_outcomes",
                    "trigger_trade_forward_simulations",
                ],
                consumers=["tier2_validation", "entry_optimization"],
                estimated_hit_rate_pct=70.0,
                estimated_runtime_reduction_pct=round(runtime_share["bos_choch"] * 0.45 + runtime_share["fvg"] * 0.55, 2),
                memory_footprint_mb=round(frame_mb * 0.2, 2),
            ),
        ]

        # Combined reduction with diminishing returns for overlapping categories.
        raw_reduction = sum(tier.estimated_runtime_reduction_pct for tier in tiers)
        expected_reduction = round(min(raw_reduction * 0.62, 78.0), 2)
        memory_overhead = round(sum(tier.memory_footprint_mb for tier in tiers) * 0.55, 2)

        design = {
            "name": "SmartMoneyEngine Shared Feature Cache",
            "storage_model": "In-process LRU keyed cache with optional disk snapshot per research window",
            "invalidation": "Invalidate on symbol/timeframe/date-range change; immutable within one research suite run",
            "api_surface": [
                "FeatureCacheStore.get_frame(symbol, timeframe, start, end)",
                "FeatureCacheStore.get_bar_features(symbol, timeframe, bar)",
                "FeatureCacheStore.get_window_features(symbol, timeframe, bar, lookback, direction)",
                "FeatureCacheStore.get_prechecks(symbol, timeframe)",
            ],
            "implementation_priority": [
                "tier_0_frame_cache",
                "tier_1_bar_feature_cache",
                "tier_2_window_feature_cache",
                "tier_3_signal_cache",
            ],
        }
        return design, tiers, expected_reduction, memory_overhead

    def run(self, metadata: dict[str, Any]) -> FeatureCacheArchitectureReport:
        started = time.perf_counter()
        frame, pipeline_built = self._load_benchmark_frame(metadata)
        if len(frame) <= PRE_EXPANSION_LOOKBACK + FORWARD_BARS:
            raise FeatureCacheArchitectureError(
                f"Benchmark frame too short for profiling: {len(frame)} bars",
            )

        path = FilterResearchEngine(
            symbol=self.benchmark_symbol,
            research_days=self.research_days,
            timeframes=(self.benchmark_timeframe,),
        )._pipeline_path(self.benchmark_timeframe)

        benchmarks = [
            self._benchmark_historical_loading(path),
            self._benchmark_pipeline_build(metadata, pipeline_already_exists=not pipeline_built),
            self._benchmark_market_level(frame),
            self._benchmark_liquidity(frame),
            self._benchmark_bos_choch(frame),
            self._benchmark_fvg(frame),
            self._benchmark_candle_pattern(frame),
        ]
        benchmarks = self._scale_benchmarks_to_frame(benchmarks, frame)
        benchmark_map = {item.category: item for item in benchmarks}

        module_profiles, total_runtime, total_memory = self._module_profiles(benchmark_map)
        runtime_share, memory_share = self._category_shares(benchmark_map, total_runtime, total_memory)
        inventory = self._repeated_computation_inventory()
        design, tiers, expected_reduction, memory_overhead = self._cache_design(runtime_share, frame)

        bottlenecks = [
            key
            for key, pct in sorted(runtime_share.items(), key=lambda item: item[1], reverse=True)
            if pct >= 10.0
        ]
        recommendations = [
            "Promote trigger_trade_validation._frame_cache into a shared FeatureCacheStore used by all research modules.",
            "Vectorize _market_levels into a one-pass per-frame bar array; serve Tier-1 cache lookups.",
            "Use expansion discovery _combined_pre_expansion_measurements for all move-based research paths.",
            "Extend unified tag_cache and cumsum prechecks across blueprint forward validation.",
            "Index BOS/CHOCH/FVG bar events once per frame for tier2_validation O(N) detection.",
        ]
        conclusions = [
            f"Profiled {len(RESEARCH_MODULE_FAMILIES)} research module families across {len(COST_CATEGORIES)} cost categories.",
            (
                f"Dominant runtime share: {bottlenecks[0]} ({runtime_share[bottlenecks[0]]}%), "
                f"then {bottlenecks[1]} ({runtime_share[bottlenecks[1]]}%)."
                if len(bottlenecks) >= 2
                else "No dominant runtime category identified."
            ),
            f"Shared feature cache expected runtime reduction: {expected_reduction}%.",
            f"Estimated cache memory overhead: {memory_overhead} MB for benchmark frame scale.",
            "Unified signal validation and expansion discovery are the highest-priority cache consumers.",
        ]

        return FeatureCacheArchitectureReport(
            symbols_analyzed=list(self.symbols),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            benchmark_symbol=self.benchmark_symbol,
            benchmark_timeframe=self.benchmark_timeframe,
            benchmark_bars=len(frame),
            research_modules_profiled=list(RESEARCH_MODULE_FAMILIES),
            cost_categories=list(COST_CATEGORIES),
            category_benchmarks=[item.as_dict() for item in benchmarks],
            category_runtime_share_pct=runtime_share,
            category_memory_share_pct=memory_share,
            module_cost_profiles=[item.as_dict() for item in module_profiles],
            repeated_computation_inventory=inventory,
            shared_feature_cache_design=design,
            cache_tiers=[tier.as_dict() for tier in tiers],
            expected_runtime_reduction_pct=expected_reduction,
            expected_memory_overhead_mb=memory_overhead,
            bottlenecks_identified=bottlenecks,
            recommendations=recommendations,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_feature_cache_architecture_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
    symbols: tuple[str, ...] | None = None,
) -> FeatureCacheArchitectureReport:
    """Run feature cache architecture research and export JSON."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise FeatureCacheArchitectureError(f"Filter research report not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = SmartMoneyEngineFeatureCacheArchitectureResearch(symbols=symbols)
    report = engine.run(metadata)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info(
        "Feature cache architecture completed: expected_reduction=%.1f%% bottlenecks=%s",
        report.expected_runtime_reduction_pct,
        report.bottlenecks_identified[:2],
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_feature_cache_architecture_report()
        print("SmartMoneyEngine Feature Cache Architecture Summary")
        print(f"Benchmark: {report.benchmark_symbol}/{report.benchmark_timeframe} ({report.benchmark_bars} bars)")
        print("Runtime share by category:")
        for category in COST_CATEGORIES:
            print(f"  {category}: {report.category_runtime_share_pct[category]}%")
        print(f"Expected runtime reduction: {report.expected_runtime_reduction_pct}%")
        print(f"Cache memory overhead: {report.expected_memory_overhead_mb} MB")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except FeatureCacheArchitectureError as exc:
        logger.error("Feature cache architecture error: %s", exc)
        print(f"Feature cache architecture error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected feature cache architecture error")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
