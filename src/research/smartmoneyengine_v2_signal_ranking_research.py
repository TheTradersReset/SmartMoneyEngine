"""
SmartMoneyEngine V2 Signal Ranking research.

Ranks V2 production-card signals by quality score and identifies top
execution-worthy archetypes. Research-only; no production modifications.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import pandas as pd

from src.research.filter_research_engine import RESEARCH_DAYS, _json_safe
from src.research.institutional_quality_validation_research import (
    InstitutionalQualityValidationResearch,
)
from src.research.liquidity_move_reconstruction_research import FORWARD_BARS
from src.research.smartmoneyengine_production_candidate_research import (
    SmartMoneyEngineProductionCandidateResearch,
)
from src.research.smartmoneyengine_v2_frequency_optimization_research import (
    SmartMoneyEngineV2FrequencyOptimizationResearch,
)
from src.research.tier2_production_validation_research import Tier2ProductionValidationResearch
from src.research.tier2_winner_loser_comparison_research import (
    ComparativeTradeRecord,
    Tier2WinnerLoserComparisonResearch,
)
from src.research.tiered_signal_framework_research import TierSignal

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_FILTER_REPORT_PATH = RESEARCH_DIR / "filter_research_report.json"
DEFAULT_V2_OPTIMIZATION_PATH = RESEARCH_DIR / "smartmoneyengine_v2_frequency_optimization.json"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "smartmoneyengine_v2_signal_ranking.json"

MIN_SAMPLE_SIZE = 30
TOP_ARCHETYPE_COUNT = 50
TOP_MODEL_COUNT = 10
ARCHETYPE_COMBO_SIZES = (4, 5, 6)

GROUPING_DIMENSIONS = (
    "symbol",
    "timeframe",
    "direction",
    "session",
    "vwap_state",
    "rsi_bucket",
    "ema_structure",
    "choch_bos_timing",
    "displacement_strength",
    "level_context",
    "liquidity_context",
    "confirmation_candle",
)


class V2SignalRankingError(Exception):
    """Raised when V2 signal ranking research fails."""


@dataclass(frozen=True)
class RankedV2Signal:
    """One V2-filtered signal with ranking dimensions and trade profile."""

    symbol: str
    bos_timestamp: str
    timeframe: str
    signal_side: str
    direction: str
    session: str
    vwap_state: str
    rsi_bucket: str
    ema_structure: str
    choch_bos_timing: str
    displacement_strength: str
    level_context: str
    liquidity_context: str
    confirmation_candle: str
    risk_points: float
    realized_pnl_points: float
    realized_rr: float
    win: bool
    hit_1r: bool
    hit_2r: bool
    hit_3r: bool
    mae_points: float
    holding_bars: int
    holding_minutes: float
    archetype_key: str

    def dimension_values(self) -> dict[str, str]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "direction": self.signal_side,
            "session": self.session,
            "vwap_state": self.vwap_state,
            "rsi_bucket": self.rsi_bucket,
            "ema_structure": self.ema_structure,
            "choch_bos_timing": self.choch_bos_timing,
            "displacement_strength": self.displacement_strength,
            "level_context": self.level_context,
            "liquidity_context": self.liquidity_context,
            "confirmation_candle": self.confirmation_candle,
        }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArchetypeMetrics:
    """Aggregated metrics for one signal archetype or group."""

    archetype_key: str
    signal_side: str | None
    grouping_dimension: str | None
    grouping_value: str | None
    sample_size: int
    signals_per_month: float
    win_rate_pct: float
    profit_factor: float | None
    expectancy: float
    hit_1r_rate_pct: float
    hit_2r_rate_pct: float
    hit_3r_rate_pct: float
    average_drawdown_points: float
    average_holding_minutes: float
    maximum_drawdown_points: float
    net_points: float
    signal_quality_score: float
    tier: str
    rejected: bool
    rejection_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class V2SignalRankingReport:
    """Full V2 signal ranking output."""

    symbols_analyzed: list[str]
    research_window_days: int
    start_date: str
    end_date: str
    v2_production_card: dict[str, Any]
    minimum_sample_size: int
    total_v2_signals: int
    grouping_dimensions: list[str]
    grouped_analysis: dict[str, list[dict[str, Any]]]
    top_50_signal_archetypes: list[dict[str, Any]]
    top_10_buy_models: list[dict[str, Any]]
    top_10_sell_models: list[dict[str, Any]]
    tier_a_archetypes: list[dict[str, Any]]
    tier_b_archetypes: list[dict[str, Any]]
    tier_c_archetypes: list[dict[str, Any]]
    tier_d_archetypes: list[dict[str, Any]]
    rejected_archetypes: list[dict[str, Any]]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SmartMoneyEngineV2SignalRankingResearch:
    """Rank V2 signals and identify top execution-worthy archetypes."""

    def __init__(
        self,
        symbols: tuple[str, ...] | None = None,
        research_days: int = RESEARCH_DAYS,
        timeframes: tuple[str, ...] = ("5M", "15M", "1H"),
        v2_optimization_path: Path | str | None = None,
    ) -> None:
        self.symbols = symbols or ("NIFTY50", "BANKNIFTY", "FINNIFTY")
        self.research_days = research_days
        self.timeframes = timeframes
        self.v2_optimization_path = Path(v2_optimization_path or DEFAULT_V2_OPTIMIZATION_PATH)
        self._frequency_engine = SmartMoneyEngineV2FrequencyOptimizationResearch(
            symbols=self.symbols,
            research_days=research_days,
            timeframes=timeframes,
        )

    @staticmethod
    def _profit_factor(pnls: list[float]) -> float | None:
        return InstitutionalQualityValidationResearch._profit_factor(pnls)

    @staticmethod
    def _maximum_drawdown(pnls: list[float]) -> float:
        return Tier2ProductionValidationResearch._maximum_drawdown(pnls)

    @staticmethod
    def _bars_to_minutes(bars: int, timeframe: str) -> float:
        minutes_map = {"5M": 5, "15M": 15, "1H": 60}
        return round(bars * minutes_map.get(timeframe, 5), 1)

    def _load_v2_card(self) -> dict[str, Any]:
        if not self.v2_optimization_path.exists():
            raise V2SignalRankingError(
                f"V2 frequency optimization export not found: {self.v2_optimization_path}",
            )
        with self.v2_optimization_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        card = payload.get("smartmoneyengine_v2_production_card")
        if not card:
            raise V2SignalRankingError("V2 production card missing from optimization export.")
        return card

    @staticmethod
    def _trait_value(trait_tags: tuple[str, ...], prefix: str) -> str:
        for tag in trait_tags:
            if tag.startswith(prefix):
                return tag.split(": ", 1)[-1] if ": " in tag else tag.replace(prefix, "").strip()
        return "Unknown"

    @staticmethod
    def _liquidity_context(trait_tags: tuple[str, ...]) -> str:
        for tag in trait_tags:
            if tag.startswith("Liquidity Distance"):
                return tag.replace("Liquidity Distance ", "")
        return "Unknown"

    @staticmethod
    def _choch_bos_timing(trait_tags: tuple[str, ...], comparative: ComparativeTradeRecord) -> str:
        for tag in trait_tags:
            if tag.startswith("CHOCH->BOS"):
                return tag.replace("CHOCH->BOS ", "")
        if comparative.choch_to_bos_minutes < 30:
            return "Fast (<30 min)"
        if comparative.choch_to_bos_minutes <= 90:
            return "Moderate (30-90 min)"
        return "Slow (>90 min)"

    @staticmethod
    def _ema_structure(flags: dict[str, bool]) -> str:
        if flags.get("ema_bull_stack"):
            return "EMA20 > EMA50 > EMA200"
        if flags.get("ema_bear_stack"):
            return "EMA20 < EMA50 < EMA200"
        return "Mixed / Unaligned"

    @staticmethod
    def _vwap_state(flags: dict[str, bool]) -> str:
        if flags.get("below_vwap"):
            return "Below VWAP"
        if flags.get("above_vwap"):
            return "Above VWAP"
        return "At VWAP"

    def _simulate_trade_profile(
        self,
        frame: pd.DataFrame,
        signal: TierSignal,
        candidate_engine: SmartMoneyEngineProductionCandidateResearch,
    ) -> dict[str, Any]:
        simulation = candidate_engine._simulate_r_hits(frame, signal)
        if not simulation:
            return {}

        entry_bar = signal.bos_bar
        end = min(len(frame) - 1, entry_bar + FORWARD_BARS)
        direction = signal.direction
        entry_price = float(frame.iloc[entry_bar]["Close"])
        risk = simulation["risk_points"]
        mae = 0.0
        holding_bars = end - entry_bar

        stop, _ = candidate_engine.trade_engine._structural_stop(
            frame,
            entry_bar,
            entry_price,
            direction,
        )
        target = candidate_engine.trade_engine._opposite_liquidity_target(
            frame,
            entry_bar,
            entry_price,
            direction,
            risk,
        )

        for index in range(entry_bar + 1, end + 1):
            bar_high = float(frame.iloc[index]["High"])
            bar_low = float(frame.iloc[index]["Low"])
            if direction == "bullish":
                mae = max(mae, max(entry_price - bar_low, 0.0))
                if bar_low <= stop or bar_high >= target:
                    holding_bars = index - entry_bar
                    break
            else:
                mae = max(mae, max(bar_high - entry_price, 0.0))
                if bar_high >= stop or bar_low <= target:
                    holding_bars = index - entry_bar
                    break

        return {
            **simulation,
            "mae_points": round(mae, 2),
            "holding_bars": holding_bars,
            "holding_minutes": self._bars_to_minutes(holding_bars, signal.timeframe),
        }

    def _build_archetype_key(self, dimensions: dict[str, str]) -> str:
        return " | ".join(f"{dim}={dimensions[dim]}" for dim in GROUPING_DIMENSIONS)

    def _collect_v2_signals(
        self,
        metadata: dict[str, Any],
        v2_card: dict[str, Any],
    ) -> list[RankedV2Signal]:
        buy_keys = self._frequency_engine._keys_from_labels(v2_card["buy_rules"]["filter_stack"])
        sell_keys = self._frequency_engine._keys_from_labels(v2_card["sell_rules"]["filter_stack"])
        no_trade_rules = list(v2_card.get("no_trade_rules", []))

        end = date.fromisoformat(metadata["end_date"])
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=self.research_days)
        )

        ranked: list[RankedV2Signal] = []
        for symbol in self.symbols:
            candidate_engine = SmartMoneyEngineProductionCandidateResearch(
                symbol=symbol,
                research_days=self.research_days,
                timeframes=self.timeframes,
            )
            comparison_engine = Tier2WinnerLoserComparisonResearch(
                symbol=symbol,
                research_days=self.research_days,
                timeframes=self.timeframes,
            )
            comparative_lookup = {
                (record.timeframe, record.bos_timestamp): record
                for record in comparison_engine._collect_records(metadata)
            }

            for timeframe_label in self.timeframes:
                path = comparison_engine.tier_engine.filter_engine._ensure_pipeline(
                    timeframe_label,
                    start,
                    end,
                )
                frame = pd.read_csv(path).reset_index(drop=True)
                filter_frame = comparison_engine.filter_context.enrich(frame)

                for signal in comparison_engine.tier_engine._detect_tier2(frame, timeframe_label):
                    key = (timeframe_label, signal.bos_timestamp)
                    comparative = comparative_lookup.get(key)
                    if comparative is None:
                        continue

                    trait_tags = comparative.trait_tags
                    if self._frequency_engine._no_trade_blocked(trait_tags, no_trade_rules):
                        continue

                    flags = candidate_engine._extended_flags(
                        frame,
                        filter_frame,
                        signal,
                        comparative,
                    )
                    side = "BUY" if signal.direction == "bullish" else "SELL"
                    required_keys = buy_keys if side == "BUY" else sell_keys
                    if not all(flags.get(key, False) for key in required_keys):
                        continue

                    profile = self._simulate_trade_profile(frame, signal, candidate_engine)
                    if not profile:
                        continue

                    dimensions = {
                        "symbol": symbol,
                        "timeframe": timeframe_label,
                        "direction": side,
                        "session": comparative.session,
                        "vwap_state": self._vwap_state(flags),
                        "rsi_bucket": comparative.rsi_band,
                        "ema_structure": self._ema_structure(flags),
                        "choch_bos_timing": self._choch_bos_timing(trait_tags, comparative),
                        "displacement_strength": comparative.displacement_strength,
                        "level_context": comparative.market_location,
                        "liquidity_context": self._liquidity_context(trait_tags),
                        "confirmation_candle": (
                            "Strong Confirmation" if flags.get("strong_confirmation") else "Weak"
                        ),
                    }

                    ranked.append(
                        RankedV2Signal(
                            symbol=symbol,
                            bos_timestamp=signal.bos_timestamp,
                            timeframe=timeframe_label,
                            signal_side=side,
                            direction=signal.direction,
                            session=dimensions["session"],
                            vwap_state=dimensions["vwap_state"],
                            rsi_bucket=dimensions["rsi_bucket"],
                            ema_structure=dimensions["ema_structure"],
                            choch_bos_timing=dimensions["choch_bos_timing"],
                            displacement_strength=dimensions["displacement_strength"],
                            level_context=dimensions["level_context"],
                            liquidity_context=dimensions["liquidity_context"],
                            confirmation_candle=dimensions["confirmation_candle"],
                            risk_points=profile["risk_points"],
                            realized_pnl_points=profile["realized_pnl_points"],
                            realized_rr=profile["realized_rr"],
                            win=profile["win"],
                            hit_1r=profile["hit_1r_before_sl"],
                            hit_2r=profile["hit_2r_before_sl"],
                            hit_3r=profile["hit_3r_before_sl"],
                            mae_points=profile["mae_points"],
                            holding_bars=profile["holding_bars"],
                            holding_minutes=profile["holding_minutes"],
                            archetype_key=self._build_archetype_key(dimensions),
                        ),
                    )
        return ranked

    @staticmethod
    def _quality_score(metrics: dict[str, Any]) -> float:
        if metrics["sample_size"] < MIN_SAMPLE_SIZE:
            return 0.0
        wr = metrics["win_rate_pct"]
        pf = metrics.get("profit_factor") or 0.0
        exp = metrics["expectancy"]
        hit_1r = metrics["hit_1r_rate_pct"]
        hit_2r = metrics["hit_2r_rate_pct"]
        hit_3r = metrics["hit_3r_rate_pct"]

        wr_score = min(wr / 70.0, 1.0) * 25.0
        pf_score = min(max(pf - 1.0, 0.0) / 2.0, 1.0) * 25.0
        exp_score = min(max(exp, 0.0) / 150.0, 1.0) * 25.0
        hit_score = min((hit_1r * 0.10 + hit_2r * 0.15 + hit_3r * 0.20) / 100.0, 1.0) * 25.0
        return round(wr_score + pf_score + exp_score + hit_score, 2)

    @staticmethod
    def _tier_for_score(score: float, sample_size: int) -> tuple[str, bool, str | None]:
        if sample_size < MIN_SAMPLE_SIZE:
            return "D", True, f"sample_size_below_{MIN_SAMPLE_SIZE}"
        if score >= 80.0:
            return "A", False, None
        if score >= 65.0:
            return "B", False, None
        if score >= 50.0:
            return "C", False, None
        return "D", False, None

    def _aggregate_signals(
        self,
        signals: list[RankedV2Signal],
        *,
        archetype_key: str,
        signal_side: str | None,
        grouping_dimension: str | None,
        grouping_value: str | None,
        research_days: int,
    ) -> ArchetypeMetrics:
        pnls = [item.realized_pnl_points for item in signals]
        total = len(signals)
        wins = sum(1 for item in signals if item.win)
        months = max(research_days / 30.4375, 1.0)
        pf = self._profit_factor(pnls)
        metrics = {
            "sample_size": total,
            "win_rate_pct": round(wins / total * 100, 2) if total else 0.0,
            "profit_factor": pf,
            "expectancy": round(mean(pnls), 2) if pnls else 0.0,
            "hit_1r_rate_pct": round(sum(1 for item in signals if item.hit_1r) / total * 100, 2)
            if total
            else 0.0,
            "hit_2r_rate_pct": round(sum(1 for item in signals if item.hit_2r) / total * 100, 2)
            if total
            else 0.0,
            "hit_3r_rate_pct": round(sum(1 for item in signals if item.hit_3r) / total * 100, 2)
            if total
            else 0.0,
        }
        score = self._quality_score(metrics)
        tier, rejected, reason = self._tier_for_score(score, total)
        side = signal_side or (signals[0].signal_side if signals else None)
        return ArchetypeMetrics(
            archetype_key=archetype_key,
            signal_side=side,
            grouping_dimension=grouping_dimension,
            grouping_value=grouping_value,
            sample_size=total,
            signals_per_month=round(total / months, 2) if total else 0.0,
            win_rate_pct=metrics["win_rate_pct"],
            profit_factor=pf,
            expectancy=metrics["expectancy"],
            hit_1r_rate_pct=metrics["hit_1r_rate_pct"],
            hit_2r_rate_pct=metrics["hit_2r_rate_pct"],
            hit_3r_rate_pct=metrics["hit_3r_rate_pct"],
            average_drawdown_points=round(mean(item.mae_points for item in signals), 2)
            if signals
            else 0.0,
            average_holding_minutes=round(mean(item.holding_minutes for item in signals), 2)
            if signals
            else 0.0,
            maximum_drawdown_points=self._maximum_drawdown(pnls),
            net_points=round(sum(pnls), 2),
            signal_quality_score=score,
            tier=tier,
            rejected=rejected,
            rejection_reason=reason,
        )

    def _grouped_analysis(
        self,
        signals: list[RankedV2Signal],
        research_days: int,
    ) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for dimension in GROUPING_DIMENSIONS:
            buckets: dict[str, list[RankedV2Signal]] = defaultdict(list)
            for item in signals:
                buckets[item.dimension_values()[dimension]].append(item)
            rows: list[dict[str, Any]] = []
            for value, bucket in buckets.items():
                metrics = self._aggregate_signals(
                    bucket,
                    archetype_key=f"{dimension}={value}",
                    signal_side=None,
                    grouping_dimension=dimension,
                    grouping_value=value,
                    research_days=research_days,
                )
                rows.append(metrics.as_dict())
            rows.sort(key=lambda row: row["signal_quality_score"], reverse=True)
            grouped[dimension] = rows
        return grouped

    def _rank_archetypes(
        self,
        signals: list[RankedV2Signal],
        research_days: int,
    ) -> list[ArchetypeMetrics]:
        buckets: dict[str, list[RankedV2Signal]] = defaultdict(list)
        for item in signals:
            dim_vals = item.dimension_values()
            for size in ARCHETYPE_COMBO_SIZES:
                for combo in combinations(GROUPING_DIMENSIONS, size):
                    if "direction" not in combo:
                        continue
                    key = " | ".join(f"{dim}={dim_vals[dim]}" for dim in combo)
                    buckets[key].append(item)

        ranked: list[ArchetypeMetrics] = []
        seen: set[str] = set()
        for key, bucket in buckets.items():
            if len(bucket) < MIN_SAMPLE_SIZE or key in seen:
                continue
            seen.add(key)
            ranked.append(
                self._aggregate_signals(
                    bucket,
                    archetype_key=key,
                    signal_side=bucket[0].signal_side,
                    grouping_dimension="multi_dimension_combo",
                    grouping_value=key,
                    research_days=research_days,
                ),
            )
        ranked.sort(
            key=lambda item: (-item.signal_quality_score, -item.sample_size),
        )
        return ranked

    @staticmethod
    def _split_tiers(archetypes: list[ArchetypeMetrics]) -> dict[str, list[dict[str, Any]]]:
        tiers: dict[str, list[dict[str, Any]]] = {
            "tier_a": [],
            "tier_b": [],
            "tier_c": [],
            "tier_d": [],
            "rejected": [],
        }
        for item in archetypes:
            payload = item.as_dict()
            if item.rejected:
                tiers["rejected"].append(payload)
            elif item.tier == "A":
                tiers["tier_a"].append(payload)
            elif item.tier == "B":
                tiers["tier_b"].append(payload)
            elif item.tier == "C":
                tiers["tier_c"].append(payload)
            else:
                tiers["tier_d"].append(payload)
        return tiers

    def run(self, metadata: dict[str, Any]) -> V2SignalRankingReport:
        started = time.perf_counter()
        v2_card = self._load_v2_card()
        research_days = metadata.get("research_window_days", self.research_days)

        signals = self._collect_v2_signals(metadata, v2_card)
        grouped = self._grouped_analysis(signals, research_days)
        all_archetypes = self._rank_archetypes(signals, research_days)
        qualifying = [item for item in all_archetypes if not item.rejected]
        top_archetypes = qualifying[:TOP_ARCHETYPE_COUNT]

        buy_models = [
            item.as_dict()
            for item in sorted(
                [row for row in qualifying if row.signal_side == "BUY"],
                key=lambda row: (-row.signal_quality_score, -row.sample_size),
            )[:TOP_MODEL_COUNT]
        ]
        sell_models = [
            item.as_dict()
            for item in sorted(
                [row for row in qualifying if row.signal_side == "SELL"],
                key=lambda row: (-row.signal_quality_score, -row.sample_size),
            )[:TOP_MODEL_COUNT]
        ]

        tiers = self._split_tiers(all_archetypes)

        conclusions = [
            f"V2 signal ranking analyzed {len(signals)} signals from the V2 production card.",
            f"Minimum sample size enforced: {MIN_SAMPLE_SIZE} (below rejected).",
            f"Unique archetypes: {len(all_archetypes)}; qualifying (n>={MIN_SAMPLE_SIZE}): {len(qualifying)}.",
            f"Tier A: {len(tiers['tier_a'])}, Tier B: {len(tiers['tier_b'])}, "
            f"Tier C: {len(tiers['tier_c'])}, Tier D: {len(tiers['tier_d'])}, "
            f"Rejected: {len(tiers['rejected'])}.",
            f"Top BUY model score: {buy_models[0]['signal_quality_score'] if buy_models else 'N/A'}.",
            f"Top SELL model score: {sell_models[0]['signal_quality_score'] if sell_models else 'N/A'}.",
        ]

        return V2SignalRankingReport(
            symbols_analyzed=list(self.symbols),
            research_window_days=research_days,
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            v2_production_card=v2_card,
            minimum_sample_size=MIN_SAMPLE_SIZE,
            total_v2_signals=len(signals),
            grouping_dimensions=list(GROUPING_DIMENSIONS),
            grouped_analysis=grouped,
            top_50_signal_archetypes=[item.as_dict() for item in top_archetypes],
            top_10_buy_models=buy_models,
            top_10_sell_models=sell_models,
            tier_a_archetypes=tiers["tier_a"],
            tier_b_archetypes=tiers["tier_b"],
            tier_c_archetypes=tiers["tier_c"],
            tier_d_archetypes=tiers["tier_d"],
            rejected_archetypes=tiers["rejected"],
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )


def generate_v2_signal_ranking_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
    v2_optimization_path: Path | str | None = None,
    symbols: tuple[str, ...] | None = None,
) -> V2SignalRankingReport:
    """Run V2 signal ranking research and export JSON."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise V2SignalRankingError(f"Filter research report not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = SmartMoneyEngineV2SignalRankingResearch(
        symbols=symbols,
        v2_optimization_path=v2_optimization_path,
    )
    report = engine.run(metadata)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info(
        "V2 signal ranking completed: signals=%s archetypes=%s tier_a=%s",
        report.total_v2_signals,
        len(report.top_50_signal_archetypes),
        len(report.tier_a_archetypes),
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_v2_signal_ranking_report()
        print("SmartMoneyEngine V2 Signal Ranking Summary")
        print(f"Total V2 signals: {report.total_v2_signals}")
        print(f"Qualifying archetypes: {len(report.top_50_signal_archetypes)}")
        print(f"Tier A: {len(report.tier_a_archetypes)} | Tier B: {len(report.tier_b_archetypes)}")
        print(f"Top BUY models: {len(report.top_10_buy_models)}")
        print(f"Top SELL models: {len(report.top_10_sell_models)}")
        if report.top_10_sell_models:
            top = report.top_10_sell_models[0]
            print(
                f"Best SELL: score={top['signal_quality_score']} "
                f"WR={top['win_rate_pct']}% PF={top['profit_factor']} "
                f"Exp={top['expectancy']}",
            )
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except V2SignalRankingError as exc:
        logger.error("V2 signal ranking error: %s", exc)
        print(f"V2 signal ranking error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected V2 signal ranking error")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
