"""
Tier-2 winner vs loser comparative research for SmartMoneyEngine.

Compares top 25% winners against bottom 25% losers across structural and
contextual traits for Tier-2 BOS Close trades. Research-only.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.context.institutional_market_narrative_engine_v2 import InstitutionalMarketNarrativeEngineV2
from src.context.market_intelligence_engine import MarketIntelligenceEngine
from src.research.filter_research_engine import (
    FilterContextBuilder,
    RESEARCH_DAYS,
    _json_safe,
)
from src.research.institutional_edge_extraction_research import (
    DISTANCE_BUCKETS,
    FRESHNESS_BUCKETS,
    FVG_SIZE_BUCKETS,
    TIMING_BUCKETS,
    InstitutionalEdgeExtractionResearch,
)
from src.research.rsi_divergence_research_engine import DivergenceType, RsiDivergenceDetector
from src.research.signal_funnel_analyzer import SignalFunnelAnalyzer
from src.research.tier2_regime_classification_research import Tier2RegimeClassificationResearch
from src.research.tier2_trade_distribution_research import TIER2_DEFINITION
from src.research.tiered_signal_framework_research import TierSignal, TieredSignalFrameworkResearch

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "tier2_winner_loser_comparison.json"

COHORT_FRACTION = 0.25
TOP_TRAIT_COUNT = 20
MIN_SIGNIFICANT_EDGE_PCT = 5.0

FEATURES_COMPARED = (
    "Displacement Strength",
    "FVG Size",
    "FVG Age",
    "FVG Retest Count",
    "CHOCH to BOS Timing",
    "Distance From Liquidity",
    "Distance From Swing High/Low",
    "Session",
    "Timeframe",
    "Market Intelligence Score",
    "Narrative Confidence",
    "Regime Classification",
    "Market Location",
    "RSI",
    "RSI Divergence",
)

MI_SCORE_BUCKETS = (
    ("MI Below 50", 0, 50),
    ("MI 50-64", 50, 65),
    ("MI 65-79", 65, 80),
    ("MI 80+", 80, 101),
)

NARRATIVE_CONFIDENCE_BUCKETS = (
    ("Confidence Below 50", 0, 50),
    ("Confidence 50-64", 50, 65),
    ("Confidence 65-79", 65, 80),
    ("Confidence 80+", 80, 101),
)


class Tier2WinnerLoserComparisonError(Exception):
    """Raised when Tier-2 winner/loser comparison fails."""


@dataclass(frozen=True)
class ComparativeTradeRecord:
    """Tier-2 trade with outcome and comparative trait tags."""

    bos_timestamp: str
    timeframe: str
    direction: str
    realized_pnl_points: float
    realized_rr: float
    win: bool
    displacement_strength: str
    fvg_size_points: float
    fvg_age_bars: int
    fvg_retest_count: int
    choch_to_bos_minutes: float
    distance_from_liquidity_points: float
    distance_from_swing_points: float
    session: str
    intelligence_score: float
    narrative_confidence: int
    regime: str
    market_location: str
    rsi: float
    rsi_band: str
    rsi_divergence: str
    trait_tags: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TraitFrequencyComparison:
    """Winner vs loser frequency for one trait label."""

    trait: str
    category: str
    winner_frequency_pct: float
    loser_frequency_pct: float
    edge_pct: float
    winner_count: int
    loser_count: int
    more_common_in: str
    statistically_significant: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Tier2WinnerLoserComparisonReport:
    """Full Tier-2 winner vs loser comparison output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    tier_definition: list[str]
    entry_method: str
    stop_loss_model: str
    cohort_fraction: float
    features_compared: list[str]
    total_signals: int
    cohort_size: int
    top_25_pct_winners_summary: dict[str, Any]
    bottom_25_pct_losers_summary: dict[str, Any]
    trait_frequency_comparison: list[dict[str, Any]]
    traits_more_common_in_winners: list[dict[str, Any]]
    traits_more_common_in_losers: list[dict[str, Any]]
    top_20_winning_traits: list[dict[str, Any]]
    top_20_losing_traits: list[dict[str, Any]]
    numeric_trait_comparison: dict[str, dict[str, Any]]
    conclusions: list[str]
    execution_time_seconds: float
    sample_top_winners: list[dict[str, Any]] = field(default_factory=list)
    sample_bottom_losers: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Tier2WinnerLoserComparisonResearch:
    """Compare Tier-2 winner and loser trait profiles."""

    def __init__(
        self,
        symbol: str = "NIFTY50",
        research_days: int = RESEARCH_DAYS,
        timeframes: tuple[str, ...] = ("5M", "15M", "1H"),
    ) -> None:
        self.symbol = symbol
        self.research_days = research_days
        self.timeframes = timeframes
        self.tier_engine = TieredSignalFrameworkResearch(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )
        self.edge_engine = InstitutionalEdgeExtractionResearch(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )
        self.regime_engine = Tier2RegimeClassificationResearch(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )
        self.intelligence_engine = MarketIntelligenceEngine(symbol=symbol)
        self.narrative_v2_engine = InstitutionalMarketNarrativeEngineV2(symbol=symbol)
        self.divergence_detector = RsiDivergenceDetector()
        self.filter_context = FilterContextBuilder()
        self.funnel_analyzer = SignalFunnelAnalyzer(symbol=symbol)

    @staticmethod
    def _bucket_label(value: float, buckets: tuple[tuple[str, float, float], ...]) -> str:
        return InstitutionalEdgeExtractionResearch._bucket_label(value, buckets)

    @staticmethod
    def _divergence_label(primary: DivergenceType) -> str:
        if primary == DivergenceType.NONE:
            return "No RSI Divergence"
        return f"RSI {primary.value}"

    @staticmethod
    def _market_location_label(location: str) -> str:
        mapping = {
            "Near Support": "Near Support",
            "Near Resistance": "Near Resistance",
            "Mid Range": "Mid Range",
        }
        return mapping.get(location, location)

    @staticmethod
    def _fvg_retest_label(count: int) -> str:
        if count == 0:
            return "FVG Retests 0"
        if count == 1:
            return "FVG Retests 1"
        return "FVG Retests 2+"

    @staticmethod
    def _trait_category(trait: str) -> str:
        prefixes = (
            ("RSI Divergence:", "RSI Divergence"),
            ("RSI:", "RSI"),
            ("Displacement:", "Displacement Strength"),
            ("FVG Size", "FVG Size"),
            ("FVG Age", "FVG Age"),
            ("FVG Retests", "FVG Retest Count"),
            ("CHOCH->BOS", "CHOCH to BOS Timing"),
            ("Liquidity Distance", "Distance From Liquidity"),
            ("Swing Distance", "Distance From Swing High/Low"),
            ("MI Score:", "Market Intelligence Score"),
            ("Narrative Confidence", "Narrative Confidence"),
            ("Regime:", "Regime Classification"),
            ("Market Location:", "Market Location"),
            ("Session:", "Session"),
            ("Timeframe:", "Timeframe"),
        )
        for prefix, category in prefixes:
            if trait.startswith(prefix):
                return category
        return "Other"

    def _build_trait_tags(self, record: ComparativeTradeRecord) -> tuple[str, ...]:
        return (
            f"Displacement: {record.displacement_strength}",
            f"FVG Size {self._bucket_label(record.fvg_size_points, FVG_SIZE_BUCKETS)}",
            f"FVG Age {self._bucket_label(float(record.fvg_age_bars), FRESHNESS_BUCKETS)}",
            self._fvg_retest_label(record.fvg_retest_count),
            (
                "CHOCH->BOS "
                + self._bucket_label(record.choch_to_bos_minutes, TIMING_BUCKETS)
            ),
            (
                "Liquidity Distance "
                + self._bucket_label(record.distance_from_liquidity_points, DISTANCE_BUCKETS)
            ),
            (
                "Swing Distance "
                + self._bucket_label(record.distance_from_swing_points, DISTANCE_BUCKETS)
            ),
            f"Session: {record.session}",
            f"Timeframe: {record.timeframe}",
            f"MI Score: {self._bucket_label(record.intelligence_score, MI_SCORE_BUCKETS)}",
            (
                "Narrative Confidence "
                + self._bucket_label(float(record.narrative_confidence), NARRATIVE_CONFIDENCE_BUCKETS)
            ),
            f"Regime: {record.regime}",
            f"Market Location: {record.market_location}",
            f"RSI: {record.rsi_band}",
            f"RSI Divergence: {record.rsi_divergence}",
        )

    def _analyze_signal(
        self,
        frame: pd.DataFrame,
        intel_frame: pd.DataFrame,
        filter_frame: pd.DataFrame,
        rsi_series: pd.Series,
        htf_lookup: pd.DataFrame,
        htf_frame_lookup: pd.DataFrame,
        timestamps: pd.Series,
        signal: TierSignal,
    ) -> ComparativeTradeRecord | None:
        detail = self.edge_engine.distribution_engine._simulate_detailed(frame, signal)
        if detail is None:
            return None

        feature = self.edge_engine._extract_features(
            frame,
            signal,
            detail.realized_pnl_points,
            detail.win,
            detail.risk_points,
        )
        if feature is None:
            return None

        bar = signal.bos_bar
        intelligence = self.intelligence_engine.evaluate_bar(intel_frame, bar)
        narrative = self.narrative_v2_engine.evaluate_signal(frame, signal, feature)

        divergence_types = self.divergence_detector.detect(filter_frame, bar, rsi_series)
        primary_div = self.divergence_detector.primary_divergence(
            divergence_types,
            signal.direction,
        )

        ts = pd.Timestamp(timestamps.iloc[bar])
        session = (
            FilterContextBuilder._session_label(ts)
            if pd.notna(ts)
            else "Unknown"
        )
        htf_1h = self.regime_engine._htf_1h_trend_at(htf_lookup, ts)
        htf_4h = str(htf_frame_lookup.iloc[bar].get("HTF_4H_Trend", "SIDEWAYS")).upper()
        htf_1d = str(htf_frame_lookup.iloc[bar].get("HTF_1D_Trend", "SIDEWAYS")).upper()
        regime = self.regime_engine.classify_regime(
            frame,
            signal,
            session,
            htf_1h,
            htf_4h,
            htf_1d,
        )

        rsi = float(intel_frame.iloc[bar]["RSI"]) if pd.notna(intel_frame.iloc[bar]["RSI"]) else 50.0
        rsi_band = FilterContextBuilder._rsi_band(rsi)
        market_location = self._market_location_label(intelligence.market_location)
        realized_rr = (
            round(detail.realized_pnl_points / detail.risk_points, 2)
            if detail.risk_points > 0
            else 0.0
        )

        draft = ComparativeTradeRecord(
            bos_timestamp=signal.bos_timestamp,
            timeframe=signal.timeframe,
            direction=signal.direction,
            realized_pnl_points=detail.realized_pnl_points,
            realized_rr=realized_rr,
            win=detail.win,
            displacement_strength=feature.displacement_strength,
            fvg_size_points=feature.fvg_size_points,
            fvg_age_bars=feature.fvg_freshness_bars,
            fvg_retest_count=feature.fvg_retests,
            choch_to_bos_minutes=feature.choch_to_bos_minutes,
            distance_from_liquidity_points=feature.distance_from_liquidity_pool_points,
            distance_from_swing_points=feature.distance_from_swing_points,
            session=session,
            intelligence_score=intelligence.intelligence_score,
            narrative_confidence=narrative.narrative_confidence,
            regime=regime,
            market_location=market_location,
            rsi=round(rsi, 2),
            rsi_band=rsi_band,
            rsi_divergence=self._divergence_label(primary_div),
            trait_tags=(),
        )
        tags = self._build_trait_tags(draft)
        return ComparativeTradeRecord(**{**draft.as_dict(), "trait_tags": tags})

    def _collect_records(self, metadata: dict[str, Any]) -> list[ComparativeTradeRecord]:
        from datetime import timedelta

        end = date.fromisoformat(metadata["end_date"])
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=self.research_days)
        )

        htf_lookup = self.regime_engine._ensure_htf_1h_lookup(start, end)
        records: list[ComparativeTradeRecord] = []

        for timeframe_label in self.timeframes:
            path = self.tier_engine.filter_engine._ensure_pipeline(timeframe_label, start, end)
            frame = pd.read_csv(path).reset_index(drop=True)
            intel_frame = self.intelligence_engine.enrich(frame)
            filter_frame = self.filter_context.enrich(frame)
            rsi_series = self.divergence_detector._compute_rsi(frame["Close"].astype(float))
            timestamps = pd.to_datetime(frame["Date"], errors="coerce")
            htf_frame_lookup = self.funnel_analyzer._build_htf_lookup(frame)

            for signal in self.tier_engine._detect_tier2(frame, timeframe_label):
                record = self._analyze_signal(
                    frame,
                    intel_frame,
                    filter_frame,
                    rsi_series,
                    htf_lookup,
                    htf_frame_lookup,
                    timestamps,
                    signal,
                )
                if record is not None:
                    records.append(record)

        records.sort(key=lambda item: pd.Timestamp(item.bos_timestamp))
        return records

    @staticmethod
    def _cohort_summary(records: list[ComparativeTradeRecord]) -> dict[str, Any]:
        if not records:
            return {"count": 0}
        return {
            "count": len(records),
            "average_pnl": round(mean(record.realized_pnl_points for record in records), 2),
            "win_rate_pct": round(sum(1 for record in records if record.win) / len(records) * 100, 2),
            "average_rsi": round(mean(record.rsi for record in records), 2),
            "average_intelligence_score": round(
                mean(record.intelligence_score for record in records),
                2,
            ),
            "average_narrative_confidence": round(
                mean(record.narrative_confidence for record in records),
                2,
            ),
            "average_fvg_size": round(mean(record.fvg_size_points for record in records), 2),
            "average_fvg_age_bars": round(mean(record.fvg_age_bars for record in records), 2),
            "average_fvg_retest_count": round(mean(record.fvg_retest_count for record in records), 2),
            "average_choch_to_bos_minutes": round(
                mean(record.choch_to_bos_minutes for record in records),
                2,
            ),
            "average_distance_from_liquidity": round(
                mean(record.distance_from_liquidity_points for record in records),
                2,
            ),
            "average_distance_from_swing": round(
                mean(record.distance_from_swing_points for record in records),
                2,
            ),
        }

    def _trait_comparisons(
        self,
        winners: list[ComparativeTradeRecord],
        losers: list[ComparativeTradeRecord],
    ) -> list[TraitFrequencyComparison]:
        winner_tags = [tag for record in winners for tag in record.trait_tags]
        loser_tags = [tag for record in losers for tag in record.trait_tags]
        all_traits = sorted(set(winner_tags) | set(loser_tags))

        comparisons: list[TraitFrequencyComparison] = []
        for trait in all_traits:
            winner_count = winner_tags.count(trait)
            loser_count = loser_tags.count(trait)
            winner_pct = round(winner_count / len(winners) * 100, 2) if winners else 0.0
            loser_pct = round(loser_count / len(losers) * 100, 2) if losers else 0.0
            edge = round(winner_pct - loser_pct, 2)
            if edge > 0:
                more_common = "Winners"
            elif edge < 0:
                more_common = "Losers"
            else:
                more_common = "Equal"
            comparisons.append(
                TraitFrequencyComparison(
                    trait=trait,
                    category=self._trait_category(trait),
                    winner_frequency_pct=winner_pct,
                    loser_frequency_pct=loser_pct,
                    edge_pct=edge,
                    winner_count=winner_count,
                    loser_count=loser_count,
                    more_common_in=more_common,
                    statistically_significant=abs(edge) >= MIN_SIGNIFICANT_EDGE_PCT,
                )
            )
        return comparisons

    @staticmethod
    def _numeric_trait_comparison(
        winners: list[ComparativeTradeRecord],
        losers: list[ComparativeTradeRecord],
    ) -> dict[str, dict[str, Any]]:
        fields = {
            "intelligence_score": "Market Intelligence Score",
            "narrative_confidence": "Narrative Confidence",
            "fvg_size_points": "FVG Size",
            "fvg_age_bars": "FVG Age",
            "fvg_retest_count": "FVG Retest Count",
            "choch_to_bos_minutes": "CHOCH to BOS Timing",
            "distance_from_liquidity_points": "Distance From Liquidity",
            "distance_from_swing_points": "Distance From Swing High/Low",
            "rsi": "RSI",
        }
        comparison: dict[str, dict[str, Any]] = {}
        for field_name, label in fields.items():
            winner_vals = [getattr(record, field_name) for record in winners]
            loser_vals = [getattr(record, field_name) for record in losers]
            winner_mean = round(mean(winner_vals), 2) if winner_vals else 0.0
            loser_mean = round(mean(loser_vals), 2) if loser_vals else 0.0
            comparison[label] = {
                "winner_mean": winner_mean,
                "loser_mean": loser_mean,
                "edge_mean": round(winner_mean - loser_mean, 2),
                "higher_in": "Winners" if winner_mean > loser_mean else "Losers",
            }
        return comparison

    def run(self, metadata: dict[str, Any]) -> Tier2WinnerLoserComparisonReport:
        """Run Tier-2 winner vs loser comparison research."""
        started = time.perf_counter()
        records = self._collect_records(metadata)
        if not records:
            raise Tier2WinnerLoserComparisonError("No Tier-2 comparative records found.")

        cohort_size = max(1, int(len(records) * COHORT_FRACTION))
        ranked = sorted(records, key=lambda item: item.realized_pnl_points, reverse=True)
        top_winners = ranked[:cohort_size]
        bottom_losers = ranked[-cohort_size:]

        comparisons = self._trait_comparisons(top_winners, bottom_losers)
        winning_sorted = sorted(comparisons, key=lambda item: item.edge_pct, reverse=True)
        losing_sorted = sorted(comparisons, key=lambda item: item.edge_pct)

        winners_significant = [
            item for item in winning_sorted if item.edge_pct > 0 and item.statistically_significant
        ]
        losers_significant = [
            item
            for item in losing_sorted
            if item.edge_pct < 0 and item.statistically_significant
        ]

        top_20_winning = [item.as_dict() for item in winning_sorted[:TOP_TRAIT_COUNT]]
        top_20_losing = [item.as_dict() for item in losing_sorted[:TOP_TRAIT_COUNT]]

        winner_summary = self._cohort_summary(top_winners)
        loser_summary = self._cohort_summary(bottom_losers)

        conclusions = [
            f"Compared top 25% winners (n={cohort_size}) vs bottom 25% losers (n={cohort_size}) "
            f"across {len(records)} Tier-2 BOS Close signals and {len(FEATURES_COMPARED)} features.",
            (
                f"Top winners avg PnL {winner_summary['average_pnl']} vs "
                f"bottom losers {loser_summary['average_pnl']}."
            ),
        ]
        if top_20_winning:
            conclusions.append(
                f"Strongest winning trait: {top_20_winning[0]['trait']} "
                f"(+{top_20_winning[0]['edge_pct']} pp edge)."
            )
        if top_20_losing:
            conclusions.append(
                f"Strongest losing trait: {top_20_losing[0]['trait']} "
                f"({top_20_losing[0]['edge_pct']} pp edge)."
            )
        conclusions.append(
            f"Significant winner traits: {len(winners_significant)}; "
            f"significant loser traits: {len(losers_significant)}."
        )

        return Tier2WinnerLoserComparisonReport(
            symbol=metadata.get("symbol", self.symbol),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            timeframes_analyzed=list(self.timeframes),
            tier_definition=TIER2_DEFINITION,
            entry_method="BOS Close Entry",
            stop_loss_model="Structural Swing SL",
            cohort_fraction=COHORT_FRACTION,
            features_compared=list(FEATURES_COMPARED),
            total_signals=len(records),
            cohort_size=cohort_size,
            top_25_pct_winners_summary=winner_summary,
            bottom_25_pct_losers_summary=loser_summary,
            trait_frequency_comparison=[item.as_dict() for item in comparisons],
            traits_more_common_in_winners=[item.as_dict() for item in winners_significant],
            traits_more_common_in_losers=[item.as_dict() for item in losers_significant],
            top_20_winning_traits=top_20_winning,
            top_20_losing_traits=top_20_losing,
            numeric_trait_comparison=self._numeric_trait_comparison(top_winners, bottom_losers),
            sample_top_winners=[record.as_dict() for record in top_winners[:8]],
            sample_bottom_losers=[record.as_dict() for record in bottom_losers[:8]],
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_tier2_winner_loser_comparison_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> Tier2WinnerLoserComparisonReport:
    """Run Tier-2 winner/loser comparison and export JSON report."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise Tier2WinnerLoserComparisonError(
            f"Filter research report not found: {metadata_path}",
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = Tier2WinnerLoserComparisonResearch(
        symbol=metadata.get("symbol", "NIFTY50"),
        research_days=metadata.get("research_window_days", RESEARCH_DAYS),
        timeframes=tuple(metadata.get("timeframes_analyzed", ("5M", "15M", "1H"))),
    )
    report = engine.run(metadata)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info(
        "Tier-2 winner/loser comparison completed: %s signals",
        report.total_signals,
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_tier2_winner_loser_comparison_report()
        print("Tier-2 Winner vs Loser Comparison Summary")
        print(f"Signals: {report.total_signals} | Cohort: {report.cohort_size} (25%)")
        print("Top winning traits:")
        for item in report.top_20_winning_traits[:5]:
            print(f"  {item['trait']}: +{item['edge_pct']} pp edge")
        print("Top losing traits:")
        for item in report.top_20_losing_traits[:5]:
            print(f"  {item['trait']}: {item['edge_pct']} pp edge")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except Tier2WinnerLoserComparisonError as exc:
        logger.error("Tier-2 winner/loser comparison error: %s", exc)
        print(f"Tier-2 winner/loser comparison error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected Tier-2 winner/loser comparison failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
