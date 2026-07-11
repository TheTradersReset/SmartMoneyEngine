"""
Liquidity sweep tradeability validation research for SmartMoneyEngine.

Validates whether post-sweep BOS Close entries with structural swing stops
are tradable. Research-only; no signals or production modifications.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.context.institutional_liquidity_map_engine import InstitutionalLiquidityMapEngine
from src.context.liquidity_narrative_engine import FvgContext, LiquidityNarrativeEngine
from src.context.market_intelligence_engine import MarketIntelligenceEngine
from src.research.filter_research_engine import RESEARCH_DAYS, FilterResearchEngine, _json_safe
from src.research.institutional_quality_validation_research import (
    InstitutionalQualityValidationResearch,
)
from src.research.liquidity_sweep_outcome_validation_research import (
    LiquiditySweepOutcomeValidationResearch,
)
from src.research.tier2_production_validation_research import Tier2ProductionValidationResearch
from src.research.trade_construction_validation_research import TradeConstructionValidationResearch
from src.research.tiered_signal_framework_research import FORWARD_BARS
from src.signals.decision_engine import DecisionEngine

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "liquidity_sweep_tradeability.json"

BOS_LOOKFORWARD_BARS = 80
TOP_CONFIG_COUNT = 20
MIN_CONFIG_SAMPLES = 3
RR_LEVELS = (1, 2, 3)


class LiquiditySweepTradeabilityError(Exception):
    """Raised when liquidity sweep tradeability validation fails."""


@dataclass(frozen=True)
class SweepTradeabilityRecord:
    """One sweep-linked BOS Close trade with reachability and context."""

    sweep_timestamp: str
    entry_timestamp: str
    timeframe: str
    sweep_type: str
    trade_direction: str
    sweep_bar: int
    entry_bar: int
    entry_price: float
    stop_price: float
    risk_points: float
    opposite_liquidity_target: float
    htf_supply_demand_target: float
    sweep_quality_classification: str
    displacement_strength: str
    choch_present: bool
    bos_present_before_entry: bool
    fvg_reclaimed: bool
    market_location: str
    hit_1r_before_sl: bool
    hit_2r_before_sl: bool
    hit_3r_before_sl: bool
    hit_opposite_liquidity_before_sl: bool
    hit_htf_supply_demand_before_sl: bool
    realized_pnl_points: float
    realized_rr: float
    win: bool
    stopped_out: bool
    configuration_key: str
    configuration_label: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TradeabilityMetrics:
    """Aggregate tradeability metrics for one cohort."""

    label: str
    trades: int
    win_rate_pct: float
    profit_factor: float | None
    expectancy: float
    average_rr: float
    net_points: float
    maximum_drawdown_points: float
    hit_1r_before_sl_pct: float
    hit_2r_before_sl_pct: float
    hit_3r_before_sl_pct: float
    hit_opposite_liquidity_before_sl_pct: float
    hit_htf_supply_demand_before_sl_pct: float
    tradable_score: float
    rank: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LiquiditySweepTradeabilityReport:
    """Full liquidity sweep tradeability validation output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    entry_method: str
    stop_loss_model: str
    total_sweeps_detected: int
    tradable_trades: int
    untradable_sweeps_no_bos: int
    overall_metrics: dict[str, Any]
    reachability_summary: dict[str, float]
    by_sweep_quality: dict[str, dict[str, Any]]
    by_displacement: dict[str, dict[str, Any]]
    by_choch_present: dict[str, dict[str, Any]]
    by_bos_present: dict[str, dict[str, Any]]
    by_fvg_reclaimed: dict[str, dict[str, Any]]
    by_market_location: dict[str, dict[str, Any]]
    top_20_sweep_configurations: list[dict[str, Any]]
    trade_records: list[dict[str, Any]]
    tradable_structures_summary: list[str]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class LiquiditySweepTradeabilityResearch:
    """Validate tradability of liquidity sweep to BOS Close sequences."""

    def __init__(
        self,
        symbol: str = "NIFTY50",
        research_days: int = RESEARCH_DAYS,
        timeframes: tuple[str, ...] = ("5M", "15M", "1H"),
    ) -> None:
        self.symbol = symbol
        self.research_days = research_days
        self.timeframes = timeframes
        self.filter_engine = FilterResearchEngine(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )
        self.construction_engine = TradeConstructionValidationResearch(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )
        self.sweep_outcome_engine = LiquiditySweepOutcomeValidationResearch(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )
        self.liquidity_map_engine = InstitutionalLiquidityMapEngine(symbol=symbol)
        self.narrative_engine = LiquidityNarrativeEngine(symbol=symbol)
        self.intelligence_engine = MarketIntelligenceEngine(symbol=symbol)

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
    def _expected_direction(sweep_type: str) -> str:
        return "bearish" if sweep_type == "Buy Side Sweep" else "bullish"

    @staticmethod
    def _quality_group(classification: str) -> str:
        mapping = {
            "Weak Sweep": "Weak",
            "Medium Sweep": "Medium",
            "Strong Sweep": "Strong",
            "Institutional Sweep": "Institutional",
        }
        return mapping.get(classification, "Unknown")

    @staticmethod
    def _displacement_group(strength: str) -> str:
        if strength in {"Strong", "Medium", "Weak"}:
            return strength
        return "None"

    @staticmethod
    def _location_group(location: str) -> str:
        if location == "Near Support":
            return "Near Support"
        if location == "Near Resistance":
            return "Near Resistance"
        return "Mid Range"

    def _find_post_sweep_bos(
        self,
        frame: pd.DataFrame,
        sweep_bar: int,
        direction: str,
    ) -> int | None:
        bos_column = "Bullish_BOS" if direction == "bullish" else "Bearish_BOS"
        end = min(len(frame) - 1, sweep_bar + BOS_LOOKFORWARD_BARS)
        for index in range(sweep_bar + 1, end + 1):
            if self._is_active(frame.iloc[index].get(bos_column)):
                return index
        return None

    def _structure_flags_between(
        self,
        frame: pd.DataFrame,
        start_bar: int,
        end_bar: int,
        direction: str,
    ) -> tuple[bool, bool, bool]:
        choch_column = "Bullish_CHOCH" if direction == "bullish" else "Bearish_CHOCH"
        bos_column = "Bullish_BOS" if direction == "bullish" else "Bearish_BOS"
        choch = any(
            self._is_active(frame.iloc[index].get(choch_column))
            for index in range(start_bar, end_bar + 1)
        )
        bos = any(
            self._is_active(frame.iloc[index].get(bos_column))
            for index in range(start_bar, end_bar)
        )
        reclaimed = False
        for index in range(start_bar, end_bar + 1):
            window = self.narrative_engine._window(frame, index)
            fvg_context, fvg_bias = self.narrative_engine._fvg_context_state(frame, index, window)
            if fvg_context == FvgContext.RECLAIMED and (
                (direction == "bullish" and fvg_bias == "bullish")
                or (direction == "bearish" and fvg_bias == "bearish")
            ):
                reclaimed = True
        return choch, bos, reclaimed

    def _simulate_trade_path(
        self,
        frame: pd.DataFrame,
        entry_bar: int,
        entry_price: float,
        direction: str,
        stop: float,
        risk: float,
        opposite_target: float,
        htf_target: float,
    ) -> dict[str, Any]:
        end = min(len(frame) - 1, entry_bar + FORWARD_BARS)
        hit_1r = hit_2r = hit_3r = False
        hit_opposite = hit_htf = False
        stopped_out = False
        pnl = 0.0
        rr = 0.0
        win = False

        for index in range(entry_bar + 1, end + 1):
            bar_high = float(frame.iloc[index]["High"])
            bar_low = float(frame.iloc[index]["Low"])

            if direction == "bullish":
                favorable = bar_high - entry_price
                stop_hit = bar_low <= stop
            else:
                favorable = entry_price - bar_low
                stop_hit = bar_high >= stop

            if not stopped_out:
                if favorable >= risk:
                    hit_1r = True
                if favorable >= risk * 2:
                    hit_2r = True
                if favorable >= risk * 3:
                    hit_3r = True

            if stop_hit:
                stopped_out = True
                pnl = -risk
                rr = -1.0
                win = False
                break

            opposite_hit = (
                bar_high >= opposite_target if direction == "bullish" else bar_low <= opposite_target
            )
            htf_hit = bar_high >= htf_target if direction == "bullish" else bar_low <= htf_target

            if opposite_hit:
                hit_opposite = True
                pnl = round(abs(opposite_target - entry_price), 2)
                rr = round(pnl / risk, 2) if risk > 0 else 0.0
                win = pnl > 0
                break

            if htf_hit:
                hit_htf = True
                pnl = round(abs(htf_target - entry_price), 2)
                rr = round(pnl / risk, 2) if risk > 0 else 0.0
                win = pnl > 0
                break
        else:
            close = float(frame.iloc[end]["Close"])
            if direction == "bullish":
                pnl = round(close - entry_price, 2)
            else:
                pnl = round(entry_price - close, 2)
            rr = round(pnl / risk, 2) if risk > 0 else 0.0
            win = pnl > 0

        return {
            "hit_1r_before_sl": hit_1r,
            "hit_2r_before_sl": hit_2r,
            "hit_3r_before_sl": hit_3r,
            "hit_opposite_liquidity_before_sl": hit_opposite,
            "hit_htf_supply_demand_before_sl": hit_htf,
            "realized_pnl_points": pnl,
            "realized_rr": rr,
            "win": win,
            "stopped_out": stopped_out,
        }

    @staticmethod
    def _configuration_key(
        quality: str,
        displacement: str,
        choch: bool,
        bos: bool,
        fvg_reclaimed: bool,
        location: str,
        sweep_type: str,
    ) -> tuple[str, str]:
        key = "|".join(
            [
                f"sweep={sweep_type}",
                f"quality={quality}",
                f"displacement={displacement}",
                f"choch={'Yes' if choch else 'No'}",
                f"bos={'Yes' if bos else 'No'}",
                f"fvg={'Yes' if fvg_reclaimed else 'No'}",
                f"location={location}",
            ]
        )
        label = (
            f"{sweep_type} | Quality {quality} | Displacement {displacement} | "
            f"CHOCH {'Yes' if choch else 'No'} | BOS {'Yes' if bos else 'No'} | "
            f"FVG Reclaimed {'Yes' if fvg_reclaimed else 'No'} | {location}"
        )
        return key, label

    def _analyze_sweep_trade(
        self,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        intel_frame: pd.DataFrame,
        sweep_bar: int,
        sweep_type: str,
        timeframe_label: str,
    ) -> SweepTradeabilityRecord | None:
        direction = self._expected_direction(sweep_type)
        entry_bar = self._find_post_sweep_bos(frame, sweep_bar, direction)
        if entry_bar is None:
            return None

        entry_price = float(frame.iloc[entry_bar]["Close"])
        stop, risk = self.construction_engine._structural_stop(
            frame,
            entry_bar,
            entry_price,
            direction,
        )
        opposite_target = self.construction_engine._opposite_liquidity_target(
            frame,
            entry_bar,
            entry_price,
            direction,
            risk,
        )
        htf_target = self.construction_engine._htf_supply_demand_target(
            frame,
            entry_bar,
            entry_price,
            direction,
            risk,
        )

        candle_map = self.liquidity_map_engine.evaluate_bar(frame, enriched, sweep_bar)
        event = candle_map.liquidity_event
        intelligence = self.intelligence_engine.evaluate_bar(intel_frame, sweep_bar)
        choch, bos_before, fvg_reclaimed = self._structure_flags_between(
            frame,
            sweep_bar,
            entry_bar,
            direction,
        )

        simulation = self._simulate_trade_path(
            frame,
            entry_bar,
            entry_price,
            direction,
            stop,
            risk,
            opposite_target,
            htf_target,
        )

        quality = self._quality_group(str(event.get("classification", "Unknown")))
        displacement = self._displacement_group(str(event.get("displacement_after_sweep", "None")))
        location = self._location_group(intelligence.market_location)
        config_key, config_label = self._configuration_key(
            quality,
            displacement,
            choch,
            bos_before,
            fvg_reclaimed,
            location,
            sweep_type,
        )

        return SweepTradeabilityRecord(
            sweep_timestamp=str(frame.iloc[sweep_bar]["Date"]),
            entry_timestamp=str(frame.iloc[entry_bar]["Date"]),
            timeframe=timeframe_label,
            sweep_type=sweep_type,
            trade_direction=direction,
            sweep_bar=sweep_bar,
            entry_bar=entry_bar,
            entry_price=round(entry_price, 2),
            stop_price=round(stop, 2),
            risk_points=round(risk, 2),
            opposite_liquidity_target=round(opposite_target, 2),
            htf_supply_demand_target=round(htf_target, 2),
            sweep_quality_classification=quality,
            displacement_strength=displacement,
            choch_present=choch,
            bos_present_before_entry=bos_before,
            fvg_reclaimed=fvg_reclaimed,
            market_location=location,
            hit_1r_before_sl=bool(simulation["hit_1r_before_sl"]),
            hit_2r_before_sl=bool(simulation["hit_2r_before_sl"]),
            hit_3r_before_sl=bool(simulation["hit_3r_before_sl"]),
            hit_opposite_liquidity_before_sl=bool(simulation["hit_opposite_liquidity_before_sl"]),
            hit_htf_supply_demand_before_sl=bool(simulation["hit_htf_supply_demand_before_sl"]),
            realized_pnl_points=float(simulation["realized_pnl_points"]),
            realized_rr=float(simulation["realized_rr"]),
            win=bool(simulation["win"]),
            stopped_out=bool(simulation["stopped_out"]),
            configuration_key=config_key,
            configuration_label=config_label,
        )

    def _collect_trades(self, metadata: dict[str, Any]) -> tuple[list[SweepTradeabilityRecord], int]:
        from datetime import timedelta

        end = date.fromisoformat(metadata["end_date"])
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=self.research_days)
        )

        trades: list[SweepTradeabilityRecord] = []
        sweep_count = 0

        for timeframe_label in self.timeframes:
            path = self.filter_engine._ensure_pipeline(timeframe_label, start, end)
            frame = pd.read_csv(path).reset_index(drop=True)
            enriched = self.liquidity_map_engine._attach_calendar_levels(frame)
            intel_frame = self.intelligence_engine.enrich(frame)

            for index in range(len(frame)):
                row = frame.iloc[index]
                for sweep_type, column in (
                    ("Buy Side Sweep", "Buy_Liquidity_Sweep"),
                    ("Sell Side Sweep", "Sell_Liquidity_Sweep"),
                ):
                    if not self._is_active(row.get(column)):
                        continue
                    sweep_count += 1
                    record = self._analyze_sweep_trade(
                        frame,
                        enriched,
                        intel_frame,
                        index,
                        sweep_type,
                        timeframe_label,
                    )
                    if record is not None:
                        trades.append(record)

        trades.sort(key=lambda item: pd.Timestamp(item.entry_timestamp))
        return trades, sweep_count

    @staticmethod
    def _pct_true(records: list[SweepTradeabilityRecord], field: str) -> float:
        if not records:
            return 0.0
        return round(sum(1 for record in records if getattr(record, field)) / len(records) * 100, 2)

    @staticmethod
    def _tradable_score(metrics: TradeabilityMetrics) -> float:
        pf = metrics.profit_factor or 0.0
        if pf == float("inf"):
            pf = 10.0
        return round(
            metrics.expectancy * 0.35
            + pf * 15.0
            + metrics.win_rate_pct * 0.25
            + metrics.hit_opposite_liquidity_before_sl_pct * 0.15
            + metrics.hit_2r_before_sl_pct * 0.10
            - metrics.maximum_drawdown_points * 0.01,
            4,
        )

    def _metrics_for_cohort(self, label: str, records: list[SweepTradeabilityRecord]) -> TradeabilityMetrics:
        if not records:
            return TradeabilityMetrics(
                label=label,
                trades=0,
                win_rate_pct=0.0,
                profit_factor=None,
                expectancy=0.0,
                average_rr=0.0,
                net_points=0.0,
                maximum_drawdown_points=0.0,
                hit_1r_before_sl_pct=0.0,
                hit_2r_before_sl_pct=0.0,
                hit_3r_before_sl_pct=0.0,
                hit_opposite_liquidity_before_sl_pct=0.0,
                hit_htf_supply_demand_before_sl_pct=0.0,
                tradable_score=0.0,
            )

        pnls = [record.realized_pnl_points for record in records]
        rrs = [record.realized_rr for record in records]
        wins = sum(1 for record in records if record.win)

        metrics = TradeabilityMetrics(
            label=label,
            trades=len(records),
            win_rate_pct=round(wins / len(records) * 100, 2),
            profit_factor=self._profit_factor(pnls),
            expectancy=round(mean(pnls), 2),
            average_rr=round(mean(rrs), 2),
            net_points=round(sum(pnls), 2),
            maximum_drawdown_points=self._maximum_drawdown(pnls),
            hit_1r_before_sl_pct=self._pct_true(records, "hit_1r_before_sl"),
            hit_2r_before_sl_pct=self._pct_true(records, "hit_2r_before_sl"),
            hit_3r_before_sl_pct=self._pct_true(records, "hit_3r_before_sl"),
            hit_opposite_liquidity_before_sl_pct=self._pct_true(
                records,
                "hit_opposite_liquidity_before_sl",
            ),
            hit_htf_supply_demand_before_sl_pct=self._pct_true(
                records,
                "hit_htf_supply_demand_before_sl",
            ),
            tradable_score=0.0,
        )
        metrics.tradable_score = self._tradable_score(metrics)
        return metrics

    def _breakdown(
        self,
        records: list[SweepTradeabilityRecord],
        field: str,
        grouper: Any,
    ) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[SweepTradeabilityRecord]] = defaultdict(list)
        for record in records:
            groups[grouper(getattr(record, field))].append(record)
        return {
            label: self._metrics_for_cohort(label, group).as_dict()
            for label, group in sorted(groups.items())
        }

    def _rank_configurations(
        self,
        records: list[SweepTradeabilityRecord],
    ) -> list[TradeabilityMetrics]:
        groups: dict[str, list[SweepTradeabilityRecord]] = defaultdict(list)
        labels: dict[str, str] = {}
        for record in records:
            groups[record.configuration_key].append(record)
            labels[record.configuration_key] = record.configuration_label

        metrics = [
            self._metrics_for_cohort(labels[key], group)
            for key, group in groups.items()
            if len(group) >= MIN_CONFIG_SAMPLES
        ]
        ranked = sorted(metrics, key=lambda item: (item.tradable_score, item.expectancy), reverse=True)
        for index, item in enumerate(ranked, start=1):
            item.rank = index
        return ranked

    def run(self, metadata: dict[str, Any]) -> LiquiditySweepTradeabilityReport:
        """Run liquidity sweep tradeability validation."""
        started = time.perf_counter()
        trades, sweep_count = self._collect_trades(metadata)
        if not trades:
            raise LiquiditySweepTradeabilityError(
                "No tradable sweep-to-BOS sequences found.",
            )

        overall = self._metrics_for_cohort("Overall", trades)
        ranked_configs = self._rank_configurations(trades)
        top_configs = ranked_configs[:TOP_CONFIG_COUNT]

        reachability = {
            "hit_1r_before_sl_pct": overall.hit_1r_before_sl_pct,
            "hit_2r_before_sl_pct": overall.hit_2r_before_sl_pct,
            "hit_3r_before_sl_pct": overall.hit_3r_before_sl_pct,
            "hit_opposite_liquidity_before_sl_pct": overall.hit_opposite_liquidity_before_sl_pct,
            "hit_htf_supply_demand_before_sl_pct": overall.hit_htf_supply_demand_before_sl_pct,
        }

        tradable_structures = [
            f"{item.label} (n={item.trades}, WR={item.win_rate_pct}%, exp={item.expectancy})"
            for item in ranked_configs
            if item.win_rate_pct >= 50 and item.expectancy > 0
        ][:10]

        conclusions = [
            f"Detected {sweep_count} sweeps; {len(trades)} tradable BOS Close sequences.",
            (
                f"Overall: WR {overall.win_rate_pct}%, PF {overall.profit_factor}, "
                f"expectancy {overall.expectancy}, avg RR {overall.average_rr}."
            ),
            (
                f"1R before SL: {overall.hit_1r_before_sl_pct}% | "
                f"2R: {overall.hit_2r_before_sl_pct}% | "
                f"3R: {overall.hit_3r_before_sl_pct}%."
            ),
            (
                f"Opposite liquidity before SL: {overall.hit_opposite_liquidity_before_sl_pct}% | "
                f"HTF supply/demand: {overall.hit_htf_supply_demand_before_sl_pct}%."
            ),
        ]
        if top_configs:
            leader = top_configs[0]
            conclusions.append(
                f"Top configuration: {leader.label} "
                f"(score {leader.tradable_score}, n={leader.trades})."
            )
        conclusions.append(f"Tradable structure profiles: {len(tradable_structures)}.")

        return LiquiditySweepTradeabilityReport(
            symbol=metadata.get("symbol", self.symbol),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            timeframes_analyzed=list(self.timeframes),
            entry_method="BOS Close Entry",
            stop_loss_model="Structural Swing SL",
            total_sweeps_detected=sweep_count,
            tradable_trades=len(trades),
            untradable_sweeps_no_bos=sweep_count - len(trades),
            overall_metrics=overall.as_dict(),
            reachability_summary=reachability,
            by_sweep_quality=self._breakdown(trades, "sweep_quality_classification", lambda x: x),
            by_displacement=self._breakdown(trades, "displacement_strength", lambda x: x),
            by_choch_present=self._breakdown(
                trades,
                "choch_present",
                lambda x: "Yes" if x else "No",
            ),
            by_bos_present=self._breakdown(
                trades,
                "bos_present_before_entry",
                lambda x: "Yes" if x else "No",
            ),
            by_fvg_reclaimed=self._breakdown(
                trades,
                "fvg_reclaimed",
                lambda x: "Yes" if x else "No",
            ),
            by_market_location=self._breakdown(trades, "market_location", lambda x: x),
            top_20_sweep_configurations=[item.as_dict() for item in top_configs],
            trade_records=[record.as_dict() for record in trades],
            tradable_structures_summary=tradable_structures,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_liquidity_sweep_tradeability_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> LiquiditySweepTradeabilityReport:
    """Run liquidity sweep tradeability validation and export JSON report."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise LiquiditySweepTradeabilityError(
            f"Filter research report not found: {metadata_path}",
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = LiquiditySweepTradeabilityResearch(
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
        "Liquidity sweep tradeability validation completed: trades=%s",
        report.tradable_trades,
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_liquidity_sweep_tradeability_report()
        print("Liquidity Sweep Tradeability Validation Summary")
        print(f"Sweeps: {report.total_sweeps_detected} | Tradable: {report.tradable_trades}")
        overall = report.overall_metrics
        print(
            f"Overall WR {overall['win_rate_pct']}% | PF {overall['profit_factor']} | "
            f"Exp {overall['expectancy']}"
        )
        print("Reachability before SL:")
        for key, value in report.reachability_summary.items():
            print(f"  {key}: {value}%")
        if report.top_20_sweep_configurations:
            top = report.top_20_sweep_configurations[0]
            print(f"Top config: {top['label']} (score={top['tradable_score']})")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except LiquiditySweepTradeabilityError as exc:
        logger.error("Liquidity sweep tradeability error: %s", exc)
        print(f"Liquidity sweep tradeability error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected liquidity sweep tradeability failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
